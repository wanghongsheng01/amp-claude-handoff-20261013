"""Independent grouped-context EB prototype; NOT an official PBCT reproduction.

Pure Python, no filesystem reads, network, fitting on import, or external dependencies.
Frozen predictive distributions score every residue and EOS iff length < 48.
"""
import copy
import hashlib
import json
import math
import random
from collections import defaultdict

AA = "ACDEFGHIKLMNPQRSTVWY"
OUTPUTS = tuple(AA) + ("<EOS>",)
BOS = "<BOS>"
CONTEXTS = tuple(AA) + (BOS,)
TERMINALS = ("Free", "Amide")
MAX_LENGTH = 48
K = len(OUTPUTS)
INDEX = {x: i for i, x in enumerate(OUTPUTS)}


def _sequence(seq):
    if not isinstance(seq, str) or len(seq) > MAX_LENGTH or any(x not in AA for x in seq):
        raise ValueError("sequence must contain only canonical AA and have length 0..48")


def _terminal(terminal):
    if terminal not in TERMINALS:
        raise ValueError("terminal must be Free or Amide")


def _context(prefix, depth):
    if depth == 0:
        return ()
    return tuple(reversed(prefix[-depth:])) + (BOS,) * max(0, depth - len(prefix))


def transitions(seq, terminal, depth):
    """No cross-sequence transitions; BOS left-padding, no EOS beyond cap."""
    _sequence(seq)
    _terminal(terminal)
    for i, aa in enumerate(seq):
        yield (_context(seq[:i], depth), terminal, INDEX[aa])
    if len(seq) < MAX_LENGTH:
        yield (_context(seq, depth), terminal, INDEX["<EOS>"])


def _rows(rows):
    out = [(r["sequence"], r["terminal"]) for r in rows]
    if not out:
        raise ValueError("nonempty fit rows required")
    for seq, terminal in out:
        _sequence(seq)
        _terminal(terminal)
    return sorted(out)


def _counts(events):
    result = {c: [0] * K for c in TERMINALS}
    for _, c, a in events:
        result[c][a] += 1
    return result


def _pooled(counts):
    return [sum(counts[c][a] for c in TERMINALS) for a in range(K)]


def _marginal(counts, alpha):
    """Integrated categorical sequence likelihood; NO multinomial coefficient."""
    return (math.lgamma(K * alpha) - math.lgamma(sum(counts) + K * alpha)
            + sum(math.lgamma(n + alpha) - math.lgamma(alpha) for n in counts))


def _partition(events, depth, config):
    """Full deterministic greedy merge path, selected under min-count and penalty.

    Rare/absent symbols start in one fixed bucket; others start as singletons.
    Every context symbol occurs in exactly one cluster, including unseen symbols.
    """
    by_symbol = {s: [0] * K for s in CONTEXTS}
    for context, _, a in events:
        by_symbol[context[depth]][a] += 1
    minimum = config["min_child_count"]
    rare = tuple(s for s in CONTEXTS if sum(by_symbol[s]) < minimum)
    groups = [((s,), by_symbol[s]) for s in CONTEXTS if s not in rare]
    if rare:
        groups.append((tuple(sorted(rare)), [sum(by_symbol[s][a] for s in rare) for a in range(K)]))
    groups.sort(key=lambda g: g[0])
    alpha = config["tree_alpha"]
    penalty = config["complexity_penalty"]
    parent = _pooled(_counts(events))
    best_score = _marginal(parent, alpha)
    best = [tuple(sorted(CONTEXTS))]
    while len(groups) > 1:
        score = sum(_marginal(g[1], alpha) for g in groups) - penalty * (len(groups) - 1)
        if all(sum(g[1]) >= minimum for g in groups) and score > best_score + 1e-12:
            best_score = score
            best = [g[0] for g in groups]
        candidates = []
        scores = [_marginal(g[1], alpha) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                merged = [a + b for a, b in zip(groups[i][1], groups[j][1])]
                gain = _marginal(merged, alpha) - scores[i] - scores[j] + penalty
                # Explicit deterministic tie-break by cluster names.
                candidates.append((-gain, groups[i][0], groups[j][0], i, j, merged))
        _, _, _, i, j, merged = min(candidates)
        symbols = tuple(sorted(groups[i][0] + groups[j][0]))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [(symbols, merged)]
        groups.sort(key=lambda g: g[0])
    return sorted(best)


def _build(events, depth, config):
    node = {"depth": depth, "counts": _counts(events), "children": []}
    if depth >= config["max_depth"] or len(events) < 2 * config["min_child_count"]:
        return node
    partition = _partition(events, depth, config)
    if len(partition) == 1:
        return node
    for symbols in partition:
        subset = [e for e in events if e[0][depth] in symbols]
        child = _build(subset, depth + 1, config)
        child["symbols"] = list(symbols)
        node["children"].append(child)
    return node


def _config(config, ngram=False):
    base = dict(max_depth=2, mode="partial", alpha=0.5, shrinkage=16.0,
                tree_alpha=0.5, complexity_penalty=1.0, min_child_count=8)
    unknown = set(config) - set(base)
    if unknown:
        raise ValueError("unknown configuration keys: " + str(sorted(unknown)))
    base.update(config)
    if base["max_depth"] not in (1, 2, 3) or isinstance(base["max_depth"], bool):
        raise ValueError("max_depth must be 1, 2, or 3")
    if base["mode"] not in ("pooled", "separate", "partial"):
        raise ValueError("invalid emission mode")
    for key in ("alpha", "tree_alpha"):
        if not math.isfinite(base[key]) or base[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    for key in ("shrinkage", "complexity_penalty"):
        if not math.isfinite(base[key]) or base[key] < 0:
            raise ValueError(key + " must be finite and nonnegative")
    if not isinstance(base["min_child_count"], int) or base["min_child_count"] < 1:
        raise ValueError("min_child_count must be positive integer")
    return base


def _emission(counts, terminal, config):
    alpha = config["alpha"]
    global_n = _pooled(counts)
    global_p = [(n + alpha) / (sum(global_n) + K * alpha) for n in global_n]
    if config["mode"] == "pooled":
        return tuple(global_p)
    local = counts[terminal]
    lam = config["shrinkage"] if config["mode"] == "partial" else 0.0
    den = sum(local) + lam + K * alpha
    return tuple((n + lam * p + alpha) / den for n, p in zip(local, global_p))


class PredictionMixin:
    def logscore(self, seq, terminal):
        _sequence(seq)
        _terminal(terminal)
        value = sum(math.log(self.probs(seq[:i], terminal)[INDEX[a]]) for i, a in enumerate(seq))
        if len(seq) < MAX_LENGTH:
            value += math.log(self.probs(seq, terminal)[-1])
        return value

    def sample(self, seed, terminal, requests):
        _terminal(terminal)
        if not isinstance(requests, int) or requests < 0:
            raise ValueError("requests must be nonnegative integer")
        rng = random.Random(seed)
        samples = []
        for _ in range(requests):
            seq = ""
            for _ in range(MAX_LENGTH):
                probs = self.probs(seq, terminal)
                threshold = rng.random()
                cumulative = 0.0
                a = K - 1
                for i, prob in enumerate(probs):
                    cumulative += prob
                    if threshold < cumulative:
                        a = i
                        break
                if a == K - 1:
                    break
                seq += OUTPUTS[a]
            samples.append(seq)
        return samples

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GroupedContextModel(PredictionMixin):
    def __init__(self, config, tree, fit_summary):
        self.config = copy.deepcopy(config)
        self.tree = copy.deepcopy(tree)
        self.fit_summary = copy.deepcopy(fit_summary)

    def leaf(self, prefix):
        _sequence(prefix)
        if len(prefix) >= MAX_LENGTH:
            raise ValueError("no next-symbol prediction after max48 truncation")
        context = _context(prefix, self.config["max_depth"])
        node = self.tree
        while node["children"]:
            symbol = context[node["depth"]]
            node = next(ch for ch in node["children"] if symbol in ch["symbols"])
        return node

    def probs(self, prefix, terminal):
        _terminal(terminal)
        return _emission(self.leaf(prefix)["counts"], terminal, self.config)

    def structure(self):
        def stripped(node):
            return {"depth": node["depth"], "symbols": node.get("symbols", []),
                    "children": [stripped(n) for n in node["children"]]}
        return stripped(self.tree)

    def with_emission(self, mode, alpha=None, shrinkage=None):
        config = dict(self.config, mode=mode)
        if alpha is not None:
            config["alpha"] = alpha
        if shrinkage is not None:
            config["shrinkage"] = shrinkage
        return GroupedContextModel(_config(config), self.tree, self.fit_summary)

    def reconfigure(self, mode, alpha=None, shrinkage=None):
        return self.with_emission(mode, alpha=alpha, shrinkage=shrinkage)

    def to_dict(self):
        return copy.deepcopy(dict(schema=1, kind="independent_grouped_context_eb", config=self.config,
                                  tree=self.tree, fit_summary=self.fit_summary))

    @classmethod
    def from_dict(cls, payload):
        if payload["schema"] != 1 or payload["kind"] != "independent_grouped_context_eb":
            raise ValueError("unsupported model schema")
        return cls(_config(payload["config"]), payload["tree"], payload["fit_summary"])


def fit(rows, config):
    config = _config(config)
    sequences = _rows(rows)
    events = [e for seq, terminal in sequences for e in transitions(seq, terminal, config["max_depth"])]
    tree = _build(events, 0, config)
    summary = dict(sequences=len(sequences), transitions=len(events),
                   terminal_sequences={c: sum(t == c for _, t in sequences) for c in TERMINALS},
                   fit_rows_sha256=hashlib.sha256(json.dumps(sequences, separators=(",", ":")).encode()).hexdigest())
    return GroupedContextModel(config, tree, summary)


class NGramModel(PredictionMixin):
    """Add-alpha terminal-conditioned ngram; longest seen context backoff.

    Backoff checks terminal-specific counts for separate mode, pooled counts for
    pooled mode; partial uses pooled support and the same EB formula as trees.
    All suffix lengths, including empty, have fit-only count tables.
    """
    def __init__(self, config, tables, fit_summary):
        self.config = copy.deepcopy(config)
        self.tables = copy.deepcopy(tables)
        self.fit_summary = copy.deepcopy(fit_summary)

    def probs(self, prefix, terminal):
        _sequence(prefix)
        _terminal(terminal)
        if len(prefix) >= MAX_LENGTH:
            raise ValueError("no next-symbol prediction after max48 truncation")
        for depth in range(self.config["max_depth"], -1, -1):
            key = "|".join(_context(prefix, depth)) if depth else ""
            counts = self.tables.get(key)
            if counts is not None:
                support = sum(counts[terminal]) if self.config["mode"] == "separate" else sum(_pooled(counts))
                if support or depth == 0:
                    return _emission(counts, terminal, self.config)
        raise AssertionError("missing root count table")

    def to_dict(self):
        return copy.deepcopy(dict(schema=1, kind="smoothed_ngram", config=self.config,
                                  tables=self.tables, fit_summary=self.fit_summary))

    @classmethod
    def from_dict(cls, payload):
        if payload["schema"] != 1 or payload["kind"] != "smoothed_ngram":
            raise ValueError("unsupported model schema")
        return cls(_config(payload["config"]), payload["tables"], payload["fit_summary"])


def fit_ngram(rows, config):
    config = _config(config)
    sequences = _rows(rows)
    tables = defaultdict(lambda: {c: [0] * K for c in TERMINALS})
    transitions_total = 0
    for seq, terminal in sequences:
        for context, c, a in transitions(seq, terminal, config["max_depth"]):
            transitions_total += 1
            for depth in range(config["max_depth"] + 1):
                tables["|".join(context[:depth])][c][a] += 1
    summary = dict(sequences=len(sequences), transitions=transitions_total,
                   terminal_sequences={c: sum(t == c for _, t in sequences) for c in TERMINALS},
                   fit_rows_sha256=hashlib.sha256(json.dumps(sequences, separators=(",", ":")).encode()).hexdigest())
    return NGramModel(config, dict(tables), summary)
