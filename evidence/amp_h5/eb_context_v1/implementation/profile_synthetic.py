"""Bounded synthetic-only throughput check, no natural data or model inputs."""
import json
import random
import time
from model import AA, fit, fit_ngram


def main():
    rng = random.Random(8675309)
    rows = []
    for i in range(279):
        length = rng.randrange(16, 41)
        # Motif repetition gives nontrivial contexts; this is not an AMP simulator.
        motif = "AKRL" if i % 2 else "GASV"
        seq = "".join(motif[j % 4] if rng.random() < .7 else rng.choice(AA) for j in range(length))
        rows.append(dict(sequence=seq, terminal="Free" if i % 2 else "Amide"))
    output = {"kind": "synthetic_only_not_natural_throughput_guarantee", "rows": len(rows), "seed": 8675309, "runs": []}
    for depth in (1, 2, 3):
        start = time.perf_counter()
        model = fit(rows, dict(max_depth=depth))
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for row in rows[:41]:
            model.logscore(row["sequence"], row["terminal"])
        score_seconds = time.perf_counter() - start
        def leaves(node):
            return sum(leaves(ch) for ch in node["children"]) if node["children"] else 1
        output["runs"].append(dict(kind="tree", depth=depth, fit_seconds=fit_seconds,
            score41_seconds=score_seconds, leaves=leaves(model.tree), transitions=model.fit_summary["transitions"]))
        start = time.perf_counter()
        ng = fit_ngram(rows, dict(mode="separate", max_depth=depth))
        output["runs"].append(dict(kind="ngram", depth=depth, fit_seconds=time.perf_counter() - start,
                                   tables=len(ng.tables)))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
