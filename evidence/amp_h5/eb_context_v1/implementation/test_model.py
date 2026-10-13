"""Synthetic-only tests; this module has no natural-data paths."""
import copy
import json
import math
import unittest

from model import (AA, BOS, CONTEXTS, INDEX, K, OUTPUTS, TERMINALS,
                   GroupedContextModel, NGramModel, fit, fit_ngram, transitions)

ROWS = [dict(sequence="AAAK", terminal="Free"), dict(sequence="AAAR", terminal="Free"),
        dict(sequence="KKKA", terminal="Amide"), dict(sequence="KKKR", terminal="Amide")]


class ModelTests(unittest.TestCase):
    def test_transitions_and_boundary(self):
        events = list(transitions("AC", "Free", 3))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0], ((BOS, BOS, BOS), "Free", INDEX["A"]))
        self.assertEqual(events[1][0], ("A", BOS, BOS))
        self.assertEqual(events[2], (("C", "A", BOS), "Free", K - 1))
        self.assertEqual(len(list(transitions("A" * 48, "Free", 3))), 48)
        self.assertFalse(any(e[2] == K - 1 for e in transitions("A" * 48, "Free", 3)))
        self.assertEqual(len(list(transitions("A" * 47, "Free", 3))), 48)
        self.assertEqual(len(list(transitions("", "Free", 3))), 1)

    def test_all_modes_normalization_and_unseen(self):
        for fitter in (fit, fit_ngram):
            for mode in ("pooled", "separate", "partial"):
                model = fitter(ROWS, dict(mode=mode, max_depth=3))
                for prefix in ("", "A", "AA", "WYY", "Z".replace("Z", "W") * 47):
                    for terminal in TERMINALS:
                        probs = model.probs(prefix, terminal)
                        self.assertEqual(len(probs), 21)
                        self.assertTrue(all(0 < p < 1 for p in probs))
                        self.assertAlmostEqual(sum(probs), 1.0, places=14)

    def test_counts_and_no_cross_sequence_edges(self):
        rows = ROWS + [dict(sequence="", terminal="Free"), dict(sequence="A" * 48, terminal="Amide")]
        model = fit(rows, dict(min_child_count=1, max_depth=3))
        totals = {c: [0] * K for c in TERMINALS}
        leaves = []
        def visit(node):
            if node["children"]:
                self.assertEqual(sorted(s for ch in node["children"] for s in ch["symbols"]), sorted(CONTEXTS))
                for c in TERMINALS:
                    for a in range(K):
                        self.assertEqual(node["counts"][c][a], sum(ch["counts"][c][a] for ch in node["children"]))
                for child in node["children"]:
                    visit(child)
            else:
                leaves.append(node)
        visit(model.tree)
        for row in rows:
            for _, c, a in transitions(row["sequence"], row["terminal"], 3):
                totals[c][a] += 1
        self.assertEqual(model.tree["counts"], totals)
        for c in TERMINALS:
            self.assertEqual([sum(n["counts"][c][a] for n in leaves) for a in range(K)], totals[c])
        self.assertEqual(sum(totals[c][-1] for c in TERMINALS), len(rows) - 1)
        self.assertEqual(model.fit_summary["transitions"], 20 + 1 + 48)
        ngram = fit_ngram(rows, dict(max_depth=3, mode="separate"))
        self.assertEqual(ngram.tables[""], totals)
        # Four nonempty short sequences and one length48 sequence start at BOS;
        # the empty sequence contributes a BOS->EOS observation, not an AA.
        self.assertEqual(sum(sum(x) for x in ngram.tables[BOS + "|" + BOS + "|" + BOS].values()), len(rows))

    def test_logscore_complete_sequence(self):
        for fitter in (fit, fit_ngram):
            model = fitter(ROWS, dict(mode="partial"))
            for seq in ("", "K", "AAAK", "W" * 47, "W" * 48):
                expected = sum(math.log(model.probs(seq[:i], "Free")[INDEX[a]]) for i, a in enumerate(seq))
                if len(seq) < 48:
                    expected += math.log(model.probs(seq, "Free")[-1])
                self.assertAlmostEqual(model.logscore(seq, "Free"), expected, places=12)
            with self.assertRaises(ValueError):
                model.probs("A" * 48, "Free")

    def test_partial_formula_and_limits(self):
        base = fit(ROWS, dict(min_child_count=1, max_depth=3))
        separate = base.with_emission("separate")
        zero = base.with_emission("partial", shrinkage=0)
        infinity = base.with_emission("partial", shrinkage=1e14)
        pooled = base.with_emission("pooled")
        for seq in ("", "A", "KK", "WYW"):
            for terminal in TERMINALS:
                self.assertEqual(separate.probs(seq, terminal), zero.probs(seq, terminal))
                for p, q in zip(infinity.probs(seq, terminal), pooled.probs(seq, terminal)):
                    self.assertAlmostEqual(p, q, places=11)
                count = base.leaf(seq)["counts"][terminal]
                gp = pooled.probs(seq, terminal)
                expected = tuple((n + 16 * g + .5) / (sum(count) + 16 + 21 * .5) for n, g in zip(count, gp))
                self.assertEqual(base.probs(seq, terminal), expected)

    def test_structure_shared_and_order_invariant(self):
        rows = ROWS * 30
        base = fit(rows, dict(max_depth=3))
        self.assertTrue(base.tree["children"])
        for mode in ("pooled", "separate", "partial"):
            other = fit(list(reversed(rows)), dict(mode=mode, max_depth=3, alpha=.1))
            self.assertEqual(base.structure(), other.structure())
        flipped = [dict(sequence=r["sequence"], terminal="Amide" if r["terminal"] == "Free" else "Free") for r in rows]
        self.assertEqual(base.structure(), fit(flipped, dict(max_depth=3)).structure())
        clone = base.reconfigure("pooled", alpha=.1)
        clone.tree["counts"]["Free"][0] += 1
        self.assertNotEqual(clone.tree["counts"], base.tree["counts"])

    def test_frozen_heldout_serialization_and_sampling(self):
        for fitter, cls in ((fit, GroupedContextModel), (fit_ngram, NGramModel)):
            model = fitter(ROWS, dict(mode="separate", max_depth=3))
            before = model.digest()
            heldout = [dict(sequence="WWYYYY", terminal="Amide")]
            model.logscore(heldout[0]["sequence"], heldout[0]["terminal"])
            generated = model.sample(123, "Free", 100)
            self.assertEqual(before, model.digest())
            self.assertTrue(all(len(s) <= 48 and set(s) <= set(AA) for s in generated))
            restored = cls.from_dict(json.loads(json.dumps(model.to_dict())))
            self.assertEqual(before, restored.digest())
            self.assertEqual(generated, restored.sample(123, "Free", 100))
            self.assertNotEqual(before, fitter(ROWS + heldout, dict(mode="separate", max_depth=3)).digest())
            # Caller rows are never retained or modified.
            self.assertEqual(ROWS[0]["sequence"], "AAAK")

    def test_ngram_backoff(self):
        model = fit_ngram([dict(sequence="AA", terminal="Free")], dict(mode="separate", max_depth=3))
        self.assertEqual(model.probs("WWW", "Free"), tuple((n + .5) / (3 + 21 * .5) for n in model.tables[""]["Free"]))
        self.assertEqual(model.probs("WWW", "Amide"), (1 / 21,) * 21)

    def test_invalid_inputs(self):
        for seq in ("B", "A" * 49, "a"):
            with self.assertRaises(ValueError):
                fit([dict(sequence=seq, terminal="Free")], {})
        with self.assertRaises(ValueError):
            fit([dict(sequence="A", terminal="unknown")], {})
        with self.assertRaises(ValueError):
            fit([], {})
        with self.assertRaises(ValueError):
            fit(ROWS, dict(alpha=0))
        with self.assertRaises(ValueError):
            fit(ROWS, dict(shrinkage=float("nan")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
