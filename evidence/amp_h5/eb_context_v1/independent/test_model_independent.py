"""Independent synthetic-only checks; does not read natural input files."""
import importlib.util
import json
import math
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eb_model_independent_target', ROOT/'implementation'/'model.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

ROWS = [dict(sequence='AC', terminal='Free'), dict(sequence='AA', terminal='Free'),
        dict(sequence='CD', terminal='Amide'), dict(sequence='C', terminal='Amide')]

class IndependentSyntheticTests(unittest.TestCase):
    def test_hand_counts_formula(self):
        model = M.fit(ROWS, dict(max_depth=1, mode='partial', alpha=.5, shrinkage=10, min_child_count=100))
        # Natural transitions: Free A3 C1 EOS2; Amide C2 D1 EOS2.
        self.assertEqual(model.fit_summary['transitions'], 11)
        counts = model.tree['counts']
        self.assertEqual(sum(counts['Free']), 6)
        self.assertEqual(sum(counts['Amide']), 5)
        for terminal, total, a_count in [('Free',6,3), ('Amide',5,0)]:
            global_p_a = (3+.5)/(11+21*.5)
            expected = (a_count+10*global_p_a+.5)/(total+10+21*.5)
            self.assertAlmostEqual(model.probs('',terminal)[0], expected, places=14)
        pooled = model.with_emission('pooled')
        self.assertEqual(pooled.probs('', 'Free'), pooled.probs('', 'Amide'))

    def test_normalization_unseen_and_limits(self):
        for depth in [1,2,3]:
            for fit in [M.fit, M.fit_ngram]:
                for mode in ['pooled','separate','partial']:
                    model = fit(ROWS, dict(max_depth=depth,mode=mode))
                    for terminal in M.TERMINALS:
                        for prefix in ['', 'W', 'WY', 'WYQ', 'A'*47]:
                            probs=model.probs(prefix,terminal)
                            self.assertEqual(len(probs),21)
                            self.assertTrue(all(math.isfinite(p) and 0<p<1 for p in probs))
                            self.assertAlmostEqual(sum(probs),1,places=13)
            model=M.fit(ROWS,dict(max_depth=depth))
            zero=model.with_emission('partial',shrinkage=0)
            separate=model.with_emission('separate')
            huge=model.with_emission('partial',shrinkage=1e14)
            pooled=model.with_emission('pooled')
            for terminal in M.TERMINALS:
                self.assertEqual(zero.probs('AC',terminal),separate.probs('AC',terminal))
                self.assertLess(max(abs(a-b) for a,b in zip(huge.probs('AC',terminal),pooled.probs('AC',terminal))),1e-12)

    def test_bos_no_crosssequence_and_cap(self):
        rows=[dict(sequence='A'*48,terminal='Free'),dict(sequence='',terminal='Amide'),dict(sequence='CD',terminal='Amide')]
        events=list(M.transitions('CD','Amide',3))
        self.assertEqual(events[0][0],('<BOS>',)*3)
        self.assertEqual(events[1][0],('C','<BOS>','<BOS>'))
        self.assertEqual(events[2][0],('D','C','<BOS>'))
        self.assertEqual(len(list(M.transitions('A'*48,'Free',3))),48)
        for fit in [M.fit,M.fit_ngram]:
            model=fit(rows,dict(max_depth=3,min_child_count=100))
            self.assertEqual(model.fit_summary['transitions'],52)
            for seq,terminal in [('', 'Free'),('CD','Amide'),('A'*47,'Free'),('A'*48,'Free')]:
                expected=sum(math.log(model.probs(seq[:i],terminal)[M.AA.index(a)]) for i,a in enumerate(seq))
                if len(seq)<48: expected+=math.log(model.probs(seq,terminal)[-1])
                self.assertAlmostEqual(model.logscore(seq,terminal),expected,places=12)
            with self.assertRaises(ValueError): model.probs('A'*48,'Free')

    def test_tree_partition_counts_and_terminal_label_invariance(self):
        rows=[dict(sequence=s*5,terminal=t) for s in ['AC','CA','DE','ED'] for t in M.TERMINALS for _ in range(5)]
        model=M.fit(rows,dict(max_depth=3,min_child_count=2,complexity_penalty=0))
        swapped=[dict(r,terminal='Amide' if r['terminal']=='Free' else 'Free') for r in rows]
        self.assertEqual(model.structure(),M.fit(swapped,model.config).structure())
        def check(node):
            if not node['children']: return
            symbols=[s for c in node['children'] for s in c['symbols']]
            self.assertEqual(sorted(symbols),sorted(M.CONTEXTS))
            for terminal in M.TERMINALS:
                self.assertEqual(node['counts'][terminal],[sum(c['counts'][terminal][a] for c in node['children']) for a in range(21)])
            for c in node['children']: check(c)
        check(model.tree)

    def test_fit_isolation_and_serialization(self):
        train=ROWS[:2]
        model=M.fit(train,dict(max_depth=1))
        before=model.digest()
        model.logscore('WYQ','Amide')
        self.assertEqual(before,model.digest())
        self.assertEqual(model.fit_summary['terminal_sequences']['Amide'],0)
        self.assertEqual(sum(model.tree['counts']['Amide']),0)
        restored=M.GroupedContextModel.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(before,restored.digest())
        self.assertEqual(restored.probs('W','Amide'),model.probs('W','Amide'))
        self.assertNotEqual(before,M.fit(ROWS,dict(max_depth=1)).digest())

if __name__=='__main__': unittest.main(verbosity=2)
