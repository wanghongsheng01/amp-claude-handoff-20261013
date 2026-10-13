"""Synthetic train-only grouping audit; never reads peptide inputs."""
import hashlib
import importlib.util
import pathlib
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('grouping_target',ROOT/'grouping.py')
G=importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

def rows(seqs):
    return [dict(sequence=s,terminal_state='FREE' if i%2==0 else 'AMD',molecule_sha256=hashlib.sha256((s+str(i)).encode()).hexdigest()) for i,s in enumerate(seqs)]

class GroupingTests(unittest.TestCase):
    def test_transitive_similarity_and_threshold(self):
        r=rows(['A'*10,'A'*7+'C'*3,'A'*4+'C'*6,'W'*10])
        report=G.grouped_folds(r,guard=False)
        self.assertEqual(len(report['components']),2)
        self.assertEqual(report['assignments'][0],report['assignments'][2])
        self.assertNotIn([0,2],report['edges'])
        self.assertIn([0,1],report['edges'])
        self.assertIn([1,2],report['edges'])
        self.assertEqual(sorted(i for c in report['components'] for i in c['indices']),list(range(4)))
        self.assertFalse(report['source_component_ids_used'])

    def test_guard_never_silently_adjusts(self):
        r=rows(['A'*10]*20)
        with self.assertRaises(G.GroupingBlocked) as caught: G.grouped_folds(r)
        self.assertTrue(any('half' in s for s in caught.exception.report['blockers']))
        self.assertEqual(caught.exception.report['component_count'],1)
        with self.assertRaises(G.GroupingBlocked): G.grouped_folds(rows(['AC','WY']))

    def test_row_permutation_preserves_group_ids_and_fold(self):
        r=rows(['A'*10,'A'*7+'C'*3,'A'*4+'C'*6,'W'*10,'G'*10,'E'*10])
        def identity_map(data):
            report=G.grouped_folds(data,guard=False)
            return {data[i]['molecule_sha256']:(group['group_id'],group['fold']) for group in report['components'] for i in group['indices']}
        self.assertEqual(identity_map(r),identity_map(list(reversed(r))))

    def test_distance_known_cases(self):
        self.assertEqual(G.distance('','ACD'),3)
        self.assertEqual(G.distance('ACD','AD'),1)
        self.assertEqual(G.distance('ACD','ADC'),2)
        self.assertEqual(G.distance('ACDEFG','WACDEFG'),1)

if __name__=='__main__': unittest.main(verbosity=2)
