"""Runner contract tests with synthetic inputs only; no model fitting/sampling."""
import hashlib
import importlib.util
import math
import pathlib
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner_independent_target',ROOT/'run.py')
R=importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.ar.configure_cpu()

def row(sequence,terminal):
    return dict(sequence=sequence,terminal_state=terminal,**{'class':str(R.TERMINALS.index(terminal))},
                molecule_sha256=hashlib.sha256((sequence+terminal).encode()).hexdigest(),
                sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())

class UniformAR:
    def eval(self): return self
    def __call__(self,inputs,labels): return R.torch.zeros((*inputs.shape,21)),None

class UniformEB:
    def probs(self,prefix,terminal): return (1/21,)*21

class RunnerTests(unittest.TestCase):
    def test_full_score_boundary_and_two_apis(self):
        rows=[row('A','FREE'),row('AC','AMD'),row('A'*47,'FREE'),row('A'*48,'AMD')]
        for model,is_ar in [(UniformAR(),True),(UniformEB(),False)]:
            scores=R.score(model,rows,is_ar)
            for s,r in zip(scores,rows):
                count=min(len(r['sequence'])+1,48)
                self.assertEqual(s['observed_tokens'],count)
                self.assertAlmostEqual(s['nll'],count*math.log(21),places=4)
                self.assertAlmostEqual(s['eos_nll'],math.log(21) if len(r['sequence'])<48 else 0,places=6)
                self.assertEqual(len(s['token_nlls']),count)

    def test_terminal_not_row_or_fold_weight(self):
        scores=[dict(terminal='FREE',nll=2,observed_tokens=2,aa_nll=1,eos_nll=1),
                dict(terminal='FREE',nll=6,observed_tokens=3,aa_nll=5,eos_nll=1),
                dict(terminal='AMD',nll=10,observed_tokens=5,aa_nll=9,eos_nll=1)]
        summary=R.aggregate(scores)
        self.assertEqual(summary['equal_terminal_sequence_nll'],7)
        self.assertEqual(summary['by_terminal']['FREE']['token_nll'],8/5)

    def test_stop_on_tie_and_one_terminal_harm(self):
        methods=['pooled','separate','partial','ngram','ar_cv','ar_historical']
        def summary(a,b): return dict(equal_terminal_sequence_nll=(a+b)/2,by_terminal={'FREE':dict(sequence_nll=a),'AMD':dict(sequence_nll=b)})
        summaries={m:summary(10,10) for m in methods}
        generated=[dict(method=m,terminal=t,sequence='ACDEFG') for m in methods for t in R.TERMINALS]
        cv={m:v for m,v in summaries.items() if m!='ar_historical'}
        result=R.decision(cv,summaries,generated,[])
        self.assertEqual(result['status'],'NOT_SUPPORTED')
        self.assertFalse(result['further_tuning_authorized'])
        summaries['partial']=summary(8,11)
        cv['partial']=summaries['partial']
        result=R.decision(cv,summaries,generated,[])
        self.assertTrue(any('worsens AMD' in s for s in result['failure_reasons']))
        summaries['partial']=summary(8,9);cv['partial']=summaries['partial']
        self.assertEqual(R.decision(cv,summaries,generated,[])['status'],'DEVELOPMENT_SUPPORT_ONLY')

if __name__=='__main__': unittest.main(verbosity=2)
