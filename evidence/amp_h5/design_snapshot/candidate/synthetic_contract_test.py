"""Stdlib-only tests, no real records, Torch imports, models, fit or inference."""
import ast
import itertools
import json
import math
from pathlib import Path
from objective_oracle import targets, losses, joint_decode, generation_request

checks=[]
def check(name, value):
    if not value: raise AssertionError(name)
    checks.append(name)

check('target original AA order and PAD', targets(['AR'])[0] == [0,1]+[20]*46)
check('48 target no PAD', targets(['A'*48])[0] == [0]*48)
for seq in ['', 'A'*49, 'AX']:
    try: targets([seq])
    except ValueError: check('reject target '+repr(seq),True)
    else: raise AssertionError(seq)
scores=[[[.2,.7,-.4],[1.,-.3,.1],[-.3,.2,1.2]]]
truth=[[1,0,2]];density=[[[.1,-.2,.03],[.4,.2,-.1]]]
a=losses(scores,truth,density);b=losses(scores*2,truth*2,density*2)
for key in ['mse_total','ce_total','kl_per_position','kl_per_record']:
    check('batch replication invariant '+key,math.isclose(a[key],b[key],abs_tol=1e-15))
check('legacy batch dependence visible',math.isclose(2*a['legacy_kl_sum_over_channels'],b['legacy_kl_sum_over_channels']))
check('class sum not implicit mean',math.isclose(a['half_onehot_square_sum'],.5*sum((v-float(i==y))**2 for row,y in zip(scores[0],truth[0]) for i,v in enumerate(row))))
shift=losses([[[v+5 for v in row] for row in scores[0]]],truth,density)
check('CE common logit shift invariant',math.isclose(a['ce_sum'],shift['ce_sum'],abs_tol=1e-14))
check('raw MSE is not softmax Brier',not math.isclose(a['half_onehot_square_sum'],shift['half_onehot_square_sum']))
negative=losses(scores,truth,[[[-.1]*3,[-.1]*3]])
check('negative sampled KL retained',negative['kl_per_position']<0)
for s in [[[0,0,0]]*3,[[2,0,0]]*3,[[0,0,2]]*3,scores[0]]:
    decoded=joint_decode(s);legal=[]
    for length in range(4):
        for prefix in itertools.product(range(2),repeat=length):
            y=list(prefix)+[2]*(3-length)
            sq=.5*sum((v-float(k==y[t]))**2 for t,row in enumerate(s) for k,v in enumerate(row))
            score=sum(s[t][y[t]] for t in range(3))
            legal.append((length,y,sq,score))
    minimum=min(x[2] for x in legal);maximum=max(x[3] for x in legal)
    selected=next(x for x in legal if x[1]==decoded['tokens'])
    check('J nearest legal onehot '+str(len(checks)),math.isclose(selected[2],minimum))
    check('J max legal categorical score '+str(len(checks)),math.isclose(selected[3],maximum))
check('J shortest exact tie empty',joint_decode([[0,0,0]]*3)['length']==0)
check('J full width allowed',joint_decode([[2,0,0]]*3)['length']==3)
check('J lowest AA tie',joint_decode([[2,2,0]]*3)['tokens']==[0,0,0])
for c in (0,1):
    req=generation_request(c,0,522)
    check('terminal-only request '+str(c),set(req)=={'class','terminal_state','sample_id','seed'})
source=Path(__file__).with_name('fvae21_adapter.py')
tree=ast.parse(source.read_text())
check('adapter no top-level Torch/model import',not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in tree.body))
check('adapter no top-level execution calls',not any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) for n in tree.body))
print(json.dumps({'status':'PASS','checks':len(checks),'names':checks,'scope':'synthetic math/AST only; adapter not executed','torch_imported':False,'real_records_read':0,'model_calls':0},ensure_ascii=False,indent=2))
