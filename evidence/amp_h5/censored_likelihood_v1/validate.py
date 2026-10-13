from pathlib import Path
import json
import numpy as np
from model import objective,fit
R=Path(__file__).resolve().parent;rng=np.random.default_rng(1729);z=np.column_stack([np.ones(30),rng.normal(size=(30,21))]);y=rng.normal(size=30);w=np.full(30,1/30);theta=rng.normal(size=22)/10;k=np.array(['point']*10+['right']*10+['left']*10)
f,g=objective(theta,z,y,w,k,True);eps=1e-6;num=np.array([(objective(theta+eps*np.eye(22)[i],z,y,w,k,True)[0]-objective(theta-eps*np.eye(22)[i],z,y,w,k,True)[0])/(2*eps)for i in range(22)])
checks={'gradient_max_error':float(np.max(abs(g-num)))};assert checks['gradient_max_error']<1e-7
p,diag=fit(z,y,w,np.array(['point']*30),False,np.zeros(22));closed=np.linalg.solve(z.T@(w[:,None]*z)+np.diag([0]+[1]*21),z.T@(w*y));checks['closedform_max_error']=float(np.max(abs(p-closed)));assert checks['closedform_max_error']<1e-5
v1=objective(theta,z,y,w,np.array(['point']*30),False);v2=objective(theta,z,y,w,np.array(['point']*30),True);assert np.array_equal(v1[1],v2[1]) and v1[0]==v2[0]
for b in [-100.,100.]:
 for kind in ['right','left']:
  loss,grad=objective(np.zeros(22),np.ones((1,22)),np.array([b]),np.ones(1),np.array([kind]),True);assert np.isfinite(loss) and np.isfinite(grad).all();assert (grad[0]<0 if kind=='right' else grad[0]>0) or grad[0]==0
checks.update(status='PASS',finite_extreme_tails=True,point_only_objectives_identical=True,synthetic_only=True)
(R/'ENGINEERING_TESTS.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
