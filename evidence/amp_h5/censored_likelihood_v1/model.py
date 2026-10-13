"""Fixed-scale penalized Gaussian working observation model, 22 coefficients."""
import collections,math
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize
ALPHABET='ACDEFGHIKLMNPQRSTVWY'
def weights(rs):
 counts=collections.defaultdict(collections.Counter)
 for r in rs:counts[r['source_family_group']][r['peptide_id']]+=1
 return np.array([1/(len(counts)*len(counts[r['source_family_group']])*counts[r['source_family_group']][r['peptide_id']])for r in rs])
def features(rs):return np.array([[r['sequence'].count(a)/len(r['sequence'])for a in ALPHABET]+[math.log(len(r['sequence']))]for r in rs])
def scale_fit(x,w):
 mean=w@x;scale=np.sqrt(w@((x-mean)**2));return mean,np.where(scale>1e-12,scale,1.)
def design(x,mean,scale):return np.column_stack([np.ones(len(x)),(x-mean)/scale])
def objective(theta,z,y,w,kinds,tail):
 mu=z@theta;err=mu-y;loss=.5*err**2;dmu=err.copy()
 if tail:
  for kind,sign in [('right',1.),('left',-1.)]:
   mask=kinds==kind;t=sign*err[mask];logp=log_ndtr(t)
   loss[mask]=-logp;dmu[mask]=-sign*np.exp(-.5*t*t-.5*np.log(2*np.pi)-logp)
 value=float(w@loss+.5*(theta[1:]@theta[1:]));grad=z.T@(w*dmu);grad[1:]+=theta[1:]
 if not np.isfinite(value) or not np.isfinite(grad).all():raise FloatingPointError('Nonfinite objective/gradient')
 return value,grad

def fit(z,y,w,kinds,tail,start):
 res=minimize(objective,start.copy(),args=(z,y,w,kinds,tail),jac=True,method='L-BFGS-B',options={'maxiter':500,'maxfun':10000,'gtol':1e-8,'ftol':1e-14,'maxls':40})
 value,grad=objective(res.x,z,y,w,kinds,tail);g=float(np.max(np.abs(grad)))
 if not res.success or g>1e-5 or not np.isfinite(res.x).all():raise RuntimeError(f'Fit failed convergence: {res.message}; gradient={g}')
 return res.x,{'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),'evaluations':int(res.nfev),'objective_within_arm_only':value,'max_abs_gradient':g}

def metrics(rs):
 y=np.array([r['target_log10']for r in rs]);p=np.array([r['prediction_log10']for r in rs]);e=p-y;w=weights(rs);c=collections.Counter(r['peptide_id']for r in rs);pw=np.array([1/(len(c)*c[r['peptide_id']])for r in rs])
 return {'rows':len(rs),'peptide_IDs':len(c),'families':len({r['sequence_family']for r in rs}),'source_groups':len({r['source_family_group']for r in rs}),'group_peptide_MAE':float(w@abs(e)),'group_peptide_RMSE':float(np.sqrt(w@(e*e))),'peptide_MAE':float(pw@abs(e)),'peptide_RMSE':float(np.sqrt(pw@(e*e))),'row_MAE':float(abs(e).mean()),'row_RMSE':float(np.sqrt((e*e).mean()))}
