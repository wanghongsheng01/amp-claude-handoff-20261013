import hashlib,json,math
from pathlib import Path
import numpy as np
import run as a
R=Path(__file__).resolve().parent;checks=[]
def test(n,c):
 assert c,n
 checks.append(n)
rs=[{'source_family_group':'A','peptide_id':1,'sequence_family':'F1'}, {'source_family_group':'A','peptide_id':1,'sequence_family':'F1'},{'source_family_group':'A','peptide_id':2,'sequence_family':'F2'},{'source_family_group':'B','peptide_id':3,'sequence_family':'F3'}]
w=a.weights(rs);test('manual_hierarchical_weights',np.allclose(w,[.125,.125,.25,.5]));test('sum_one',abs(sum(w)-1)<1e-14)
y=np.array([0.,2.,4.,6.]);test('manual_weighted_constant',abs(w@y-4.25)<1e-14)
for r,v in zip(rs,y):r.update(target_log10=0.,prediction_log10=v)
m=a.metrics(rs);test('manual_group_peptide_MAE',m['group_peptide_macro_MAE']==4.25);test('manual_group_peptide_RMSE',abs(m['group_peptide_macro_RMSE']-math.sqrt(22.5))<1e-14);test('manual_row_MAE',m['row_MAE']==3.0);test('weights_not_row_or_peptide_macro',m['group_peptide_macro_MAE']!=m['peptide_macro_MAE']!=m['row_MAE'])
x=np.array([[0.],[1.]]);target=np.array([0.,2.]);ww=np.array([.5,.5]);fit=a.ridge_fit(x,target,ww,1.);pred=a.ridge_predict(fit,x)
test('ridge_closed_form',np.allclose(pred,[.5,1.5]));test('ridge_train_only_scaling',np.allclose(fit['mu'],[.5])and np.allclose(fit['scale'],[.5]));shift=a.ridge_fit(x,target+17,ww,1.)
test('ridge_target_shift',np.allclose(a.ridge_predict(shift,x),pred+17));fit2=a.ridge_fit(np.ones((2,1)),target,ww,1.);test('ridge_constant_feature',np.allclose(a.ridge_predict(fit2,np.ones((2,1))),[1.,1.]))
fs=a.features([{'sequence':'AK'},{'sequence':'AA'}]);test('features_exact_composition',fs[0,0]==.5 and fs[0,a.ALPHABET.index('K')]==.5 and fs[1,0]==1. and abs(fs[0,-1]-math.log(2))<1e-14)
# All natural fold data are frozen; no model constructed or trained in these tests.
d=json.loads((R/'FOLDS.json').read_text());test('fixed17folds',sum(len(v['folds'])for v in d['units'].values())==17)
p=json.loads((R/'PROTOCOL.json').read_text());test('steps1700',p['cnn']['epochs']*17==p['limits']['maximum_CNN_updates']==1700);test('limit600s',p['limits']['wall_seconds']==600)
(R/'ENGINEERING_TESTS.json').write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'no_model_constructed_or_trained':True,'runner_sha256':a.sha(R/'run.py')},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
