"""Independent fixed-weight synthetic edge assertions; no training loop or data reads."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'witten_cpu_v1'))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from model import WittenCNN,encode,KerasStyleAdam
m=WittenCNN().double().eval()
with torch.no_grad():
 for p in m.parameters():p.zero_()
 m.conv1.weight[0,0,0]=1;m.conv2.weight[0,0,0]=1
 m.dense1.weight[0,64]=1;m.dense2.weight[0,0]=1;m.output.weight[0,0]=1
x=torch.tensor(encode(['A','C','C'*99+'A']),dtype=torch.float64)
pred=m(x);expect=torch.tensor([[1.],[0.],[0.]],dtype=torch.float64)
assert torch.equal(pred,expect),(pred,expect)
loss=torch.nn.functional.mse_loss(pred,torch.tensor([[2.],[0.],[0.]],dtype=torch.float64));loss.backward()
assert abs(loss.item()-1/3)<1e-15
assert abs(m.output.weight.grad[0,0].item()+2/3)<1e-15
p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64));o=KerasStyleAdam([p]);p.grad=torch.tensor([1e-9],dtype=torch.float64);o.step()
expected=1-.001*(1-.999)**.5/(1-.9)*(.1*1e-9)/((.001*1e-18)**.5+1e-7)
assert abs(p.item()-expected)<1e-15
out={'status':'PASS','model_sha256':hashlib.sha256((root/'witten_cpu_v1/model.py').read_bytes()).hexdigest(),'execution':'independent fixed-weight forward/backward and one scalar Adam operation; no naturaldata, no training loop','checks':{'padding_pool_flatten_position':[1,0,0],'scalar_MSE':loss.item(),'MSE_output_weight_gradient':m.output.weight.grad[0,0].item(),'tiny_gradient_epsilon_hat_Adam_abs_error':abs(p.item()-expected)},'threads':[torch.get_num_threads(),torch.get_num_interop_threads()]}
(root/'chem_cnn_audit_v1/CNN_INDEPENDENT_NUMERIC.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
