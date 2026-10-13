import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,platform,resource,traceback,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent
START=time.perf_counter()
RESULT={'status':'RUNNING','checks':{},'failures':[]}
def save():
    RESULT['wall_seconds']=time.perf_counter()-START
    RESULT['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    (ROOT/'RESULT.json').write_text(json.dumps(RESULT,indent=2))
def check(name,condition,detail):
    RESULT['checks'][name]={'pass':bool(condition),'detail':detail}
    save()
    if not condition: raise AssertionError(name)
def timeout(*_):raise TimeoutError('Frozen 120 second cap')
signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
try:
    frozen=json.loads((ROOT/'FREEZE.json').read_text())
    for name,digest in frozen['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Freeze mismatch '+name)
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    from model import WittenCNN, KerasStyleAdam,encode,ALPHABET
    cfg=json.loads((ROOT/'CONFIG.json').read_text());seed=cfg['seed']
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    RESULT['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
    check('encoding_padding',encode(['AC']).shape==(1,100,21) and encode(['AC'])[0].sum()==2 and not encode(['AC'])[0,:,20].any() and not encode(['AC'])[0,2:].any(),'Zero padding; channel21 unused')
    check('truncation',encode(['A'*101]).sum()==100,'Author encoder truncates at100; natural admission must separately enforce <100')
    for s in ['X','a']:
        try:encode([s]);ok=False
        except KeyError:ok=True
        check('reject_'+s,ok,'Source raises KeyError for unknown/noncanonical/lowercase')
    model=WittenCNN().double().eval()
    check('parameters',sum(p.numel() for p in model.parameters())==183069,{n:p.numel() for n,p in model.named_parameters()})
    # NumPy reference uses channels-last kernels; separate implementation, not torch operators.
    def conv(x,w,b):
        w=w.transpose(2,1,0)
        return np.stack([np.einsum('bkc,kco->bo',x[:,i:i+w.shape[0]],w)+b for i in range(x.shape[1]-w.shape[0]+1)],axis=1)
    def pool(x):return np.maximum(x[:,0:2*(x.shape[1]//2):2],x[:,1:2*(x.shape[1]//2):2])
    def ref(x):
        p={n:t.detach().numpy() for n,t in model.named_parameters()}
        z=pool(np.maximum(conv(np.pad(x,((0,0),(5,5),(0,0))),p['conv1.weight'],p['conv1.bias']),0))
        z=pool(np.maximum(conv(z,p['conv2.weight'],p['conv2.bias']),0)).reshape(x.shape[0],1536)
        for n in ['dense1','dense2']:z=np.maximum(z@p[n+'.weight'].T+p[n+'.bias'],0)
        return z@p['output.weight'].T+p['output.bias']
    # Continuous synthetic input avoids sparse one-hot max-pool ties in derivative tests.
    x=rng.normal(size=(2,100,21));xt=torch.tensor(x,dtype=torch.float64)
    error=float(np.max(np.abs(model(xt).detach().numpy()-ref(x))))
    check('numpy_forward',error<1e-9,{'max_abs_error':error,'tolerance':1e-9})
    model.zero_grad();model(xt).sum().backward()
    gradchecks=[]
    for n,p in model.named_parameters():
        idx=np.unravel_index(int(p.grad.abs().argmax()),tuple(p.shape));analytical=float(p.grad[idx]);v=float(p[idx]);h=1e-6
        with torch.no_grad():p[idx]=v+h
        plus=float(ref(x).sum())
        with torch.no_grad():p[idx]=v-h
        minus=float(ref(x).sum())
        with torch.no_grad():p[idx]=v
        numerical=(plus-minus)/(2*h);err=abs(analytical-numerical)/max(1,abs(analytical),abs(numerical))
        gradchecks.append({'parameter':n,'index':[int(i) for i in idx],'autograd':analytical,'numpy_finite_difference':numerical,'scaled_error':err})
    check('numpy_gradient',max(t['scaled_error'] for t in gradchecks)<2e-5,gradchecks)
    p=torch.nn.Parameter(torch.tensor([.25,-.3],dtype=torch.float64));opt=KerasStyleAdam([p]);q=p.detach().numpy().copy();m=np.zeros(2);v=np.zeros(2)
    for t in range(1,5):
        g=np.array([1e-6*t,-.2/t]);p.grad=torch.tensor(g);opt.step();m=.9*m+.1*g;v=.999*v+.001*g*g;q-=.001*np.sqrt(1-.999**t)/(1-.9**t)*m/(np.sqrt(v)+1e-7)
    err=float(np.max(np.abs(p.detach().numpy()-q)))
    check('adam_numpy',err<1e-12,{'max_abs_error':err,'steps':4,'tiny_gradient_tests_epsilon_placement':True})
    sequences=[''.join(rng.choice(list(ALPHABET),size=int(rng.integers(5,100)))) for _ in range(32)]
    target=np.array([[.5+2*sum(a in 'KR' for a in s)/len(s)-sum(a in 'DE' for a in s)/len(s)] for s in sequences],dtype=np.float32)
    xx=torch.tensor(encode(sequences));yy=torch.tensor(target)
    torch.manual_seed(seed);model=WittenCNN();opt=KerasStyleAdam(model.parameters());lossfn=torch.nn.MSELoss()
    model.eval();initial=float(lossfn(model(xx),yy));model.train()
    history=[]
    def step():
        opt.zero_grad(set_to_none=True);loss=lossfn(model(xx),yy)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
        opt.step();return float(loss.detach())
    for i in range(80):history.append(step())
    model.eval();final=float(lossfn(model(xx),yy));a=model(xx).detach();b=model(xx).detach()
    check('synthetic_loss',final/initial<.9,{'initial_eval_mse':initial,'final_eval_mse':final,'ratio':final/initial,'training_losses_all80':history})
    check('eval_determinism',torch.equal(a,b),'Repeated evaluation exactly equal')
    model.train();a=model(xx).detach();b=model(xx).detach()
    check('dropout_active',not torch.equal(a,b),'Two training forwards differ under dropout')
    for _ in range(5):step()
    ts=time.perf_counter()
    for _ in range(20):step()
    training_seconds=time.perf_counter()-ts
    model.eval();ts=time.perf_counter()
    with torch.no_grad():
        for _ in range(20):model(xx)
    inference_seconds=time.perf_counter()-ts
    RESULT['throughput']={'batch_size':32,'warmup_steps':5,'timed_steps':20,'training_seconds':training_seconds,'train_seconds_per_step':training_seconds/20,'train_examples_per_second':640/training_seconds,'inference_seconds':inference_seconds,'inference_examples_per_second':640/inference_seconds,'training_steps_total':105,'natural_epochs_extrapolation':'ceil(N/32)*epochs*train_seconds_per_step; excludes IO/evaluation and is not measured natural throughput'}
    check('cpu_single_thread',all(p.device.type=='cpu' for p in model.parameters()) and torch.get_num_threads()==torch.get_num_interop_threads()==1,'CPU only; BLAS env1')
    save();check('resource_cap',RESULT['wall_seconds']<120 and RESULT['peak_rss_bytes']<1500000000,{'wall_seconds':RESULT['wall_seconds'],'peak_rss_bytes':RESULT['peak_rss_bytes']})
    RESULT['status']='PASS_SYNTHETIC_ONLY'
except BaseException as e:
    RESULT['status']='FAILED';RESULT['failures'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
finally:
    save();print(json.dumps({'status':RESULT['status'],'wall_seconds':RESULT['wall_seconds'],'peak_rss_bytes':RESULT['peak_rss_bytes'],'failures':RESULT['failures']}))
if RESULT['status']!='PASS_SYNTHETIC_ONLY':raise SystemExit(1)
