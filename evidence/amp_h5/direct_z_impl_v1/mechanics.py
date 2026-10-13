"""Direct-z frozen mathematical and random mechanisms. No import-time execution."""
import hashlib
import json
import math
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch.nn import functional as F
from architecture import UNet_conditional


def tensor_digest(t):
    t=t.detach().cpu().contiguous()
    h=hashlib.sha256(str(t.dtype).encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0')
    h.update(t.numpy().tobytes());return h.hexdigest()

def state_digest(model):
    state=model.state_dict() if hasattr(model,'state_dict') else model
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode()+b'\0'+tensor_digest(v).encode())
    return h.hexdigest()

def finite(x,name):
    if not bool(torch.isfinite(x).all()):raise FloatingPointError('nonfinite '+name)
    return x

def positive_sigma(logsigma):
    assert logsigma.dtype==torch.float32
    sigma=finite(logsigma.exp(),'float32 exp(logsigma)')
    if not bool((sigma>0).all()):raise FloatingPointError('float32 sigma underflow/nonpositive')
    return sigma

def initialize_models():
    """Private fork; R first, Z shared trunk exact, boundary mapping explicit."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(322);before=torch.get_rng_state().clone()
        r=UNet_conditional(c_in=2,c_out=2,time_dim=256,num_class1=2,num_class2=2)
        after_r=torch.get_rng_state().clone()
        z=UNet_conditional(c_in=1,c_out=1,time_dim=256,num_class1=2,num_class2=2)
        after_z=torch.get_rng_state().clone()
    rs,zs=r.state_dict(),z.state_dict()
    for k,v in zs.items():
        if k=='inc.double_conv.0.weight':v.copy_(rs[k][:,:1]*math.sqrt(2))
        elif k in ('outc.weight','outc.bias'):v.copy_(rs[k][:1])
        else:
            assert v.shape==rs[k].shape
            v.copy_(rs[k])
    z.load_state_dict(zs)
    assert sum(p.numel() for p in r.parameters())==18611970
    assert sum(p.numel() for p in z.parameters())==18611329
    common={k:v for k,v in rs.items() if k not in ('inc.double_conv.0.weight','outc.weight','outc.bias')}
    assert state_digest(common)==state_digest({k:v for k,v in zs.items() if k in common})
    return {'R':r,'Z':z},{'generator_initial':before,'generator_after_R':after_r,'generator_after_Z':after_z,
        'R_sha256':state_digest(r),'Z_sha256':state_digest(z),'common_sha256':state_digest(common)}

def normalization(q):
    assert q.dtype==torch.float32 and q.ndim==4 and q.shape[1:]==(2,16,48)
    finite(q,'posterior');positive_sigma(q[:,1])
    a=q.numpy().astype(np.float64);mu,logs=a[:,0],a[:,1]
    rmean=a.mean(axis=(0,2,3));rstd=a.std(axis=(0,2,3),ddof=0)
    zm=float(mu.mean());zv=float(np.mean(mu*mu+np.exp(2*logs))-zm*zm)
    zs=math.sqrt(zv) if zv>0 else float('nan')
    values=np.r_[rmean,rstd,zm,zs]
    if not np.isfinite(values).all() or (rstd<=1e-8).any() or not zs>1e-8:
        raise FloatingPointError('normalization constant/nonfinite')
    return {'R_mean':rmean.tolist(),'R_std':rstd.tolist(),'Z_mean':zm,'Z_std':zs,'n':len(q),'fit_dtype':'float64','ddof':0}

def normalize(target,arm,stats):
    if arm=='R':
        assert target.shape[1:]==(2,16,48)
        m=torch.tensor(stats['R_mean'],dtype=torch.float32)[None,:,None,None]
        s=torch.tensor(stats['R_std'],dtype=torch.float32)[None,:,None,None]
    else:
        assert arm=='Z' and target.shape[1:]==(1,16,48)
        m,s=stats['Z_mean'],stats['Z_std']
    return finite((target-m)/s,'normalized target')

def inverse_normalize(u,arm,stats):
    if arm=='R':
        assert u.shape[1:]==(2,16,48)
        m=torch.tensor(stats['R_mean'],dtype=torch.float32)[None,:,None,None]
        s=torch.tensor(stats['R_std'],dtype=torch.float32)[None,:,None,None]
    else:
        assert arm=='Z' and u.shape[1:]==(1,16,48)
        m,s=stats['Z_mean'],stats['Z_std']
    return finite(m+s*u,'inverse normalized target')

def to_decoder_z(target,arm,epsilon=None):
    finite(target,'generated target')
    if arm=='Z':
        assert epsilon is None and target.shape[1:]==(1,16,48)
        return target[:,0]
    assert arm=='R' and target.shape[1:]==(2,16,48) and epsilon.shape==target[:,0].shape
    return finite(target[:,0]+positive_sigma(target[:,1])*finite(epsilon,'final reparameterization epsilon'),'decoder z')

def noise_schedule():
    beta=torch.linspace(1e-4,.02,999,dtype=torch.float32);alpha=1-beta
    return beta,alpha,torch.cumprod(alpha,0)

def optimizer_and_scheduler(model):
    opt=torch.optim.AdamW(model.parameters(),lr=.005,betas=(.9,.999),eps=1e-5,weight_decay=.01,foreach=False,fused=False)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.005,total_steps=1000,pct_start=.3,
        anneal_strategy='cos',cycle_momentum=True,base_momentum=.85,max_momentum=.95,
        div_factor=25.,final_div_factor=10000.,three_phase=False,last_epoch=-1)
    return opt,sch

def scheduler_seal():
    """Derive exact torch schedule without model execution, RNG or optimizer updates."""
    dummy=torch.nn.Module()
    dummy.register_parameter('scalar',torch.nn.Parameter(torch.zeros(())))
    # Constant dummy parameter: scheduler math only, no random/model execution.
    opt,sch=optimizer_and_scheduler(dummy)
    rows=[]
    for step in range(1001):
        sch.last_epoch=step
        lr=sch.get_lr()[0]
        rows.append({'schedule_index':step,'lr':lr,'beta1':opt.param_groups[0]['betas'][0]})
    payload=json.dumps(rows,separators=(',',':')).encode()
    return rows,hashlib.sha256(payload).hexdigest()

class TrainingRandom:
    """Shared row/t/dropout/2-channel noise; Z additionally consumes fresh q epsilon."""
    def __init__(self,arm):
        assert arm in ('R','Z');self.arm=arm
        self.g={k:torch.Generator(device='cpu').manual_seed(s) for k,s in
            [('batches',322),('t',3322),('noise',4322),('dropout',5322)]}
        if arm=='Z':self.g['q_epsilon']=torch.Generator(device='cpu').manual_seed(6322)
        self.order=torch.empty(0,dtype=torch.long);self.cursor=0
    def states(self):return {k:g.get_state().clone() for k,g in self.g.items()}
    def next(self):
        if self.cursor>=len(self.order):self.order=torch.randperm(279,generator=self.g['batches']);self.cursor=0
        ids=self.order[self.cursor:self.cursor+8];self.cursor+=len(ids);n=len(ids)
        t=torch.randint(1,999,(n,),generator=self.g['t'])
        drop=bool(torch.rand((),generator=self.g['dropout'])<.1)
        fullnoise=torch.randn((n,2,16,48),generator=self.g['noise'],dtype=torch.float32)
        eps=torch.randn((n,16,48),generator=self.g['q_epsilon'],dtype=torch.float32) if self.arm=='Z' else None
        return {'ids':ids,'t':t,'dropout':drop,'full_noise':fullnoise,'noise':fullnoise if self.arm=='R' else fullnoise[:,:1],
            'q_epsilon':eps}

def random_record(d):
    return {'ids':d['ids'].tolist(),'t':d['t'].tolist(),'dropout':d['dropout'],
        'full_noise_sha256':tensor_digest(d['full_noise']),'first_channel_noise_sha256':tensor_digest(d['full_noise'][:,:1]),
        'q_epsilon_sha256':tensor_digest(d['q_epsilon']) if d['q_epsilon'] is not None else None}

class GenerationRandom:
    def __init__(self,terminal):
        assert terminal in ('FREE','AMD');self.terminal=terminal
        self.g=torch.Generator(device='cpu').manual_seed(7422 if terminal=='FREE' else 7423)
    def noise(self):return torch.randn((16,2,16,48),generator=self.g,dtype=torch.float32)
    def state(self):return self.g.get_state().clone()

def seal_randomness():
    seal={'training':{},'generation':{},'reparameterization':{}};states={}
    for arm in ('R','Z'):
        stream=TrainingRandom(arm)
        for k,s in stream.states().items():states[f'{arm}_train_initial_{k}']=s
        seal['training'][arm]=[random_record(stream.next()) for _ in range(1000)]
        assert sum(len(r['ids']) for r in seal['training'][arm])==7972
        for k,s in stream.states().items():states[f'{arm}_train_final_{k}']=s
    for r,z in zip(seal['training']['R'],seal['training']['Z']):
        assert all(r[k]==z[k] for k in ('ids','t','dropout','full_noise_sha256','first_channel_noise_sha256'))
    for terminal,seed in [('FREE',8422),('AMD',8423)]:
        stream=GenerationRandom(terminal);states[terminal+'_generation_initial']=stream.state()
        hashes=[tensor_digest(stream.noise())] # x_998
        hashes.extend(tensor_digest(stream.noise()) for t in range(998,1,-1)) # noise at t998..2; t1 zero
        assert len(hashes)==998
        seal['generation'][terminal]={'initial_noise_sha256':hashes[0],
            'reverse_noise_sha256':dict(zip(map(str,range(998,1,-1)),hashes[1:])), 't1_noise':'constant_zero_no_draw'}
        states[terminal+'_generation_final']=stream.state()
        g=torch.Generator(device='cpu').manual_seed(seed);states[terminal+'_reparam_initial']=g.get_state().clone()
        epsilon=torch.randn((16,16,48),generator=g,dtype=torch.float32)
        seal['reparameterization'][terminal]=tensor_digest(epsilon);states[terminal+'_reparam_final']=g.get_state()
    seal['generator_states_sha256']={k:tensor_digest(s) for k,s in states.items()}
    return seal,states

def training_target(q,random,arm,stats):
    v=q[random['ids']]
    if arm=='Z':v=(v[:,0]+positive_sigma(v[:,1])*random['q_epsilon'])[:,None]
    return normalize(finite(v,'training target'),arm,stats)

def training_step(model,opt,sch,q,labels,arm,stats,random,schedule,*,before_forward=lambda:None,after_update=lambda:None):
    """Exact natural step; synthetic harness must use this without changing scheduler1000."""
    _,_,abar=schedule
    u=training_target(q,random,arm,stats);t=random['t'];noise=random['noise']
    a=abar[t][:,None,None,None]
    xt=finite(a.sqrt()*u+(1-a).sqrt()*noise,'noised target')
    opt.zero_grad(set_to_none=True);model.train()
    lr,beta1=opt.param_groups[0]['lr'],opt.param_groups[0]['betas'][0]
    before_forward()
    prediction=finite(model(xt,t,None if random['dropout'] else labels[random['ids']]),'training prediction')
    loss=finite(F.mse_loss(prediction,noise,reduction='mean'),'epsilon MSE');loss.backward()
    missing=[]
    for name,p in model.named_parameters():
        if p.grad is None:missing.append(name)
        else:finite(p.grad,'gradient '+name)
    assert missing==(['label_emb.weight'] if random['dropout'] else []),missing
    opt.step();after_update()
    for name,p in model.named_parameters():finite(p,'parameter '+name)
    for state in opt.state.values():
        for name,value in state.items():
            if torch.is_tensor(value):finite(value,'AdamW '+name)
    sch.step()
    return {'mse':float(loss.detach()),'elements':noise.numel(),'objects':len(random['ids']),'lr_used':lr,
        'beta1_used':beta1,'lr_next':opt.param_groups[0]['lr'],'gradients_finite':True,'parameters_finite':True,
        'missing_gradients':missing,'target_sha256':tensor_digest(u),'noised_target_sha256':tensor_digest(xt)}

@torch.no_grad()
def reverse_step(model,x,t,labels,noise,schedule,*,before_forward=lambda:None,observations=None):
    beta,alpha,abar=schedule
    assert 1<=t<=998 and x.shape==noise.shape
    before_forward()
    prediction=model(x,torch.full((len(x),),t,dtype=torch.long),labels)
    if observations is not None:observations['prediction']=prediction
    finite(prediction,'reverse prediction')
    result=(x-((1-alpha[t])/torch.sqrt(1-abar[t]))*prediction)/torch.sqrt(alpha[t])
    if t>1:result=result+torch.sqrt(beta[t])*noise
    if observations is not None:observations['u_after_step']=result
    return finite(result,'reverse u')
