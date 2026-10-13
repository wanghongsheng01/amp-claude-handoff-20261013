"""CPU architecture-equivalent port of cached QMAP/Witten source; no data loading."""
import math
import numpy as np
import torch
from torch import nn
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'  # Author set order not stable: explicit permutation.
INDEX = {a:i for i,a in enumerate(ALPHABET)}

def encode(sequences):
    out = np.zeros((len(sequences),100,21), dtype=np.float32)
    for b,s in enumerate(sequences):
        for i,a in enumerate(s[:100]):
            out[b,i,INDEX[a]]=1.0  # Unknown/lowercase raises KeyError as source does.
    return out

class WittenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(21,64,5)
        self.conv2=nn.Conv1d(64,64,5)
        self.dense1=nn.Linear(1536,100)
        self.dense2=nn.Linear(100,20)
        self.output=nn.Linear(20,1)
        self.dropout=nn.Dropout(.5)
        # Keras default Glorot uniform kernels and zero biases; RNG differs.
        for m in (self.conv1,self.conv2,self.dense1,self.dense2,self.output):
            nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (100,21):
            raise ValueError('Expected (batch,100,21)')
        x=x.transpose(1,2)
        x=torch.nn.functional.pad(x,(5,5))
        x=torch.relu(self.conv1(x));x=torch.nn.functional.max_pool1d(x,2,2)
        x=torch.relu(self.conv2(x));x=torch.nn.functional.max_pool1d(x,2,2)
        x=x.transpose(1,2).contiguous().flatten(1) # Keras channels-last Flatten.
        x=self.dropout(x)
        x=torch.relu(self.dense1(x));x=torch.relu(self.dense2(x))
        return self.output(x)

class KerasStyleAdam(torch.optim.Optimizer):
    """Dense Adam with TF/Keras epsilon-hat placement; no decay/AMSGrad/clipping."""
    def __init__(self,params,lr=.001,betas=(.9,.999),eps=1e-7):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.is_sparse: raise ValueError('Sparse gradients unsupported')
                s=self.state[p]
                if not s: s.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                s['t']+=1
                s['m'].mul_(b1).add_(p.grad,alpha=1-b1)
                s['v'].mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
                alpha=group['lr']*math.sqrt(1-b2**s['t'])/(1-b1**s['t'])
                p.addcdiv_(s['m'],s['v'].sqrt().add_(group['eps']),value=-alpha)
