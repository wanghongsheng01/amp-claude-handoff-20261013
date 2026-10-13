"""UNEXECUTED H5 adapter draft. Import has no Torch/model/data side effects.

Root smoke protocol + independent preflight + START are required before any
function here is called with real tensors/model. No training driver is provided.
Original sources/checkpoints are not edited; q inputs and flow are inherited.
"""

def make_fresh_model(vae_model_class):
    """Construction only after authorized smoke. Both arms clone ONE fresh state."""
    return vae_model_class(out_dim=21, con_dim=3, latent_size=16,
                           sqz_prior=False, cond_drop=False,
                           use_prior_flow=True, device='cpu')


def forward_objective(model, target, residue_g, abs_scaled_h, arm):
    """Explicit inherited forward, only 21-output head and declared objective.

    target [B,48]; residue_g [B,3,48] contains original scaled train descriptors
    with zero tail; abs_scaled_h [B,9] keeps original abs interaction. Both are
    posterior/training inputs, NEVER supplied to DDPM sampling. Decoder g=None.
    No logsigma clipping, negative-KL clamping, beta schedule, or new noise rule.
    """
    import torch
    import torch.nn.functional as F
    if arm not in ('raw_onehot_mse', 'categorical_ce'):
        raise ValueError('unregistered arm')
    b, t = target.shape
    if t != 48 or residue_g.shape != (b,3,48) or abs_scaled_h.shape != (b,9):
        raise ValueError('input shape')
    v = model.vae
    if v.sqz_prior or not v.use_prior_glow or v.decoder.out_proj.out_channels != 21:
        raise ValueError('architecture mismatch')
    x = v.word_emb(target.T).permute(1,2,0)
    zq, mu, logs = v.encoder(x, v.g_pre_net(residue_g), abs_scaled_h)
    score_bkt = v.decoder(zq)
    zp = v.prior_flow(zq, residue_g)
    logq = torch.distributions.Normal(mu, logs.exp()).log_prob(zq)
    logp = v.prior_dist.log_prob(zp)
    density_sum = (logq-logp).sum()
    scores = score_bkt.transpose(1,2)
    if scores.shape != (b,48,21):
        raise ValueError('categorical output shape')
    onehot = F.one_hot(target,num_classes=21).to(dtype=scores.dtype)
    half_square_sum = .5*(scores-onehot).square().sum()
    ce_sum = F.cross_entropy(score_bkt,target,reduction='sum')
    denominator = b*48
    reconstruction_sum = half_square_sum if arm == 'raw_onehot_mse' else ce_sum
    loss = (reconstruction_sum+density_sum)/denominator
    return {'loss':loss,'scores':scores,'zq':zq,'zp':zp,'mu':mu,'logsigma':logs,
            'logq':logq,'logp':logp,'density_sum':density_sum,
            'half_onehot_square_sum':half_square_sum,'ce_sum':ce_sum,
            'position_denominator':denominator,
            'legacy_kl_sum_over_channels':density_sum/16,
            'kl_per_record':density_sum/b,'kl_per_position':density_sum/denominator,
            'beta':1.0}


def decode_joint(score_btk):
    """Truth-free same J rule for both heads; scalar/firstPAD not silently reused."""
    import torch
    if score_btk.ndim != 3 or score_btk.shape[1:] != (48,21):
        raise ValueError('score shape')
    body, aa = score_btk[:,:,:20].max(-1)
    padding = score_btk[:,:,20]
    base = padding.sum(1,keepdim=True)
    length_scores = base + torch.cat((torch.zeros_like(base),
                                     (body-padding).cumsum(1)),dim=1)
    length = length_scores.argmax(1)
    tokens = torch.where(torch.arange(48,device=aa.device)[None,:] < length[:,None],
                         aa, torch.full_like(aa,20))
    return tokens,length,length_scores
