#!/usr/bin/env python3
"""Independent synthetic checks. No work on import; requires explicit root window."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--core', required=True)
    ap.add_argument('--core-sha256', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--scoring', required=True)
    ap.add_argument('--scoring-sha256', required=True)
    ap.add_argument('--root-serial-window', required=True)
    args = ap.parse_args()
    assert args.root_serial_window == 'AUTHORIZED_SYNTHETIC_ONLY'
    assert digest_file(args.core) == args.core_sha256
    assert digest_file(args.plan) == args.plan_sha256
    assert digest_file(args.config) == args.config_sha256
    assert digest_file(args.scoring) == args.scoring_sha256
    plan = json.loads(Path(args.plan).read_text())
    assert plan['outer_seconds'] == 90
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    import numpy as np
    spec = importlib.util.spec_from_file_location('independently_checked_ar_core', args.core)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sspec = importlib.util.spec_from_file_location('independently_checked_scoring', args.scoring)
    scoring = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(scoring)
    core.configure_cpu()
    checks = []
    count = {'actual_AR_forward': 0, 'optimizer_updates': 0}

    def budget():
        if time.monotonic() - started >= 90:
            raise TimeoutError('Independent synthetic90s exhausted; no extension')

    def counted_forward(module, args):
        budget()
        count['actual_AR_forward'] += 1
        assert count['actual_AR_forward'] <= 1867

    def passed(name, details=None):
        checks.append({'check': name, 'passed': True, 'details': details})

    result = {'status': 'RUNNING', 'model_natural_calls': 0, 'root_window': args.root_serial_window,
              'core_sha256': args.core_sha256, 'plan_sha256': args.plan_sha256}
    try:
        # Exact initialization and correspondence, independent of candidate digest helper.
        models, initial = core.initial_models()
        for m in models.values():
            m.register_forward_pre_hook(counted_forward)
        ast, bst, cst = [models[k].state_dict() for k in 'ABC']
        assert {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()} == {
            'A': 66597, 'B': 66597, 'C': 62501}
        for key, value in cst.items():
            av = ast[key][:, :40] if key == 'lstm.weight_ih_l0' else ast[key]
            assert av.contiguous().numpy().tobytes() == value.contiguous().numpy().tobytes()
        for key in ast:
            assert ast[key].contiguous().numpy().tobytes() == bst[key].contiguous().numpy().tobytes()
        eg = torch.Generator(device='cpu').manual_seed(7222)
        expected = torch.empty(256, 16).uniform_(-.125, .125, generator=eg)
        assert torch.equal(expected, ast['lstm.weight_ih_l0'][:, 40:])
        passed('parameter_counts_and_exact_shared_initialization')

        # Reproduce the pre-forward receipt seal using only old row metadata/indices.
        import csv
        config = json.loads(Path(args.config).read_text())
        for entry in config['bindings']:
            assert digest_file(entry['path']) == entry['sha256']
        def labels_only(path):
            with Path(path).open(newline='') as handle:
                return torch.tensor([int(r['class']) for r in csv.DictReader(handle)])
        trainlabels = labels_only(config['inputs']['train'])
        devlabels = labels_only(config['inputs']['development'])
        oldtrace = [json.loads(s) for s in Path(config['inputs']['batch_trace']).read_text().splitlines()]
        assert len(oldtrace) == 1000
        ng = torch.Generator(device='cpu').manual_seed(222)
        natural_batches, cursor = [], 279
        for i, row in enumerate(oldtrace, 1):
            if cursor >= 279:
                order, cursor = torch.randperm(279, generator=ng), 0
            indices = order[cursor:cursor+32].tolist()
            cursor += len(indices)
            assert row['step'] == i and row['row_indices'] == indices
            natural_batches.append(indices)
        natural_maps = core.derangements(trainlabels, 1000, 2422)[0]
        natural_devmaps = core.derangements(devlabels, 8, 6422)[0]
        natural_uniforms = core.uniform_bank()[0]
        result['deterministic_preparation'] = {
            'initial_state_sha256': {a: core.state_digest(m) for a,m in models.items()},
            'batch_indices_sha256': hashlib.sha256(json.dumps(natural_batches,separators=(',',':')).encode()).hexdigest(),
            'train_mappings_digest': core.digest(natural_maps),
            'dev_mappings_digest': core.digest(natural_devmaps),
            'uniforms_digest': core.digest(natural_uniforms)}
        passed('deterministic_pre_forward_receipt_seal')
        # Exercise candidate hash/shape/row-identity admission without a forward.
        candidate_dir=Path(args.core).parent
        author_seal=json.loads((candidate_dir/'AUTHOR_STATIC_CHECK.json').read_text())
        for path, expected_sha in author_seal['hashes'].items():
            assert digest_file(path)==expected_sha
        sys.modules['core']=core
        cspec=importlib.util.spec_from_file_location('independently_checked_contracts',candidate_dir/'contracts.py')
        contracts=importlib.util.module_from_spec(cspec)
        cspec.loader.exec_module(contracts)
        cacheout=out/'cache_preflight';cacheout.mkdir()
        prepstart=time.monotonic()
        cached=contracts.prepare(config,cacheout)
        assert torch.equal(cached['maps'],natural_maps) and torch.equal(cached['devmaps'],natural_devmaps)
        assert torch.equal(cached['uniforms'],natural_uniforms)
        assert [b.tolist() for b in cached['batches']]==natural_batches
        result['cache_preparation_seconds']=time.monotonic()-prepstart
        del cached
        passed('cached_shapes_dtypes_finite_rows_and_preparation_parity')

        # L48 explicitly requires EOS49; tail padding must never affect the objective.
        batch = core.teacher_batch(['A', 'R' * 47, 'N' * 48], [0, 1, 0])
        assert batch['inputs'].shape == (3, 49)
        assert batch['mask'].sum(1).tolist() == [2, 48, 49]
        assert batch['inputs'][:, 0].tolist() == [21] * 3
        assert torch.equal(batch['inputs'][:, 1:], batch['targets'][:, :-1])
        assert batch['targets'][2, 48].item() == 20
        logits = (torch.arange(3*49*21, dtype=torch.float32).reshape(3, 49, 21) % 37) / 11
        parts, _ = core.loss_parts(logits, batch['targets'], batch['mask'])
        independent_ce = torch.logsumexp(logits, -1) - logits.gather(-1, batch['targets'][..., None]).squeeze(-1)
        assert torch.allclose(parts['loss'], independent_ce[batch['mask']].sum() / 99)
        assert int(parts['aa_tokens']) == 96 and int(parts['eos_tokens']) == 3
        changed = logits.clone()
        changed[~batch['mask']] = torch.arange(21, dtype=logits.dtype) * 999
        other, _ = core.loss_parts(changed, batch['targets'], batch['mask'])
        assert torch.equal(parts['loss'], other['loss'])
        passed('EOS49_PAD_mask_and_independent_loss_denominator', {'observed': 99, 'AA': 96, 'EOS': 3})

        # Observe the actual LSTM boundary, not only extend_z's returned shape.
        z = torch.arange(3*16*48, dtype=torch.float32).reshape(3, 16, 48) / 10000
        captured = []
        hook = models['A'].lstm.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach().clone()))
        models['A'](batch['inputs'], batch['labels'], core.extend_z(z))
        hook.remove()
        actual = captured.pop()
        assert actual.shape == (3, 49, 56)
        assert torch.equal(actual[:, :48, 40:], z.transpose(1, 2))
        assert torch.equal(actual[:, 48, 40:], torch.zeros(3, 16))
        assert bool((actual[0, 2:48, 40:] != 0).any())  # no true-length mask on latent
        passed('actual_LSTM_z_boundary_and_no_length_mask')

        labels = torch.tensor([0, 0, 1, 1, 1])
        maps, _, _ = core.derangements(labels, 7, 912422)
        for mapping in maps:
            assert sorted(mapping.tolist()) == list(range(5))
            assert all(mapping[i] != i for i in range(5))
            assert torch.equal(labels[mapping], labels)
        for invalid in (torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])):
            try:
                core.derangements(invalid, 1, 912422)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('singleton not rejected')
        fingerprints = torch.arange(5*16*48).reshape(5, 16, 48)
        wrong = fingerprints[maps[0]]
        assert torch.equal(wrong[[0]], fingerprints[maps[0][[0]]])  # singleton target minibatch allowed
        assert all(torch.equal(wrong[i], fingerprints[int(maps[0][i])]) for i in range(5))
        passed('full_group_derangement_and_singleton_rejection')

        class Scripted:
            latent = False

            def __init__(self):
                self.position = 0
                self.inputs = []

            def __call__(self, tokens, labels, z, hidden):
                p = self.position
                self.position += 1
                self.inputs.append(tokens.clone())
                raw = torch.full((7, 1, 21), -1000.0)
                raw[:, :, 0] = 0
                raw[0, 0, 0], raw[0, 0, 20] = -1000, 0
                if p == 48:
                    raw[2, 0, 0], raw[2, 0, 20] = -1000, 0
                if p == 0:
                    raw[3:6, 0, 1] = 0
                    raw[6, 0] = 0  # near1 uniform must map to EOS20, never21
                else:
                    raw[3:7, 0, 0], raw[3:7, 0, 20] = -1000, 0
                return raw, None

        u = torch.full((7, 49), .25)
        u[3, 0] = .5
        u[4, 0] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
        u[5, 0] = 0
        u[6, 0] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
        u_before = u.clone()
        scripted = Scripted()
        sampled = core.decode(scripted, torch.zeros(7, dtype=torch.long), uniforms=u)
        assert sampled['status'] == ['empty_sequence', 'cap_nonEOS', 'eos', 'eos', 'eos', 'eos', 'empty_sequence']
        assert [len(s) for s in sampled['sequences']] == [0, 48, 48, 1, 1, 1, 0]
        assert sampled['raw_tokens'][3:7, 0].tolist() == [1, 1, 0, 20]
        assert torch.equal(u, u_before)
        assert not bool(sampled['active_before'][0, 1:].any())
        assert all(x[0, 0].item() == 20 for x in scripted.inputs[1:])
        assert sampled['tokens'][1, 48].item() == 0 and sampled['tokens'][2, 48].item() == 20
        calls = [0]

        def injected_timeout():
            calls[0] += 1
            if calls[0] == 3:
                raise TimeoutError('synthetic timeout')

        failed = core.decode(Scripted(), torch.zeros(7, dtype=torch.long), uniforms=u, before_forward=injected_timeout)
        assert failed['failure']['type'] == 'TimeoutError'
        assert len(failed['status']) == 7 and failed['status'][1:3] == ['error', 'error']
        assert failed['sequences'][1:3] == [None, None]
        passed('inverseCDF_edges_absorption_EOS49_cap_and_timeout')

        # Independent score formulas and fixed-denominator/undefined boundaries.
        from fractions import Fraction
        from copy import deepcopy
        def raw_dev(number):
            d = {}
            for arm, prefix in (('A','S'),('B','S'),('A','mismatch_S')):
                for seed in range(1422,1430):
                    matches = number if arm == 'A' and prefix == 'S' else 0
                    rows = [{'matched_true_AA':matches,'true_AA':20,'accuracy':matches/20} for _ in range(41)]
                    d[f'{arm}_{prefix}{seed}'] = {'reconstruction':{'complete':True,'rows':rows,'accuracy_equal_sequence':matches/20}}
            d['C_plain'] = {'reconstruction':{'accuracy_equal_sequence':0,'rows':[
                {'matched_true_AA':0,'true_AA':20,'accuracy':0} for _ in range(41)]}}
            return d
        boundary = raw_dev(1)
        assert scoring.information_gate(boundary)['status'] == 'PASS'
        below = deepcopy(boundary)
        below['A_S1422']['reconstruction']['rows'][0]['matched_true_AA'] = 0
        below['A_S1422']['reconstruction']['rows'][0]['accuracy'] = 0
        below['A_S1422']['reconstruction']['accuracy_equal_sequence'] = 40/41/20
        assert scoring.information_gate(below)['status'] == 'FAIL'
        assert scoring.edit_distance('ARN', 'AN') == 1
        assert scoring.edit_distance('ARN', 'ARQ') == 1
        assert scoring.edit_distance('', 'ARN') == 3
        aa = scoring.frozen.feature_vector('AA')
        rr = scoring.frozen.feature_vector('RR')
        expected_mmd = sum((a-b)**2 for a,b in zip(aa,rr))
        comp = scoring.frozen.compare_summaries(scoring.frozen.sequence_summary(['AA']),scoring.frozen.sequence_summary(['RR']))
        assert abs(expected_mmd-4/3) < 1e-12
        assert abs(comp['mmd2_linear_biased_fixed_features']-expected_mmd) < 1e-12
        assert comp['dipeptide_jensen_shannon_bits'] == 1.0
        tiny_train = [{'sequence':'ARNDC','terminal_state':'FREE'}, {'sequence':'QEGHI','terminal_state':'AMD'}]
        tiny_dev = [{'sequence':'ARNDC','terminal_state':'FREE'}, {'sequence':'QEGHI','terminal_state':'AMD'}]
        allslots = [{'arm':a,'terminal':t,'slot':i,'sequence':'ARNDC' if t=='FREE' else 'QEGHI',
                     'status':'historical_ok' if a=='D' else 'eos'} for a in 'ABCD' for t in ('FREE','AMD') for i in range(16)]
        equal = scoring.score_generation(deepcopy(allslots),tiny_train,tiny_dev,budget)
        assert equal['status'] == 'FAIL'  # strict comparisons cannot pass ties
        for mutation, expected_status in ((('A','empty_sequence',''), 'FAIL_CANDIDATE'),
                                           (('B','empty_sequence',''), 'NOT_EVALUABLE'),
                                           (('B','not_executed',None), 'NOT_EVALUABLE'),
                                           (('B','error',None), 'NOT_EVALUABLE'),
                                           (('A','eos','Z'), 'FAIL_CANDIDATE')):
            rows=deepcopy(allslots)
            chosen=next(r for r in rows if r['arm']==mutation[0] and r['terminal']=='FREE')
            chosen.update(status=mutation[1],sequence=mutation[2])
            scored=scoring.score_generation(rows,tiny_train,tiny_dev,budget)
            assert scored['status']==expected_status
            assert all(scored['summary'][a][t]['requests']==16 for a in 'ABCD' for t in ('FREE','AMD'))
        capped=deepcopy(allslots);capped[0].update(status='cap_nonEOS',sequence='A'*48)
        scored=scoring.score_generation(capped,tiny_train,tiny_dev,budget)
        assert scored['summary']['A']['FREE']['main_complete']
        assert scored['summary']['A']['FREE']['cap_nonEOS_rate']==1/16
        passed('metrics_exact_thresholds_strict_ties_and_failure_denominators')

        # Bounded throughput: exclusively synthetic sequence content and latents.
        synth_seqs = []
        for i in range(279):
            length = (1, 5, 47, 48)[i % 4]
            synth_seqs.append(''.join(core.AA[(i+j) % 20] for j in range(length)))
        synth_labels = torch.tensor([0]*126 + [1]*153)
        full = core.teacher_batch(synth_seqs, synth_labels)
        mu = torch.arange(279*16*48, dtype=torch.float32).reshape(279, 16, 48) / 1e6
        sigma = torch.full_like(mu, float(np.exp(-1)))
        permutations, _, _ = core.derangements(synth_labels, 100, 912422)
        orders = [torch.arange(i, min(i+32, 279)) for i in range(0, 279, 32)]
        training = {}
        a_bank_digests = []
        for arm, model in models.items():
            setup0 = time.monotonic()
            g = torch.Generator(device='cpu').manual_seed(914422)
            opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},out/('synthetic_initial_'+arm+'.pt'))
            setup_seconds = time.monotonic()-setup0
            t0 = time.monotonic()
            with (out / ('synthetic_train_' + arm + '.jsonl')).open('x') as trace:
                for step in range(100):
                    budget()
                    idx = orders[step % len(orders)]
                    bank_hash = None
                    if arm in 'AB':
                        epsilon = torch.randn((279, 16, 48), generator=g)
                        bank = mu + sigma * epsilon
                        bank_hash = core.digest(bank)
                        if arm == 'A':
                            a_bank_digests.append(bank_hash)
                        else:
                            assert bank_hash == a_bank_digests[step]
                        chosen = bank[permutations[step]][idx] if arm == 'B' else bank[idx]
                        latent = core.extend_z(chosen)
                    else:
                        latent = None
                    opt.zero_grad(set_to_none=True)
                    pred, _ = model(full['inputs'][idx], full['labels'][idx], latent)
                    lp, _ = core.loss_parts(pred, full['targets'][idx], full['mask'][idx])
                    lp['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
                    count['optimizer_updates'] += 1
                    trace.write(json.dumps({'step': step+1, 'indices': idx.tolist(), 'sampled_bank_sha256': bank_hash,
                                            'loss': float(lp['loss'].detach())}) + '\n')
                    trace.flush()
                    counterpath=out/'SYNTHETIC_COUNTERS.json'
                    tmp=counterpath.with_suffix('.tmp')
                    tmp.write_text(json.dumps(count))
                    tmp.replace(counterpath)
            duration = time.monotonic()-t0
            save0 = time.monotonic()
            torch.save(model.state_dict(), out / ('synthetic_final_' + arm + '.pt'))
            save_duration = time.monotonic()-save0
            training[arm] = {'100_update_seconds_including_trace': duration,
                             'setup_seconds':setup_seconds,
                             'checkpoint_seconds': save_duration,
                             'projected1000_seconds_2x_plus_IO_and5': 20*duration+setup_seconds+save_duration+5}
        passed('100_step_AB_same_complete_sampled_bank_digest', {'steps': 100})

        eval_start = time.monotonic()
        dev = core.teacher_batch(synth_seqs[:41], [0]*18+[1]*23)
        zz41 = core.extend_z(mu[:41])
        conditions = ['A']*18 + ['B']*9 + ['C']
        with torch.no_grad():
            for ci, arm in enumerate(conditions):
                model = models[arm]
                zz = zz41 if model.latent else None
                teacher_logits,_=model(dev['inputs'], dev['labels'], zz)
                _,token_ce=core.loss_parts(teacher_logits,dev['targets'],dev['mask'])
                np.savez(out/('synthetic_teacher_%02d.npz'%ci),logits=teacher_logits.numpy(),
                    probabilities=teacher_logits.softmax(-1).numpy(),target_tokens=dev['targets'].numpy(),
                    inputs=dev['inputs'].numpy(),observed_mask=dev['mask'].numpy(),token_ce=token_ce.numpy(),labels=dev['labels'].numpy())
                inputs = torch.full((41, 1), 21, dtype=torch.long)
                hidden = None
                raw = torch.empty(41, 49, 21)
                for p in range(49):
                    logits, hidden = model(inputs, dev['labels'], None if zz is None else zz[:, p:p+1], hidden)
                    raw[:, p] = logits[:, 0]
                    inputs = logits[:, :, :20].argmax(-1)  # synthetic full upper-bound workload
                np.savez(out / ('synthetic_dev_%02d.npz' % ci), logits=raw.numpy(),probabilities=raw.softmax(-1).numpy(),
                    tokens=np.zeros((41,49),dtype=np.int64),raw_tokens=np.zeros((41,49),dtype=np.int64),
                    active_before=np.ones((41,49),dtype=bool),executed_positions=np.ones(49,dtype=bool))
            for arm, model in models.items():
                inputs = torch.full((32, 1), 21, dtype=torch.long)
                labels32 = torch.tensor([0]*16+[1]*16)
                zz = core.extend_z(mu[:32]) if model.latent else None
                hidden = None
                raw = torch.empty(32, 49, 21)
                for p in range(49):
                    logits, hidden = model(inputs, labels32, None if zz is None else zz[:, p:p+1], hidden)
                    raw[:, p] = logits[:, 0]
                    inputs = logits[:, :, :20].argmax(-1)
                np.savez(out / ('synthetic_gen_' + arm + '.npz'), logits=raw.numpy(),probabilities=raw.softmax(-1).numpy(),
                    tokens=np.zeros((32,49),dtype=np.int64),raw_tokens=np.zeros((32,49),dtype=np.int64),
                    active_before=np.ones((32,49),dtype=bool),uniforms=np.full((32,49),.5))
        eval_seconds = time.monotonic()-eval_start
        # Full128 distinct48-AA output scoring against279 synthetic48-AA references.
        def unique_seq(i):
            return ''.join(core.AA[(i//(20**j))%20] for j in range(4))+''.join(core.AA[(i+j)%20] for j in range(44))
        score_train=[{'sequence':unique_seq(i),'terminal_state':'FREE' if i<126 else 'AMD'} for i in range(279)]
        score_dev=[{'sequence':unique_seq(500+i),'terminal_state':'FREE' if i<18 else 'AMD'} for i in range(41)]
        score_slots=[{'arm':a,'terminal':t,'slot':i,'sequence':unique_seq(1000+ai*32+ti*16+i),
                      'status':'historical_ok' if a=='D' else 'eos'}
                     for ai,a in enumerate('ABCD') for ti,t in enumerate(('FREE','AMD')) for i in range(16)]
        score_start=time.monotonic()
        score_result=scoring.score_generation(score_slots,score_train,score_dev,budget)
        (out/'SYNTHETIC_SCORING.json').write_text(json.dumps(score_result,allow_nan=False))
        score_seconds=time.monotonic()-score_start
        projected_eval=2*(eval_seconds+score_seconds)+10
        projection_pass=all(x['projected1000_seconds_2x_plus_IO_and5']<=120 for x in training.values()) and projected_eval<=120
        result.update(status='PASS' if projection_pass else 'THROUGHPUT_BLOCKED', checks=checks,
                      training_benchmark=training, full_dev_gen_raw_export_seconds=eval_seconds,
                      scoring_seconds=score_seconds,projected_evaluation_and_scoring_seconds=projected_eval,
                      scoring_benchmark_executed=True, natural_admission=False,
                      synthetic_checks='PASS',throughput_checks='PASS' if projection_pass else 'BLOCKED',counts=count)
    except Exception as exc:
        import traceback
        result.update(status='FAILED', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(),
                      checks=checks, counts=count, natural_admission=False)
    finally:
        result['wall_seconds'] = time.monotonic()-started
        result['core_hash_unchanged'] = digest_file(args.core) == args.core_sha256
        result['plan_hash_unchanged'] = digest_file(args.plan) == args.plan_sha256
        result['discard_all_synthetic_weights'] = True
        (out/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'], 'counts': count}))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
