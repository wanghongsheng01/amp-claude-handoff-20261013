#!/usr/bin/env python3
"""Independent synthetic worker. Inert until an exact root window is supplied.

No candidate import, model construction or RNG occurs at module import time.
Natural data are used only for read-only contracts; optimizer data are synthetic.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True)
    ap.add_argument('--window-sha256', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert file_sha(args.window) == args.window_sha256
    window = json.loads(Path(args.window).read_text())
    assert window['status'] == 'AUTHORIZED_SINGLE_SYNTHETIC_WINDOW'
    assert window['outer_seconds'] == 180 and window['forward_max'] == 132
    assert window['natural_execution_authorized'] is False
    out = Path(args.out).resolve()
    assert out == Path(window['output']).resolve() and out.is_dir()
    assert not (out / 'RESULT.json').exists()
    candidate = Path(window['candidate_root']).resolve()
    assert candidate.name == 'direct_z_impl_v1' and candidate.parent.name == 'amp_h5'
    for item in window['bindings']:
        assert file_sha(item['path']) == item['sha256'], item['path']
    assert any(Path(b['path']).resolve() == Path(__file__).resolve() for b in window['bindings'])
    config = json.loads(Path(window['candidate_config']).read_text())
    result = {'status': 'RUNNING', 'natural_forward': 0, 'actual_model_forward': 0,
              'optimizer_updates': 0, 'checks': [], 'window_sha256': args.window_sha256,
              'candidate_config_sha256': file_sha(window['candidate_config']),
              'worker_sha256': file_sha(__file__), 'natural_execution_authorized': False}

    def check_time():
        if time.monotonic() - started >= 180:
            raise TimeoutError('Synthetic180s exhausted; no retry or extension')

    def passed(name, details=None):
        check_time()
        result['checks'].append({'check': name, 'status': 'PASS', 'details': details})
        save_json(out / 'PROGRESS.json', result)

    phase = {'name': 'preparation'}

    def count_forward(module, inputs):
        check_time()
        result['actual_model_forward'] += 1
        assert result['actual_model_forward'] <= 132
        save_json(out / 'COUNTERS.json', {
            'actual_model_forward': result['actual_model_forward'],
            'optimizer_updates': result['optimizer_updates'], 'phase': phase['name'],
            'natural_forward': 0})

    try:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        global_initial = torch.get_rng_state().clone()
        sys.path.insert(0, str(candidate))
        mech = load_module('mechanics', candidate / 'mechanics.py')
        contracts = load_module('contracts', candidate / 'contracts.py')
        scoring = load_module('score', candidate / 'score.py')
        result['inherited_lock'] = contracts.verify_lock()
        assert torch.equal(global_initial, torch.get_rng_state())
        result['environment'] = {'python': sys.version, 'torch': torch.__version__,
                                 'numpy': np.__version__, 'threads': torch.get_num_threads(),
                                 'interop_threads': torch.get_num_interop_threads()}
        assert torch.__version__ == config['torch_version']

        def td(t):
            a = t.detach().cpu().contiguous()
            h = hashlib.sha256(str(a.dtype).encode() + b'\0' + str(tuple(a.shape)).encode() + b'\0')
            h.update(a.numpy().tobytes())
            return h.hexdigest()

        def sd(state):
            h = hashlib.sha256()
            for key, value in sorted(state.items()):
                h.update(key.encode() + b'\0' + td(value).encode())
            return h.hexdigest()

        # Exact read-only contracts. Nothing below passes these values to training.
        real_start = time.monotonic()
        contracts.verify(config)
        real = contracts.load_inputs(config)
        a = real['q'].numpy().astype(np.float64)
        means, stds = a.mean(axis=(0, 2, 3)), a.std(axis=(0, 2, 3), ddof=0)
        zm = a[:, 0].mean()
        zs = np.sqrt((a[:, 0] ** 2 + np.exp(2 * a[:, 1])).mean() - zm ** 2)
        ns = real['normalization']
        assert np.array_equal(means, np.asarray(ns['R_mean']))
        assert np.array_equal(stds, np.asarray(ns['R_std']))
        assert abs(zm - ns['Z_mean']) <= 1e-12 and abs(zs - ns['Z_std']) <= 1e-12
        result['readonly_contract_seconds'] = time.monotonic() - real_start
        passed('cache_identity_train_only_analytic_moments_and_reused_C_uniforms')

        # Initialization does not consume the surrounding global RNG stream.
        setup_start = time.monotonic()
        models, identity = mech.initialize_models()
        model_setup_seconds = time.monotonic() - setup_start
        assert torch.equal(global_initial, torch.get_rng_state())
        rs, zs_state = models['R'].state_dict(), models['Z'].state_dict()
        assert {arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()} == {
            'R': 18611970, 'Z': 18611329}
        boundary = ('inc.double_conv.0.weight', 'outc.weight', 'outc.bias')
        for key in zs_state:
            assert 'embed_level_' not in key
            expected = (rs[key][:, :1] * math.sqrt(2) if key == boundary[0]
                        else rs[key][:1] if key in boundary[1:] else rs[key])
            assert torch.equal(zs_state[key], expected), key
        for arm, model in models.items():
            model.register_forward_pre_hook(count_forward)
        initial_digests = {arm: sd(model.state_dict()) for arm, model in models.items()}
        assert identity['R_sha256'] == initial_digests['R']
        assert identity['Z_sha256'] == initial_digests['Z']
        passed('effective_capacity_and_exact_initialization_mapping', initial_digests)

        # Reproduce all private streams independently, including the batch tail.
        seal, states = mech.seal_randomness()
        independent_training = {}
        independent_states = {}
        for arm in ('R', 'Z'):
            gs = {k: torch.Generator(device='cpu').manual_seed(seed) for k, seed in
                  [('batches', 322), ('t', 3322), ('noise', 4322), ('dropout', 5322)]}
            if arm == 'Z':
                gs['q_epsilon'] = torch.Generator(device='cpu').manual_seed(6322)
            for key, gen in gs.items():
                independent_states[f'{arm}_train_initial_{key}'] = td(gen.get_state())
            records = []
            cursor, order = 279, None
            for step in range(1000):
                check_time()
                if cursor == 279:
                    order, cursor = torch.randperm(279, generator=gs['batches']), 0
                ids = order[cursor:min(cursor + 8, 279)]
                cursor += len(ids)
                t = torch.randint(1, 999, (len(ids),), generator=gs['t'])
                drop = bool(torch.rand((), generator=gs['dropout']) < .1)
                noise = torch.randn((len(ids), 2, 16, 48), generator=gs['noise'])
                eps = torch.randn((len(ids), 16, 48), generator=gs['q_epsilon']) if arm == 'Z' else None
                records.append({'ids': ids.tolist(), 't': t.tolist(), 'dropout': drop,
                                'full_noise_sha256': td(noise), 'first_channel_noise_sha256': td(noise[:, :1]),
                                'q_epsilon_sha256': td(eps) if eps is not None else None})
            assert records == seal['training'][arm]
            assert sum(len(row['ids']) for row in records) == 7972
            assert any(len(row['ids']) == 7 for row in records)
            independent_training[arm] = canonical_sha(records)
            for key, gen in gs.items():
                independent_states[f'{arm}_train_final_{key}'] = td(gen.get_state())
        for terminal, seed, reparam_seed in [('FREE', 7422, 8422), ('AMD', 7423, 8423)]:
            g = torch.Generator(device='cpu').manual_seed(seed)
            independent_states[terminal + '_generation_initial'] = td(g.get_state())
            hashes = []
            for draw in range(998):
                hashes.append(td(torch.randn((16, 2, 16, 48), generator=g)))
            assert hashes[0] == seal['generation'][terminal]['initial_noise_sha256']
            assert dict(zip(map(str, range(998, 1, -1)), hashes[1:])) == seal['generation'][terminal]['reverse_noise_sha256']
            independent_states[terminal + '_generation_final'] = td(g.get_state())
            g = torch.Generator(device='cpu').manual_seed(reparam_seed)
            independent_states[terminal + '_reparam_initial'] = td(g.get_state())
            assert td(torch.randn((16, 16, 48), generator=g)) == seal['reparameterization'][terminal]
            independent_states[terminal + '_reparam_final'] = td(g.get_state())
        assert independent_states == seal['generator_states_sha256']
        assert {k: td(v) for k, v in states.items()} == independent_states
        save_json(out / 'INDEPENDENT_RANDOM_SEAL.json', seal)
        torch.save(states, out / 'PRIVATE_GENERATOR_STATES.pt')
        schedules, schedule_sha = mech.scheduler_seal()
        assert len(schedules) == 1001
        frozen_A = torch.load(config['inputs']['A_checkpoint'], map_location='cpu', weights_only=True)
        assert frozen_A['updates'] == 1000 and sd(frozen_A['model']) == frozen_A['state_sha256']
        schedule = mech.noise_schedule()
        common_state = {k: v for k, v in rs.items() if k not in boundary}
        result['deterministic_preparation'] = {
            'R_initial_sha256': initial_digests['R'], 'Z_initial_sha256': initial_digests['Z'],
            'shared_trunk_sha256': sd(common_state),
            'random_seal_sha256': canonical_sha(seal),
            'initial_rng_sha256': {k: td(v) for k, v in identity.items() if torch.is_tensor(v)},
            'scheduler_sequence_sha256': schedule_sha,
            'noise_schedule_sha256': [td(t) for t in schedule],
            'normalization': ns, 'torch_version': torch.__version__,
            'A_state_sha256': sd(frozen_A['model']), 'uniforms_sha256': td(real['uniforms']),
            'posterior279_sha256': td(real['q'])}
        result['independent_stream_checks'] = {
            'training_records_canonical_sha256': independent_training,
            'generator_states_sha256': independent_states}
        passed('independent_full_private_stream_seal_and_exposure_counts')

        # Deterministic synthetic target; real q is never used by optimizer work.
        grid = torch.arange(279 * 16 * 48, dtype=torch.float32).reshape(279, 16, 48)
        q = torch.stack(((grid % 101 - 50) / 50,
                         -.5 + (grid % 31 - 15) / 100), dim=1)
        labels = torch.tensor([0] * 126 + [1] * 153, dtype=torch.long)
        stats = mech.normalization(q)
        for arm, target in [('R', q[:8]), ('Z', q[:8, :1])]:
            recovered = mech.inverse_normalize(mech.normalize(target, arm, stats), arm, stats)
            assert torch.allclose(recovered, target, atol=2e-7, rtol=2e-7)
        for bad in (-1000., 1000., float('nan')):
            try:
                mech.positive_sigma(torch.tensor([bad], dtype=torch.float32))
            except (FloatingPointError, AssertionError):
                pass
            else:
                raise AssertionError('Invalid sigma accepted')
        for bad in (torch.zeros_like(q), torch.full_like(q, float('nan'))):
            try:
                mech.normalization(bad)
            except (FloatingPointError, AssertionError):
                pass
            else:
                raise AssertionError('Invalid normalization accepted')
        rz = torch.stack((torch.full((2, 16, 48), 2.), torch.zeros(2, 16, 48)), dim=1)
        assert torch.equal(mech.to_decoder_z(rz, 'R', torch.full((2, 16, 48), 3.)), torch.full((2, 16, 48), 5.))
        assert torch.equal(mech.to_decoder_z(rz[:, :1], 'Z'), rz[:, 0])
        for bad in (-1000., 1000.):
            broken = rz.clone(); broken[:, 1] = bad
            try:
                mech.to_decoder_z(broken, 'R', torch.ones(2, 16, 48))
            except FloatingPointError:
                pass
            else:
                raise AssertionError('R decoder boundary silently accepted zero/inf sigma')
        passed('float32_sigma_failure_and_single_inverse_transform')

        # Arithmetic oracle is a non-model callable; does not consume forwards.
        schedule = mech.noise_schedule()
        beta, alpha, abar = schedule
        assert torch.equal(beta, torch.linspace(1e-4, .02, 999))
        class ConstantPrediction:
            def __call__(self, x, t, labels):
                return torch.ones_like(x) * .25
        x = torch.ones(2, 1, 16, 48) * .75
        noise = torch.ones_like(x) * 2
        for t in (998, 1):
            y = mech.reverse_step(ConstantPrediction(), x, t, torch.tensor([0, 1]), noise, schedule)
            expected = (x - ((1 - alpha[t]) / torch.sqrt(1 - abar[t])) * .25) / torch.sqrt(alpha[t])
            if t > 1:
                expected += beta[t].sqrt() * noise
            assert torch.equal(y, expected)
        passed('independent_ancestral_reverse_arithmetic_and_t1_no_noise')

        # Scripted logits exercise frozen decode boundaries without a model call.
        ar = contracts.load_ar_core(config)
        class ScriptedDecoder:
            latent = True
            def __init__(self): self.position = 0; self.inputs = []; self.latents = []
            def __call__(self, tokens, labels, latent, hidden):
                p = self.position; self.position += 1
                self.inputs.append(tokens.clone()); self.latents.append(latent.clone())
                logits = torch.full((4, 1, 21), -1000.)
                logits[:, :, 0] = 0
                logits[0, 0, 0] = -1000; logits[0, 0, 20] = 0
                if p == 48:
                    logits[1, 0, 0] = -1000; logits[1, 0, 20] = 0
                if p == 0: logits[3, 0] = 0
                return logits, None
        scripted = ScriptedDecoder()
        uniforms = torch.full((4,49), .25)
        uniforms[3,0] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
        fake_z = torch.arange(4*16*48, dtype=torch.float32).reshape(4,16,48) / 10000
        decoded = ar.decode(scripted, torch.tensor([0,0,1,1]), z=fake_z, uniforms=uniforms)
        assert decoded['status'] == ['empty_sequence','eos','cap_nonEOS','empty_sequence']
        assert decoded['sequences'] == ['', 'A'*48, 'A'*48, '']
        assert decoded['steps'] == 49 and decoded['raw_tokens'][3,0] == 20
        assert all(torch.equal(x[:,0], torch.tensor([20,0,0,20])) for x in scripted.inputs[1:])
        assert torch.equal(scripted.latents[0][:,0], fake_z[:,:,0])
        assert torch.equal(scripted.latents[48], torch.zeros(4,1,16))
        assert not bool(decoded['active_before'][0,1:].any())
        assert not bool(decoded['active_before'][3,1:].any())
        passed('scripted_EOS49_absorption_cap_CDF_near1_and_latent49_zero')

        # Scorer definition parity and failure denominator; no model calls.
        train = [{'sequence': 'ARNDCEQGHILKMFPSTWYV'[:5 + i % 16],
                  'terminal_state': 'FREE' if i < 126 else 'AMD'} for i in range(279)]
        dev = [{'sequence': 'ARNDC', 'terminal_state': 'FREE' if i < 18 else 'AMD'} for i in range(41)]
        slots = [{'arm': arm, 'terminal': terminal, 'slot': j,
                  'sequence': 'ARNDC', 'status': 'eos'}
                 for arm in ('R', 'Z', 'C') for terminal in ('FREE', 'AMD') for j in range(16)]
        scored = scoring.score_generation(copy.deepcopy(slots), train, dev, check_time)
        assert scored['status'] == 'FAIL_MECHANISM_SCREEN'
        for arm in ('R', 'Z', 'C'):
            for terminal in ('FREE', 'AMD'):
                s = scored['summary'][arm][terminal]
                assert s['requests'] == s['completed_n'] == s['valid_nonempty_n'] == 16
                assert s['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features'] <= 1e-28
                assert s['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits'] == 0
                assert s['copy_train_n'] == 16 and s['duplicates_rate'] == 15 / 16
        for arm, expected in [('Z', 'FAIL_CANDIDATE'), ('R', 'NOT_EVALUABLE'), ('C', 'NOT_EVALUABLE')]:
            for seq, status in [('', 'empty_sequence'), (None, 'not_executed'), ('X', 'eos')]:
                changed = copy.deepcopy(slots)
                i = next(i for i, row in enumerate(changed) if row['arm'] == arm)
                changed[i].update(sequence=seq, status=status)
                test = scoring.score_generation(changed, train, dev, check_time)
                assert test['status'] == expected
                assert len(test['slots']) == 96 and test['summary'][arm]['FREE']['requests'] == 16
        original_compare = scoring.frozen.compare_summaries
        try:
            for candidate_value, baseline_value, expected in [(.9, 1., 'SCREEN_POSITIVE_HOLD'),
                    (.900000000000001, 1., 'FAIL_MECHANISM_SCREEN'), (0., 0., 'FAIL_MECHANISM_SCREEN')]:
                calls = {'n': 0}
                def injected_compare(a, b):
                    arm = ('R', 'Z', 'C')[calls['n'] // 2]
                    calls['n'] += 1
                    return {'mmd2_linear_biased_fixed_features': candidate_value if arm == 'Z' else baseline_value,
                            'dipeptide_jensen_shannon_bits': .1 if arm == 'Z' else .2}
                scoring.frozen.compare_summaries = injected_compare
                assert scoring.score_generation(copy.deepcopy(slots), train, dev, check_time)['status'] == expected
        finally:
            scoring.frozen.compare_summaries = original_compare
        passed('R_Z_C_96_slot_denominators_empty_failure_ties_zero_and_threshold')

        # Whole32-step benchmarks; never select a faster subset or change schedule1000.
        benchmark = {}
        for arm in ('R', 'Z'):
            phase['name'] = arm + '_synthetic_training'
            setup = time.monotonic()
            opt, sch = mech.optimizer_and_scheduler(models[arm])
            setup_seconds = model_setup_seconds + time.monotonic() - setup
            stream = mech.TrainingRandom(arm)
            train_start = time.monotonic()
            conditional_seen = False
            captured = {}
            prediction_hook = models[arm].register_forward_hook(
                lambda module, inputs, output: captured.update(prediction=output.detach().clone()))
            with (out / (arm + '_synthetic_training.jsonl')).open('x') as handle:
                for step in range(1, 33):
                    step_start = time.monotonic()
                    rnd = stream.next()
                    record = mech.random_record(rnd)
                    assert record == seal['training'][arm][step - 1]
                    def after_update():
                        result['optimizer_updates'] += 1
                        save_json(out / 'COUNTERS.json', {
                            'actual_model_forward': result['actual_model_forward'],
                            'optimizer_updates': result['optimizer_updates'], 'phase': phase['name'],
                            'natural_forward': 0})
                    row = mech.training_step(models[arm], opt, sch, q, labels, arm, stats, rnd, schedule,
                                             after_update=after_update)
                    assert row['objects'] == len(record['ids'])
                    assert row['elements'] == len(record['ids']) * (2 if arm == 'R' else 1) * 16 * 48
                    squared = (captured['prediction'].double() - rnd['noise'].double()).square()
                    independent_mse = float(squared.sum() / squared.numel())
                    assert math.isclose(row['mse'], independent_mse, rel_tol=2e-6, abs_tol=1e-7)
                    assert row['lr_used'] == schedules[step - 1]['lr']
                    assert row['beta1_used'] == schedules[step - 1]['beta1']
                    conditional_seen |= not rnd['dropout']
                    handle.write(json.dumps({'step': step, **record, **row,
                                             'seconds': time.monotonic() - step_start,
                                             'elapsed_stage_seconds': time.monotonic() - train_start,
                                             'rng_after_sha256': {k: td(s) for k, s in stream.states().items()}},
                                            allow_nan=False) + '\n')
                    handle.flush()
                    check_time()
            prediction_hook.remove()
            training_seconds = time.monotonic() - train_start
            assert conditional_seen
            save_start = time.monotonic()
            torch.save({'synthetic_only': True, 'completed_synthetic_updates': 32,
                        'model': models[arm].state_dict(), 'optimizer': opt.state_dict(),
                        'scheduler': sch.state_dict(), 'rng_states': stream.states(),
                        'state_sha256': sd(models[arm].state_dict()),
                        'ema_state_sha256': sd(models[arm].state_dict())}, out / (arm + '_SYNTHETIC_ONLY.pt'))
            save_json(out / (arm + '_SYNTHETIC_SAVE_CHECK.json'), {
                'state_sha256': sd(models[arm].state_dict()), 'synthetic_only': True})
            # Include the ten small monitoring writes that occur in natural1000.
            for milestone in range(100, 1001, 100):
                save_json(out / (arm + '_SYNTHETIC_MONITORS.json'), {'step': milestone, 'synthetic_only': True})
            checkpoint_seconds = time.monotonic() - save_start
            benchmark[arm] = {'training32_seconds': training_seconds,
                              'setup_seconds_conservative': setup_seconds,
                              'checkpoint_seconds': checkpoint_seconds,
                              'training1000_forecast_seconds': 1.2 * (1000 / 32 * training_seconds) + setup_seconds + checkpoint_seconds + 20}
            del opt, sch
            phase['name'] = arm + '_synthetic_generation'
            models[arm].eval()
            stream = mech.GenerationRandom('FREE')
            full = stream.noise()
            gx = full if arm == 'R' else full[:, :1]
            generation_start = time.monotonic()
            with (out / (arm + '_synthetic_reverse.jsonl')).open('x') as handle:
                for t in range(998, 966, -1):
                    step_start = time.monotonic()
                    full = stream.noise()
                    noise_sha = td(full)
                    assert noise_sha == seal['generation']['FREE']['reverse_noise_sha256'][str(t)]
                    noise = full if arm == 'R' else full[:, :1]
                    gx = mech.reverse_step(models[arm], gx, t, torch.zeros(16, dtype=torch.long), noise, schedule)
                    handle.write(json.dumps({'t': t, 'full_noise_sha256': noise_sha, 'u_sha256': td(gx),
                                             'finite': True, 'seconds': time.monotonic() - step_start}) + '\n')
                    handle.flush()
                    check_time()
            generation_seconds = time.monotonic() - generation_start
            benchmark[arm].update({'generation32_seconds': generation_seconds,
                                  'generation1996_forecast_seconds': 1.2 * (1996 / 32 * generation_seconds) + 30})
            passed(arm + '_full_path_synthetic32_training_and_reverse', benchmark[arm])
        result['benchmark'] = benchmark
        assert result['optimizer_updates'] == 64 and result['actual_model_forward'] == 128
        assert torch.equal(global_initial, torch.get_rng_state()), 'Unisolated global RNG consumption'
        result['synthetic_checks'] = True
        result['throughput_checks'] = all(x['training1000_forecast_seconds'] <= 720 and
                                          x['generation1996_forecast_seconds'] <= 1440 for x in benchmark.values())
        if not result['throughput_checks']:
            raise RuntimeError('Measured conservative throughput does not fit frozen phase caps')
        check_time()
        result['status'] = 'PASS'
    except Exception as exc:
        result['status'] = 'BLOCKED'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['retry_allowed'] = False
        save_json(out / 'RESULT.json', result)
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
