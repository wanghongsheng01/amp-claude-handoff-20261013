#!/usr/bin/env python3
"""Read-only author report for a closed zero-training recovery, with split provenance.

Never imports the recovery/model runner, deserializes weights, or reads source CSVs.
Original formal E/L/G remain NOT_EVALUABLE_INTERRUPTION regardless of this report.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def check_manifest(directory, before):
    manifest = read(directory / 'manifest.json')
    if manifest is None:
        return ['manifest missing']
    return [x['path'] for x in manifest.get('files', [])
            if Path(x['path']).name != x['path'] or before.get(x['path']) != x['sha256']]


def file_snapshot(directory):
    return {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}


def export_schema(path, expected_rows):
    if not path.exists():
        return {'present': False, 'rows': 0, 'complete_identity_schema': False}
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    ok = (len(rows) == expected_rows and [r.get('row_id') for r in rows] == [str(i) for i in range(expected_rows)]
          and all(r.get('terminal') in ('FREE', 'AMD') and r.get('status') == 'ok' for r in rows))
    return {'present': True, 'sha256': sha(path), 'rows': len(rows), 'expected_rows': expected_rows,
            'complete_identity_schema': ok, 'terminal_counts': dict(collections.Counter(r.get('terminal') for r in rows)),
            'status_counts': dict(collections.Counter(r.get('status') for r in rows)),
            'quality_metrics_computed': False, 'truth_and_predictions_not_emitted': True}


def phase_cost(contract, status, job, generation):
    old = contract['cost_provenance']
    train_sum = old['original_vae_stage_seconds'] + old['original_ddpm_stage_seconds']
    recovered_wall = job.get('seconds')
    lower = train_sum + old['original_generation_seconds_lower_bound'] + recovered_wall if recovered_wall is not None else None
    return {'original_training': {'VAE_seconds': old['original_vae_stage_seconds'], 'DDPM_seconds': old['original_ddpm_stage_seconds'],
                'recorded_stage_subtotal_seconds': train_sum, 'source': contract['original_run_directory']},
            'original_interrupted_generation': {'seconds_lower_bound': old['original_generation_seconds_lower_bound'],
                'exact_end_or_total_unknown': True, 'FREE_partial_requests': 16, 'FREE_reverse_steps_each_at_least': 799,
                'AMD_start_evidence': False, 'final_sequences': 0},
            'new_recovery': {'new_training_updates': status.get('training_updates_this_attempt'),
                'setup_seconds': status.get('setup_seconds'), 'generation_seconds': generation.get('elapsed_seconds') if generation else None,
                'export_seconds': status.get('export_seconds'), 'process_seconds_before_final_status_io': status.get('process_seconds_before_final_status_io'),
                'supervisor_job_seconds': recovered_wall, 'caps': contract['budget_seconds'],
                'generation_cap_exceeded_recorded': status.get('generation_cap_exceeded'), 'outer_cap_exceeded_recorded': status.get('outer_cap_exceeded')},
            'recorded_disjoint_path_cost_lower_bound_seconds': lower,
            'lower_bound_definition': 'Original two training stage times + observed original partial generation lower bound + new supervisor job wall time. No double-count of new setup/generation/export inside its wall time.',
            'excluded_or_unknown': 'Original unlogged work and setup/export gap, smoke, DDPM maturity diagnostic, zero-update entry failure, review/report costs are not included. This is not total research compute or equal-compute evidence.'}


def run(request_path, output):
    request_path = Path(request_path).resolve()
    request = read(request_path)
    if sha(__file__) != request['script_sha256']:
        raise ValueError('author script binding differs')
    for item in request['fixed_bindings']:
        p = Path(item['path'])
        if not p.is_absolute() or p.is_symlink() or sha(p) != item['sha256']:
            raise ValueError('fixed binding differs: ' + str(p))
    original = Path(request['original_run_directory']).resolve()
    recovery = Path(request['recovery_run_directory']).resolve()
    supervisor_path = Path(request['supervisor_status_path']).resolve()
    if original != ROOT / 'runs/formal_v2/categorical_ce' or recovery != ROOT / 'runs/ce_checkpoint_recovery_v1':
        raise ValueError('phase roots differ from author request scope')
    if supervisor_path != ROOT / 'runs/ce_checkpoint_recovery_execution_v1/SUPERVISOR_STATUS.json':
        raise ValueError('supervisor scope differs')
    supervisor = read(supervisor_path)
    if supervisor is None or supervisor.get('state') not in ('completed', 'completed_with_failures', 'failed'):
        raise ValueError('recovery supervisor is not terminal; do not read live natural outputs')
    supervisor_hash = sha(supervisor_path)
    if supervisor.get('plan_sha256') != request['recovery_plan_sha256']:
        raise ValueError('executed recovery plan binding differs')
    jobs = [j for j in supervisor.get('jobs', []) if j.get('id') == 'ce_checkpoint_recovery_v1']
    if len(jobs) != 1 or jobs[0].get('state') not in ('completed', 'failed', 'launcher_failed'):
        raise ValueError('recovery job has no terminal state')
    job = jobs[0]
    argv = job['argv']
    if Path(argv[argv.index('--output')+1]).resolve() != recovery:
        raise ValueError('supervisor actual output differs')
    contract = read(Path(request['contract_path']))
    input_errors = []
    expected_original = {x['path']: x['sha256'] for x in contract['original_25_file_snapshot']}
    actual_original = {str(p) for directory in (original, Path(contract['original_supervisor_directory'])) for p in directory.iterdir() if p.is_file()}
    if set(expected_original) != actual_original or len(expected_original) != 25:
        input_errors.append('original 25-file membership differs')
    for path, expected_sha in expected_original.items():
        if sha(path) != expected_sha:
            input_errors.append(path)
    if input_errors:
        raise ValueError('original bound artifacts changed: ' + json.dumps(input_errors))
    before_old, before_new = file_snapshot(original), file_snapshot(recovery)
    status = read(recovery / 'status.json') or {'state': 'missing'}
    blockers = []
    if status.get('state') != 'recovery_completed' or job.get('exit_code') != 0:
        blockers.append('new recovery not successfully completed; do not infer completion from partial artifacts')
    if status.get('training_updates_this_attempt') != 0:
        blockers.append('zero-training status not established')
    copied_training = [name for name in before_new if name in ('vae_train.jsonl', 'ddpm_train.jsonl', 'dev_monitor.jsonl',
        'train_posterior_parameters.npz', 'train_feature_scaler.json') or name.endswith('.pt')]
    if copied_training:
        blockers.append('training artifacts unexpectedly present in recovery directory')
    phase_provenance = read(recovery / 'phase_provenance.json')
    if phase_provenance != contract['phase_provenance']:
        blockers.append('saved phase provenance differs from frozen recovery contract')
    manifest_errors = check_manifest(recovery, before_new)
    if manifest_errors:
        blockers.append('recovery manifest missing or inconsistent')
    helper_path = Path(request['numeric_helper_path'])
    if helper_path != ROOT / 'implementation/author_terminal_engineering_readout_v1.py':
        raise ValueError('unexpected numeric helper')
    spec = importlib.util.spec_from_file_location('h5_author_saved_artifact_helpers', helper_path)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    # The imported helper is a pure artifact reader; it has no torch import/model work.
    np = h.np
    vae = h.json_lines(original / 'vae_train.jsonl'); ddpm = h.json_lines(original / 'ddpm_train.jsonl')
    training = {'source_directory': str(original), 'never_copied_to_recovery': True,
                'vae': h.curves(vae, ['loss', 'reconstruction', 'kl', 'lr', 'duration_seconds', 'elapsed_stage_seconds'], 1000),
                'ddpm': h.curves(ddpm, ['mse', 'lr', 'duration_seconds', 'elapsed_stage_seconds'], 1000),
                'saved_dev_curve': h.json_lines(original / 'dev_monitor.jsonl'), 'checkpoint_selection': False}
    latent = {'train_cache_source': str(original / 'train_posterior_parameters.npz'), 'train': {}, 'generated': {}, 'failure_tensors': {}}
    with np.load(original / 'train_posterior_parameters.npz', allow_pickle=False) as z:
        values, labels, ids = z['values'], z['labels'], z['row_ids']
        if len(values) != 279 or ids.tolist() != list(range(279)) or len(labels) != 279 or not np.isin(labels, [0, 1]).all():
            raise ValueError('train-cache row identity differs')
        latent['train'] = {'pooled': h.parameter_summary(values), 'by_terminal': {
            t: {'rows': int((labels == i).sum()), **h.parameter_summary(values[labels == i])} for i, t in enumerate(('FREE', 'AMD'))}}
    for terminal in ('FREE', 'AMD'):
        path = recovery / f'generated_posterior_{terminal}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                saved_parameters = z['values']
                entry = h.parameter_summary(saved_parameters)
                entry['rows'] = len(saved_parameters)
            complete_path = recovery / f'generated_{terminal}.npz'
            if complete_path.exists():
                with np.load(complete_path, allow_pickle=False) as z:
                    entry['saved_z'] = h.moments(z['z']); entry['saved_decoder_raw'] = h.moments(z['raw'])
                    entry['posterior_values_equal_across_saved_files'] = bool(np.array_equal(z['posterior_parameters'], saved_parameters, equal_nan=True))
            entry['same_terminal_scale_comparison'] = h.compare_ranges(entry, latent['train']['by_terminal'][terminal])
            entry['pooled_scale_comparison'] = h.compare_ranges(entry, latent['train']['pooled'])
            latent['generated'][terminal] = entry
        fail = recovery / f'generation_failure_{terminal}.npz'
        if fail.exists():
            with np.load(fail, allow_pickle=False) as z:
                latent['failure_tensors'][terminal] = {'phase': read(recovery / f'generation_failure_{terminal}.json'),
                    'tensors': {key: h.moments(z[key]) for key in z.files}}
    rows = []
    if (recovery / 'generated.csv').exists():
        with (recovery / 'generated.csv').open(newline='') as fh:
            rows = [{key: row[key] for key in ('sample_id', 'terminal', 'terminal_sample_id', 'seed', 'status', 'length')} for row in csv.DictReader(fh)]
    expected = {(t, str(i)) for t in ('FREE', 'AMD') for i in range(16)}
    actual = [(r['terminal'], r['terminal_sample_id']) for r in rows]
    slot_coverage = len(actual) == 32 and set(actual) == expected
    if not slot_coverage:
        blockers.append('not all32 preallocated recovery slots are preserved exactly once')
    generation_summary = read(recovery / 'generation_summary.json')
    generation = {'source_directory': str(recovery), 'attempt_id': 'ce_checkpoint_recovery_v1',
        'expected_requests': 32, 'unique_slot_coverage': slot_coverage, 'slots': rows,
        'status_counts': dict(collections.Counter(r['status'] for r in rows)), 'summary': generation_summary,
        'errors': h.json_lines(recovery / 'generation_errors.jsonl'), 'progress': h.json_lines(recovery / 'generation_progress.jsonl'),
        'placeholder_warning': 'In an interrupted recovery, not_executed CSV slots may contain already-started partial computation; consult progress, never infer zero cost.'}
    exports = {role: export_schema(recovery / f'{role}_reconstruction.csv', n) for role, n in [('train', 279), ('dev', 41)]}
    identity = read(recovery / 'model_identity_after.json')
    integrity = read(recovery / 'input_integrity_after.json')
    if not identity or identity.get('both_loaded_verified_and_unchanged') is not True:
        blockers.append('unchanged loaded model identities not established')
    elif identity.get('before') != contract['checkpoint_tensor_digests'] or identity.get('after') != contract['checkpoint_tensor_digests']:
        blockers.append('reported loaded model digests differ from original checkpoint bindings')
    if not integrity or integrity.get('original_errors') or integrity.get('core_errors'):
        blockers.append('recovery input-integrity closeout not established')
    if before_old != file_snapshot(original) or before_new != file_snapshot(recovery) or supervisor_hash != sha(supervisor_path):
        raise RuntimeError('input artifacts changed during readout; no report emitted')
    report = {'report_role': 'AUTHOR_RECOVERY_ENGINEERING_READOUT_NOT_INDEPENDENT_APPROVAL',
        'original_E': 'NOT_EVALUABLE_INTERRUPTION', 'original_L': 'NOT_EVALUABLE_INTERRUPTION', 'original_G': 'NOT_EVALUABLE_INTERRUPTION',
        'RECOVERY_ENGINEERING': 'REQUIRES_INDEPENDENT_POSTFLIGHT', 'PROMOTE': 'HOLD', 'AR_ADEQUACY': 'NOT_ESTABLISHED',
        'original_training_directory': str(original), 'new_generation_directory': str(recovery),
        'phase_provenance': phase_provenance, 'unexpected_training_artifacts_in_recovery': copied_training,
        'status': status, 'supervisor_job': job,
        'training': training, 'latent': latent, 'generation': generation, 'reconstruction_export_schema': exports,
        'cost': phase_cost(contract, status, job, generation_summary), 'model_identity_after': identity, 'input_integrity_after': integrity,
        'author_blockers': blockers, 'recovery_manifest_errors': manifest_errors,
        'original_artifact_hashes': before_old, 'recovery_artifact_hashes': before_new,
        'supervisor_sha256': supervisor_hash, 'request_sha256': sha(request_path), 'script_sha256': sha(__file__),
        'numeric_helper_sha256': sha(helper_path), 'input_stable_during_read': True,
        'limitations': ['No model calls, weight deserialization, natural source CSV reads, new sequence-quality calculation or test access.',
            'NumPy exp of saved logsigma is labelled derived diagnostic, not torch bitwise replay; actual finite-positive gate is recorded by the frozen generation path.',
            'Pooled and terminal-matched moments/ranges are descriptive correlated coordinates, not distribution fit or activity evidence.',
            'Saved dev curve is transcribed without checkpoint or parameter selection; export checks concern schema only.',
            'Original interrupted attempt remains incomplete; same-seed generation restart adds disclosed cost and violates original no-retry contract under a prospective amendment.',
            'No next-arm execution or scientific claim is authorized by this author report.']}
    output = Path(output).resolve()
    if output.exists() or output.parent != ROOT / 'feasibility':
        raise ValueError('fresh author report directory under feasibility required')
    output.mkdir()
    dump(output / 'RECOVERY_AUTHOR_READOUT.json', report)
    dump(output / 'ORIGINAL_SAVED_TRAINING_CURVES.json', training)
    cost = report['cost']
    text = ['# CE recovery author engineering readout', '',
        f"Recovery state: {status.get('state')}; supervisor exit: {job.get('exit_code')}.",
        'Original E/L/G: NOT_EVALUABLE_INTERRUPTION. Recovery engineering requires independent postflight.',
        f"Saved recovery statuses: {generation['status_counts']}; complete32-slot coverage: {slot_coverage}.",
        f"Training updates in recovery: {status.get('training_updates_this_attempt')}.", '',
        f"Original VAE/DDPM recorded stage subtotal: {cost['original_training']['recorded_stage_subtotal_seconds']:.6f} seconds.",
        f"Original partial generation: at least {cost['original_interrupted_generation']['seconds_lower_bound']:.6f} seconds; exact end unknown.",
        f"New recovery wall time: {job.get('seconds')} seconds; its setup/generation/export subphases are reported separately in JSON, not summed again.",
        '', 'Detailed finite/scale diagnostics, saved curves, source hashes and export-schema checks are in RECOVERY_AUTHOR_READOUT.json.',
        'Finite values do not establish quality, distribution fit or activity. No sequence-quality metrics were computed. PROMOTE remains HOLD.']
    (output / 'READOUT.md').write_text('\n'.join(text)+'\n')
    dump(output / 'MANIFEST.json', {'files': [{'path': p.name, 'sha256': sha(p)} for p in sorted(output.iterdir()) if p.is_file()]})
    print(json.dumps({'output': str(output), 'recovery_state': status.get('state'), 'author_blockers': blockers, 'model_calls': 0}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.request, a.output)
