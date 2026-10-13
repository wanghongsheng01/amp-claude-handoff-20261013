"""Analytic zero-epsilon reference; no model, RNG, sampling, or test access."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEDULE = Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/diagnostics_h4_h2_generation_bridge_v1/run/SCHEDULE.npz')
SAVED = HERE / 'SAVED_NUMERIC_AUDIT.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with np.load(SCHEDULE, allow_pickle=False) as schedule:
    beta = schedule['beta'].astype(np.float64)
    alpha = schedule['alpha'].astype(np.float64)
assert len(beta) == len(alpha) == 999
assert np.all((alpha > 0) & (alpha < 1))

# Reverse steps 998,...,1. Added noise is independent N(0, beta[t])
# except at t=1, where the implementation uses zero new noise.
variance = initial_part = 1.0
noise_part = 0.0
for t in range(998, 0, -1):
    added = float(beta[t]) if t > 1 else 0.0
    variance = variance / float(alpha[t]) + added
    initial_part /= float(alpha[t])
    noise_part = noise_part / float(alpha[t]) + added

# Independent closed-form check. Noise at t is amplified by steps t-1,...,1.
prefix = [1.0]
for t in range(1, 999):
    prefix.append(prefix[-1] / float(alpha[t]))
closed_form = prefix[998] + math.fsum(float(beta[t]) * prefix[t-1] for t in range(2, 999))
assert math.isclose(variance, closed_form, rel_tol=1e-12)
reference_std = math.sqrt(variance)
saved = json.loads(SAVED.read_text())
comparison = []
for arm, values in saved['h5_smoke'].items():
    for terminal, result in values['generated'].items():
        for coordinate in ('mu', 'logsigma'):
            summary = result[coordinate]
            comparison.append(dict(source='h5_smoke', arm=arm, terminal=terminal,
                coordinate=coordinate, empirical_population_std=summary['std'],
                std_over_analytic_reference=summary['std']/reference_std,
                min=summary['min'], max=summary['max'], n=summary['n']))
for terminal, result in saved['h4_saved_generation'].items():
    for coordinate in ('mu', 'logsigma'):
        summary = result[coordinate]
        comparison.append(dict(source='h4_saved_generation', terminal=terminal,
            coordinate=coordinate, empirical_population_std=summary['std'],
            std_over_analytic_reference=summary['std']/reference_std,
            min=summary['min'], max=summary['max'], n=summary['n']))

output = dict(
    identity='AMP_H5_ZERO_PREDICTOR_ANALYTIC_REFERENCE',
    no_model_calls=True, no_model_loads=True, no_rng=True,
    no_sampling=True, no_training=True, no_test_reads=True,
    assumptions=['epsilon prediction identically zero at every reverse step',
                 'initial x has independent N(0,1) coordinates',
                 'fresh reverse noises independent of initial x and each other',
                 'last t=1 step adds no new noise'],
    frozen_schedule=dict(length=999, reverse_first_t=998, reverse_last_t=1,
                         beta_first=float(beta[0]), beta_last=float(beta[-1]),
                         source_storage_dtype='float32', scalar_calculation_dtype='float64',
                         alpha0_used_in_this_recurrence=False),
    recurrence='V_after = V_before / alpha[t] + (beta[t] if t > 1 else 0)',
    closed_form='prod(j=1..998,1/alpha[j]) + sum(t=2..998,beta[t]*prod(j=1..t-1,1/alpha[j]))',
    analytic=dict(terminal_mean=0.0, terminal_variance=variance, terminal_std=reference_std,
                  initial_variance_contribution=initial_part,
                  reverse_noise_variance_contribution=noise_part,
                  initial_only_std=math.sqrt(initial_part),
                  reverse_noise_only_std=math.sqrt(noise_part),
                  closed_form_variance=closed_form,
                  recurrence_closed_form_abs_difference=abs(variance-closed_form)),
    empirical_comparison=comparison,
    interpretation_limits=['Similar empirical scale does not establish zero or near-zero model predictions.',
                           'This is not a causal diagnosis of overflow or an identified training defect.',
                           'H4 and smoke are saved outputs from different training maturity and seeds/batches.',
                           'Fixed-representation finite-step training remains the proposed maturity test, not an executed result here.'],
    input_sha256={str(SCHEDULE):sha(SCHEDULE),str(SAVED):sha(SAVED)})
(HERE / 'ZERO_PREDICTOR_ANALYTIC.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(output['analytic'], indent=2))
