"""Stdlib mathematical oracle only; no tensors, models, data, fitting or sampling."""
import math

AA = 'ARNDCEQGHILKMFPSTWYV'
PAD = 20
CLASS_MAP = {0: 'FREE', 1: 'AMD'}


def targets(sequences):
    if not sequences:
        raise ValueError('empty training batch')
    out = []
    for sequence in sequences:
        if not 1 <= len(sequence) <= 48 or any(a not in AA for a in sequence):
            raise ValueError('training target outside canonical 1..48 contract')
        out.append([AA.index(a) for a in sequence] + [PAD] * (48-len(sequence)))
    return out


def logsumexp(row):
    m = max(row)
    return m + math.log(math.fsum(math.exp(x-m) for x in row))


def losses(scores, truth, density_difference):
    """Proposal, not a frozen H5 execution protocol: B*T normalization, beta=1.

    MSE is half squared distance to raw one-hot, summed over classes. It is NOT
    softmax Brier. CE is padded independent-position categorical NLL, NOT a
    normalized legal-prefix likelihood. Density is [B,C,T] sampled logq-logp.
    """
    b, t, k = len(scores), len(scores[0]), len(scores[0][0])
    if not b or not t or k < 2 or len(truth) != b or len(density_difference) != b:
        raise ValueError('shape')
    c = len(density_difference[0])
    if not c:
        raise ValueError('zero latent channels')
    mse = ce = 0.0
    for s, y in zip(scores, truth):
        if len(s) != t or len(y) != t:
            raise ValueError('ragged batch')
        for row, target in zip(s, y):
            if len(row) != k or not 0 <= target < k or not all(math.isfinite(v) for v in row):
                raise ValueError('score/target')
            mse += .5 * math.fsum((v-float(i == target))**2 for i,v in enumerate(row))
            ce += logsumexp(row)-row[target]
    density = []
    for record in density_difference:
        if len(record) != c or any(len(channel) != t for channel in record):
            raise ValueError('density shape')
        density.extend(v for channel in record for v in channel)
    if not all(math.isfinite(v) for v in density):
        raise ValueError('nonfinite density')
    d = math.fsum(density)
    denominator = b*t
    return {'half_onehot_square_sum': mse, 'ce_sum': ce,
            'density_difference_sum': d, 'position_denominator': denominator,
            'legacy_kl_sum_over_channels': d/c,
            'kl_per_record': d/b, 'kl_per_position': d/denominator,
            'mse_per_position': mse/denominator, 'ce_per_position': ce/denominator,
            'mse_total': (mse+d)/denominator, 'ce_total': (ce+d)/denominator}


def joint_decode(scores):
    """Truth-free shared J. CE constrained MAP; MSE nearest legal one-hot.

    Earliest boundary and lowest content class win exact ties. Empty/full-width
    are retained. This does not produce calibrated probabilities for MSE.
    """
    width, k = len(scores), len(scores[0])
    pad = k-1
    if any(len(row) != k or not all(math.isfinite(x) for x in row) for row in scores):
        raise ValueError('scores')
    aa = [max(range(pad), key=lambda a: row[a]) for row in scores]
    acc = math.fsum(row[pad] for row in scores)
    length_scores = [acc]
    for row, a in zip(scores, aa):
        acc += row[a]-row[pad]
        length_scores.append(acc)
    length = max(range(width+1), key=lambda n: length_scores[n])
    return {'length': length, 'tokens': aa[:length]+[pad]*(width-length),
            'length_scores': length_scores}


def generation_request(terminal_class, sample_id, seed):
    if terminal_class not in CLASS_MAP or not 0 <= sample_id < 16 or not isinstance(seed, int):
        raise ValueError('generation request')
    # No target, true length, individual descriptor, source ID or reconstruction.
    return {'class': terminal_class, 'terminal_state': CLASS_MAP[terminal_class],
            'sample_id': sample_id, 'seed': seed}
