"""Pure standard-library RMS metric primitives; no data loading or scoring on import.

Feature differences are prediction minus reference in the supplied frozen 421
coordinates. Invalid/empty feature inputs produce missing values, never zeros.
These functions do not load a model, choose samples, or change the supplied text.
"""
from __future__ import annotations

import math
from decimal import Decimal
from numbers import Real
from typing import Callable, Iterable

AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(AA)
_BLOCKS = (("length1", 0, 1), ("aa20", 1, 21), ("dipeptide400", 21, 421))


def _levenshtein(reference: str, prediction: str) -> int:
    """Unit insertion/deletion/substitution distance, with linear working memory."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def pair_core(reference: str, prediction: str) -> dict:
    """Literal positional accuracy plus edit and length metrics.

    Each target position is one denominator unit. Missing predictions are wrong;
    extra predictions are accounted for by edit/length metrics, not accuracy.
    Empty reference accuracy is undefined (None), including the both-empty case.
    No alphabet, case, whitespace, padding, or alignment normalization is applied.
    """
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise TypeError("reference and prediction must be strings")
    target = len(reference)
    correct = sum(a == b for a, b in zip(reference, prediction))
    distance = _levenshtein(reference, prediction)
    maximum_length = max(target, len(prediction))
    signed_length = len(prediction) - target
    return {
        "target_residues": target,
        "correct_target_positions": correct,
        "nonpad_accuracy": correct / target if target else None,
        "whole_exact": reference == prediction,
        "levenshtein_raw": distance,
        "levenshtein_normalized": distance / maximum_length if maximum_length else 0.0,
        "length_signed": signed_length,
        "length_absolute": abs(signed_length),
    }


def _finite_number(value) -> float | None:
    # Reject bools/numeric strings rather than silently interpreting them as scores.
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    try:
        converted = float(value)
    except (ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _input_problem(value, role: str) -> str | None:
    if not isinstance(value, str):
        return role + "_not_string"
    if not value:
        return role + "_empty"
    if len(value) > 48:
        return role + "_length_gt48"
    if not set(value) <= _AA_SET:
        return role + "_noncanonical"
    return None


def _missing_features(status: str) -> dict:
    return {"valid": False, "status": status, "delta421": None,
            "blocks": {name: None for name, _, _ in _BLOCKS}}


def _feature_values(sequence: str, feature_fn: Callable) -> list[float]:
    values = list(feature_fn(sequence))
    if len(values) != 421:
        raise ValueError("feature_fn must return exactly 421 coordinates")
    converted = [_finite_number(value) for value in values]
    if any(value is None for value in converted):
        raise ValueError("feature_fn returned nonnumeric or nonfinite coordinates")
    return converted


def feature_delta(reference: str, prediction: str, feature_fn: Callable) -> dict:
    """Apply the caller's frozen feature_fn only to two valid canonical strings.

    No model/feature implementation is imported here. Blocks use coordinates
    [0:1], [1:21], [21:421], without rescaling or removing their frozen weights.
    Feature exceptions are retained as a typed status with all numeric outputs
    missing; there is no retry, repair, imputation, or partial-block success.
    """
    problems = [_input_problem(reference, "reference"),
                _input_problem(prediction, "prediction")]
    problems = [problem for problem in problems if problem is not None]
    if problems:
        return _missing_features("invalid_input:" + ";".join(problems))
    try:
        left = _feature_values(reference, feature_fn)
        right = _feature_values(prediction, feature_fn)
        delta = [p - r for r, p in zip(left, right)]
        if not all(math.isfinite(value) for value in delta):
            return _missing_features("nonfinite_delta")
        blocks = {}
        for name, start, stop in _BLOCKS:
            block = delta[start:stop]
            summary = {"signed_sum": math.fsum(block),
                       "L1": math.fsum(abs(value) for value in block),
                       "squared_L2": math.fsum(value * value for value in block)}
            if not all(math.isfinite(value) for value in summary.values()):
                return _missing_features("nonfinite_block_summary")
            blocks[name] = summary
    except Exception as exc:
        return _missing_features("feature_error:" + type(exc).__name__)
    return {"valid": True, "status": "ok", "delta421": delta, "blocks": blocks}


def summarize_numeric(values: Iterable, total_expected: int | None = None) -> dict:
    """Mean over finite numeric values, retaining all explicit and missing slots.

    None/nonfinite/nonnumeric values (including bools) are missing. total_expected
    can include unobserved slots, but cannot shrink the supplied slot count.
    The mean is a valid-only mean, not an imputed mean over the total denominator.
    """
    observed = list(values)
    if total_expected is None:
        total = len(observed)
    else:
        if isinstance(total_expected, bool) or not isinstance(total_expected, int):
            raise TypeError("total_expected must be an integer or None")
        if total_expected < len(observed):
            raise ValueError("total_expected cannot be smaller than observed slots")
        total = total_expected
    valid = [number for value in observed if (number := _finite_number(value)) is not None]
    mean = None
    if valid:
        scale = max(abs(value) for value in valid)
        mean = math.fsum(value / scale for value in valid) / len(valid) * scale if scale else 0.0
    return {"mean": mean, "n_valid": len(valid), "n_missing": total - len(valid), "total": total}
