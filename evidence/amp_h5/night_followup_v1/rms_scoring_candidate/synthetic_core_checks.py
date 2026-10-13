"""Synthetic-only checks for rms_metric_core; no natural data, models, or RNG."""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

from rms_metric_core import AA, feature_delta, pair_core, summarize_numeric


def run_checks() -> dict:
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def raises(name, expected, function):
        try:
            function()
        except expected:
            check(name, True)
        else:
            check(name, False)

    check("canonical_AA_order", AA == "ACDEFGHIKLMNPQRSTVWY")
    extra = pair_core("AC", "ACD")
    check("extra_not_penalized_in_target_accuracy", extra["nonpad_accuracy"] == 1.0)
    check("extra_target_denominator", extra["target_residues"] == extra["correct_target_positions"] == 2)
    check("extra_edit_and_normalization", extra["levenshtein_raw"] == 1 and extra["levenshtein_normalized"] == 1 / 3)
    check("extra_length_and_not_exact", extra["length_signed"] == extra["length_absolute"] == 1 and not extra["whole_exact"])
    missing = pair_core("ACD", "AC")
    check("missing_residue_in_denominator", missing["nonpad_accuracy"] == 2 / 3 and missing["target_residues"] == 3)
    check("missing_length_sign", missing["length_signed"] == -1 and missing["length_absolute"] == 1)
    check("missing_edit", missing["levenshtein_raw"] == 1 and missing["levenshtein_normalized"] == 1 / 3)
    shifted = pair_core("ACD", "AACD")
    check("position_accuracy_is_not_alignment", shifted["correct_target_positions"] == 1 and shifted["nonpad_accuracy"] == 1 / 3)
    check("edit_can_differ_from_position_errors", shifted["levenshtein_raw"] == 1 and shifted["levenshtein_normalized"] == 0.25)
    check("substitution_unit_cost", pair_core("AC", "AD")["levenshtein_raw"] == 1)
    check("transposition_is_two_edits", pair_core("AC", "CA")["levenshtein_raw"] == 2)
    both_empty = pair_core("", "")
    check("both_empty_accuracy_missing", both_empty["target_residues"] == 0 and both_empty["nonpad_accuracy"] is None)
    check("both_empty_exact_and_zero_distance", both_empty["whole_exact"] and both_empty["levenshtein_raw"] == both_empty["levenshtein_normalized"] == 0)
    check("empty_reference_nonempty_prediction", pair_core("", "A")["nonpad_accuracy"] is None and pair_core("", "A")["levenshtein_raw"] == 1)
    check("empty_prediction_all_target_positions_wrong", pair_core("AC", "")["correct_target_positions"] == 0 and pair_core("AC", "")["nonpad_accuracy"] == 0)
    check("pair_text_is_not_normalized", pair_core("aX", "AX")["correct_target_positions"] == 1)
    raises("pair_rejects_nonstring", TypeError, lambda: pair_core(None, "A"))
    for left, right in (("A", "CCCC"), ("ACD", "DCA"), ("", "AC"), ("AC", "AC")):
        l, r = pair_core(left, right), pair_core(right, left)
        check("edit_symmetry_" + repr((left, right)), l["levenshtein_raw"] == r["levenshtein_raw"])

    calls = []

    def synthetic_feature(sequence):
        calls.append(sequence)
        vector = [0.0] * 421
        vector[0] = float(len(sequence))
        vector[1], vector[2] = float(sequence.count("A")), float(sequence.count("C"))
        vector[21], vector[22] = float(sequence.count("AC")), float(sequence.count("CA"))
        return vector

    delta = feature_delta("AC", "CAA", synthetic_feature)
    check("feature_called_once_per_valid_string", calls == ["AC", "CAA"])
    check("feature_valid_and421", delta["valid"] and delta["status"] == "ok" and len(delta["delta421"]) == 421)
    check("delta_is_prediction_minus_reference", delta["delta421"][:3] == [1.0, 1.0, 0.0] and delta["delta421"][21:23] == [-1.0, 1.0])
    check("length_block", delta["blocks"]["length1"] == {"signed_sum": 1.0, "L1": 1.0, "squared_L2": 1.0})
    check("aa_block", delta["blocks"]["aa20"] == {"signed_sum": 1.0, "L1": 1.0, "squared_L2": 1.0})
    check("dipeptide_cancellation_not_zero_distance", delta["blocks"]["dipeptide400"] == {"signed_sum": 0.0, "L1": 2.0, "squared_L2": 2.0})
    check("three_blocks_partition_signed_sum", sum(x["signed_sum"] for x in delta["blocks"].values()) == sum(delta["delta421"]))
    check("three_blocks_partition_L1", sum(x["L1"] for x in delta["blocks"].values()) == sum(abs(x) for x in delta["delta421"]))
    check("three_blocks_partition_squared_L2", sum(x["squared_L2"] for x in delta["blocks"].values()) == sum(x * x for x in delta["delta421"]))
    reverse = feature_delta("CAA", "AC", synthetic_feature)
    check("feature_direction_reversal", reverse["delta421"] == [-x for x in delta["delta421"]])
    equal = feature_delta("AC", "AC", synthetic_feature)
    check("legitimate_equal_features_are_zero_and_valid", equal["valid"] and equal["delta421"] == [0.0] * 421)

    for i, (left, right) in enumerate((("", "A"), ("A", ""), ("", ""), ("A*", "C"),
                                      ("ac", "A"), ("A", "A" * 49), (None, "A"))):
        before = len(calls)
        result = feature_delta(left, right, synthetic_feature)
        check("invalid_null_" + str(i), not result["valid"] and result["delta421"] is None and all(v is None for v in result["blocks"].values()))
        check("invalid_no_feature_calls_" + str(i), len(calls) == before)
        check("invalid_reason_retained_" + str(i), result["status"].startswith("invalid_input:"))

    malformed = feature_delta("A", "C", lambda _: [0.0] * 420)
    check("wrong_dimension_is_missing", malformed["delta421"] is None and malformed["status"] == "feature_error:ValueError")
    nonfinite = feature_delta("A", "C", lambda _: [math.nan] + [0.0] * 420)
    check("nonfinite_feature_is_missing", not nonfinite["valid"] and nonfinite["delta421"] is None)
    string_feature = feature_delta("A", "C", lambda _: ["0"] * 421)
    check("numeric_string_feature_rejected", not string_feature["valid"])

    def failing_feature(_):
        raise RuntimeError("synthetic failure")

    check("feature_exception_status_retained", feature_delta("A", "C", failing_feature)["status"] == "feature_error:RuntimeError")
    overflow = feature_delta("A", "C", lambda s: [-1e308 if s == "A" else 1e308] + [0.0] * 420)
    check("delta_overflow_missing", overflow["status"] == "nonfinite_delta" and overflow["delta421"] is None)

    check("summary_valid_mean", summarize_numeric([1, 3]) == {"mean": 2.0, "n_valid": 2, "n_missing": 0, "total": 2})
    mixed = [1, None, math.nan, math.inf, -math.inf, "3", True, 3]
    check("summary_explicit_missing_denominator", summarize_numeric(mixed) == {"mean": 2.0, "n_valid": 2, "n_missing": 6, "total": 8})
    check("summary_unobserved_expected_slots", summarize_numeric(mixed, 10) == {"mean": 2.0, "n_valid": 2, "n_missing": 8, "total": 10})
    check("summary_all_missing", summarize_numeric([None, math.nan]) == {"mean": None, "n_valid": 0, "n_missing": 2, "total": 2})
    check("summary_no_slots", summarize_numeric([]) == {"mean": None, "n_valid": 0, "n_missing": 0, "total": 0})
    check("summary_expected_but_unobserved", summarize_numeric([], 3) == {"mean": None, "n_valid": 0, "n_missing": 3, "total": 3})
    check("summary_zero_is_valid", summarize_numeric([0.0])["mean"] == 0.0 and summarize_numeric([0.0])["n_valid"] == 1)
    check("summary_Decimal_values", summarize_numeric([Decimal("1.5"), Decimal("2.5")])["mean"] == 2.0)
    check("summary_finite_large_mean", summarize_numeric([1e308, 1e308])["mean"] == 1e308)
    raises("summary_cannot_shrink_denominator", ValueError, lambda: summarize_numeric([1, None], 1))
    raises("summary_rejects_negative_expected", ValueError, lambda: summarize_numeric([], -1))
    raises("summary_rejects_boolean_expected", TypeError, lambda: summarize_numeric([], True))
    raises("summary_rejects_fractional_expected", TypeError, lambda: summarize_numeric([], 1.5))
    check("outputs_JSON_without_NaN", bool(json.dumps([extra, both_empty, delta, malformed, summarize_numeric(mixed)], allow_nan=False)))
    return {"status": "PASS", "n_checks": len(checks), "checks": checks,
            "scope": "Synthetic literals only; no RNG, model, RDKit, natural cohort or test access",
            "files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("rms_metric_core.py"))]}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
