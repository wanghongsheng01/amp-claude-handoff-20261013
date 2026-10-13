"""Metadata-only final delivery builder. Never loads arrays/models/data cohorts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

H = Path(__file__).resolve().parents[1]
OUTPUT = H / 'DELIVERY_H5_FOUR_ARM_SOURCE_DIAGNOSTIC_FINAL_v1.json'
GROUPS = {
    'handoff_and_identity': [
        'HANDOFF_H5_FOUR_ARM_AND_SOURCE_DIAGNOSTIC_FINAL.zh.md',
        'protocol/IDENTITY_RECEIPT.json',
        'protocol/FORMAL_PROTOCOL.json',
        'protocol/FORMAL_BUDGET_RULE.json',
        'protocol/FORMAL_BUDGET_CALCULATION.json',
        'protocol/CE_CHECKPOINT_RECOVERY_AMENDMENT_v1.zh.md',
        'feasibility/H5_ROOT_PREDECESSOR_AND_FOUR_ARM_PRESERVATION_FINAL_v1.json',
        'feasibility/H5_FINAL_PROCESS_SNAPSHOT_v1.json',
        'protocol/AUTOMATION_STATUS.json',
    ],
    'four_arm_results_and_independent_reviews': [
        'feasibility/CE_RECOVERY_FOUR_ARM_DELIVERY_v1.json',
        'feasibility/ce_recovery_four_arm_readout_v4_v1/AUTHOR_READOUT.zh.md',
        'feasibility/ce_recovery_four_arm_full_metrics_addendum_v1/FULL_METRICS_AND_NATIVE_CAP_v1.zh.md',
        'review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.json',
        'review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.zh.md',
        'review/independent/AR_FORMAL_POSTFLIGHT_REVIEW.json',
        'review/independent/CE_V2_INTERRUPTION_FORENSIC_v2.json',
        'review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json',
        'review/independent/MSE_V2_POSTFLIGHT_v1.json',
        'review/independent/SCALAR_V2_POSTFLIGHT_v1.json',
    ],
    'latent_saved_data_and_route_evidence': [
        'feasibility/latent_match_diagnostic_v1/DELIVERY_MANIFEST.json',
        'feasibility/latent_same_n_group_reference_v1/DELIVERY_MANIFEST.json',
        'feasibility/latent_same_n_group_reference_v1/READOUT.zh.md',
        'review/latent_route_static_v1/MANIFEST.json',
        'review/latent_comparability_availability_v1/MANIFEST.json',
        'review/independent/LATENT_SAME_N_GROUP_REFERENCE_POSTFLIGHT_v1.json',
        'review/independent/REPARAMETERIZATION_BYTE_LEVEL_ADDENDUM_v1.json',
        'review/literature/LATENT_SCALE_PRIOR_PRIMARY_BRIEF_v1.json',
        'review/literature/AGGREGATE_POSTERIOR_PRECEDENTS_ADDENDUM_v1.json',
        'review/literature/AGGREGATE_POSTERIOR_PRECEDENTS_ADDENDUM_v1.zh.md',
    ],
    'source_diagnostic_frozen_scope_and_actual_execution': [
        'implementation/ce_source_replacement_v2.py',
        'protocol/CE_SOURCE_REPLACEMENT_CONTRACT_v2.json',
        'protocol/CE_SOURCE_REPLACEMENT_PLAN_v2.json',
        'protocol/CE_SOURCE_REPLACEMENT_REQUEST_SLOTS_v2.json',
        'protocol/CE_SOURCE_REPLACEMENT_SAMPLING_SEEDS_v2.json',
        'protocol/ce_source_replacement_evaluation_v2/PREPARATION_MANIFEST.json',
        'review/independent/CE_SOURCE_REPLACEMENT_CANDIDATE_REVIEW_v1.json',
        'review/independent/CE_SOURCE_REPLACEMENT_R_S1_REPAIR_REVIEW_v2.json',
        'review/independent/CE_SOURCE_REPLACEMENT_PREFLIGHT_v2.json',
        'review/literature/SOURCE_BCD_EXACT_CONTRACT_SCIENTIFIC_REVIEW_v2.json',
        'runs/ce_source_replacement_v2/manifest.json',
        'runs/ce_source_replacement_v2/CLOSING_LEDGER.json',
        'runs/ce_source_replacement_execution_v2/SUPERVISOR_STATUS.json',
        'review/independent/CE_SOURCE_REPLACEMENT_POSTFLIGHT_v2.json',
        'feasibility/ce_source_replacement_author_terminal_readout_v2/MANIFEST.json',
    ],
    'source_diagnostic_results_and_independent_science': [
        'feasibility/CE_SOURCE_REPLACEMENT_EVALUATION_DELIVERY_v2.json',
        'feasibility/ce_source_replacement_author_readout_v2/AUTHOR_READOUT.zh.md',
        'feasibility/ce_source_replacement_author_readout_v2/GATE_AND_SCOPE_INTERPRETATION.json',
        'feasibility/ce_source_replacement_evaluation_execution_v2/EVALUATION_RESULT.json',
        'feasibility/ce_source_replacement_evaluation_execution_v2/AUTHOR_READOUT_RESULT.json',
        'review/literature/SOURCE_REPLACEMENT_FINAL_SCIENTIFIC_REVIEW_v2.json',
        'review/literature/SOURCE_REPLACEMENT_FINAL_SCIENTIFIC_REVIEW_v2.zh.md',
    ],
    'cost_and_append_only_erratum': [
        'feasibility/cost_opportunity_ledger_v4/MANIFEST.json',
        'feasibility/cost_opportunity_ledger_v4/COST_OPPORTUNITY_LEDGER_V4.json',
        'feasibility/cost_addendum_source_replacement_v2_v1/MANIFEST.json',
        'feasibility/cost_addendum_source_replacement_v2_v1/COST_ADDENDUM.json',
        'feasibility/cost_addendum_source_replacement_v2_v1/READOUT.zh.md',
        'feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.json',
        'feasibility/cost_addendum_source_c_erratum_v1/ERRATUM_SOURCE_C_v1.zh.md',
    ],
}

def binding(relative):
    p = H / relative
    b = p.read_bytes()
    if p.suffix not in {'.json', '.md', '.py'}:
        raise ValueError('Metadata-only allowlist')
    if p.suffix == '.json':
        json.loads(b)
    return {'relative_path': relative, 'path': str(p), 'bytes': len(b),
            'sha256': hashlib.sha256(b).hexdigest()}

def main():
    if OUTPUT.exists():
        raise FileExistsError('Final index is append-only; do not overwrite')
    groups = {g: [binding(p) for p in paths] for g, paths in GROUPS.items()}
    source_review = json.loads((H / 'review/literature/SOURCE_REPLACEMENT_FINAL_SCIENTIFIC_REVIEW_v2.json').read_text())
    four_review = json.loads((H / 'review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.json').read_text())
    data = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope_id': 'AMP_H5_FOUR_ARM_AND_SOURCE_DIAGNOSTIC_FINAL_V1',
        'status': 'BOUNDED_EXPERIMENTS_AND_SEPARATE_SCIENTIFIC_REVIEWS_COMPLETE',
        'subject': 'Antimicrobial peptides; original 3DHumanRehab/AMP',
        'train_dev_test_boundary': {'train': 279, 'dev': 41, 'test37_access': False},
        'original_CE_E_L_G': 'PERMANENTLY_NOT_EVALUABLE_INTERRUPTION',
        'recovery_descriptive_L': {'status': 'PASS', 'passed': 4, 'total': 4},
        'recovery_descriptive_G': {'status': 'FAIL', 'failed_subconditions': 8, 'total': 18},
        'AR_ADEQUACY': 'NOT_ESTABLISHED', 'PROMOTE': 'HOLD',
        'four_arm_requests_completed': 128,
        'source_diagnostic': {'natural_attempts': 1, 'new_requests': 96, 'reused_rows': 32,
            'observations': 128, 'new_completed': 96, 'new_failed': 0,
            'scope_is_posthoc_descriptive': True, 'new_training_seeds': 0,
            'B': 'Empirical training posterior replay; not novelty.',
            'C': 'Per-coordinate within-terminal permutation of actual sampled B decoder-z.',
            'D': 'Frozen CE native prior conditioned on reference training g; extra residue/length-template information; excluded from c-only rankings.',
            'gate_status': 'NOT_APPLICABLE_SOURCE_DIAGNOSTIC'},
        'claims_supported': ['Observed CE21+J reconstruction improvement in this fixed-config pilot.',
            'Current pipeline did not establish overall generation superiority.',
            'Source replacement does not establish a unique DDPM bottleneck or novel method.'],
        'claims_not_established': ['General or statistical superiority', 'Adequately tuned strongest baseline',
            'New method novelty', 'Antimicrobial efficacy', 'Safety', 'Prospective wetlab validation', 'Submission readiness'],
        'costs_seconds': {'model_job_wall_lower_bound': '9999.154718747828206',
            'known_non_model_analysis_separate': '53.70622158146245758',
            'original_CE_total_and_unlogged_review': None,
            'nested_times_not_summed': True, 'exact_total_known': False},
        'historical_status_rule': 'Earlier delivery indices retain PENDING as historical snapshots; independently completed reviews are bound here without rewriting them.',
        'cost_erratum_rule': 'Read the original cost package together with ERRATUM_SOURCE_C_v1: four source-description locations were incorrect; all numeric totals unchanged.',
        'missing_evidence': ['Saved dev posterior latent arrays', 'Multiple training seeds', 'Independent final-test evaluation (intentionally sealed)',
            'AR tuning adequacy', 'Original interrupted CE total time/termination cause', 'Resolved wetlab N/material conflicts'],
        'runtime': {'model_jobs_complete': True, 'new_model_runs_authorized_by_this_index': False,
            'background_monitor_created': False, 'tool_outage_is_current_blocker': False,
            'A10_authenticated_session_used': False, 'credentials_read_or_created': False,
            'upload_push_submission': False, 'backup': 'BACKUP_BLOCKED'},
        'independent_science_review_states': {
            'four_arm': {k: four_review[k] for k in ['status', 'review_status', 'blocking_issues'] if k in four_review},
            'source': {k: source_review[k] for k in ['status', 'review_status', 'blocking_issues', 'blocking_scientific_mismatches', 'source_diagnostic_closeout_complete'] if k in source_review}},
        'binding_count': sum(map(len, groups.values())), 'evidence': groups,
        'builder': binding('feasibility/build_h5_final_delivery_v1.py'),
        'builder_model_calls': 0, 'builder_test_reads': 0,
    }
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(OUTPUT), 'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      'binding_count': data['binding_count']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
