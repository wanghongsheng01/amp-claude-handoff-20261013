"""Read saved summaries/source only; no model, fitting or dataset reads."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
R=P.parent.parent
files=['runner/run_rebuild_v3r2.py','corrected_code/AMP_VAEflow/modules/vae.py','corrected_code/AMP_VAEflow/modules/flow_base.py','diagnostics_h0_fixed_checkpoint_v1/run/SUMMARY.json','diagnostics_h1_readout_v1/run/SUMMARY.json','diagnostics_h2_readout_length_v1/run/SUMMARY.json','diagnostics_h2_readout_length_v1/run/DECISIONS.json','diagnostics_h3_nonlinear_content_v1/run/DECISIONS.json','diagnostics_h4_h2_generation_bridge_v1/AUTHOR_READOUT.zh.md','diagnostics_h4_h2_generation_bridge_v1/AUTHOR_METRIC_FACTS.json','runner/dbaasp_v4_cpu_pilot_001/vae_throughput.json','runner/dbaasp_v4_cpu_pilot_001/ddpm_throughput.json','runner/dbaasp_v4_cpu_pilot_001/vae_validation.json','runner/dbaasp_v4_cpu_pilot_001/ddpm_validation.json']
def read(p):return json.loads((R/p).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mode=['mean','posterior_seed_322','posterior_seed_1322','posterior_seed_2322']
h0=read(files[3]);h1=read(files[4]);h2=read(files[5]);h3=read(files[7]);h4=read(files[9])
out={'state':'STATIC_DESIGN_ONLY_NOT_EXECUTION_PRELOCK','sources':[{'path':p,'sha256':sha(R/p)} for p in files],
'h0_val':{m:{k:h0['val/'+m]['all'][k] for k in ['correct_nonpad','n_nonpad','original_mse']} for m in mode},
'h0_matched_terminal_length_shuffle':h0['val/perm_terminal_length']['paired_against_mean']['moved_subset'],
'h1_val_categorical':{m:{k:h1['val/'+m+'/categorical_21'][k] for k in ['correct_nonpad','nonpad','premature_pad_positions','length_mae','whole_rendered_exact']} for m in mode},
'h2_val_ce_j':{m:{k:h2['val/'+m+'/ridgeCE/J'][k] for k in ['rendered_aligned_correct','nonpad','length_mae','whole_rendered_exact']} for m in mode},
'h3_self_gates':{k:v['final']['conditional_combined_gate'] for k,v in h3['hypotheses'].items()},
'h4_recorded_summary':{'scalar_valid1':sum(x['valid_1_48_count'] for x in h4['long_rows'] if x['head']=='original_scalar'),'h2_valid1':sum(x['valid_1_48_count'] for x in h4['long_rows'] if x['head']=='H2_CE_J'),'h2_valid5':sum(x['valid_5_48_count'] for x in h4['long_rows'] if x['head']=='H2_CE_J'),'description':'mixed; MMD five increase/AMD-DDPM decrease; validation dipeptideJS all six decrease; lengthKS four decrease/two increase'},
'cpu_historical_cost':{'vae1000':read(files[10]),'ddpm1000':read(files[11])},'new_models_executed':0,'test_reads':0,'backup':'BACKUP_BLOCKED'}
# The 19-row moved subset contains only aggregate metrics, never underlying records.
with (P/'EXISTING_EVIDENCE.json').open('x') as f:f.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('sources',len(files),'evidence_sha',sha(P/'EXISTING_EVIDENCE.json'))
