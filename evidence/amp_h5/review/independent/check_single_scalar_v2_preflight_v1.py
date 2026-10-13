#!/usr/bin/env python3
"""Static/hash-only scalar preflight; no natural model or quality evaluation."""
from pathlib import Path
import ast,datetime,hashlib,json
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');H=W/'amp_h5';O=H/'review/independent'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
P=H/'protocol/SINGLE_SCALAR_V2_PLAN.json';C=H/'protocol/formal_configs_v2/scalar_legacy.json';S=H/'implementation/supervise.py';L=H/'implementation/launch_detached_supervisor_v1.py';V=H/'implementation/vae_pipeline_formal_v2.py'
def main():
 out=O/'SINGLE_SCALAR_V2_PREFLIGHT_CHECKS_v1.json'
 if out.exists():raise FileExistsError('Never replace existing review')
 p=read(P);c=read(C);draft=read(H/'protocol/staged_remaining_vae_drafts_v1/SCALAR_V2_PLAN_DRAFT.json');m=read(O/'MSE_V2_POSTFLIGHT_v1.json');checks=[]
 def ck(n,x):checks.append({'name':n,'pass':bool(x)})
 ck('exact_final_plan',sha(P)=='69e082e8a5c172f8ca2ed421fdd47caaac4015bea70dcffe612a750a7554b501')
 ck('24_unique_canonical_binding_hashes',len(p['bindings'])==len({str(Path(b['path']).resolve()) for b in p['bindings']})==24 and all(str(Path(b['path']).resolve())==b['path'] and sha(b['path'])==b['sha256'] for b in p['bindings']))
 ck('unchanged_config_runner',sha(C)=='3de741fc97dfb2d61965cc88609dd7026c831c5458641362d1ae3c06081286c0' and c['implementation_sha256']==sha(V)=='e97d2a766774461793cba1860455e4cb4c8d689656b60becc326a9f11297ca9b')
 ck('unchanged_supervisor_launcher',sha(S)=='598ce5a9565140cbde52a7db141a12a5d71293f533f92d5ef60bb8c46e1853d3' and sha(L)=='f2afb79a841c6c40cda13547808f070d9592439069e670a19d880f08ae472b21')
 ck('MSE_postflight_engineering_eligibility',sha(O/'MSE_V2_POSTFLIGHT_v1.json')=='a0a410526b6a252ba1f73fd3957dea5ee20413bb6e5c264fca5d34a4eec97b71' and m['status']=='PASS' and m['scope']=='SINGLE_MSE_ENGINEERING_ONLY' and m['next_stage_judgment']['next_frozen_scalar_plan_eligible_on_engineering_only'] and not m['next_stage_judgment']['scalar_execution_authorized_by_this_postflight'])
 ck('MSE_postflight70_sources_unchanged',len(m['source_bindings'])==70 and all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in m['source_bindings']))
 ck('one_exact_early_draft_job',p['scope_id']=='AMP_H5_SINGLE_SCALAR_V2' and p['jobs']==[draft['proposed_job']] and p['jobs'][0]['id']=='scalar_legacy' and p['output_root']==draft['proposed_output_root'])
 ck('all8_early_draft_bindings_retained',len(draft['frozen_bindings'])==8 and all(b in p['bindings'] for b in draft['frozen_bindings']))
 ck('fresh_exact_workspace_outputs',p['workspace']==str(W) and p['output_root']==str(H/'runs/single_scalar_legacy_execution_v2') and p['jobs'][0]['argv'][-1]==str(H/'runs/formal_v2/scalar_legacy') and not Path(p['output_root']).exists() and not Path(p['jobs'][0]['argv'][-1]).exists())
 ck('fixed1000_1000_32',c['arm']=='scalar_legacy' and c['vae_updates']==c['ddpm_updates']==1000 and c['samples_per_terminal']==16 and p['fixed_method_contract']['total_requests']==32)
 ck('fixed_seed_contract',all(c[k]==v for k,v in {'initialization_seed':222,'shuffle_seed':222,'vae_noise_seed':222,'ddpm_seed':322,'sampling_seeds':[422,423]}.items()))
 ck('fixed_caps',all(c[k]==p['budgets_seconds'][k]==draft['budgets_seconds'][k]==v for k,v in {'vae_seconds_cap':1920,'ddpm_seconds_cap':1020,'training_seconds_cap':2940,'generation_seconds_cap':1800}.items()) and p['outer_timeout_seconds']==p['jobs'][0]['outer_timeout_seconds']==4920)
 ck('fixed_batch_schedule',c['vae_batch_size']==32 and c['ddpm_batch_size']==8 and c['noise_steps']==999 and c['ddpm_lr']==.005 and c['optimizer_contract']['kl_beta']==1)
 ck('exact_four_train_dev_hashes_no_test',all('test' not in c[k].lower() and sha(c[k])==c[k+'_sha256'] for k in ('train','train_cohort','dev','dev_cohort')) and c['train_rows']==279 and c['dev_rows']==41 and c['train_idx_start']==0 and c['dev_idx_start']==279)
 ck('fixed_dev_monitor_no_selection',c['do_eval'] is True and c['monitor_interval']==100 and p['fixed_method_contract']['fixed_endpoint_no_checkpoint_selection'])
 ck('CPU1_serial_lock_contract',p['execution_environment']['device']=='cpu' and p['execution_environment']['torch_threads']==p['execution_environment']['torch_interop_threads']==1 and p['execution_environment']['serial_task_lock']=='H5_PROCESS.lock')
 tree=ast.parse(V.read_text());u={n.name:ast.unparse(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 ck('scalar_head_and_inherited_loss_normalization',"out_dim=1 if arm == 'scalar_legacy' else 21" in u['make_model'] and '(raw - target).square().sum() / (target != 20).sum()' in u['objective'] and 'kl = density / 16' in u['objective'])
 ck('native_scalar_round_clamp_firstPAD_retained','raw.clamp(0, 20).round().long()' in u['decode'] and 'row.index(20) if 20 in row else 48' in u['decode'])
 ck('terminal_only_same_generator','prediction = ddpm(x, t, labels)' in u['generate'] and "['terminal class only']" in u['generate'])
 ck('no_checkpoint_input_or_resume',"'warm-start/checkpoint/test inputs prohibited'" in u['validate_config'] and "'output must be absent or empty; no overwrite or implicit resume'" in u['run'])
 ck('whole_pipeline_not_single_factor','whole-pipeline' in p['scope'] and 'not a single-factor ablation' in p['scope'] and 'density sum / 16' in p['fixed_method_contract']['loss_contract'])
 ck('original_CE_NE_and_recovery_deviation',all(p['scope_boundaries']['original_CE_'+k]=='NOT_EVALUABLE_INTERRUPTION' for k in ('E','L','G')) and 'no-retry deviation' in p['scope_boundaries']['CE_recovery'] and m['original_protocol_E_L_G']=='NOT_EVALUABLE')
 ck('no_quality_permission_other_arm_or_submission',not p['scope_boundaries']['quality_based_changes_or_permission'] and not p['scope_boundaries']['other_model_arms_in_this_plan'] and not p['scope_boundaries']['submission_in_this_plan'] and p['PROMOTE']=='HOLD' and p['scope_boundaries']['AR_ADEQUACY']=='NOT_ESTABLISHED')
 ck('one_attempt_no_automatic_continuation','One fresh scalar_legacy attempt only.' in p['failure_policy'] and 'No automatic subsequent model run.' in p['failure_policy'] and 'not authorization to launch again' in p['transport_uncertainty_policy'])
 ck('adaptive_costs_and_original_unknowns_preserved','not equal actual compute' in p['cost_disclosure'] and 'MSE execution' in p['cost_disclosure'])
 ck('plan_unchanged_at_end',sha(P)=='69e082e8a5c172f8ca2ed421fdd47caaac4015bea70dcffe612a750a7554b501')
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_SCOPED_PREFLIGHT_CHECKS' if all(x['pass'] for x in checks) else 'HOLD','checks':checks,'failed_checks':[x for x in checks if not x['pass']],'plan_sha256':sha(P),'audit_script_sha256':sha(__file__),'clipping_interpretation':'Generic no-clipping text prohibits new sampler/overflow repair. The frozen scalar decoder clamp(0,20),torch.round and first-PAD truncation are required native readout and stay unchanged.','reviewer_actions':{'stdlib_only':True,'model_constructions':0,'model_forwards':0,'training_updates':0,'natural_CSV_parsing':0,'test37_reads':0,'quality_metrics_read_or_used':False},'limitations':['Root fresh load/process check required immediately before launch.','Whole-pipeline legacy comparator, not same-head single-factor loss comparison.','No future run, retry or original CE outcome restoration is authorized.']}
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'path':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
