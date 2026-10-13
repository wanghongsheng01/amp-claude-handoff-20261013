#!/usr/bin/env python3
"""Immutable MSE plan preflight. Stdlib only; no models or natural data parsing."""
from pathlib import Path
import ast, datetime, hashlib, json
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');H=W/'amp_h5';O=H/'review/independent'
P=H/'protocol/SINGLE_MSE_V2_PLAN.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 dest=O/'SINGLE_MSE_V2_PREFLIGHT_CHECKS_v2.json'
 if dest.exists():raise FileExistsError('Existing review must remain unchanged')
 p=read(P); cfgp=H/'protocol/formal_configs_v2/raw_onehot_mse.json';cfg=read(cfgp)
 cep=H/'protocol/formal_configs_v2/categorical_ce.json';ce=read(cep)
 runner=H/'implementation/vae_pipeline_formal_v2.py';sup=H/'implementation/supervise.py';launch=H/'implementation/launch_detached_supervisor_v1.py'
 postp=O/'CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json';post=read(postp)
 checks=[]
 def ck(n,x,d=None):checks.append({'name':n,'pass':bool(x),**({'detail':d} if d is not None else {})})
 ck('exact_final_plan',sha(P)=='a1c24efa2164c5cd9fcca2e46154fac8af44df2aec5f363226da168a374ae802')
 ck('all19_binding_hashes',len(p['bindings'])==19 and all(sha(b['path'])==b['sha256'] for b in p['bindings']))
 ck('unique_canonical_bindings',len({str(Path(b['path']).resolve()) for b in p['bindings']})==19 and all(str(Path(b['path']).resolve())==b['path'] for b in p['bindings']))
 ck('unchanged_runner',sha(runner)==cfg['implementation_sha256']=='e97d2a766774461793cba1860455e4cb4c8d689656b60becc326a9f11297ca9b')
 ck('unchanged_frozen_MSE_config',sha(cfgp)=='861d530bce7529826ed481697083a4efee016cce24878d85dced240b28d9d004')
 ck('CE_MSE_configs_only_arm_diff',[k for k in sorted(set(cfg)|set(ce)) if cfg.get(k)!=ce.get(k)]==['arm'])
 ck('same_reviewed_supervisor_launcher',sha(sup)=='598ce5a9565140cbde52a7db141a12a5d71293f533f92d5ef60bb8c46e1853d3' and sha(launch)=='f2afb79a841c6c40cda13547808f070d9592439069e670a19d880f08ae472b21')
 ck('eligible_scoped_engineering_only',sha(postp)=='2450c62a2553d854685fee7dde300a2ee794746ad8af36bc4db814b9a472c484' and post['status']=='PASS' and post['scope']=='RECOVERY_ENGINEERING_ONLY' and post['next_stage_judgment']['next_frozen_MSE_plan_eligible_on_engineering_only'] and not post['next_stage_judgment']['MSE_execution_authorized_by_this_postflight'])
 ck('postflight35_source_bindings_unchanged',len(post['source_bindings'])==35 and all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in post['source_bindings']))
 expected=[str(Path(cfg['source_root'])/'env/bin/python'),'-u','-B',str(runner),'run','--config',str(cfgp),'--output',str(H/'runs/formal_v2/raw_onehot_mse')]
 ck('one_exact_fresh_MSE_job',len(p['jobs'])==1 and p['jobs'][0]['id']=='raw_onehot_mse' and p['jobs'][0]['argv']==expected and p['scope_id']=='AMP_H5_SINGLE_MSE_V2')
 ck('exact_workspace_fresh_output_paths',p['workspace']==str(W) and p['output_root']==str(H/'runs/single_raw_onehot_mse_execution_v2') and not Path(p['output_root']).exists() and not Path(expected[-1]).exists())
 ck('fixed1000_1000_32_requests',cfg['vae_updates']==cfg['ddpm_updates']==1000 and cfg['samples_per_terminal']==16 and p['fixed_method_contract']['total_requests']==32)
 ck('unchanged_seeds',all(cfg[k]==v for k,v in {'initialization_seed':222,'shuffle_seed':222,'vae_noise_seed':222,'ddpm_seed':322,'sampling_seeds':[422,423]}.items()))
 ck('unchanged_caps',all(cfg[k]==p['budgets_seconds'][k]==v for k,v in {'vae_seconds_cap':1920,'ddpm_seconds_cap':1020,'training_seconds_cap':2940,'generation_seconds_cap':1800}.items()) and p['outer_timeout_seconds']==p['jobs'][0]['outer_timeout_seconds']==4920)
 ck('fixed_batch_schedule_objective_contract',cfg['vae_batch_size']==32 and cfg['ddpm_batch_size']==8 and cfg['noise_steps']==999 and cfg['ddpm_lr']==.005 and cfg['optimizer_contract']['raw_onehot_mse_prefactor']==.5 and cfg['optimizer_contract']['kl_beta']==1)
 ck('exact_train_dev_input_hashes_only',all('test' not in cfg[k].lower() and sha(cfg[k])==cfg[k+'_sha256'] for k in ('train','dev','train_cohort','dev_cohort')) and cfg['train_rows']==279 and cfg['dev_rows']==41 and cfg['train_idx_start']==0 and cfg['dev_idx_start']==279)
 ck('fixed_descriptive_dev_monitor',cfg['do_eval'] is True and cfg['monitor_interval']==100 and p['fixed_method_contract']['fixed_endpoint_no_checkpoint_selection'])
 ck('CPU_serial_one_thread_contract',p['execution_environment']['device']=='cpu' and p['execution_environment']['torch_threads']==p['execution_environment']['torch_interop_threads']==1 and p['execution_environment']['serial_task_lock']=='H5_PROCESS.lock')
 tree=ast.parse(runner.read_text());units={x.name:ast.unparse(x) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
 ck('fresh_21head_and_MSE_half_raw_objective',"out_dim=1 if arm == 'scalar_legacy' else 21" in units['make_model'] and '0.5 * (raw - onehot).square().sum() / denom' in units['objective'] and 'kl = density / denom' in units['objective'])
 ck('terminal_only_same_generator_no_quality_invocation','ddpm(x, t, labels)' in units['generate'] and 'reconstruction_export' in units['run'] and 'evaluate_generation' not in units['run'])
 ck('no_resume_checkpoint_input_and_no_output_overwrite',"'warm-start/checkpoint/test inputs prohibited'" in units['validate_config'] and "'output must be absent or empty; no overwrite or implicit resume'" in units['run'])
 ck('scope_preserves_original_NE_and_deviation',all(p['scope_boundaries']['original_CE_'+s]=='NOT_EVALUABLE_INTERRUPTION' for s in ('E','L','G')) and 'no-retry deviation' in p['scope_boundaries']['CE_recovery'] and post['original_protocol_E_L_G']=='NOT_EVALUABLE')
 ck('no_scalar_no_quality_permission_no_submission',not p['scope_boundaries']['scalar_in_this_plan'] and not p['scope_boundaries']['quality_based_changes_or_permission'] and not p['scope_boundaries']['submission_in_this_plan'] and p['PROMOTE']=='HOLD' and p['scope_boundaries']['AR_ADEQUACY']=='NOT_ESTABLISHED')
 ck('single_attempt_and_unknown_transport_not_retry','One fresh MSE attempt only' in p['failure_policy'] and 'not authorization to launch again' in p['transport_uncertainty_policy'])
 ck('known_adaptive_extra_costs_disclosed','additional CE recovery' in p['cost_disclosure'] and 'not equal actual compute' in p['cost_disclosure'])
 draft=read(H/'protocol/staged_remaining_vae_drafts_v1/MSE_V2_PLAN_DRAFT.json')
 ck('early_draft_job_retained',p['draft_derivation']['config_or_code_modified'] is False and p['draft_derivation']['proposed_job_copied_without_change'] is True)
 ck('plan_stable_at_end',sha(P)=='a1c24efa2164c5cd9fcca2e46154fac8af44df2aec5f363226da168a374ae802')
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_SCOPED_PREFLIGHT_CHECKS' if all(c['pass'] for c in checks) else 'HOLD','checks':checks,'failed_checks':[c for c in checks if not c['pass']],'plan_sha256':sha(P),'audit_script_sha256':sha(__file__),'audit_correction':{'preserved_v1':'SINGLE_MSE_V2_PREFLIGHT_CHECKS_v1.json','reason':'Reviewer v1 string probe used label instead of actual labels vector. Static source confirms constant terminal labels only; production code/plan/config unchanged.'},'reviewer_actions':{'stdlib_only':True,'model_constructions':0,'model_forwards':0,'training_updates':0,'natural_CSV_parsing':0,'test37_reads':0},'limitations':['Root must perform fresh process/load check before launching.','Hash-based source review is not a filesystem-access sandbox.','No scientific advantage, original CE protocol restoration, scalar authorization or future retry follows from these checks.']}
 with dest.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'path':str(dest),'sha256':sha(dest)}))
if __name__=='__main__':main()
