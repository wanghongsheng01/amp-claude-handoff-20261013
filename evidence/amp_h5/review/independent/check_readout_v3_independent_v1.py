"""Independent synthetic-only phase/readout contract review; no natural run reads."""
from pathlib import Path
import sys,json,hashlib,tempfile,copy
H=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=H/'review/independent'
sys.dont_write_bytecode=True;sys.path.insert(0,str(H/'implementation'))
import synthetic_readout_v3_tests as fixtures
from synthetic_readout_v2_tests import standard_package,ar_package
v=fixtures.v3
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];findings=[]
def ck(n,ok,d=None):
 checks.append({'name':n,'pass':bool(ok),**({'detail':d} if d is not None else {})})
 if not ok:raise AssertionError(n)
def fixture_case(fn):
 with tempfile.TemporaryDirectory(prefix='h5_readout_independent_',dir='/tmp') as td:
  root=Path(td).resolve();c,source,recovery,ev=fixtures.ready_fixture(root);return fn(root,c,source,recovery,ev)
def rewrite_receipt(root,c,change):
 receipt=json.loads((root/'postflight.json').read_text());change(receipt);fixtures.put(root,'postflight.json',receipt)
 contract=json.loads(c.read_text());contract['recovery_postflight_receipt']=v.v1.record(root/'postflight.json');fixtures.put(root,c.name,contract)
ck('frozen_code_and_schema_identity',sha(H/'implementation/build_readout_v3.py')=='7905f92e2a62a3f3e47295177842a9c5f667d83d6c932ea240257c85c69aa19d' and sha(H/'implementation/build_readout_v3.contract.schema.json')=='564557f6b3a25ca0215576174a4e5ff43fa3b7e1a635095016e40d4bab5acbaa')
def receipt_checkpoint_drift(root,c,source,recovery,ev):
 rewrite_receipt(root,c,lambda r:r['source_bindings'][-1].update(sha256='0'*64))
 f=fixtures.build(c);ck('checkpoint_file_hash_mismatch_blocks_recovery',not f['recovery_engineering_postflight']['pass'] and f['recovery_engineering']['state']=='NOT_EVALUABLE' and f['recovery_descriptive_screens']['G']['state']=='NOT_EVALUABLE')
fixture_case(receipt_checkpoint_drift)
def missing_status_binding(root,c,source,recovery,ev):
 rewrite_receipt(root,c,lambda r:r.update(source_bindings=[x for x in r['source_bindings'] if Path(x['path']).name!='status.json']))
 f=fixtures.build(c);ck('missing_status_binding_blocks_recovery',not f['recovery_engineering_postflight']['pass'])
fixture_case(missing_status_binding)
def new_generation_drift(root,c,source,recovery,ev):
 (recovery/'generated.csv').write_text((recovery/'generated.csv').read_text()+'\n')
 f=fixtures.build(c);ck('generation_hash_drift_blocks_postflight_and_metrics',not f['recovery_engineering_postflight']['pass'] and not any(g['source_binding_pass'] for g in f['generation_observations'] if g['method']==v.CE))
fixture_case(new_generation_drift)
def wrong_training_phase(root,c,source,recovery,ev):
 p=json.loads((recovery/'phase_provenance.json').read_text());p['training']['source_directory']=str(root/'wrong');fixtures.put(recovery,'phase_provenance.json',p)
 f=fixtures.build(c);ck('phase_training_root_mismatch_blocks_recovery',not f['CE_recovery_attempt']['recorded_completion'] and f['recovery_engineering']['state']=='NOT_EVALUABLE')
fixture_case(wrong_training_phase)
def separate_packages(root,c,source,recovery,ev):
 contract=json.loads(c.read_text());paths={m:e['generation_attempt_root'] for m,e in contract['methods'].items()}
 ce,ar=root/'ce_only',root/'ar_only';previous=root/'fixture/evaluation'
 standard_package(ce,previous,[v.CE],paths);ar_package(ar,previous,paths)
 contract['evaluation_directories']=[str(ce),str(ar)];fixtures.put(root,c.name,contract)
 f=fixtures.build(c);ck('no_cross_package_G_splicing',f['recovery_descriptive_screens']['G']['state']=='NOT_EVALUABLE' and f['recovery_descriptive_screens']['G']['records']==[] and len(f['generation_observations'])==4)
fixture_case(separate_packages)
def duplicates(root,c,source,recovery,ev):
 contract=json.loads(c.read_text());contract['evaluation_directories']=[str(ev),str(ev)];fixtures.put(root,c.name,contract)
 try:fixtures.build(c)
 except ValueError:ck('duplicate_observations_rejected',True)
 else:ck('duplicate_observations_rejected',False)
fixture_case(duplicates)
def caps(root,c,source,recovery,ev):
 status=json.loads((recovery/'status.json').read_text());status['process_seconds_before_final_status_io']=1981;fixtures.put(recovery,'status.json',status)
 # Bind changed status to the synthetic receipt so only budget guard is tested.
 rewrite_receipt(root,c,lambda r:r.update(source_bindings=[v.v1.record(recovery/'status.json') if Path(x['path']).name=='status.json' else x for x in r['source_bindings']]))
 f=fixtures.build(c);ck('recorded_outer_overrun_blocks_recovery',f['recovery_engineering']['state']=='NOT_EVALUABLE' and not f['CE_recovery_attempt']['checks']['outer_within1980'])
 ck('original_gates_always_NE_and_unknown_total_kept',all(f['original_protocol'][k]=='NOT_EVALUABLE' for k in 'ELG') and f['original_protocol']['original_total_wall_seconds'] is None and f['phase_costs']['original_CE_amendment_record']['interrupted_generation']['bound']=='lower_bound_only')
fixture_case(caps)
def manifest_drift(root,c,source,recovery,ev):
 recorded=[v.v1.record(p) for p in ev.iterdir() if p.is_file() and p.name!='MANIFEST.json'];fixtures.put(ev,'MANIFEST.json',{'files':recorded})
 before=fixtures.build(c);g=json.loads((ev/'generation_metrics.json').read_text());g['groups'][0]['valid_1_48_count']=0;fixtures.put(ev,'generation_metrics.json',g)
 after=fixtures.build(c)
 expected=next(x['sha256'] for x in recorded if Path(x['path']).name=='generation_metrics.json');actual=v.v1.record(ev/'generation_metrics.json')['sha256']
 ck('manifest_drift_bug_reproduced',expected!=actual and after['recovery_descriptive_screens']['G']['state']=='PASS' and after['input_snapshot_stable'])
 findings.append({'id':'R3-1','severity':'blocking_for_effective_descriptive_gate_transcription','location':'build_readout_v3.py build -> frozen v2.evaluate_artifacts; evaluation MANIFEST consumed but not verified',
 'evidence':{'baseline_G':before['recovery_descriptive_screens']['G']['state'],'after_stale_manifest_G':after['recovery_descriptive_screens']['G']['state'],'after_valid1_count':after['generation_observations'][0]['metrics']['valid_1_48_count'],'input_snapshot_stable':after['input_snapshot_stable'],'recorded_metric_sha256':expected,'actual_metric_sha256':actual},
 'impact':'A result JSON drifted from its existing evaluation package manifest can still be transcribed as an eligible PASS; same-read stability does not establish the saved package integrity. Raw input CSV binding remains correct but does not bind the metric/gate outputs.',
 'minimal_fix':'New candidate wrapper must validate existing MANIFEST against every evaluation JSON it actually consumes: exact local member, SHA256 and optional bytes, reject missing/duplicate/mismatch entries. Preserve raw data but package/group interpretation and effective screens become NOT_EVALUABLE. Do not follow external/source/pt/npz/split entries and do not modify frozen evaluator/v1/v2/v3.'})
fixture_case(manifest_drift)
def schema_gap(root,c,source,recovery,ev):
 raw=json.loads(c.read_text());raw.pop('context_evidence');raw['unexpected_extra']='ignored';fixtures.put(root,c.name,raw)
 v.load_contract(c)
 findings.append({'id':'R3-2','severity':'nonblocking_for_current_valid_template','location':'build_readout_v3.py load_contract vs contract schema required/additionalProperties',
 'evidence':'Missing context_evidence plus unexpected_extra is accepted by runtime loader despite schema required context_evidence and additionalProperties=false.',
 'impact':'The supplied schema is a declaration, not fully enforced by this CLI. Actual template includes required fields, so this does not invalidate its scope.',
 'minimal_fix':'In the next candidate either enforce required/top-level/method/binding keys with standard library checks or label schema validation as a separate required step.'})
 ck('schema_runtime_validation_gap_reproduced',True)
fixture_case(schema_gap)
ck('no_torch_numpy_loaded','torch' not in sys.modules and 'numpy' not in sys.modules)
report={'status':'HOLD_FOR_NEW_READOUT_CANDIDATE','scope':'Static and synthetic readout only; not a model or recovery-result review','checks':checks,'checks_passed':len(checks),'author10tests_rerun':'PASS; READOUT_V3_AUTHOR_TEST_RERUN_v1.json','findings':findings,
 'accepted_behaviors':['Phase-specific source/reconstruction bindings','Independent postflight scope/id/root/source hashes match planned interface','Original E/L/G hardcoded NOT_EVALUABLE, PROMOTE HOLD','No fake absent method groups or zero observations','No gate splicing across separate metric packages','Known original stages and partial-generation lower bound kept separately from unknown original wall time','No checkpoint,posterior,data-split file opened by readout'],
 'future_postflight_interface':{'status':'PASS only if confirmed','scope':'RECOVERY_ENGINEERING_ONLY','generation_attempt_id':'ce_checkpoint_recovery_v1','generation_attempt_root':'absolute recovery root','training_source_root':'absolute original training root','source_bindings':'absolute path + sha256 for recovery status/generated and original vae_state.pt/ddpm_ema_state.pt; extra evidence permitted','original_v2_engineering_gate':'NOT_EVALUABLE_INTERRUPTION','original_protocol_E_L_G':'NOT_EVALUABLE','PROMOTE':'HOLD'},
 'frozen_source_hashes':{n:sha(H/'implementation'/n) for n in ('build_readout_v3.py','build_readout_v3.contract.schema.json','build_readout_v2.py','build_readout.py')},
 'reviewer_actions':{'natural_run_reads':0,'model_calls':0,'weight_loads':0,'natural_metric_recomputation':0,'test37_reads':0,'frozen_files_changed':0},'PROMOTE':'HOLD'}
p=O/'READOUT_V3_INDEPENDENT_REVIEW_v1.json'
with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'findings':[(x['id'],x['severity']) for x in findings],'report':str(p),'sha256':sha(p)}))
