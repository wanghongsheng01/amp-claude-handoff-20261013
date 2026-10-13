from pathlib import Path
import ast,csv,json,hashlib,importlib.util,sys,datetime,copy
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');O=W/'amp_h5/review/independent';old=W/'amp_h5/implementation/vae_pipeline_formal.py';new=W/'amp_h5/implementation/vae_pipeline_formal_v2.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks=[]
def ck(n,b):checks.append({'name':n,'pass':bool(b)});assert b,n
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
module=load('review_h5_formal_v2',new)
cfg=json.loads((W/'amp_h5/protocol/formal_configs_v2/categorical_ce.json').read_text())
validator_path=Path(cfg['source_root'])/'corrected_code/AMP_VAEflow/tasks/dataset_loader/dataset_utils/amp_input_contract.py'
validator=load('review_bound_input_contract',validator_path);module.validate_file=validator.validate_file
ck('no_torch_or_numpy_import','torch' not in sys.modules and 'numpy' not in sys.modules)
def units(p):return {getattr(n,'name',f'node{i}'):ast.dump(n,include_attributes=False) for i,n in enumerate(ast.parse(p.read_text()).body)}
x,y=units(old),units(new);changed=[k for k in x if x[k]!=y[k]];ck('only3_contract_units_changed',set(changed)=={'read_records','validate_config','run'} and x.keys()==y.keys())
config_details={}
for arm in ('categorical_ce','raw_onehot_mse','scalar_legacy'):
 op=W/f'amp_h5/protocol/formal_configs/{arm}.json';np=W/f'amp_h5/protocol/formal_configs_v2/{arm}.json';a=json.loads(op.read_text());b=json.loads(np.read_text());changed_cfg=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k));module.validate_config(b)
 ck(arm+'_only_binding_fields_changed',changed_cfg==['dev_idx_start','implementation_sha256','train_idx_start'] and b['implementation_sha256']==sha(new))
 config_details[arm]={'path':str(np),'sha256':sha(np),'changed_keys':changed_cfg}
records={}
for role,start,n in [('train',0,279),('dev',279,41)]:
 rr=module.read_records(cfg[role],cfg[role+'_sha256'],cfg[role+'_rows'],cfg[role+'_cohort'],cfg[role+'_cohort_sha256'],cfg[role+'_idx_start']);records[role]=rr
 ck(role+'_actual_bound_loader_and_idx_pass',len(rr)==n and [r.idx for r in rr]==list(range(start,start+n)))
ck('actual_train_dev_no_sequence_overlap',not ({r.protein for r in records['train']}&{r.protein for r in records['dev']}))
F=O/'contract_v2_fixtures';F.mkdir(exist_ok=True)
D=[{'idx':279+i,'protein':seq,'describe':','.join(['[0 0 0]']*len(seq)),'point_describe':json.dumps([0]*9),'class':i} for i,seq in enumerate(['ARNDCE','GQHILK'])]
C=[{'sequence':r['protein'],'class':r['class'],'terminal_state':('FREE','AMD')[r['class']]} for r in D]
def write(p,rows,cols):
 with p.open('w',newline='') as f:z=csv.DictWriter(f,fieldnames=cols);z.writeheader();z.writerows(rows)
def call(name,d,c,start=279,desc_hash=None,cohort_hash=None,rows=2):
 dp=F/(name+'_descriptor.csv');cp=F/(name+'_cohort.csv');write(dp,d,['idx','protein','describe','point_describe','class']);write(cp,c,['sequence','class','terminal_state'])
 return module.read_records(dp,desc_hash or sha(dp),rows,cp,cohort_hash or sha(cp),start)
ck('synthetic_valid_global_idx_pass',len(call('good',D,C))==2)
def reject(name,d=D,c=C,**kw):
 try:call(name,d,c,**kw)
 except ValueError:ck(name+'_rejected',True)
 else:ck(name+'_rejected',False)
a=copy.deepcopy(D);a[0]['idx']=0;reject('wrong_idx',d=a)
reject('wrong_idx_start',start=0)
reject('wrong_order',c=list(reversed(C)))
a=copy.deepcopy(C);a[0]['sequence']='ARNDCA';reject('wrong_sequence',c=a)
a=copy.deepcopy(C);a[0]['class']=1;reject('wrong_class',c=a)
a=copy.deepcopy(C);a[0]['terminal_state']='AMD';reject('wrong_terminal',c=a)
reject('wrong_descriptor_hash',desc_hash='0'*64);reject('wrong_cohort_hash',cohort_hash='0'*64);reject('wrong_count',rows=3)
for key,v in [('train_idx_start',279),('dev_idx_start',0),('dev_idx_start',True)]:
 c=copy.deepcopy(cfg);c[key]=v
 try:module.validate_config(c)
 except ValueError:ck('config_reject_'+key+'_'+str(v),True)
 else:ck('config_reject_'+key+'_'+str(v),False)
ck('no_model_lib_import_after_all_checks','torch' not in sys.modules and 'numpy' not in sys.modules)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_SCOPED_FORMAL_V2_CONTRACT_REPAIR','checks':checks,'source_sha256':sha(new),'prior_source_sha256':sha(old),'changed_top_level_units':changed,'configs':config_details,'validator_source':{'path':str(validator_path),'sha256':sha(validator_path)},'real_loaders':{'train':{'rows':279,'idx_first':records['train'][0].idx,'idx_last':records['train'][-1].idx},'dev':{'rows':41,'idx_first':records['dev'][0].idx,'idx_last':records['dev'][-1].idx}},'synthetic_fixture_scope':'Invented two short canonical alphabet strings with zero descriptors; no biological quality or model work. Stored only in reviewer fixture directory.','actual_source_data_changed':False,'no_reordering':True,'model_imports':0,'model_calls':0,'training_updates':0,'test_reads':0,'execution_authorized_by_this_report':False,'limitations':'Approval eligibility for new indexed-loader version only. Parent still needs exact plan and independent scoped receipt; old failure and old receipt preserved.'}
p=O/'FORMAL_V2_CONTRACT_REVIEW.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'report_sha256':sha(p),'source_sha256':sha(new)}))
