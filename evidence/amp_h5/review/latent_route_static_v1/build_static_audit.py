#!/usr/bin/env python3
"""Read-only static audit; stdlib only, no model/RNG/cohort reads."""
import ast
import csv
import hashlib
import json
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
S = Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011')
C = S / 'corrected_code'
CE = H / 'runs/formal_v2/categorical_ce'
REC = H / 'runs/ce_checkpoint_recovery_v1'
MSE = H / 'runs/formal_v2/raw_onehot_mse'
FILES = {
 'F': H/'implementation/vae_pipeline_formal_v2.py',
 'RECOVERY': H/'implementation/ce_checkpoint_recovery_v1.py',
 'V': C/'AMP_VAEflow/modules/vae.py',
 'D': C/'AMP_VAEflow/modules/diff.py',
 'FLOW': C/'AMP_VAEflow/modules/flow_base.py',
 'ATTN': C/'AMP_VAEflow/modules/attention_model.py',
 'ROOTDIFF': C/'diffusion.py',
 'TASKDIFF': C/'AMP_VAEflow/tasks/train_diffusion.py',
 'VAE_YAML': C/'AMP_VAEflow/configs/train_vae.yaml',
 'CSV_LOADER': C/'AMP_VAEflow/tasks/dataset_loader/dataset_utils/csv_dataset.py',
 'OLD': S/'runner/run_rebuild_v3r2.py',
 'H4': S/'diagnostics_h4_h2_generation_bridge_v1/h4_run.py',
 'BUILD': S/'expanded_preprocessing/build_descriptors_v2.py',
 'REP': S/'expanded_preprocessing/molecular_representation.py',
 'SOURCE_MANIFEST': S/'runner/source_copy_manifest.json',
 'CONFIG': CE/'config.json',
 'SCALER': CE/'train_feature_scaler.json',
 'RUNTIME': CE/'runtime_and_source.json',
 'DDPM_SUMMARY': CE/'ddpm_training_summary.json',
 'DDPM_CURRENT_META': CE/'ddpm_state.json',
 'DDPM_EMA_META': CE/'ddpm_ema_state.json',
 'REC_SCALER': REC/'restored_scaler_identity.json',
 'REC_LOADED': REC/'loaded_state_identity.json',
 'REC_RUNTIME': REC/'runtime_and_source.json',
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def evidence(alias, first, last=None):
    last = first if last is None else last
    lines = FILES[alias].read_text().splitlines()
    assert 1 <= first <= last <= len(lines)
    return {'source_id': alias, 'path': str(FILES[alias]), 'first_line': first,
            'last_line': last, 'sha256': BEFORE[alias],
            'excerpt': '\n'.join(f'{i}: {lines[i-1]}' for i in range(first,last+1))}

def npz_schema(path):
    # Parse only each .npy header. No tensor payload is loaded or transformed.
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as fh:
                assert fh.read(6) == b'\x93NUMPY'
                version = tuple(fh.read(2))
                n = struct.unpack('<H' if version == (1,0) else '<I', fh.read(2 if version == (1,0) else 4))[0]
                header = ast.literal_eval(fh.read(n).decode('latin1'))
            arrays[name[:-4]] = {'shape': list(header['shape']), 'dtype': header['descr'],
                                'fortran_order': header['fortran_order']}
    return {'path': str(path), 'sha256': digest(path), 'arrays': arrays}

BEFORE = {key:digest(path) for key,path in FILES.items()}
T = {key:path.read_text() for key,path in FILES.items() if path.suffix == '.py'}
J = {key:json.loads(path.read_text()) for key,path in FILES.items() if path.suffix == '.json'}
for alias, txt in T.items():
    ast.parse(txt, filename=str(FILES[alias]))

FINDINGS = []
def finding(key, classification, title, conclusion, limitation, refs):
    FINDINGS.append({'id':key,'classification':classification,'title':title,
                     'conclusion':conclusion,'limitation':limitation,
                     'evidence':[evidence(*r) for r in refs]})

finding('S01','STATIC_MATCH_WITH_DECLARED_TRANSFORM','Descriptor scaler and abs boundary',
 'H5 fits population mean/std on train descriptors only (279 global vectors; 4868 actual residues in saved scaler), uses scale=1 for constant columns, transforms actual residues then pads g with zero. h is abs(zscore(global)); g keeps signs. abs is applied once in H5 tensorize because it calls the encoder directly, not VAEModel.forward. Original wrapper also abs(global); H4 reconstruction runner uses the same declared train zscore. No latent parameter standardizer is in this path.',
 'abs is non-invertible and discards global descriptor sign relative to the mean, but is inherited/declared. The reconstruction pipeline does not need an inverse feature transform. Saved scaler metadata is checked without rereading train/dev/cohort records; historical preprocessing faithfulness is not claimed.',
 [('F',154,186),('F',198,202),('F',347,363),('V',365,382),('OLD',87,112),('OLD',144,153),('SCALER',1,40),('SCALER',321,322)])
finding('S02','STATIC_MATCH','DDPM target and latent axes',
 'Encoder mu and logsigma are [B,16,48]; H5 stacks them at dim=1 as [B,2,16,48]. Channel0=mu, channel1=log standard deviation, positions=last axis including padded positions. DDPM predicts injected epsilon with mean MSE, not mu/z/variance. Generated channels are split by the identical convention and sigma=exp(logsigma); no factor 0.5 belongs here.',
 'This excludes an identified axis or log-variance interpretation mismatch in inspected code; it does not prove DDPM has learned the target distribution. Padding positions are modelled, not masked out.',
 [('V',114,127),('F',347,363),('F',376,378),('F',400,409),('F',502,518),('OLD',334,350),('ROOTDIFF',155,167)])
finding('S03','STATIC_MATCH','No latent inverse standardization or inverse-flow omission',
 'The DDPM cache stores native mu/logsigma without standardization, whitening, clipping or abs. Training adds diffusion noise in that coordinate system and generation consumes it directly. Therefore no missing latent inverse-scaler is identified. The posterior-parameter DDPM samples decoder-coordinate z=mu+sigma*epsilon and should go directly to decoder; inverse flow belongs to a different prior route.',
 'Standardizing these latent channels or clipping log scales would change the experiment and requires a new version; static consistency does not establish target adequacy.',
 [('F',347,378),('F',400,409),('F',502,524),('F',238,245),('H4',125,148)])
finding('S04','INHERITED_ENDPOINT_CONVENTION','999 coefficients versus 998 reverse updates',
 'Both train and sample use a length999 linear beta[0..998] schedule (1e-4 to .02), alpha=1-beta, abar=cumprod(alpha). Training randint(1,999) gives t=1..998; generation reverses 998..1, conditional epsilon prediction, variance beta[t] and zero final noise at t=1. There is no t=999 access and no train/sample index range mismatch.',
 'The inherited schedule has abar[0]=.9999, not 1. The loop ends at index0 without an extra t0-to-clean transition. This is a precise endpoint convention/departure from an exactly clean endpoint, not evidence that it caused current latent mismatch. No correction or trajectory rerun was performed.',
 [('F',366,369),('F',401,404),('F',483,501),('CONFIG',23),('CONFIG',38),('ROOTDIFF',92,103),('ROOTDIFF',114,131),('OLD',353,361),('H4',135,146)])
finding('S05','DECLARED_INACTIVE_AVERAGING','EMA at update1000 is current-weight copy',
 'The inherited EMA helper resets to current state before step2000. H5 updates it after each optimizer step and returns the EMA model for generation. At the frozen1000 endpoint, CE current/EMA metadata contain identical tensor-state digests a339af1652047bd0aee90cc3d343f5c177a0e625dd12a4cb21c4b1b418f4c94c. Recovery loads the saved EMA state and checks that digest.',
 'Calling this checkpoint averaged weights would be inaccurate. This is declared historical warmup, not a discovered checkpoint mix-up. Weight binaries were not loaded in this audit; equality is supported by existing metadata plus exact helper logic.',
 [('D',11,36),('F',382,385),('F',411,429),('DDPM_CURRENT_META',1,13),('DDPM_EMA_META',1,13),('RECOVERY',163,169),('RECOVERY',200,216),('REC_LOADED',1,10)])
finding('S06','ENTRYPOINT_CONTRACT_DIFFERENCE','Classifier-free guidance differs between source entrypoints',
 'H5 invokes only ddpm(x,t,labels): conditional-only, with no unconditional blend. The repository-root diffusion.py defaults cfg_scale=-1 and inference explicitly passes -1, equivalent to this conditional-only branch. The separate AMP_VAEflow/tasks/train_diffusion.py defaults7 and its inference passes3, uses16 channels with spatial axes2×length and random length10..35; H5 uses2 channels with spatial axes16×48, following the root/H4 convention. These are distinct configurations, not interchangeable replicas.',
 'Do not call H5 faithful to every original entrypoint or CFG3. No absent CFG application is identified relative to the frozen H5/H4 contract. Source config classes3 versus H5 terminal classes2 is an intentional label-task change; historical class biological meaning is not inferred.',
 [('F',382),('F',469,496),('ROOTDIFF',42,64),('ROOTDIFF',105,121),('ROOTDIFF',261,280),('TASKDIFF',42,62),('TASKDIFF',103,123),('TASKDIFF',283,295),('H4',135,141)])
finding('S07','STATIC_REACHABILITY_CONFIRMED','Terminal label mapping and dropout',
 'Preprocessing maps FREE=0 and AMD=1. H5 verifies descriptor/cohort class+terminal identity, carries labels to the posterior cache, trains a2-class embedding, and allocates FREE thenAMD by enumeration. Only label_emb(y) reaches the active UNet; hierarchical embed_level_1/2 are defined but get_label_emb is not called. Label dropping is one Python Bernoulli(.1) decision per minibatch, not per-example and not VAE descriptor dropout. VAE cond_drop=False, active WN dropout defaults0. Generation requests use only terminal label plus random tensors.',
 'This proves reachable software conditioning, not measured conditional fidelity or synthesized terminal chemistry. Global/residue descriptors are used for encoding the training sequences, not supplied at generation. Full cohort binding was not re-executed in this audit.',
 [('REP',15,19),('BUILD',99,116),('F',147,150),('F',185,192),('F',357,378),('F',381,407),('F',432,439),('F',480,491),('D',158,169),('D',212,221),('V',333,347),('FLOW',78,100)])
finding('S08','PRIOR_CONTROL_BOUNDARY','N(0,I) applies to flow-space zp, not necessarily decoder z',
 'With use_prior_flow=True, training computes zp=prior_flow(z,g), evaluates Normal(0,1) density at zp, and decodes z. For fixed lawful g, sampling that learned conditional prior requires zp~N(0,I), z=prior_flow(zp,g,reverse=True), then decoder(z). g is a sequence-derived3×48 descriptor tensor. The original H4 empirical-condition-flow route explicitly selects a training descriptor and applies reverse flow.',
 'There is no specified terminal-to-g generator in frozen H5. g=None is callable in WN but drops a conditioning signal present during training, and cannot be silently equated with the learned conditional prior. Selecting training g is empirical-conditioned prior sampling; feeding decoder a standard normal z directly is a standard-normal latent control and has no terminal input of its own.',
 [('F',189,202),('V',174,193),('V',290,303),('FLOW',120,133),('FLOW',234,254),('FLOW',278,291),('BUILD',113,116),('H4',125,128),('V',142,146),('F',238,245)])
finding('S09','NO_IDENTIFIED_JACOBIAN_OMISSION','Active coupling has zero log-determinant',
 'The active ResidualCouplingBlock instantiates mean_only=True couplings; logs are zeros, so each additive coupling logdet is zero. Channel flip has unit absolute determinant. Ignoring the returned logdet in the block is therefore consistent for this active architecture.',
 'This statement is limited to the inspected mean_only block. A different affine/non-volume-preserving flow would require its own density accounting.',
 [('FLOW',192,196),('FLOW',234,254),('FLOW',278,291),('V',174,177),('F',200,203)])
finding('S10','SAVED_EVIDENCE_LIMIT','Epsilon not saved in CE recovery or MSE generated NPZ',
 'Completed CE recovery/MSE generated arrays save posterior_parameters,z,raw,tokens,lengths,length_scores; posterior files save values only. The final randn_like(mu) is intended as a fresh standard-normal draw, but uses the same seeded generator stream after initial/reverse DDPM draws. The saved files have no original epsilon/sigma/mu/logsigma keys; mu/logsigma can be sliced from posterior_parameters, while sigma and epsilon would be derived/reconstructed.',
 'Algebraic (z-mu)/exp(logsigma) is a floating-point residual, not guaranteed bit-exact original noise. Replaying seed consumption would be new RNG execution and was not done. Shared seeds alone do not prove strict statistical independence or identical realization across arms. decoder(epsilon) is a new control, not a replay of saved z.',
 [('F',480,524),('RECOVERY',151,156),('RECOVERY',215,216)])
finding('S11','REFERENCE_NOT_GENERATION','Mean reconstruction and empirical posterior reference boundaries',
 'Saved train/dev reconstruction CSVs export deterministic mu decoding only. raw_prediction_tokens48 versus prediction_tokens48 are output representations of the same mean decode, not independent mean/sample branches. A posterior resampling reference using cached training mu/logsigma retains sequence-derived posterior information; it is an empirical aggregated-posterior reference, not independent prior generation.',
 'Mean reconstruction has privileged observed sequence+descriptors. Neither this score nor a training-posterior resampling control establishes independent generation quality. Any new control must be named, prelocked and counted separately; this audit launches none.',
 [('F',347,363),('F',551,565),('F',591,608),('H4',129,134)])
finding('S12','CAUSAL_LIMIT','Finite generation is not latent matching, and mismatch is not bug proof',
 'The frozen sampler exponentiates unbounded generated logσ, checks finite/positive sigma and finite z/raw, and records failures without clipping or replacement. Finite successful outputs establish engineering execution only. No active-route channel/logscale/scaler/EMA-label mismatch was identified in the inspected path.',
 'This static audit cannot establish the cause of reported distribution mismatch or distinguish DDPM fitting limits, posterior modelling difficulty, decoder sensitivity and objective effects. No new quality statistics, forward passes or models were run. CE-vs-MSE and scalar-vs21-head contrasts also differ in the declared objective normalization; scalar-toCE is not a single-component causal claim.',
 [('F',205,223),('F',238,245),('F',502,546)])

SCHEMAS=[]
for base in (CE,MSE):
    SCHEMAS.append(npz_schema(base/'train_posterior_parameters.npz'))
for base in (REC,MSE):
    for terminal in ('FREE','AMD'):
        for prefix in ('generated_posterior','generated'):
            SCHEMAS.append(npz_schema(base/f'{prefix}_{terminal}.npz'))
CSV_HEADERS=[]
for base in (REC,MSE):
    for role in ('train','dev'):
        path=base/f'{role}_reconstruction.csv'
        with path.open() as stream:
            header=next(csv.reader(stream))
        CSV_HEADERS.append({'path':str(path),'sha256':digest(path),'header':header,
                            'row_values_parsed':False})

CHECKS=[]
def check(name,result): CHECKS.append({'check':name,'pass':bool(result)})
check('frozen_runner_hash_matches_config',BEFORE['F']==J['CONFIG']['implementation_sha256'])
check('source_manifest_hash_matches_CE_runtime',BEFORE['SOURCE_MANIFEST']==J['RUNTIME']['source_manifest_sha256'])
source_map={x['relative_path']:x for x in J['SOURCE_MANIFEST']['files']}
runtime_map={x['path']:x['sha256'] for x in J['RUNTIME']['source_files']}
for alias in ('V','D','FLOW','ATTN','ROOTDIFF','TASKDIFF','VAE_YAML','CSV_LOADER'):
    rel=str(FILES[alias].relative_to(C))
    check(alias+'_hash_matches_manifest_and_recorded_runtime',BEFORE[alias]==source_map[rel]['corrected_sha256']==runtime_map[str(FILES[alias])])
check('recovery_runtime_binding_same',all(J['RUNTIME'][k]==J['REC_RUNTIME'][k] for k in ('torch','numpy','python','threads','interop_threads','device','source_manifest_sha256')))
check('CE_current_ema_recorded_tensor_digest_identical',J['DDPM_CURRENT_META']['tensor_state_sha256']==J['DDPM_EMA_META']['tensor_state_sha256'])
check('CE_DDPM_endpoint_1000',J['DDPM_SUMMARY']['updates']==J['DDPM_SUMMARY']['ema_step']==1000 and J['DDPM_SUMMARY']['fixed_endpoint_reached'])
check('recovery_loaded_ema_identity',J['REC_LOADED']['before']['ddpm_ema_state']==J['REC_LOADED']['expected']['ddpm_ema_state']==J['DDPM_EMA_META']['tensor_state_sha256'])
check('recovery_scaler_identity_no_refit',J['REC_SCALER']['sha256']==BEFORE['SCALER'] and not J['REC_SCALER']['fit_called'])
check('train_only_scaler_metadata',J['SCALER']['fit_scope']=='train only' and J['SCALER']['fit_record_ids']==list(range(279)) and J['SCALER']['global']['n']==279 and J['SCALER']['residue']['n']==4868)
check('NPZ_headers_have_no_epsilon',all('epsilon' not in d['arrays'] for d in SCHEMAS))
check('train_cache_shape',all(d['arrays']['values']['shape']==[279,2,16,48] for d in SCHEMAS[:2]))
check('generated_param_shape',all(d['arrays'].get('values',d['arrays'].get('posterior_parameters'))['shape']==[16,2,16,48] for d in SCHEMAS[2:]))
check('CSV_headers_identical',all(d['header']==CSV_HEADERS[0]['header'] for d in CSV_HEADERS))
check('inspected_python_syntax_valid',True)
check('source_files_unchanged_during_audit',all(digest(path)==BEFORE[key] for key,path in FILES.items()))
assert all(c['pass'] for c in CHECKS), CHECKS

now=datetime.now(timezone.utc).isoformat()
dump('LATENT_ROUTE_STATIC_AUDIT.json',{
 'created_utc':now, 'project':'AMP H5 antimicrobial peptide end-to-end pilot',
 'status':'STATIC_ROUTE_CONSISTENT_WITH_DECLARED_DIFFERENCES_CAUSE_UNRESOLVED',
 'scope':'Frozen code/config/CE completed training+recovery metadata and schema-only completed CE/MSE artifacts. No live scalar content, test/cohort records, model imports, forwards, new random draws or quality calculations.',
 'confirmed_active_route_implementation_bugs':[],
 'absence_of_found_bug_is_not_proof_of_absence':True,
 'causal_claims_authorized':False,'PROMOTE':'HOLD',
 'findings':FINDINGS,
 'control_names':{
  'decoder_of_standard_normal':'standard-normal latent control; no terminal-conditioned prior claim',
  'decoder_of_train_mu':'privileged posterior-mean reconstruction reference',
  'decoder_of_selected_train_mu_plus_sigma_epsilon':'empirical training-posterior resampling reference',
  'inverse_flow_of_standard_normal_given_empirical_g':'empirical-descriptor-conditioned flow prior',
  'frozen_H5_generator':'terminal-conditioned posterior-parameter DDPM followed by posterior draw and decoder'},
 'reviewer_actions':{'model_imports':0,'weight_loads':0,'neural_forwards':0,'training_updates':0,'new_generation':0,'RNG_reconstructions':0,'quality_recalculations':0,'test_or_cohort_records_read':0,'live_scalar_contents_read':0,'frozen_files_modified':0},
 'remaining_limits':['Recorded metadata/hashes and static paths cannot replace full end-to-end execution proof.','Existing observed latent mismatch is not itself proof of a coding bug.','No new head, normalization, noise schedule, EMA warmup or CFG setting is proposed or applied.','New controls, if later authorized, require fixed inputs/denominators and independent review; they cannot retroactively change original32-slot outcomes.']})
dump('SAVED_ARRAY_SCHEMAS.json',{'npz':SCHEMAS,'reconstruction_csv_headers':CSV_HEADERS,
 'inspection':'NPY headers only; no tensor values loaded or numerical quality calculated. SHA256 is byte-level I/O only.'})
dump('SOURCE_BINDINGS.json',{'created_utc':now,'files':[{'source_id':key,'path':str(path),'sha256':BEFORE[key]} for key,path in FILES.items()],
 'historical_source_relation':[{**source_map[str(FILES[a].relative_to(C))],'source_id':a} for a in ('V','D','FLOW','ATTN','ROOTDIFF','TASKDIFF','VAE_YAML','CSV_LOADER')]})
dump('VALIDATION.json',{'status':'PASS','checks':CHECKS,'check_count':len(CHECKS),'scope':'Static syntax/hash/metadata/schema assertions only; not model execution validation.'})

def link(alias,line): return f'[{alias}:{line}]({FILES[alias]}:{line})'
report=f'''AMP H5 潜变量路径静态独立审计（v1）

本次未在已检查的冻结 H5 路径中发现 μ/logσ 通道交换、将 logσ 当作 log 方差、遗漏 latent 逆标准化、错载 EMA 或端基标签映射错误。这个结论只排除了已检查的具体实现不一致；不是“代码完全正确”的证明，更不能把已观察到的 latent 匹配差直接归因为某个算法组件。当前比较仍为 HOLD。

只读范围：冻结 runner、源配置、已完成 CE 训练/恢复元数据、CE/MSE 保存数组的 header。未读取 test、cohort 记录或 live scalar 内容；模型导入、权重加载、forward、训练、采样、RNG 重建和质量重算均为 0。现有文件未修改。共 {len(CHECKS)} 项静态校验通过。

| 路径 | 核实事实 | 判断边界 |
|---|---|---|
| 输入标准化 | train279 全局向量、4868 实际残基拟合 population zscore；global 再 abs，residue 保留正负，padding 后置为0 | abs 为历史继承的非可逆信息折叠；不是 latent scaler，也不应在生成时逆变换 |
| DDPM 对象 | `[B,2,16,48]`，channel0=μ，channel1=logσ，末轴=位置；预测注入的 epsilon | 没有证据表明通道/位置错换；参数与位置（含 padding）全部建模 |
| latent 到 decoder | 生成 σ=exp(logσ)，z=μ+σε，直接 decoder(z) | 这条 posterior-parameter DDPM 路线没有漏 inverse flow；latent 也没有预先标准化 |
| noise schedule | 999 个系数，train t=1..998，reverse998..1，共998次；最后噪声置0 | 无999越界；abar0=.9999，未额外执行t0→完全clean，是继承端点约定，效果未知 |
| EMA | helper 在2000更新前复制当前权重；CE1000的 current/EMA tensor digest相同 | 不能称已发挥平滑平均效果；不是已发现的checkpoint混用 |
| CFG | H5仅调用有条件预测，无unconditional blend；源root diffusion.py同为cfg=-1 | 另一个 tasks/train_diffusion.py 是cfg=3推理、16通道、随机长度；不是相同入口合同 |
| 条件 | FREE0 / AMD1；DDPM 每批10%丢标签；生成只有端基与随机输入 | 软件条件可达不等于端基控制有效；序列字符串不能独自证明化学末端状态 |

精确实现入口：{link('F',154)}（scaler）、{link('F',347)}（cache）、{link('F',400)}（epsilon目标）、{link('F',480)}（生成）、{link('D',27)}（EMA）、{link('D',212)}（label embedding）。完整行范围、摘录和文件 SHA256 已随每条 finding 保存于 LATENT_ROUTE_STATIC_AUDIT.json。

标准正态所在的坐标必须区分。训练先把 decoder 坐标 z 送入 `prior_flow(z,g)` 得到 zp，再对 zp 求 N(0,I) 密度；decoder 接收 z。给定合法 g，所对应的 learned conditional prior 应是 zp~N 后逆 flow 回 z。活动 flow 使用 mean_only=True 的加性耦合，logdet为0，因此此处忽略返回的 logdet 不是漏 Jacobian 的证据。见 {link('F',198)}、{link('V',193)}、{link('FLOW',234)}、{link('FLOW',278)}。

g 是由训练序列逐残基内容生成的3维descriptor，不是端基 c 的函数；冻结H5没有定义 c→g 的生成机制。`g=None` 虽在底层可调用，但跳过了训练使用的条件，不可直接叫同一 learned prior。H4明确另设“抽取训练g后inverse flow”的经验条件路线。直接 `decoder(epsilon)` 只能叫 standard-normal latent control；decoder没有接terminal，不能继承原DDPM的端基条件性。见 {link('BUILD',113)}、{link('H4',125)}、{link('V',142)}。

保存证据方面，CE恢复与MSE的generated NPZ均没有epsilon原始键；有posterior_parameters、z、raw、tokens、lengths、length_scores。posterior NPZ只含values。最终epsilon来自同一有状态RNG在逆扩散后的新randn_like调用，代码不据mu/logσ数值计算噪声；但本次没有做RNG重建，也不把相同种子当成跨臂逐位一致证明。`(z−μ)/exp(logσ)`仅是浮点派生残差，不保证还原原始epsilon。

现存train/dev重建都是μ的确定性解码。两个token字段是同一解码的表示，不能当作mean/sample两分支。训练后验重采样也保留了训练序列派生信息，应称 empirical training-posterior reference；它能作为诊断参考，却不是独立先验生成。标准正态control、训练后验reference、经验g的flow prior、端基DDPM生成应保留各自分母与来源，不能互换名称。

本审计未提出或应用新head、latent缩放、clipping、CFG、EMA或schedule调参。finite输出只表明数值/工程执行成功；究竟是DDPM拟合不足、后验分布难学、decoder敏感性或目标影响，仍需独立、预锁的小范围证据区分，不能靠静态检查断言原因。

冻结runner SHA256：`{BEFORE['F']}`。源文件与元数据完整绑定见 SOURCE_BINDINGS.json；保存数组结构见 SAVED_ARRAY_SCHEMAS.json；检查见 VALIDATION.json。
'''
(HERE/'READOUT.zh.md').write_text(report)
manifest={'created_utc':now,'scope':'new append-only static audit deliverables','files':[]}
for p in sorted(HERE.iterdir()):
    if p.is_file() and p.name!='MANIFEST.json':
        manifest['files'].append({'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size})
dump('MANIFEST.json',manifest)
print(json.dumps({'status':'PASS','checks':len(CHECKS),'findings':len(FINDINGS),'source_bindings':len(FILES),
 'audit_sha256':digest(HERE/'LATENT_ROUTE_STATIC_AUDIT.json'),'manifest_sha256':digest(HERE/'MANIFEST.json')},ensure_ascii=False))
