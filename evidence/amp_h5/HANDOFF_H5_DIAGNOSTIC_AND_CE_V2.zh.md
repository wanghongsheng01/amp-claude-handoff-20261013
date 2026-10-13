# AMP H5 数值诊断完成与单 CE v2 运行交接

唯一题目仍是抗菌肽 AMP。所有新产物位于本目录；旧 AMP 源、H0–H4和稿件只读，test37未读。此交接不是完整四臂比较完成声明，PROMOTE=HOLD。

## 当前正在运行

2026-10-11 16:48:18 UTC，单 CE v2 在独立回执后启动：supervisor PID **60744**，模型子进程 PID **60747**，工具 session **47076**。监督记录为 `runs/single_ce_execution_v2/SUPERVISOR_STATUS.json`；模型输出为 `runs/formal_v2/categorical_ce/`。已有真实VAE更新记录，不能重复启动。已为该supervisor设置生命周期内临时caffeinate，进程结束即解除，没有修改系统设置。

配置仍是 fresh VAE1000 → 自身train编码 → fresh DDPM1000 → FREE16/AMD16请求；CPU单线程、串行。VAE1920秒、DDPM1020秒、共同训练2940秒、生成1800秒、外层4920秒。不复用诊断权重，不续训、不clip、不补采。尚未报告正式CE终点或质量结果。

实际状态以持续写入的supervisor、训练日志和最终status为准。交接时的只读快照见 `protocol/HANDOFF_RUNTIME_SNAPSHOT.json`。不要使用旧 `FORMAL_PLAN.json`、旧 `SINGLE_CE_PLAN.json` 或旧输出目录重新运行。

## 已完成的真实结果

1. H4成功checkpoint保存末步核验：两组已存档16行批次，20项比较全部逐bit一致；包括预测、逆扩散、exp、重参数化和scalar解码。仅验证存档t=1，不称完整998步新重放。
2. 固定CE20表示的DDPM成熟度诊断：fresh同一条1000更新轨迹，检查点20/1000，配对噪声、四个预锁请求。1000训练用558.689秒，四请求91.691秒，监督总651.827秒。第20步两个请求溢出，第1000步两个请求均得到有限正sigma及有限z。所有失败保留，诊断状态为COMPLETE_WITH_NUMERICAL_FAILURES，exit0仅表示协议完整执行。
3. 独立后审50项通过；第1000步FREE/AMD logσ分别约[-2.085,1.084]/[-2.184,.583]，sigma[.124,2.955]/[.113,1.791]，z RMS1.1605/.9567，训练q(z)解析RMS1.1083。没有此前数百量级放大；不能据此证明分布吻合、唯一根因已隔离、新VAE1000稳定或n=16批次稳定。旧smoke的OneCycle总步数20不同，本诊断总步数1000，不能称原smoke精确重放。
4. AR正式1000更新已完成，32请求全部合法5–48且每端基16条unique，未观测到train exact-copy。FREE/AMD dev MMD²为0.011567/0.007362。dev teacher NLL在200步2.6153，最终3.1698，训练最终1.6431，存在过拟合迹象；保留预锁1000末点，AR_ADEQUACY=NOT_ESTABLISHED，不称最强基线已击败。

## 必须保留的入口失败与修复

第一次单CE入口 `runs/single_ce_execution_v1` 在训练前失败（1.074秒、0更新、32请求not_executed）。开发集原始idx为279–319，旧guard误要求0–40；41行内容和行序完全吻合。其development_read=false是成功校验后才置位的记录缺陷，实际上已打开dev文件。

新版本只显式绑定train idx起点0、dev起点279，并修正读取尝试/校验成功状态。没有重排、改split、改数据、重算descriptor或改变模型。独立22项输入契约检查通过，AST确认只有read_records/validate_config/run三个契约函数变化；模型、loss、训练、随机、采样和导出函数完全相同。三VAE配置仅增加两个idx起点及更新实现hash。旧失败与旧回执完整保留；这是一项明示的训练前契约修复版本，不能抹去新增尝试成本。

## 主线程继续点

先查看当前单CE v2进程和日志，待其自然完成，再由实现作者之外的审查者核查1000/1000更新、全部32请求、数值/尺度/预算、数据与代码hash。仅依据工程可行性决定是否为原冻结v2 MSE与scalar配置新建逐臂计划及独立回执；不得根据重建或MMD好坏改模型、门槛或种子。二者目前均未授权、未启动。

若继续，使用 `protocol/formal_configs_v2/` 和 `implementation/vae_pipeline_formal_v2.py`，保持相同契约修复；生成各自全新输出。原AR结果继续引用 `runs/formal_v1/terminal_lstm`，不要重训或误当缺失。后续评价需要显式引用v2三个VAE路径与v1 AR路径；原评价器CLI支持分别给样本路径，旧readout builder的单run-root路径假设需要明确的只读路径映射或新版本，不能伪造/覆盖原结果。

原L/G门、128请求完整分母和所有指标不变；缺失/失败如实NOT_EVALUABLE。额外成熟度诊断、入口失败及适应性分阶段执行成本单列，不能声称实际等计算或无额外开发。尚未建立新颖性、强基线充分性、抗菌活性或投稿就绪；旧湿实验不验证新H5。

## 核心证据入口

- `feasibility/ddpm_maturity_author/READOUT.zh.md`：作者数值读出。
- `review/independent/DDPM_MATURITY_POSTFLIGHT_REVIEW.zh.md`、`DDPM_MATURITY_SCALE_ASSESSMENT.json`：独立诊断后审与尺度判断。
- `review/independent/SINGLE_CE_V2_PREFLIGHT_RECEIPT.json`：当前唯一运行的限定回执。
- `protocol/SINGLE_CE_V2_PLAN.json`：当前计划，SHA256 `ce193e133e86d995c21e4596412ce08ee61bb2be3cb47b2549d4f11f5c86f2d6`。
- `feasibility/formal_v2_contract_repair/REPAIR.zh.md`、`review/independent/FORMAL_V2_CONTRACT_REVIEW.json`：0更新入口缺陷及修复证据。
- `feasibility/ar_formal_readout/AUTHOR_READOUT.zh.md`、`review/independent/AR_FORMAL_POSTFLIGHT_REVIEW.json`：AR完整读出与独审。
- `protocol/PREDECESSOR_CHECK_AFTER_DIAGNOSTIC.json`：本轮复核旧H4/H3/H2三个manifest下64/45/32文件，全部未变；不冒称本代理复核了全部历史507文件或全部稿件。

无夜间自动化已创建；一次工具尝试因委派线程不支持而失败，主线程后续明确无需创建并持续跟进。没有远端登录、凭据操作、上传、push或投稿，BACKUP_BLOCKED。Caliber自行按原负载门等待，不因其等待改动本实验预算。
