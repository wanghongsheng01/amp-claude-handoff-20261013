# AMP H5：后验坐标与溢出路径独立只读审计

2026-10-11；`/root/h5_literature`。未加载模型、未调用神经前向、未训练、未生成、未读test。读取已保存smoke后验、H4重放中间量与train缓存；只做张量统计和保存值之间的代数复算。可复跑脚本 `audit_saved_numbers.py`，全量数值及输入哈希见 `SAVED_NUMERIC_AUDIT.json`。

**结论：没有发现 logσ/logvariance 二倍混淆、posterior 反标准化遗漏、通道交换或 CFG 放大这一类源码差异。smoke 的最终 DDPM 后验参数在 exp 之前已经达到数百；随后 float32 的 exp(logσ) 确实溢出。这个定位没有解释为什么 DDPM 会输出这些数，也不足以把根因直接定为欠训练。必须完成父任务安排的已知H4成功checkpoint在新路径的最小一致性核验，之后再决定长跑。**

## 坐标与目标逐段核对

下表旧源根 `R=/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011`；新源 `H=amp_h5/implementation/vae_pipeline_formal.py`。文件哈希由 `SOURCE_BINDINGS.json` 绑定。

| 阶段 | 精确源码位置 | 独立判断 |
|---|---|---|
| encoder输出语义 | `R/corrected_code/AMP_VAEflow/modules/vae.py:123–127,292`；`H:198–201` | 原encoder将32通道拆为16个μ和16个logs，采样为`mu+eps*exp(logs)`，q也定义为`Normal(mu,exp(logs))`。因此logs是log标准差，不是log方差。 |
| latent维度 | `vae.py:110,123,172–177`；`H:358–361` | 每个位置有16latent，共48位置；z为`B×16×48`，不是每条肽一个16维向量。DDPM为`B×2×16×48`，channel0=μ，channel1=logσ。 |
| 原全forward返回转置 | `vae.py:303,329`；旧runner `run_rebuild_v3r2.py:344–347`；`H:357–361` | 原full forward返回`B×48×16`，旧runner stack后permute恢复`B×2×16×48`；H5直接取encoder的`B×16×48`再stack。两种布局一致，不应再加一次旧permute。 |
| 输入描述符标准化 | 旧runner`:85–110,144–153`；`H:154–186` | train拟合9维global与3维residue的population zscore；global再abs；residue只标准化真实位置、padding保持0。三smoke保存的mean/scale/n/fit_record_ids与H4逐值相等。它们是encoder输入g/h的标准化。 |
| DDPM训练目标 | 旧runner`:334–363,384–417`；`H:347–363,376–409` | 两者扩散的都是原始μ和logσ，未标准化posterior target，也没有对应反标准化。不能把输入feature scaler用到DDPM输出上。要新增posterior标准化是新训练目标合同，不能当无影响修补。 |
| 前向加噪与逆公式 | 旧runner`:353–363,694–700`；H4 `h4_run.py:110,135–146`；`H:366–369,401–409,483–516` | 同线性beta 1e-4→0.02、999槽；train t=1…998，reverse 998…1；均预测epsilon，reverse用sqrt(beta)噪声且t=1不加随机噪声。原始beta0/索引约定共同继承，不是H5新差异。 |
| posterior到decoder | 旧runner`:706–713`；H4`:144–150`；`H:502–518,238–245` | 均先exp(logσ)，再`z=mu+eps*scale`，直接decoder。这里不需要逆flow：目标本来就是q坐标的μ/logσ；在此添inverse-flow会改变坐标语义并要求源g。 |
| CFG | 原 `R/corrected_code/diffusion.py:117–121,262–264`；旧runner`:698`；H4`:139`；`H:491` | 原仓库只有cfg_scale>0才做unconditional/conditional lerp，原调用-1关闭它。旧成功runner、H4和H5都只做一次带terminal标签的预测，没有CFG放大。训练10%丢标签不是生成期CFG被激活的证据。 |
| prior flow与KL | `flow_base.py:240–248,276–291`；`vae.py:292–297`；`H:200–218` | coupling固定mean_only=True，logdet为0，忽略返回logdet与该构造相容。flow自己的logs=0不是encoder的logs。scalar沿用density_sum/16；新21头density_sum/(B×48)为明确改动，B32时KL缩小96倍，不得说所有臂目标尺度相同。 |
| EMA | `R/corrected_code/AMP_VAEflow/modules/diff.py:27–36`；`H:385,411–422` | 到2000更新前为整状态copy；旧1000步和smoke20步的“EMA”均不是成熟指数平均。此语义共同继承，不是本次新尺度转换。 |

feature边界还有实现形式差别：旧runner先将global cast float32再abs；H5先在Python float64中abs，encoder再cast float32。对有限float值的abs与cast这里应交换相容；本审仅核源码和已保存scaler，未调用encoder验证输出完全一致。旧encode还运行未使用的decoder/flow，H5只取encoder；eval无dropout时不应改变μ/logσ，但最终同checkpoint数值回归须由执行核验完成。

## 保存张量给出的实际定位

每个smoke terminal只有1请求，logσ有768元素；H4每组16请求。下列H4来自当时保存的真实重放张量，本轮没有重新生成。

| 来源 | 训练缓存logσ范围 | FREE最终logσ范围 | AMD最终logσ范围 | FREE/AMD `exp(logσ)=+inf` 数 |
|---|---:|---:|---:|---:|
| CE smoke20 | −3.2073…2.1637 | −942.9824…655.7025 | −714.0074…636.8910 | 250/326 |
| raw MSE smoke20 | −3.2085…2.1641 | −941.5716…660.4686 | −715.3682…638.3248 | 250/327 |
| scalar smoke20 | −3.2082…2.1638 | −943.0933…655.6302 | −714.3741…636.7850 | 249/326 |
| 旧H4成功状态 | −0.2633…0.2537 | −1.4465…1.1499 | −2.1222…1.9135 | 0/0 |

所有smoke保存的最终μ/logσ仍有限；μ也达到约−717…854，说明大幅变化不限于第二通道。float32 exp上溢阈值约88.7228，故记录的首个非有限检查点是 `H:511–513`，异常大的输入在此之前已经存在。下溢也同时存在（CE FREE/AMD有255/201个scale为0）；当前检查先因inf抛错，不能把未到达的后续underflow检查说成没有underflow。decoder、scalar取整和J尚未运行，不能造成此处首次非有限。

作为纯保存值代数，额外算了错误假设`exp(0.5*logs)`：CE FREE/AMD仍有150/205个inf。这个反例只说明“除2”既不符合encoder语义，也不能消除当前大范围溢出；没有将此错误变体用于采样或训练，不建议采用它。clip/cast float64同样不能被说成已修复DDPM分布。

H4 FREE/AMD各自通过五项逐bit复算：保存的最后reverse输入/预测/噪声重算最后输出相同；最后输出=最终posterior；exp(logσ)=保存scale；μ+保存epsilon×scale=保存latent；保存raw_replay=raw_original。latent、hidden、raw均有限。这里证明保存的末段链条数值自洽，**不是本轮完整998步重放，也不是H5新函数已通过旧checkpoint回归**。

## 已排除与尚未排除

已排除到源码/保存值范围：logs被定义成logvariance；后验目标做过标准化却忘了逆变换；H5显式增加CFG；decoder/J先引发此smoke溢出；新feature统计与旧H4不一致。不存在这些证据，不等于整个实现已经完全无错。

尚未隔离：20步VAE/20步DDPM与旧1000/1000状态差异、OneCycle总周期变化、DDPM训练seed322与旧223差异、smoke生成batch1/seed422及423与H4 batch16/seed524及534差异、每个逆步预测误差及累积量。smoke只保存末态和进度时刻，没有全逆链张量，无法定位从哪一步开始偏离，也不能用末步MSE≈1单独判根因。三臂相近失败提示共享流程值得检查，但它们不是隔离因子的因果控制。

最小先行验证应使用旧H4确切VAE/DDPM-EMA、noise schedule、CPU/float32、batch16、两个原seed及原条件，在H5的采样路径保留至少初态、第一步、几个固定步和末态与输出；既然原H4有成功回放材料，这比直接长跑三个新模型更有鉴别力。通过后只说明新采样路径与旧状态兼容；不保证新21头表示的DDPM可学或1000步足够。若需要比较encoder入口，则同旧checkpoint、同冻结train对象的μ/logσ取值一致性应单列，不与生成检查混为一项。本审不启动上述执行。
