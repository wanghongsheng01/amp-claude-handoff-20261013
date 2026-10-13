# AMP H5 对 H4 成功路径的静态核对

状态：代码静态核对与最小检查器已准备，尚未执行检查器，不能写成数值一致性已通过。三条 VAE 正式训练保持暂停。本文件没有以“增加训练即可修复”解释 smoke 失败。

## 当前结论

H5 相对已成功的本地 H4 路径，未发现 logσ/logvar 混用、DDPM 目标通道交换、遗漏 latent 逆缩放、β/α 日程或 CFG 强度的新差异。这个结论只来自以下已定位代码，待固定权重与已保存输入的一步数值重放核验。

**原 GitHub 仓库有两个 CFG 配置不同的入口，必须分别指明**：`corrected_code/AMP_VAEflow/tasks/train_diffusion.py:105` 的 sample 默认 cfg_scale=7，`:284` 主推理调用显式 cfg_scale=3；`:120–123` 将 conditional 与 unconditional noise prediction 作 lerp。另一个顶层入口 `corrected_code/diffusion.py:106` 默认 cfg_scale=-1，`:264,280,296` 的三个显式调用也为-1，关闭CFG。H4 成功路径、run_rebuild_v3r2、旧 complete_posttraining 所调用的 run_rebuild，以及 H5 均只执行 conditional forward，与顶层入口的关闭配置一致。因此不存在 H5 相对 H4 新引入的CFG差异，亦不能笼统声称本地重建偏离“原仓库唯一CFG配置”。不得现在以 CFG=3 替换协议采样并宣称修复；那会是另一个干预。

## 逐项代码核对

下列旧路径均相对源根 `.../amp_commsbio_20261011/rebuild_experiments_20261011/`；H5 相对当前 `amp_h5/implementation/`。

| 项目 | 旧成功路径 | H5 | 静态判定及限制 |
| --- | --- | --- | --- |
| encoder 参数语义 | `modules/vae.py` 的 FVAEEncoder 将通道分成 m、logs，并以 `m+randn*exp(logs)` 采样；Normal 的 scale 也是 exp(logs) | 直接调用同一 encoder，使用 mu、logsigma，Normal scale=exp(logsigma) | 均为 log 标准差，不是 log 方差；没有 0.5 因子 |
| DDPM target 来源 | v3r2 `encode` 从 FVAE infer 返回的 mu/logs `[B,48,16]`，stack 后 permute 成 `[B,2,16,48]` | `posterior_parameters` 从 encoder 取得 mu/logs `[B,16,48]`，直接 stack 成 `[B,2,16,48]` | 通道0=mu、1=logσ；没有将 prior-flow 的 z_p 当 target |
| descriptor 与 latent 尺度 | train-only population zscore 用于全局 h 和逐残基 g；全局 h 再 abs；encode 输出 mu/logs 原值 | 同样 train-only zscore、全局 abs；mu/logs 原值 | 两者都没有 latent 参数 zscore、clip 或后验逆缩放步骤；不能凭 exp 溢出推断遗漏 inverse scaler |
| h 转换 | v3r2 先 float32 再 abs；encoder 再 float | H5 先 Python double abs，构造 float64，encoder float | 对有限实数 abs 与该单次 float32 舍入交换；不是 logσ 的尺度变换。尚未在此检查中重跑 encoder |
| 训练噪声目标 | epsilon prediction：t均匀取1..998，xt=sqrt(abar_t)*target+sqrt(1−abar_t)*epsilon，MSE | 相同公式、范围、MSE | 不含 v-prediction、x0-prediction 或方差学习 head |
| UNet 形状 | 本地成功 checkpoint 为 c_in=c_out=2，空间16×48、time_dim256、2类 | 相同 | 原 GitHub 脚本还使用 `[B,16,2,L]` 与随机 L；该原仓库路径并非本地 H4 checkpoint 路径 |
| β/α 日程 | beta=linspace(1e−4,.02,999)，alpha=1−beta，abar=cumprod(alpha) | 相同 | 下标0不参与训练t或逆链；都是998次逆更新 |
| reverse 均值与噪声 | `(x−(1−alpha)/sqrt(1−abar)*pred)/sqrt(alpha)+sqrt(beta)*noise`；t=1 noise=0 | 同式分成两个赋值 | 使用 beta_t 方差，未改成 posterior beta-tilde；待检查器按源码 AST 逐位比较 |
| 条件与 CFG | H4 `h4_run.py:139` 和 v3r2 `:698` 都只有 `ddpm(x,t,labels)` | 同样一个 conditional forward | H4/H5无CFG差异；仓库顶层入口关闭CFG，tasks入口默认7/main3，需分别披露 |
| label dropout | v3r2训练时Python random<.1置None | 相同算法和比例 | H5的DDPM seed=322；旧成功run由training_seed222+1得到223。随机流不同，不能把两运行当逐步配对 |
| 优化器及 schedule | AdamW lr .005，eps1e−5，OneCycleLR total_steps=1000，EMA .995 | formal拟定同1000；smoke total_steps=20 | 20步OneCycle被压缩，不能视为1000步路径的前20步；这一差异不说明更多训练必然恢复 |
| EMA | 同一 `modules/diff.py`，step<2000时每步直接复制当前参数 | 同一实现 | 旧1000与smoke20均尚未进入指数平均阶段；不能把旧成功归因于平滑而把smoke失败归因于未平滑 |
| posterior 重参数化 | H4 `mu=x[:,0],log=x[:,1]`；scale=exp(log)；z=mu+epsilon*scale | 同样 | 无 clipping、缩放补偿、逆flow或g/h条件进入DDPM生成 |
| decoder | H4在固定latent上先WN再out_proj；原scalar clamp/round/firstPAD | H5 decoder调用的内部也是WN+out_proj；scalar读出相同 | CE/MSE的21类输出与J是计划干预；最小parity用旧scalar验证公共路径 |
| sampling seed/批量 | H4 FREE=524、AMD=534，各16；全部来自保存归档 | H5 FREE422、AMD423；smoke各1、formal拟各16 | 预先声明的变化，不是旧请求精确重放；batch形状改变也会改变抽样流 |
| 初始化与VAE噪声 | 旧run固定seed222后构建模型，再顺延全局RNG | H5同seed构建模型，另把posterior训练噪声重置222以锁CE/MSE配对 | 这是明确的训练随机流差异；不是输出坐标混淆，也不能从单seed定位失败根因 |

`complete_posttraining.py` 是更早200/200运行在元数据保存失败后的零训练继续脚本，导入的是 `run_rebuild.py`。它加载已绑定 EMA 后调用该 runner 的 `generate`；该函数也没有 CFG。H4本次成功所绑定的是 `runner/dbaasp_v4_cpu_pilot_001` 的1000/1000权重，不能把两个历史运行混为一个。

## 最小已保存输入检查

`h4_saved_step_parity.py` 和 `PARITY_CONFIG.json` 已固定：只读旧 VAE/EMA state_dict、H4 schedule 与 FREE/AMD 两个保存归档。每个归档保留原16行 batch，以避免改变 batch 形状而引入数值变化；这不是新生成32条序列。

调用预算：2次DDPM conditional forward、2次scalar decoder forward、0次encoder、0训练、0新随机序列噪声、0完整逆链。每个terminal只重算归档最后t=1的一步：

1. 用旧checkpoint和保存的 last_input 重算 prediction，对照保存 prediction。
2. 在同一坐标上执行旧/H5原始源码抽取的 posterior target packing，检查通道、维度与值。
3. 直接执行旧/H5源码 AST 中的 reverse 更新赋值，与保存 final parameters 逐位比较。
4. 使用保存 epsilon 计算 exp(logσ)、z，对照保存 scale/latent。
5. 使用H5 scalar decode入口和旧VAE，比较保存 raw及raw tokens，不导出新序列字符串。

严格要求所有比较按位相同（含符号位）、finite、权重不变、无梯度、模型构造之后RNG不变。120秒上限；不放宽容差，不重试。运行环境锁H4原CPU单线程Python3.12.13/NumPy1.26.4/Torch2.2.2。

检查通过也只能关闭“已保存最后一步与decoder公共路径不一致”这一局部假说；不证明所有998步、H5自身新权重、训练充分性、分布质量或生物活性。若不通过，保留差异位置，先定位实现，不启动新长训。

本次只做了AST与来源/配置静态准备，没有执行模型或打开train/dev/test CSV。
