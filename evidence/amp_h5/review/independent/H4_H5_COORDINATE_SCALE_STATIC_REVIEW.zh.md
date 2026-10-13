# H4 → H5 后验坐标与尺度独立静态复核

结论：所审代码中没有发现 `log σ` / `log variance`、轴顺序、后验标准化逆变换或 DDPM 预测目标的迁移错误。**三个 VAE 正式长跑保持 HOLD**。这是源码一致性结论，不能把六个烟测生成失败解释为“只是欠训练”，也不能保证 1000 步会恢复。

| 核对项 | 证据与判断 |
|---|---|
| 后验参数 | 原 encoder 为 `z = μ + ε exp(logs)`，Normal 的 scale 也是 `exp(logs)`；H4/H5 相同。`logs` 是 log 标准差，不该任意加 0.5。 |
| 轴与通道 | 原 FVAE wrapper 先把 B×16×48 转成 B×48×16，旧 encode 再转回；H5 直接 stack encoder 的 B×16×48。最终均 B×2×16×48，0=μ，1=log σ。 |
| 特征尺度 | 两边只对实际 descriptor 做 train-only population zscore，常数列 scale=1；PAD 特征后补 0；global 取绝对值。DDPM 的 μ/log σ 从未另做标准化，所以不存在代码中遗漏的 latent inverse scaling。 |
| 训练目标 | 两边对原始 μ/log σ 加噪，并预测 ε，MSE 目标一致。 |
| 反向采样 | 均线性 beta、999 个 schedule 元素、998→1、纯端基条件预测、beta 方差噪声、末步零噪声。H5 两条赋值与旧单表达式运算顺序对应。 |
| Decoder 坐标 | posterior-parameter DDPM 输出的是 q 后验参数，采出的 z 直接入 decoder。inverse flow 是另一种生成方法的路径，不应混用。 |
| EMA | 两边复用原 helper；2000 步前是 current model 的拷贝。 |
| 已声明差别 | CE/MSE 共用 KL density/(B×48)，scalar 保留 /16。只有 CE↔MSE 支持单组件归因，scalar 是整条路线对照。 |

原 1000/1000 运行存档的 train log σ 范围为 [-0.263324, 0.253727]；旧 FREE 生成为 [-1.446530, 1.149901]，AMD 为 [-2.122235, 1.913505]。新烟测已保存的数值显示 log σ 可达数百，超过 float32 exp 上界约 88.72。这确认数值失效位置，**不确认原因**：模型、训练轮数、随机种子与轨迹均不同。

下一步仅做哈希绑定旧 checkpoint 与旧存档张量的一致性检查，保留 pred、reverse state、sigma、z、raw decoder 数值。若只复算存档末步，它只能排除这一局部算子/坐标差异，不能声称重放了完整 998 步，也不能证明全新 CE/MSE 模型可稳定生成。任何重参数化、标准化、schedule 调整或 clip 都应建新版本、对相关臂统一并重新预锁。

本复核没有加载模型、运行 forward、启动训练/新生成或读取 test。旧原始 posterior x 已在烟测保存；先前“旧代码未保存 x”的说法已撤回，正式副本增加的是失败阶段摘要。

机器可读证据、源码 SHA256 和行号见 `H4_H5_COORDINATE_SCALE_STATIC_REVIEW.json`。AR-only 执行单独依据 `AR_ONLY_PREFLIGHT_RECEIPT.json`，不放行完整 FORMAL_PLAN。
