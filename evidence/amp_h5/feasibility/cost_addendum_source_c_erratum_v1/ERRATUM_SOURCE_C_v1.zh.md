AMP H5 成本包 C 来源措辞勘误

原包 `cost_addendum_source_replacement_v2_v1` 中，“打乱已有DDPM潜变量坐标”应更正为：**在每个末端组内，对实际采样的 B 经验训练后验 decoder-z，逐坐标跨 16 个样本 slot 置换。** C 并未置换已有 DDPM-z。

依据：冻结实现 `ce_source_replacement_v2.py` 第 184–185 行先形成 B 的采样 z，再传入 `permute_columns`；第 141 行执行 `C_z[i,ch,pos] = B_z[permutation[ch,pos,i],ch,pos]`。冻结合同第 99、134、136 行给出同一来源及映射。

需要结合本勘误解读的位置：

- `READOUT.zh.md:29` 的上述中文短语。
- `COST_ADDENDUM.json:129` 的 `/new_model_diagnostic/method_boundary`，其中 `C coordinate-permuted existing DDPM z` 应改读为对实际采样 B 经验训练后验 decoder-z 的逐坐标、端内跨 slot 置换。
- 封存构建脚本 `build_addendum.py:180` 和 `:257` 中对应的两处文本模板。脚本保留原样，不重跑。

这是来源表述勘误。原包及其全部 7 个清单成员均未修改，模型执行、成本、请求数、质量与门控结论不变；本次未运行模型、重算成本或质量、读取 cohort/test。

绑定原 READOUT SHA256：`85c5bceae49bc503c36c16d97ca5de9464c691e431b15d4c5a2e9c9ecf2fa967`。

绑定原 MANIFEST SHA256：`771045ffd8deb4490eeaafee4fa72b562d76d64cac608799c6e94da58b3c4f3b`。

逐字段替换文本及源码哈希见同目录 `ERRATUM_SOURCE_C_v1.json`。
