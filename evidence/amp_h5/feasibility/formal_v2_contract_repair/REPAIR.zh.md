# formal_v2：训练前输入合同修复

formal_v1 在开发集输入校验阶段失败，未创建模型、保存权重或执行更新。旧输出 `runs/formal_v1/categorical_ce` 与 `runs/single_ce_execution_v1` 保持原样；本次只准备独立版本，不自动重试。

根因是旧读取器把每个文件的 `idx` 都要求为从零开始的行位置。已绑定 train 源编号为 0–278，dev 源编号为 279–319。作者用标准库 CSV 与哈希重新核对：train 的 279 行和 dev 的 41 行在 descriptor/cohort 两文件中均逐位置序列、类别、端基一致，键唯一且多重集合相等。映射就是恒等行位置，无需重排、重编号或修改数据。详见 `BINDING_EVIDENCE.json`。

新文件 `implementation/vae_pipeline_formal_v2.py` 仅改变三个函数：

- `read_records` 增加显式 `idx_start`，要求源编号等于起点加行位置，原文件哈希、内容、类别、端基和计数校验保留。
- `validate_config` 强制 train 起点为整数 0、dev 起点为整数 279。
- `run` 传入已绑定起点；在读取前落盘 attempted 标记，成功后落盘 validated 标记。兼容字段 `development_read` 从读取尝试开始就置真。attempted 表示开始读取流程，不表示校验已经通过。

旧失败输出的 `development_read=false` 是原状态字段赋值时机的记录，不能解释为旧运行未打开开发集；该原始记录不回写。

三臂新配置位于 `protocol/formal_configs_v2/`，仅新增相同起点绑定并更新实现哈希。种子、预算、优化器、损失、模型、更新数、数据路径、数据哈希及评价安排均未修改。开发集重建输出的 `row_id` 仍为 split 内 0–40；源 `idx` 279–319 不被改写。

标准库验证已完成：三个配置通过、train279/dev41 读取合同通过、错误起点和布尔起点被拒绝。源码 AST 对比表明其余全部函数及 FeatureScaler 类完全相同，包括模型、损失、训练、采样、随机数与重建导出。见 `STATIC_PROOF.json` 与 `formal_v1_to_v2.patch`。

本次没有导入 torch/numpy，没有训练、模型前向、生成、质量评价或 test 读取。新 supervisor plan、独立复审和新输出目录由 root 管理；此文件不授权后续模型运行。
