# AMP H5 条件 β-warmup：仅候选准备

尚未执行模型、训练、自然数据评分或读取 test。实际 ROOT_LAUNCH_AUTHORIZATION.json 和 run_v1 均不存在。仅 root 在完整 R/M/S 入门门槛、独立预审与新鲜负载检查全部通过后才可单次启动；ARGV.json 是冻结命令模板，不是授权。

唯一目标改动是冻结 objective 的 CE + beta × 原归一化 density-KL；beta=min(1,(step−1)/200)，共 1000 步，step1=0、step200=.995、step201=1。保留原 train_vae、Adam、梯度裁剪、LR、seed222、batch32、数据/描述符与 J，以及每100步一次的原固定 dev mean monitor。后者总计20个 encoder batch、20个 decoder batch；不得误记为零 forward。没有 DDPM、生成或额外 posttraining q/RMS；新的权重不能复用旧 q，后续 RMS 另锁。

冻结代码的 init_runtime 会遍历整个旧 manifest，其中包含旧 test_dataset.csv 名称；未据文件名认定它等于封存 test37。本 wrapper 使用独立等价 VAE code-only bootstrap，真验 manifest 和82项 py/pyc/yaml/yml；9项非代码成员明确未读未复验，不用虚构 SHA 返回。指定 train279/dev41 在未来授权运行时仍通过冻结 read_records 真验。

预期配对：初始 tensor hash/scaler 严格相等；每一步 batch indices、行数、preforward RNG hash、LR 对历史 CE trace 严格相等；首个未加权 CE/KL/density 精确相等。保留原 vae_train.jsonl，另存全部 β/加权及未加权损失、batch tensor hash 和完整 CPU RNG 状态。新增审计 I/O 不抽随机数，不加 forward，计入预算。

历史 trace 没有原始每步 epsilon 数组、完整 RNG 状态或 batch tensor hash。因此只能凭相同实现/环境/初始状态、既有 RNG hash 与调用顺序核配对，不能声称旧/新 epsilon 文件已逐元素验证。参数、z、梯度和最终权重随干预改变；这不是配对失败。纯静态准备也不能保证实际配对或吞吐，任何失败均 NOT_EVALUABLE，保留部分记录且不重试。

VAE1920秒、outer2100秒，固定1000更新；不得延长或选最佳 dev checkpoint。2100秒信号中止后可能仍有保存失败状态/部分权重的清理 I/O，需照实记账并由 root 外部 supervisor 控制；不将超时清理宣称为预算内。完成训练的状态仅是 completed_vae_only_not_scientific_success。科学资源规则以绑定的 root guards 为准：同 dev41/同8epsilon 的 S +5pp、M不降、M长度MAE不升（基准0）、S macro不降；后续测量须另锁。

标准库纯数学/AST/合成receipt检查 PASS 1045；额外8项历史 trace/文件状态 metadata 检查通过。未作真实模型 smoke。独立预审尚待完成。旧 CE 是已观察的历史对照；不称实际累计算力相等、独立seed复现、因果证明、创新、显著性、生成优势或抗菌功效。
