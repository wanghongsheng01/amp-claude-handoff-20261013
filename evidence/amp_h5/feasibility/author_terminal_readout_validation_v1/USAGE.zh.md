# 待终态执行的作者工程读出

脚本只读取预先绑定的运行目录与监督状态，不载入模型、不反序列化权重、不读取源 train/dev/test CSV、不计算新序列质量指标。checkpoint 文件只按字节哈希核验。

CE v2 结束后可执行以下命令。现在不执行；脚本也会拒绝仍运行中的监督状态。输出必须使用新目录，不能覆盖旧报告或写入原运行目录。

```sh
/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python -B \
  /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/implementation/author_terminal_engineering_readout_v1.py \
  --request /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/protocol/CE_V2_AUTHOR_TERMINAL_READOUT_REQUEST_v1.json \
  --output /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/feasibility/ce_v2_author_terminal_readout_v1
```

输出包括全部已保存 VAE/DDPM 曲线、原 dev monitor 曲线、train posterior 全体和同端基分组的尺度、生成 posterior/z/decoder 输出的有限性、失败张量、32 个预分配请求的状态与实际预算。NumPy 对已保存 logσ 求 exp 的结果明确标为派生诊断，不声称与 torch.exp 逐位一致。训练 q(z) 参照是解析混合边缘矩，未进行额外抽样。

报告不自行设置尺度阈值、不比较序列优势、不选 checkpoint、不改变后续配置，也不发放下一臂许可。MSE/scalar 草案在 `protocol/staged_remaining_vae_drafts_v1/`，刻意缺少监督器可执行所需的 `jobs`、`output_root`，且 `authorized=false`。须待 CE 独立工程后审后，root 另建正式单臂 plan 并由独审签发新回执。

`VALIDATION.json` 记录 8 项纯合成数组/临时文件检查，包含运行中拒绝、终态读取、非有限值计数、重复输出拒绝；零模型调用，没有在真实 CE 输出上执行脚本。
