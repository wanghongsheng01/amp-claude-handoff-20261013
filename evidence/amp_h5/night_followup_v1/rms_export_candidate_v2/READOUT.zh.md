# R/M/S 导出候选 v2（未执行）

范围固定为同一 CE checkpoint 的 train279 + dev41：M320 与8个固定种子 S2560，共2880请求；R320独立绑定。train后验缓存复用，新dev后验仅编码32+9两批，所有M重新解码并与历史序列及48 tokens逐位相同。每模式/seed为train9批、dev2批，共99 decoder批；无训练/DDPM/priorflow。

M=decoder(mu)。S来自learned approximate q(z|R,g,h)，经同一decoder/J；无posterior flow或teacher forcing。原70.90%为dev posterior-mean非PAD残基micro准确率，不能改叫抽样率。g/h来自已知R进入encoder；重建不是terminal-only新肽生成。此前dev latent确实缺失，本次若成功新cache只属于runs/ce_rms_reconstruction_v2，不回填旧报告。

1422..1429各自独立CPU Generator，一次抽完整320×16×48，先保存epsilon及初末RNG状态、原始输入seal，再构建模型。modelinit seed222与dev posterior_parameters内部seed0工作RNG均fork隔离，编码器内部丢弃的41条z会登记，非第9个S种子。decoder不消耗RNG。

唯一正式首batch M train0..31兼吞吐pilot，包含在2880中，不重做；仅按finite/历史parity及预锁时间投影决定能否继续。inner120从最早Python计时到manifest完成后检查；outer180涵盖完整进程。最后状态、独立closing ledger和supervisor exit需联合判定；异常不能将已有模型调用虚报为零。全失败/未执行槽保留，禁止retry、选seed、补采或按质量续跑。

v1未执行，因旧bootstrap遍历91项且包含4个legacy dataset CSV而保留停用。v2只换runtime校验：真实验82个py/yaml/yml/pyc及manifest本身，9个非code文件明确跳过并记录declared hash未验证。不同于sealed expanded test37，不打开这些旧CSV；不通过伪SHA冒称已经验过。encoder/decode/epsilon科学函数AST不变。

30项纯stdlib合成/AST检查通过，包括2880身份、empty/cap、生产token顺序/PAD、均值parity负例、失败张量可保留、postfinal异常不归零、码文件过滤及v1-v2函数一致。无torch导入、自然随机数、数组评分或新模型执行。独立预审和根代理exact plan/单次负载检查仍为执行前提。评分由独立只读脚本另以300秒预算完成，不因RDKit失败重跑模型。
