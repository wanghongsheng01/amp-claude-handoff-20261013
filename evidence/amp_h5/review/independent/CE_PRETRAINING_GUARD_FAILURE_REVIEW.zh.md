# 单 CE 预训练身份检查失败独审

已确认是 **descriptor 全局编号与 split 局部位置混用**。train 的 idx 是0–278；dev 的 idx 是279–319。旧 guard 要求 `r.idx == i`，因此 dev 第0行因279≠0失败。dev 的41行 sequence、class、terminal 按现有顺序全部一一对应，无重复键、无行重排问题；不需要改数据或排序。

异常发生在 train_vae 之前。目录仅有 config、runtime、status、manifest 和32个全为not_executed的生成slot，没有权重、训练日志或scaler产物。结合源码调用顺序与traceback，确认0次VAE更新、0次DDPM更新、0次生成。原错误与原回执必须保留。

原status中的development_read=false不表示未读dev：两文件已经打开、哈希和解析，只有验证成功后才会改此标记。新版本应区分read_attempted与validated。此前静态/合成预审未覆盖真实dev的全局编号约定，是审查覆盖缺口，不能归因于用户数据错乱。

可接受的修复仅限新版本显式绑定train_idx_start=0、dev_idx_start=279，并继续校验`idx=start+i`及每行内容/端基；不移除身份检查，不改源CSV，不重排行，不改重建导出的split局部row_id。三个VAE配置统一绑定新实现。模型、目标、随机调用、种子、优化器、预算不变。先做无模型的真实绑定loader检查及合成错ID/错序/错内容拒绝，再以新输出目录、新独审回执启动一次fresh单CE。必须披露额外一次预训练契约修复尝试。

本轮独审只读四个已绑定train/dev CSV及现有产物，未读test、未加载或调用模型。MSE/scalar及完整三臂仍未授权。
