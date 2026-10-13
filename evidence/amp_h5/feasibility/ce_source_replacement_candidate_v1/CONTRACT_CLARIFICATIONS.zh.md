# 新 B/C/D 合同澄清（无自然执行）

最终合同固定 B32+C32+D32 全部开启，共96新输入，复用DDPM32单列。历史DRAFT的optional/maximum/if-enabled只是此前设计状态，不适用于本合同。最终方法ID为CE_DDPM_REUSED、EMPIRICAL_POSTERIOR_B、COORDINATE_PERMUTATION_C、REFERENCE_G_PRIOR_D；原DRAFT中文与名称映射保留，不改旧文件。

D使用原CE learned conditional prior和明确训练reference-g，仍非terminal-only候选。相同B索引不意味着B和D条件信息相同，B posterior还承载训练x/h。C只排列B已采样decoder-z，不能指定单一训练真值或g。所有输入先封存，D的z在inverse后封存；2次inverse、2次forward roundtrip、6次decoder，零encoder/DDPM/训练。固定容差1e-5，错误g可能同样roundtrip，所以独立g来源与padding字节核验保留。

准入前已有协议96槽声明；准入失败由其加监督stderr保存requested96/notexecuted96/actual-model0，可能没有run CSV，该情形不可进行普通质量评价。准入后任何D失败保留全分母并终止，绝不退回B/C-only成功。预分配empty/pending length=0，冻结reader已用纯合成128行（含failed、pending、empty）检验。

内层120秒从首次import time之后的最早Python时钟起计，包含其后导入、argparse、准入、计算、通常导出及finally核验；before-final-IO必须<=120。额外严格检查manifest写完后若超过120，status/summary改为failed、manifest重建、退出2。最初解释器/time导入以及最后closing ledger/stdout由outer180整体约束。CLOSING_LEDGER单独引用manifest SHA且自身不进入manifest，独审必须单独绑定它，并联合status、closing、supervisor exit/总wall判断，不能只取某个中途completed值。

四臂科学终审已精确绑定，仅表示原四臂科学封口；它显式不授权新scope。当前候选含26项toy/AST检查PASS，没有自然latent读取、自然RNG重建、模型import/forward或质量计算。仍须根代理负载检查、正式plan和独审精确receipt后才能单次执行；旧A/B64仍不执行。
