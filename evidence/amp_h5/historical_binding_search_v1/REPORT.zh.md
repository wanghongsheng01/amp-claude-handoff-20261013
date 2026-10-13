# AMP 27 条肽历史绑定有界检索

结论：本次范围内未恢复任何一条可完整核验的“历史生成运行 → 逐候选筛选 → 原始送样”链（0/27）。27/27 均找到序列或样本编号对应；供应商与 MIC 材料仍然支持候选合成和报告的测量结果。不能据此推断湿实验不存在，也不能将论文序列提及升级为具体模型运行归属。

## 范围与可复核性

- 共检索 229 个文本单元，1,818,747 字节；1822 条命中包含不同副本、工作表及重复记录，**不是独立证据数量**。单元为文本文件、DOCX XML 段落部件或 XLSX 工作表。
- 源码固定快照 `e7486fb714cbf5c663cc01052e42fb63d4ba3a40`，62 个文本单元，目标序列／ID 命中 0。
- 论文备份索引固定提交 `a658aeb03f17bd765e2696a0baa299ba864e54dc`：201 个索引文件，23 个本地可用，其中 9 个直接文件内容被解析/搜索，另搜索已下载 v1.1.0 ZIP 中符合规则的文本成员。201 不能声称全内容已检索。
- 两个精确授权根目录都无 `.git`。查到了 v1.0.0 / v1.1.0 文件及版本包，没有遍历提交历史；未联网获取远程分支或未下载内容。固定提交是快照标识，不是实验日期。
- 包括已下载 Feishu DOCX、两个 MIC 工作簿、108 份供应商报告的既有文本提取、既有 PDF 文本、既有候选截图转录。I-TASSER ZIP 仅 5 张 GIF，无可检索序列文本。
- 按协议限制：每文本单元≤2 MiB、总文本≤50 MiB、≤1000 单元、容器≤20 MiB；未新增 OCR、读取权重、读取 test37、扫描其他项目或对话目录。开发提示、AGENTS 和凭据路径排除。
- 在同一个冻结来源集合上修复了断词匹配和来源清单覆盖，再验证结果；没有扩展来源或发起新一轮外部检索。

## 逐环节解释

1. **生成归属**：未恢复可把这 27 条绑定到历史模型/checkpoint/seed/condition 的生成输出。源码默认路径、默认参数和论文方法叙述只能说明设计或声明。
2. **筛选**：既有截图转录有候选 ID/序列；pp3-1 的 F 数量视觉不确定性仍保留。未恢复逐候选阶段分数及完整生成池→筛选列表。派生转录中精确字符串命中不能抹去原像素的不确定性。
3. **合成**：27 条 COA 序列和产品编号、订单、批次，以及对应的质量报告材料存在。这是供应商交付记录；未恢复生成运行→原始送样申请的绑定。
4. **MIC**：两工作簿中的 27 个编号对应供应商编号。它们不是两个独立实验；样本批次→板孔、实验日期、独立生物学重复 N 仍未得到历史绑定。
5. **论文和结构流程**：pp2-8 `FMPVVLTKLFGVLHFRKIY` 在历史正文和准备流程出现；pp1-5 `LKFTFIKGKFGKSSYY` 在准备流程出现，amp9.10 DOCX 中 `LKFTFIKGK- FGKSSYY` 为显式标记的排版去断词命中，两个既有 PDF 文本中的跨行断词也单独标记。这些是候选描述，不是生成运行日志。未把其他近似字符或手动修改当作精确匹配。

## 日期与版本

`TIMESTAMP_FIELDS.json` 保存实际字段及语义，`CONTAINER_INVENTORIES.json` 保存 ZIP 原始时间元组（时区未知）。20240818 DOCX 的 created=`2024-08-12T11:25:00Z`、modified=`2024-08-18T09:11:00Z` 是可编辑文档元数据；Feishu 导出的 DOCX created 字段为 2026-10-11 的导出期时间。论文包有 `2025-11-08` 版本报告日期；COA 有供应商认证日期。以上都不能替代生成时间或 MIC 测定日期。未取得提交时间，未猜测。

## 27 条分类表

全部分类为“仅序列或编号匹配”；“在本次范围连序列/ID都未找到”=0，“完整链强关联”=0。计数包含重复副本，精确、断词规范化和 ID 类型在逐条命中中分别标注。

|样本|完整序列|序列命中|ID命中|分类|
|---|---|---:|---:|---|
|pp1-1|`LNLPKILFCSAPRTRLPGNWI`|4|63|仅匹配|
|pp1-2|`KAFFLLYFAVPFGPWS`|4|63|仅匹配|
|pp1-3|`CVVSLFKSKKGGLRKHGVPFKL`|4|63|仅匹配|
|pp1-4|`LLWIMLAYRLAFHRNWNAGGWKLFAVV`|4|63|仅匹配|
|pp1-5|`LKFTFIKGKFGKSSYY`|10|63|仅匹配|
|pp1-7|`AVCLILGLFNWGK`|4|63|仅匹配|
|pp1-8|`CGHLFPYWLYLGPKTPW`|4|63|仅匹配|
|pp2-1|`CPRTRVGSKT`|4|63|仅匹配|
|pp2-2|`LKLLWLIKCSG`|4|63|仅匹配|
|pp2-3|`AKLSFGKPFANKSKYW`|4|63|仅匹配|
|pp2-4|`CNNLKLHICVKKFARYYLYGPTPYY`|4|63|仅匹配|
|pp2-5|`RETGRGKRDKGNIGTALGCFRMI`|4|63|仅匹配|
|pp2-6|`VGPGKKFLPK`|4|63|仅匹配|
|pp2-7|`FTGGKGRDWGGHSSRKKKSLGFA`|4|63|仅匹配|
|pp2-8|`FMPVVLTKLFGVLHFRKIY`|10|63|仅匹配|
|pp3-1|`CHLKLAILFNFLKFLKQFLGNLTYV`|4|64|仅匹配|
|pp3-2|`HGKFFKYFNLQVCLGLARALKDGG`|4|63|仅匹配|
|pp3-3|`RPKQIYFFAHLRLIFGLKI`|4|63|仅匹配|
|pp3-4|`CLGPKLYNNWQGLGGL`|4|63|仅匹配|
|pp3-5|`NWQPRHFLADTFCPKLKGQILR`|4|63|仅匹配|
|pp3-6|`MPKGIFFALKLLWKNFEKAI`|4|63|仅匹配|
|pp3-7|`PKPFLWAKSDLFVKL`|4|63|仅匹配|
|pp3-9|`GFFPGFIVLGCKTGFTPTY`|4|63|仅匹配|
|pp3-11|`FKGVAKARHRLWFL`|4|63|仅匹配|
|pp3-12|`GFPGDVFKAKAAKA`|4|63|仅匹配|
|pp3-13|`KTFKKKLFLKLFYILHTPTY`|4|63|仅匹配|
|pp3-14|`CFANFQSKLHREARKLMGGAKKWPY`|4|63|仅匹配|

## 交付与限制

`HITS.json` 联结 `SEARCH_UNITS.json` 提供路径、成员、行/段/单元格定位、SHA-256、可用 Git blob/快照版本。`PARTIAL_EDGE_EVIDENCE_27.json` 按样本保存各边证据，`BACKUP_INDEX_COVERAGE.json` 列出未下载/未检索条目；`SOURCE_MANIFEST.json` 和源文件末次核验记录支持只读校验。

没有改动原稿或启动新训练。没有增加生物效力、方法优越性或投稿就绪声明。新本地产物远端备份受限制，状态 `BACKUP_BLOCKED`。
