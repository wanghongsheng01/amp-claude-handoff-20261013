# 编译与复现入口（不是新运行授权）

## 可以独立阅读/检查的内容

README→实验账本→具体结果JSON/CSV→独立审查。MANIFEST SHA256只证明包内字节一致，不证明科学结论成立。先前独立审计的哈希绑定保留；新的打包审计不升级自然实验的证据等级。

## Overleaf / LaTeX

当前候选根目录 `manuscripts/review_candidate/source/`；Overleaf选择 **pdfLaTeX + main.tex**，参考文献BibTeX/plainnat。`supplement.tex`单独编译；`cover_letter.tex`仍是未发送草稿；旧`decision.tex`须结合`manuscripts/latest_policy/`勘误阅读。不要以编译成功作为投稿批准。

本地只做文件编译：TeX Live的latexmk/pdflatex、bibtex和标准宏包；推荐 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`，关闭shell-escape。本次在独立拷贝目录进行编译，结果见`verification/LATEX_COMPILATION.json`。历史v6r3的入口是`main.tex`，其依赖IEEEtran.cls自带许可头；不重写原稿。

## 可选的离线数据复现（此交接不自动运行）

`cd data/fixed200` 后 `python3 scripts/clean_data.py --output rebuilt`。只需Python标准库，重新解析字符串、构造家族/来源组和标签资格；化学结构状态是显式冻结上游输入，不暗中重跑分子分析。既有CLI复现逐字节一致的证据随包保存。不会调用网络或模型。

## 模型实现与历史环境

作者实现位于`evidence/amp_h5/implementation/`，以及AR/direct-z/native/censored/EB目录的runner/model/score文件。`environment/requirements.lock.txt`是实际原环境锁文件，`ENVIRONMENT_CHECK.json`为历史机器证据。主要自然执行使用Python3.12、PyTorch CPU、NumPy/SciPy/Bio/RDKit（依任务）；不将macOS环境锁误称跨平台必然可复现。

各协议记录执行命令、真实预算、配置/源数据哈希。入口可能包含原Mac绝对路径、旧源模块和一次性STARTED保护；尚未跨机器移植验证。不能直接替换路径或抹去hash guard再声称重现。先核NOT_INCLUDED并补足合法依赖，另行冻结移植计划，独立预审后才可能运行。不要读取未受信pickle；direct-z原weights_only安全加载失败仍未闭合。

本轮只读/复制/编译，无训练、无自然模型采样、无test37、无新依赖安装。仓库无Actions、自动训练hook或子模块。
