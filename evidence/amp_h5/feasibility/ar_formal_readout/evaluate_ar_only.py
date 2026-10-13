#!/usr/bin/env python3
"""Evaluate the completed H5 AR run only, with frozen H4/H5 metric functions.

No model/checkpoint import, training, generation, other-arm placeholders or
comparative gates. Only the two exact hash-bound train/development files open.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "implementation/evaluate_h5.py"
EXPECTED_EVALUATOR = "b894a7c56cd5c59c5fb77c046baa3c9873ae390d244082dd1f09998459e85417"
METRICS = ROOT / "implementation/frozen_metrics_v1.py"
EXPECTED_METRICS = "f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480"
DATA = Path("/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/expanded_cohort/release_v2/runner_train_val")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}


def dump(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    run, out = Path(args.run_root).resolve(), Path(args.output).resolve()
    if digest(EVALUATOR) != EXPECTED_EVALUATOR or digest(METRICS) != EXPECTED_METRICS:
        raise ValueError("frozen evaluator dependency drift")
    if (out / "RUN_FACTS.json").exists():
        raise ValueError("AR readout already exists; no overwrite/retry")
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_h5_evaluator", EVALUATOR)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    filenames = ["config.json", "status.json", "manifest.json", "generated.csv", "train_nll.json", "dev_nll.json",
                 "training/RUN.json", "training/MONITORS.json", "training/steps.jsonl", "generation/GENERATION.json"]
    source_paths = [run / name for name in filenames] + [EVALUATOR, METRICS, Path(__file__), DATA / "train.csv", DATA / "val.csv"]
    before = [record(p) for p in source_paths]
    read = lambda name: json.loads((run / name).read_text())
    config, status = read("config.json"), read("status.json")
    training, generation = read("training/RUN.json"), read("generation/GENERATION.json")
    manifest = read("manifest.json")
    for name in filenames:
        if name == "manifest.json":
            continue
        entries = [e for e in manifest["files"] if e["path"] == name]
        if len(entries) != 1 or digest(run / name) != entries[0]["sha256"]:
            raise ValueError("consumed completed-run artifact manifest mismatch: " + name)
    if (status.get("state") != "completed" or training.get("status") != "COMPLETED_FIXED_STEPS"
            or training.get("completed_steps") != 1000 or config.get("updates") != 1000
            or generation.get("requests") != 32 or generation.get("terminal_counts") != {"FREE": 16, "AMD": 16}
            or generation.get("seeds") != {"FREE": 422, "AMD": 423} or status.get("test_read") is not False):
        raise ValueError("completed AR fixed endpoint/request contract mismatch")
    if (Path(config["train"]).resolve() != (DATA / "train.csv").resolve()
            or Path(config["dev"]).resolve() != (DATA / "val.csv").resolve()
            or config["train_sha256"] != h.TRAIN_SHA or config["dev_sha256"] != h.DEV_SHA):
        raise ValueError("only exact approved train279/development41 references allowed")
    train = h.dataset(DATA / "train.csv", "train", h.TRAIN_SHA, 279)
    development = h.dataset(DATA / "val.csv", "development", h.DEV_SHA, 41)
    h.split_isolation(train, development)
    groups = h.sample_groups([run / "generated.csv"], ["terminal_LSTM"], 16,
                             {"terminal_LSTM": {"FREE": 422, "AMD": 423}})
    metrics, audit = h.evaluate_generation(groups, train, development)
    metrics["scope"] = "AR-only descriptive readout; no comparison or comparative gate was evaluated"
    nll = h.ar_nll_diagnostics([run / "train_nll.json", run / "dev_nll.json"])
    monitors = read("training/MONITORS.json")
    if [m["step"] for m in monitors] != list(range(100, 1001, 100)) or any(m.get("used_for_selection") is not False for m in monitors):
        raise ValueError("fixed monitor endpoint/selection contract mismatch")
    monitor_rows = [{"step": m["step"], "train_nll_nats": m["train"]["mean_nll"],
                     "development_nll_nats": m["development"]["mean_nll"],
                     "train_teacher_forced_accuracy": m["train"]["teacher_forced_token_accuracy"],
                     "development_teacher_forced_accuracy": m["development"]["teacher_forced_token_accuracy"],
                     "used_for_selection": False} for m in monitors]
    facts = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "completed AR-only descriptive feasibility",
             "training_updates": 1000, "training_seed": 222, "parameter_count": training["parameter_count"],
             "attempts": 32, "attempts_per_terminal": 16, "generation_seeds": {"FREE": 422, "AMD": 423},
             "architecture": training["architecture"], "optimizer": training["optimizer"],
             "generation_information": "requested terminal and model-generated prefix/randomness only",
             "no_rejection_or_resampling": generation["no_rejection_or_resampling"],
             "costs": {"recorded_training_pipeline_seconds": training["total_pipeline_seconds_before_manifest"],
                       "recorded_training_update_seconds": training["train_update_seconds"],
                       "recorded_monitor_seconds": training["monitoring_seconds"],
                       "recorded_generation_seconds": generation["wall_seconds_before_manifest"],
                       "recorded_method_process_seconds": status["process_seconds"],
                       "evaluation_seconds_before_output": time.perf_counter() - start},
             "training_and_generation_status": {"state": status["state"], "training_state": training["status"]},
             "comparisons_performed": 0, "other_method_observations_created": 0,
             "L_and_G_comparison": "NOT_EVALUATED_NO_MATCHED_FORMAL_VAE_ARMS",
             "AR_ADEQUACY": "NOT_ESTABLISHED", "PROMOTE": "HOLD", "independent_execution_audit": "separate; not claimed here",
             "operations": {"train_rows_read": 279, "development_rows_read": 41, "test_reads": 0,
                            "model_calls": 0, "training_updates": 0, "generation_calls": 0, "checkpoint_loads": 0},
             "limitations": ["single fitted seed and 16 requests per terminal; no significance or independent confirmation",
                             "reference development set has been reused; no test37 access",
                             "MMD421/JS/length/copy/nearest metrics are sequence-distribution descriptions, not antimicrobial activity or safety",
                             "teacher-forced NLL/accuracy are AR likelihood diagnostics, not generation reconstruction accuracy or VAE objective comparisons",
                             "all fixed monitor points retained; no checkpoint selection, continuation or new adequacy threshold",
                             "VAE formal arms are absent; no comparison or superiority claim", "BACKUP_BLOCKED"]}
    dump(out / "GENERATION_METRICS.json", metrics)
    dump(out / "SAMPLE_AUDIT.json", audit)
    dump(out / "TEACHER_FORCED_NLL.json", nll)
    dump(out / "MONITOR_READOUT.json", monitor_rows)
    dump(out / "RUN_FACTS.json", facts)
    fields = ["terminal", "requests", "valid1", "valid5", "unique_valid1", "same_terminal_copy", "any_terminal_copy",
              "mean_length_all", "train_mmd2", "development_mmd2", "train_aa_js", "development_aa_js",
              "train_dipeptide_js", "development_dipeptide_js", "train_length_ks", "development_length_ks"]
    flat = []
    for g in metrics["groups"]:
        row = {"terminal": g["terminal"], "requests": g["requests"], "valid1": g["valid_1_48_count"],
               "valid5": g["valid_5_48_count"], "unique_valid1": g["valid_1_48"]["unique_count"],
               "same_terminal_copy": g["exact_copy_train_count_all_requests"],
               "any_terminal_copy": g["exact_sequence_copy_any_train_terminal_count"],
               "mean_length_all": g["all_output_length"]["mean"]}
        for ref in ("train", "development"):
            c = g["valid_1_48_vs_" + ref]
            row.update({ref + "_mmd2": c["mmd2_linear_biased_fixed_features"], ref + "_aa_js": c["aa_jensen_shannon_bits"],
                        ref + "_dipeptide_js": c["dipeptide_jensen_shannon_bits"], ref + "_length_ks": c["length_ks_statistic_no_pvalue"]})
        flat.append(row)
    with (out / "AR_METRICS.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)
    fmt = lambda x: "undefined" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x))
    lines = ["# AMP H5：terminal LSTM 单臂描述性读出", "",
       "已记录 fixed final 1000 更新、32 个生成请求（FREE/AMD 各16），全部原请求保留。该页只评价已完成的 AR；没有创建另外三臂观测，也没有方法比较。",
       "**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD；L/G = NOT_EVALUATED_NO_MATCHED_FORMAL_VAE_ARMS。**", "",
       "| terminal | 请求 | 合法1–48 | 合法5–48 | unique合法1–48 | 同端基train-copy | 任意train-copy | 全请求平均长度 | train MMD² | dev MMD² |",
       "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in flat:
        lines.append("| " + " | ".join(fmt(row[k]) for k in fields[:10]) + " |")
    lines += ["", "MMD²使用冻结421特征，分布指标仅以合法1–48集合为分母；有效数与全请求分母一并报告。", "",
       "| terminal | train AA-JS | dev AA-JS | train二肽JS | dev二肽JS | train长度KS | dev长度KS |",
       "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in flat:
        lines.append("| " + " | ".join(fmt(row[k]) for k in ["terminal"] + fields[10:]) + " |")
    lines += ["", "全部AA/二肽L1、entropy、长度分布、同端基和任意train最近邻、逐样本合法性均保留于 GENERATION_METRICS.json / SAMPLE_AUDIT.json。无exact-copy只说明这32个输出中未观测到该事件，不证明不存在记忆。", "",
       "| teacher-forced数据 | token分母 | NLL/token (nats) | teacher-forced token accuracy |",
       "| --- | ---: | ---: | ---: |"]
    for value in nll:
        lines.append("| " + " | ".join(fmt(value[k]) for k in ("role", "token_count", "nll_per_target_token_nats", "teacher_forced_token_accuracy")) + " |")
    lines += ["", "固定每100更新的全部十个监测点如下。仅描述已记录曲线，没有选择最优checkpoint或新增充分性阈值。", "",
       "| step | train NLL/token | dev NLL/token | train teacher-forced accuracy | dev teacher-forced accuracy |",
       "| ---: | ---: | ---: | ---: | ---: |"]
    for row in monitor_rows:
        lines.append("| " + " | ".join(fmt(row[k]) for k in ("step", "train_nll_nats", "development_nll_nats", "train_teacher_forced_accuracy", "development_teacher_forced_accuracy")) + " |")
    c = facts["costs"]
    lines += ["", f"记录成本：训练pipeline {c['recorded_training_pipeline_seconds']:.6f}s；生成 {c['recorded_generation_seconds']:.6f}s；方法进程 {c['recorded_method_process_seconds']:.6f}s。它们范围不同，不能相加声称独立外层墙钟或equal compute。",
       "", "独立执行审计由另一代理进行，本页不代表其结论。评价复用冻结函数，仅打开已绑定train279/dev41；没有test37、模型调用、重采或原输出修改。",
       "", "单seed、复用dev41和小样本不能证明AR已充分优化、强基线优势、显著性、抗菌活性、安全性或可投稿；旧湿实验不验证此新模型。BACKUP_BLOCKED：本地哈希不是远端备份。", "",
       "完整指标：[GENERATION_METRICS.json](GENERATION_METRICS.json)；输入绑定：[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)；输出清单：[MANIFEST.json](MANIFEST.json)。"]
    with (out / "AUTHOR_READOUT.zh.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    preservation = [{**entry, "after_sha256": digest(entry["path"]), "unchanged": digest(entry["path"]) == entry["sha256"]} for entry in before]
    if not all(item["unchanged"] for item in preservation):
        raise ValueError("read-only input preservation failed")
    dump(out / "SOURCE_BINDINGS.json", {"inputs": before, "preservation": preservation,
         "run_manifest_consumed_files_verified": True, "checkpoint_files_opened": False,
         "source_preservation_pass": True, "evaluator_functions": ["dataset", "split_isolation", "sample_groups", "evaluate_generation", "ar_nll_diagnostics"]})
    dump(out / "MANIFEST.json", {"files": [record(p) for p in sorted(out.iterdir()) if p.is_file()],
         "scope": "AR-only descriptive evaluation; no method comparison", "PROMOTE": "HOLD"})
    print(json.dumps({"output": str(out), "groups": len(metrics["groups"]), "attempts": len(audit),
                      "summary": flat, "source_preservation_pass": True, "PROMOTE": "HOLD"}))


if __name__ == "__main__":
    main()
