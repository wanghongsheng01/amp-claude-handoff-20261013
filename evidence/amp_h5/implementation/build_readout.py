#!/usr/bin/env python3
"""Summarize existing AMP H5 artifacts only: no fitting, generation or metrics.

Never imports torch/evaluate_h5 and never opens paths stored in input configs.
The recorded engineering checks are a completeness screen, not an independent
source/RNG/condition audit. Scientific promotion is always HOLD in this pilot.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

METHODS = {
    "categorical_ce": "CE21_DDPM", "raw_onehot_mse": "MSE21_DDPM",
    "scalar_legacy": "scalar_DDPM", "terminal_lstm": "terminal_LSTM",
}
TERMINALS = ("FREE", "AMD")
SUCCESS = {"ok", "empty_sequence", "no_eos_at_limit"}


def record(path):
    payload = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value):
    if value is None:
        return "NOT_EVALUABLE"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def dig(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def reject_nonfinite_json(value):
    raise ValueError("nonfinite JSON constant: " + value)


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records, self.problems = {}, []

    def read(self, relative, kind="json", required=True):
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("artifact path escaped run root")
        # Only hardcoded artifact paths are passed here. Config paths are data.
        if not path.is_file():
            if required:
                self.problems.append({"path": relative, "problem": "MISSING"})
            return None
        try:
            payload = path.read_bytes()
            self.records[relative] = {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            text = payload.decode("utf-8-sig")
            if kind == "json":
                return json.loads(text, parse_constant=reject_nonfinite_json)
            if kind == "jsonl":
                return [json.loads(line, parse_constant=reject_nonfinite_json) for line in text.splitlines() if line.strip()]
            if kind == "csv":
                return list(csv.DictReader(io.StringIO(text, newline="")))
            if kind == "binding_only":
                return self.records[relative]
            raise ValueError("unsupported artifact kind")
        except (ValueError, OSError, UnicodeError) as exc:
            self.problems.append({"path": relative, "problem": "UNREADABLE", "error": str(exc)})
            return None


def log_endpoints(rows, fields):
    if rows is None:
        return {"status": "NOT_EVALUABLE", "records": None}
    if not rows:
        return {"status": "EMPTY_LOG", "records": 0}
    result = {"status": "DESCRIPTIVE_ONLY", "records": len(rows),
              "first": {k: rows[0].get(k) for k in ("step", *fields)},
              "last": {k: rows[-1].get(k) for k in ("step", *fields)},
              "interpretation": "fixed recorded checkpoints only; no new adequacy threshold or endpoint selection"}
    return result


def sample_inventory(rows, method, requests):
    if rows is None:
        return {"state": "NOT_EVALUABLE", "rows": None, "all_slots_present": False, "all_executed": False}
    expected = {f"{method}_{terminal}_{i:03d}" for terminal in TERMINALS for i in range(requests)}
    ids = [row.get("sample_id") for row in rows]
    correct = (len(rows) == len(expected) and len(set(ids)) == len(ids) and set(ids) == expected
               and all(r.get("method") == method and r.get("terminal") in TERMINALS for r in rows))
    statuses = dict(sorted(Counter(r.get("status", "MISSING_STATUS") for r in rows).items()))
    complete = all(r.get("status") in SUCCESS for r in rows)
    return {"state": "RECORDED_COMPLETE" if correct and complete else "NOT_EVALUABLE",
            "rows": len(rows), "expected_rows": len(expected), "all_slots_present": correct,
            "all_executed": complete, "status_counts": statuses,
            "note": "slot/status bookkeeping only; no sequence-quality recomputation"}


def method_facts(inputs, directory, method):
    get = lambda file, **kwargs: inputs.read(directory + "/" + file, **kwargs)
    status = get("status.json")
    config = get("config.json")
    manifest = get("manifest.json")
    status, config = status or {}, config or {}
    requests = config.get("samples_per_terminal", 16)
    if not isinstance(requests, int) or requests <= 0:
        requests = 16
    samples = sample_inventory(get("generated.csv", kind="csv"), method, requests)
    checks = {"recorded_state_completed": status.get("state") == "completed",
              "formal_purpose": config.get("purpose") == "formal_pilot",
              "full_formal_slot_budget": requests == 16,
              "all_slots_present": samples["all_slots_present"],
              "all_generation_executed": samples["all_executed"],
              "manifest_present": isinstance(manifest, dict),
              "test_read_recorded_false": status.get("test_read") is False}
    costs = {"process_seconds": status.get("process_seconds"),
             "training_cap_seconds": config.get("training_seconds_cap"),
             "generation_cap_seconds": config.get("generation_seconds_cap")}
    updates, curves, nll = {}, {}, None
    if method != "terminal_LSTM":
        vae = get("vae_training_summary.json") or {}
        ddpm = get("ddpm_training_summary.json") or {}
        generation = get("generation_summary.json") or {}
        initialization = get("vae_initialization.json") or {}
        get("train_reconstruction.csv", kind="binding_only")
        get("dev_reconstruction.csv", kind="binding_only")
        for stage, summary in (("vae", vae), ("ddpm", ddpm)):
            requested, completed = config.get(stage + "_updates"), summary.get("updates")
            updates[stage] = {"requested": requested, "completed": completed}
            checks[stage + "_fixed_endpoint"] = (isinstance(requested, int) and requested > 0 and completed == requested
                                                     and summary.get("fixed_endpoint_reached") is True)
            checks[stage + "_finite_logged_losses"] = True
        costs.update(training_seconds=status.get("training_seconds"), generation_seconds=generation.get("elapsed_seconds"),
                     vae_stage_seconds=vae.get("elapsed_seconds"), ddpm_stage_seconds=ddpm.get("elapsed_seconds"),
                     posterior_encoding_and_ddpm_setup_seconds=ddpm.get("posterior_encoding_and_setup_seconds"))
        train_log = get("vae_train.jsonl", kind="jsonl")
        ddpm_log = get("ddpm_train.jsonl", kind="jsonl")
        dev_log = get("dev_monitor.jsonl", kind="jsonl")
        checks["vae_finite_logged_losses"] = bool(train_log) and all(finite(r.get("loss")) for r in train_log)
        checks["ddpm_finite_logged_losses"] = bool(ddpm_log) and all(finite(r.get("mse")) for r in ddpm_log)
        checks["vae_log_updates_complete"] = bool(train_log) and [r.get("step") for r in train_log] == list(range(1, (config.get("vae_updates") or 0) + 1))
        checks["ddpm_log_updates_complete"] = bool(ddpm_log) and [r.get("step") for r in ddpm_log] == list(range(1, (config.get("ddpm_updates") or 0) + 1))
        curves = {"vae_training": log_endpoints(train_log, ("loss", "reconstruction", "kl")),
                  "ddpm_training": log_endpoints(ddpm_log, ("mse",)),
                  "development_monitor": log_endpoints(dev_log, ("nonpad_token_accuracy", "length_mae", "mean_reconstruction_objective"))}
        checks["training_cap_recorded_clear"] = status.get("training_cap_exceeded") is False
        binding = {"implementation_sha256": config.get("implementation_sha256"),
                   "initial_tensor_state_sha256": initialization.get("tensor_state_sha256"),
                   "vae_parameters": initialization.get("parameters"),
                   "decoder_head_parameters": initialization.get("decoder_head_parameters"),
                   "ddpm_parameters": ddpm.get("parameters")}
    else:
        training = get("training/RUN.json") or {}
        generation = get("generation/GENERATION.json") or {}
        train_log = get("training/steps.jsonl", kind="jsonl")
        monitors = get("training/MONITORS.json")
        requested, completed = config.get("updates"), training.get("completed_steps")
        updates["ar"] = {"requested": requested, "completed": completed}
        checks["ar_fixed_endpoint"] = (isinstance(requested, int) and requested > 0 and requested == completed
                                       and training.get("status") == "COMPLETED_FIXED_STEPS")
        checks["ar_finite_logged_losses"] = bool(train_log) and all(finite(r.get("loss")) for r in train_log)
        checks["ar_log_updates_complete"] = bool(train_log) and [r.get("step") for r in train_log] == list(range(1, (requested or 0) + 1))
        costs.update(training_seconds=training.get("total_pipeline_seconds_before_manifest"),
                     train_update_seconds=training.get("train_update_seconds"),
                     monitoring_seconds=training.get("monitoring_seconds"),
                     generation_seconds=generation.get("wall_seconds_before_manifest"),
                     train_cap_overrun_seconds=training.get("cap_overrun_seconds"),
                     generation_cap_overrun_seconds=generation.get("cap_overrun_seconds_before_manifest"))
        curves = {"ar_training": log_endpoints(train_log, ("loss", "observed_tokens")),
                  "ar_fixed_monitor": log_endpoints(monitors, ("train", "development"))}
        nll = {"train": get("train_nll.json"), "development": get("dev_nll.json"),
               "scope": "AR teacher-forced diagnostic only; not a VAE objective comparison"}
        binding = {"implementation_sha256": config.get("implementation_sha256"),
                   "entry_sha256": config.get("entry_sha256"),
                   "initial_tensor_state_sha256": training.get("initial_tensor_state_sha256"),
                   "final_tensor_state_sha256": training.get("final_tensor_state_sha256")}
    for name in ("training", "generation"):
        measured, cap = costs.get(name + "_seconds"), costs.get(name + "_cap_seconds")
        checks[name + "_within_recorded_cap"] = finite(measured) and finite(cap) and measured <= cap
    checks["no_missing_method_artifacts"] = not any(p["path"].startswith(directory + "/") for p in inputs.problems)
    passed = all(checks.values())
    return {"method": method, "directory": directory, "recorded_state": status.get("state", "MISSING"),
            "engineering_record_state": "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" if passed else "NOT_EVALUABLE",
            "engineering_record_checks": checks, "status_error": status.get("error"),
            "updates": updates, "costs": costs, "samples": samples, "bindings": binding,
            "config_sha256_reported_by_runner": status.get("config_sha256"),
            "curves": curves, "ar_nll": nll,
            "engineering_scope": "summarizes recorded status/steps/caps/slots/finite objectives; does not independently audit model math, all gradients, source conditioning or RNG"}


def read_evaluation(inputs):
    data = {name: inputs.read("evaluation/" + name + ".json") for name in
            ("generation_metrics", "reconstruction_metrics", "exploratory_gates", "run", "MANIFEST")}
    data["ar_teacher_forced_nll"] = inputs.read("evaluation/ar_teacher_forced_nll.json")
    generation = data.get("generation_metrics") or {}
    reconstruction = data.get("reconstruction_metrics") or {}
    groups = {(r.get("method"), r.get("terminal")): r for r in generation.get("groups", [])}
    rec = {(r.get("method"), r.get("split")): r for r in reconstruction.get("groups", [])}
    rows = []
    for method in METHODS.values():
        for terminal in TERMINALS:
            g = groups.get((method, terminal))
            rows.append({"method": method, "terminal": terminal, "metric_state": "AVAILABLE_DESCRIPTIVE" if g else "NOT_EVALUABLE",
                "requests": dig(g, "requests"), "valid1": dig(g, "valid_1_48_count"), "valid5": dig(g, "valid_5_48_count"),
                "unique_valid1": dig(g, "valid_1_48", "unique_count"),
                "same_terminal_copy_count": dig(g, "exact_copy_train_count_all_requests"),
                "any_terminal_copy_count": dig(g, "exact_sequence_copy_any_train_terminal_count"),
                "train_mmd2": dig(g, "valid_1_48_vs_train", "mmd2_linear_biased_fixed_features"),
                "development_mmd2": dig(g, "valid_1_48_vs_development", "mmd2_linear_biased_fixed_features"),
                "empty_count": dig(g, "empty_count"), "raw_full_metrics": g})
    rec_rows = []
    for method in list(METHODS.values())[:3]:
        g = rec.get((method, "development"))
        rec_rows.append({"method": method, "metric_state": "AVAILABLE_DESCRIPTIVE" if g else "NOT_EVALUABLE",
                        "rows": dig(g, "rows"), "micro": dig(g, "rendered_nonpad_micro"),
                        "macro": dig(g, "rendered_nonpad_macro"), "sequence_mean": dig(g, "per_sequence_mean_nonpad_accuracy"),
                        "length_mae": dig(g, "length_mae"), "exact_count": dig(g, "whole_rendered_exact_count"),
                        "raw_full_metrics": g})
    return {"recorded_evaluation_status": dig(data.get("run"), "status"),
            "run": data.get("run"), "generation_rows": rows, "development_reconstruction_rows": rec_rows,
            "raw_gates": data.get("exploratory_gates"), "ar_nll": data.get("ar_teacher_forced_nll")}


def gate_status(raw):
    if not isinstance(raw, dict):
        return "NOT_EVALUABLE"
    if raw.get("status") != "EVALUATED":
        return "NOT_EVALUABLE"
    if isinstance(raw.get("pass"), bool):
        return "PASS" if raw["pass"] else "FAIL"
    return "NOT_EVALUABLE"


def build(root):
    inputs = Inputs(root)
    methods = [method_facts(inputs, d, m) for d, m in METHODS.items()]
    evaluation = read_evaluation(inputs)
    engineering = all(m["engineering_record_state"] == "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" for m in methods)
    evaluation_complete = evaluation["recorded_evaluation_status"] == "COMPLETE" and all(
        r["metric_state"] != "NOT_EVALUABLE" for r in evaluation["generation_rows"] + evaluation["development_reconstruction_rows"]) and not any(
            p["path"].startswith("evaluation/") for p in inputs.problems)
    raw_gates = evaluation.get("raw_gates") or {}
    raw_l = gate_status(raw_gates.get("reconstruction_ce_vs_matched_mse"))
    raw_g = gate_status(raw_gates.get("generation_ce_vs_terminal_ar"))
    effective_l = raw_l if engineering and evaluation_complete else "NOT_EVALUABLE"
    effective_g = raw_g if engineering and evaluation_complete else "NOT_EVALUABLE"
    method_ok = {m["method"]: m["engineering_record_state"] == "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" for m in methods}
    for row in evaluation["generation_rows"] + evaluation["development_reconstruction_rows"]:
        row["interpretation_state"] = row["metric_state"] if method_ok.get(row["method"]) and evaluation_complete else "NOT_EVALUABLE"
    known_costs = [m["costs"]["process_seconds"] for m in methods if finite(m["costs"].get("process_seconds"))]
    facts = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "AMP H5 existing-artifact author summary only",
        "run_root": str(inputs.root), "source": record(__file__), "methods": methods, "evaluation": evaluation,
        "engineering_E": {"state": "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" if engineering and evaluation_complete else "NOT_EVALUABLE",
                          "recorded_methods_complete": engineering, "recorded_evaluation_complete": evaluation_complete,
                          "independent_audit_claim": False},
        "raw_L": raw_l, "raw_G": raw_g, "effective_L": effective_l, "effective_G": effective_g,
        "AR_ADEQUACY": "NOT_ESTABLISHED", "PROMOTE": "HOLD",
        "cost_summary": {"known_method_process_seconds_sum": sum(known_costs), "methods_with_process_timing": len(known_costs),
                         "all_four_method_timings_present": len(known_costs) == 4,
                         "evaluation_seconds": dig(evaluation.get("run"), "wall_seconds"),
                         "note": "sum of recorded method process timings; not independent outer wall time, not equal compute; smoke/review costs are excluded here and must be separately disclosed"},
        "missing_or_unreadable_artifacts": inputs.problems, "consumed_artifacts": list(inputs.records.values()),
        "hash_scope": "hashes bind consumed artifact bytes; preexisting manifests are recorded, not independently revalidated; weight files are not opened",
        "limits": ["single training seed and one fixed configuration/endpoint per method; reused dev41; no significance or independent confirmation",
                   "AR adequacy fixed NOT_ESTABLISHED; curves are descriptive and cannot add an adequacy threshold after results",
                   "proxy gates are not activity, safety, novelty, strongest-baseline superiority, submission-readiness or a wetlab result",
                   "same numeric sampling seeds across models do not make draws paired; actual training compute differs",
                   "CE/MSE holds nominal beta fixed, not effective regularization strength; scalar changes target/capacity/scale/readout together",
                   "H0-H4/manuscript unchanged by this summarizer; unknown wetlab N and conflicts not inferred; no formal submission",
                   "BACKUP_BLOCKED: local hashes are not remote backup"],
        "operations": {"model_calls": 0, "metric_recomputations": 0, "split_files_opened": 0,
                       "test_files_opened": 0, "checkpoint_files_loaded": 0}}
    return facts


def render(facts):
    ev = facts["evaluation"]
    lines = ["# AMP H5 作者结果读出", "",
        "本文件只汇总既有运行与评价记录；没有训练、生成、指标重算或 split 读取。",
        f"工程 E：**{facts['engineering_E']['state']}**；raw L：**{facts['raw_L']}**；raw G：**{facts['raw_G']}**。",
        f"依完整性前提解释的 L/G：**{facts['effective_L']} / {facts['effective_G']}**。",
        "**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD。** 单次开发 pilot 不支持强基线优势、活性、新颖性或可投稿结论。", "",
        "以下八组固定列出；NOT_EVALUABLE 表示缺失/不可解释，不以零填补。MMD² 的有效集合为合法 1–48 输出，应与全请求有效数同读。", "",
        "| 方法 | terminal | 请求 | 合法1–48 | 合法5–48 | unique合法1–48 | copy同端基/任意train | train MMD² | dev MMD² | 指标状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for r in ev["generation_rows"]:
        lines.append("| " + " | ".join(fmt(x) for x in (r["method"], r["terminal"], r["requests"], r["valid1"], r["valid5"],
            r["unique_valid1"], fmt(r["same_terminal_copy_count"])+" / "+fmt(r["any_terminal_copy_count"]), r["train_mmd2"], r["development_mmd2"], r["interpretation_state"])) + " |")
    lines += ["", "VAE 的下表仅为 dev posterior mean 的最终可见序列重建。macro 是 target 支持的 AA 类别 recall 均值，PAD 不入分母；sequence mean 另列。", "",
              "| 方法 | dev对象数 | 内容micro | AA类macro | sequence mean | 长度MAE | 整串exact | 指标状态 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in ev["development_reconstruction_rows"]:
        lines.append("| " + " | ".join(fmt(r[k]) for k in ("method", "rows", "micro", "macro", "sequence_mean", "length_mae", "exact_count", "interpretation_state")) + " |")
    lines += ["", "| 方法 | 运行状态 | 记录工程状态 | 完成/计划更新 | 训练秒 | 生成秒 | 进程秒 | 训练上限秒 |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for m in facts["methods"]:
        updates = "; ".join(f"{stage} {fmt(v['completed'])}/{fmt(v['requested'])}" for stage, v in m["updates"].items())
        c = m["costs"]
        lines.append("| " + " | ".join(fmt(x) for x in (m["method"], m["recorded_state"], m["engineering_record_state"], updates,
             c.get("training_seconds"), c.get("generation_seconds"), c.get("process_seconds"), c.get("training_cap_seconds"))) + " |")
        if m["status_error"]:
            lines.extend(["", f"{m['method']} 保留的执行错误：{fmt(m['status_error'])}"])
    lines += ["", "训练曲线仅转录固定首末记录，不据此新增充分性阈值或选择终点：", ""]
    for m in facts["methods"]:
        for name, c in m["curves"].items():
            first, last = c.get("first"), c.get("last")
            # Compact AR monitors to likelihood values, preserving the full source in JSON.
            if name == "ar_fixed_monitor" and first and last:
                first = {"step": first.get("step"), "train_nll": dig(first, "train", "mean_nll"), "dev_nll": dig(first, "development", "mean_nll")}
                last = {"step": last.get("step"), "train_nll": dig(last, "train", "mean_nll"), "dev_nll": dig(last, "development", "mean_nll")}
            lines.append(f"- {m['method']} / {name}：{fmt(c.get('records'))} 条；首 {json.dumps(first, ensure_ascii=False)}；末 {json.dumps(last, ensure_ascii=False)}。")
    lines += ["", "AR teacher-forced NLL 只用于该自回归模型的诊断，不与 VAE 训练目标横比：", "",
              "| 数据 | target token数 | mean NLL (nats) |",
              "| --- | ---: | ---: |"]
    ar = next(m for m in facts["methods"] if m["method"] == "terminal_LSTM")
    for role in ("train", "development"):
        nll = dig(ar, "ar_nll", role)
        lines.append("| " + " | ".join(fmt(v) for v in (role, dig(nll, "token_count"), dig(nll, "mean_nll"))) + " |")
    lines += ["", "原始 L/G 条件值仅转录如下；完整性门未成立时，即使 raw 为 PASS 也不作为有效科学通过：", "",
              "| raw门 | 端基 | 条件 | CE值 | 比较 | 参照值 | 原始条件pass |",
              "| --- | --- | --- | ---: | --- | ---: | --- |"]
    gates = ev.get("raw_gates") or {}
    for name, key in (("L", "reconstruction_ce_vs_matched_mse"), ("G", "generation_ce_vs_terminal_ar")):
        checks = dig(gates, key, "checks") or {}
        scopes = {"dev_mean": checks} if name == "L" else checks
        if not scopes:
            lines.append(f"| {name} | NOT_EVALUABLE | 缺少条件 | NOT_EVALUABLE | — | NOT_EVALUABLE | NOT_EVALUABLE |")
        for terminal, values in scopes.items():
            for metric, check in values.items():
                lines.append("| " + " | ".join(fmt(v) for v in (name, terminal, metric, check.get("left"),
                    check.get("operator"), check.get("right"), check.get("pass"))) + " |")
    lines += ["", "AR 的 AA/EOS 分解及各条曲线源记录保留于 EXECUTION_FACTS.json。",
              "", "来源与 raw L/G 各项条件、完整八组分布指标、缺失项、成本记录及源文件 SHA256 见 [EXECUTION_FACTS.json](EXECUTION_FACTS.json)；本次汇总文件哈希见 [READOUT_MANIFEST.json](READOUT_MANIFEST.json)。",
              "", "记录工程状态只核 status/更新日志/预算/槽位/有限损失及所需文件存在性；并非对全部梯度、源码、条件信息与 RNG 的独立复审。哈希绑定本脚本读取的产物字节，未重验旧 manifest 内全部权重。", ""]
    lines += ["- " + limitation for limitation in facts["limits"]]
    if facts["missing_or_unreadable_artifacts"]:
        lines += ["", "缺失或不可读产物：", ""]
        lines += [f"- {p['path']}：{p['problem']}" for p in facts["missing_or_unreadable_artifacts"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if not root.is_dir():
        parser.error("run root must already exist")
    outputs = [root / n for n in ("AUTHOR_READOUT.zh.md", "EXECUTION_FACTS.json", "READOUT_MANIFEST.json")]
    if any(p.exists() for p in outputs):
        raise ValueError("readout outputs already exist; no implicit overwrite")
    facts = build(root)
    outputs[0].write_text(render(facts))
    outputs[1].write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "author existing-artifact readout",
                "source": record(__file__), "outputs": [record(p) for p in outputs[:2]],
                "inputs": facts["consumed_artifacts"], "independent_review": "PENDING", "PROMOTE": "HOLD"}
    outputs[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run_root": str(root), "engineering_E": facts["engineering_E"]["state"],
                      "raw_L": facts["raw_L"], "raw_G": facts["raw_G"], "PROMOTE": "HOLD",
                      "AR_ADEQUACY": "NOT_ESTABLISHED", "missing_artifacts": len(facts["missing_or_unreadable_artifacts"])}))


if __name__ == "__main__":
    main()
