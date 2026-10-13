#!/usr/bin/env python3
"""Read-only cross-version H5 artifact summary; no model or metric evaluation.

Explicit method paths replace the v1 single-root assumption. Missing methods are
availability entries only: generation/reconstruction arrays contain exclusively
observations already present in hash-bound evaluation artifacts. Output is fresh.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V1 = HERE / "build_readout.py"
EXPECTED_V1 = "75012cfc99e73f7d08c5b69cb0e9cf2d0f74acc28b98f6db8203e4b8536486dd"
if hashlib.sha256(V1.read_bytes()).hexdigest() != EXPECTED_V1:
    raise ValueError("frozen v1 summary dependency changed")
spec = importlib.util.spec_from_file_location("readout_v1_frozen", V1)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
METHODS = tuple(v1.METHODS.values())
TERMINALS = v1.TERMINALS
TRAIN_SHA = "a232401fb46bb9419cf8d690f21476447b853862fd23622e81247b88add0b660"
DEV_SHA = "65fdc7d402240235698d30aa769de3ea45c577cbcb28931970571ee808098ec1"


class MappedInputs:
    """Resolve virtual method/file keys only under explicitly supplied roots."""
    def __init__(self, method_paths):
        if set(method_paths) != set(METHODS):
            raise ValueError("all four explicit method paths required, even if paths do not yet exist")
        self.paths = {k: Path(p).resolve() for k, p in method_paths.items()}
        if len(set(self.paths.values())) != 4:
            raise ValueError("method roots must be distinct")
        self.readers = {m: v1.Inputs(p) for m, p in self.paths.items()}
        self.problems, self.records = [], {}

    def read(self, virtual_path, **kwargs):
        method, separator, relative = virtual_path.partition("/")
        if not separator or method not in self.readers:
            raise ValueError("unregistered method artifact path")
        reader = self.readers[method]
        before = len(reader.problems)
        value = reader.read(relative, **kwargs)
        self.problems.extend({**p, "path": method + "/" + p["path"]} for p in reader.problems[before:])
        if relative in reader.records:
            self.records[virtual_path] = reader.records[relative]
        return value


def normalize_records(records):
    if not isinstance(records, list):
        return []
    return [r for r in records if isinstance(r, dict) and isinstance(r.get("path"), str)
            and isinstance(r.get("sha256"), str)]


def evaluate_artifacts(directory, method_inputs):
    """Read saved evaluator results; never evaluate sequences or compare metrics."""
    root = Path(directory).resolve()
    reader = v1.Inputs(root)
    # APFS may be case-insensitive: probe actual names, not two is_file calls.
    names = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    standard = "generation_metrics.json" in names
    ar_only = "GENERATION_METRICS.json" in names
    if standard and ar_only:
        raise ValueError("ambiguous mixed evaluation schema: " + str(root))
    if not standard and not ar_only:
        return {"directory": str(root), "schema": "MISSING", "state": "NOT_EVALUABLE",
                "generation": [], "reconstruction": [], "raw_gates": None,
                "method_bindings": {}, "problems": [{"path": str(root), "problem": "NO_EVALUATION_ARTIFACTS"}], "records": []}
    if standard:
        generation = reader.read("generation_metrics.json") or {}
        reconstruction = reader.read("reconstruction_metrics.json") or {}
        run = reader.read("run.json") or {}
        gates = reader.read("exploratory_gates.json")
        reader.read("MANIFEST.json")
        reader.read("ar_teacher_forced_nll.json", required=False)
        source_records = normalize_records(run.get("sample_inputs", [])) + normalize_records(run.get("reconstruction_inputs", []))
        state_complete = run.get("status") == "COMPLETE"
        references_bound = (v1.dig(run, "train", "sha256") == TRAIN_SHA and
                            v1.dig(run, "development", "sha256") == DEV_SHA)
        schema = "frozen_h5_evaluator"
    else:
        generation = reader.read("GENERATION_METRICS.json") or {}
        reconstruction, gates = {}, None
        run = reader.read("RUN_FACTS.json") or {}
        binding = reader.read("SOURCE_BINDINGS.json") or {}
        reader.read("MANIFEST.json")
        reader.read("TEACHER_FORCED_NLL.json")
        source_records = normalize_records(binding.get("inputs", []))
        state_complete = (run.get("scope") == "completed AR-only descriptive feasibility" and
                          binding.get("source_preservation_pass") is True and run.get("attempts") == 32)
        refs = {r["sha256"] for r in source_records}
        references_bound = TRAIN_SHA in refs and DEV_SHA in refs
        schema = "ar_only_descriptive_v1"
    gen, rec = generation.get("groups", []), reconstruction.get("groups", [])
    if not isinstance(gen, list) or not isinstance(rec, list):
        raise ValueError("evaluation groups must be arrays")
    if any(not isinstance(g, dict) or g.get("method") not in METHODS or g.get("terminal") not in TERMINALS for g in gen):
        raise ValueError("unexpected generation group identity")
    if any(not isinstance(g, dict) or g.get("method") not in METHODS[:3] or g.get("split") not in ("train", "development") for g in rec):
        raise ValueError("unexpected reconstruction group identity")
    groups_methods = {g["method"] for g in gen + rec}
    bindings = {}
    for method in groups_methods:
        wanted = method_inputs.paths[method] / "generated.csv"
        saved = [r for r in source_records if Path(r["path"]).resolve() == wanted]
        actual = method_inputs.records.get(method + "/generated.csv")
        generation_bound = len(saved) == 1 and actual is not None and saved[0]["sha256"] == actual["sha256"]
        reconstruction_bound = True
        for group in [g for g in rec if g["method"] == method]:
            filename = "train_reconstruction.csv" if group["split"] == "train" else "dev_reconstruction.csv"
            wanted = method_inputs.paths[method] / filename
            saved_recon = [r for r in source_records if Path(r["path"]).resolve() == wanted]
            actual_recon = method_inputs.records.get(method + "/" + filename)
            reconstruction_bound = reconstruction_bound and len(saved_recon) == 1 and actual_recon is not None and saved_recon[0]["sha256"] == actual_recon["sha256"]
        bindings[method] = {"generation_input_matches_explicit_run": generation_bound,
                            "reconstruction_inputs_match_explicit_run": reconstruction_bound,
                            "reference_hashes_match_locked_train_dev": references_bound,
                            "pass": generation_bound and reconstruction_bound and references_bound}
    return {"directory": str(root), "schema": schema,
            "state": "RECORDED_COMPLETE" if state_complete and not reader.problems else "NOT_EVALUABLE",
            "generation": gen, "reconstruction": rec, "raw_gates": gates, "run": run,
            "method_bindings": bindings, "problems": reader.problems, "records": list(reader.records.values())}


def build(method_paths, evaluation_paths):
    inputs = MappedInputs(method_paths)
    methods = [v1.method_facts(inputs, method, method) for method in METHODS]
    for item in methods:
        item["explicit_run_path"] = str(inputs.paths[item["method"]])
        item["source_version_note"] = "explicit path only; no implicit current/latest run substitution"
    packages = [evaluate_artifacts(p, inputs) for p in evaluation_paths]
    seen_gen, seen_rec = set(), set()
    gen, rec = [], []
    status_by_method = {m["method"]: m for m in methods}
    for package in packages:
        for kind, source, seen, dest, identity_fields in (
                ("generation", package["generation"], seen_gen, gen, ("method", "terminal")),
                ("reconstruction", package["reconstruction"], seen_rec, rec, ("method", "split"))):
            for group in source:
                key = tuple(group[k] for k in identity_fields)
                if key in seen:
                    raise ValueError("duplicate evaluation observation; select one explicit source: " + repr(key))
                seen.add(key)
                bound = package["method_bindings"].get(group["method"], {}).get("pass", False)
                method_complete = status_by_method[group["method"]]["engineering_record_state"] == "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW"
                dest.append({"method": group["method"], identity_fields[1]: group[identity_fields[1]],
                    "source_evaluation": package["directory"], "source_binding_pass": bound,
                    "interpretation_state": "AVAILABLE_DESCRIPTIVE" if bound and method_complete and package["state"] == "RECORDED_COMPLETE" else "NOT_EVALUABLE",
                    "metrics": group})
    availability = []
    for method in METHODS:
        available_gen = sorted(g["terminal"] for g in gen if g["method"] == method)
        available_rec = sorted(g["split"] for g in rec if g["method"] == method)
        availability.append({"method": method, "run_path": str(inputs.paths[method]),
             "recorded_run_state": status_by_method[method]["recorded_state"],
             "generation_groups_present": available_gen, "reconstruction_splits_present": available_rec,
             "observations_complete": available_gen == sorted(TERMINALS) and (method == "terminal_LSTM" or available_rec == ["development", "train"]),
             "state": "AVAILABLE_PARTIAL_OR_COMPLETE" if available_gen or available_rec else "NOT_EVALUABLE",
             "absent_observation_policy": "no synthetic group or row is created"})
    all_methods_complete = all(m["engineering_record_state"] == "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" for m in methods)
    all_obs = (seen_gen == {(m, t) for m in METHODS for t in TERMINALS} and
               seen_rec == {(m, s) for m in METHODS[:3] for s in ("train", "development")} and
               all(g["interpretation_state"] == "AVAILABLE_DESCRIPTIVE" for g in gen + rec))
    # A raw gate set may only control the full comparison if its own single
    # saved evaluation covered all observations; merging partial gates forbidden.
    complete_gate_packages = [p for p in packages if p["schema"] == "frozen_h5_evaluator" and
       {(g["method"], g["terminal"]) for g in p["generation"]} == {(m, t) for m in METHODS for t in TERMINALS} and
       {(g["method"], g["split"]) for g in p["reconstruction"]} == {(m, s) for m in METHODS[:3] for s in ("train", "development")}]
    authoritative = complete_gate_packages[0] if len(complete_gate_packages) == 1 else None
    raw_l = v1.gate_status(v1.dig(authoritative, "raw_gates", "reconstruction_ce_vs_matched_mse"))
    raw_g = v1.gate_status(v1.dig(authoritative, "raw_gates", "generation_ce_vs_terminal_ar"))
    eligible = all_methods_complete and all_obs and authoritative is not None
    known_costs = [m["costs"]["process_seconds"] for m in methods if v1.finite(m["costs"].get("process_seconds"))]
    records = list(inputs.records.values()) + [r for p in packages for r in p["records"]]
    # Detect changes during this read-only snapshot; running files remain a
    # partial progress snapshot even when each individual read was successful.
    preserved = [{"path": r["path"], "before_sha256": r["sha256"],
                  "unchanged_at_close": Path(r["path"]).is_file() and v1.record(r["path"])["sha256"] == r["sha256"]} for r in records]
    stable = all(r["unchanged_at_close"] for r in preserved)
    eligible = eligible and stable
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
       "scope": "AMP H5 explicit-path author summary of saved artifacts only",
       "source": v1.record(__file__), "frozen_summary_dependency": v1.record(V1),
       "path_mapping": {m: str(p) for m, p in inputs.paths.items()}, "methods": methods,
       "method_availability": availability, "generation_observations": gen, "reconstruction_observations": rec,
       "observation_counts": {"generation_groups": len(gen), "reconstruction_groups": len(rec),
                               "synthetic_missing_observations": 0},
       "evaluation_packages": packages,
       "engineering_E": {"state": "RECORDED_PASS_PENDING_INDEPENDENT_REVIEW" if eligible else "NOT_EVALUABLE",
                         "all_method_records_complete": all_methods_complete, "all_bound_observations_complete": all_obs,
                         "single_complete_gate_package_present": authoritative is not None, "input_snapshot_stable": stable,
                         "independent_audit_claim": False},
       "raw_L": raw_l, "raw_G": raw_g, "effective_L": raw_l if eligible else "NOT_EVALUABLE",
       "effective_G": raw_g if eligible else "NOT_EVALUABLE",
       "comparison_status": "RECORDED_COMPLETE_SCREEN" if eligible else "NOT_EVALUABLE_INCOMPLETE_COMPARISON",
       "AR_ADEQUACY": "NOT_ESTABLISHED", "PROMOTE": "HOLD",
       "costs": {"known_method_process_seconds_sum": sum(known_costs), "methods_with_process_timing": len(known_costs),
                 "extra_development_costs": "not added here; preserve smoke, maturity diagnostic, v1 zero-update input failure, staged execution and review separately",
                 "scope": "sum of recorded method timings only; no equal-compute or independent outer-wall-clock claim"},
       "missing_or_unreadable_artifacts": inputs.problems + [x for p in packages for x in p["problems"]],
       "consumed_artifacts": records, "input_preservation_checks": preserved,
       "limits": ["missing MSE/scalar or other observations remain absent; missingness is not a measured zero or a failed generation sample",
                  "single fitted seed, repeated dev41 use, fixed final checkpoint; no significance or independent confirmation",
                  "AR adequacy stays NOT_ESTABLISHED; fixed-monitor curves create no new adequacy gate or endpoint selection",
                  "no activity/safety/novelty/strongest-baseline superiority/submission-readiness claim; old wetlab does not validate H5",
                  "recorded engineering completeness is not an independent audit of model math, all gradients, RNG or conditioning",
                  "v2 input idx contract repair does not erase prior zero-update failed entry or extra adaptive diagnostic cost",
                  "BACKUP_BLOCKED; no submission, remote backup or credential action"],
       "operations": {"model_calls": 0, "metric_recomputations": 0, "split_files_opened": 0,
                      "test_files_opened": 0, "checkpoint_files_loaded": 0}}


def render(facts):
    fmt, dig = v1.fmt, v1.dig
    lines = ["# AMP H5 跨版本作者读出", "",
       "本文件仅汇总显式路径上的既有产物，没有训练、生成、指标重算或 split 读取。缺失方法只在可用性表标记；后面的观测表只包含实际存在的评价记录。", "",
       f"工程 E：**{facts['engineering_E']['state']}**；有效 L/G：**{facts['effective_L']} / {facts['effective_G']}**。",
       "**AR_ADEQUACY = NOT_ESTABLISHED；PROMOTE = HOLD。**", "",
       "| 方法 | 显式运行路径 | 记录状态 | 已有生成端基 | 已有重建split | 可用性 |",
       "| --- | --- | --- | --- | --- | --- |"]
    for a in facts["method_availability"]:
        lines.append("| " + " | ".join(fmt(x) for x in (a["method"], a["run_path"], a["recorded_run_state"],
           ",".join(a["generation_groups_present"]) or "无", ",".join(a["reconstruction_splits_present"]) or "无", a["state"])) + " |")
    lines += ["", f"已有生成组数：{len(facts['generation_observations'])}；没有为缺失组创建请求或样本。MMD²仍以合法1–48集合为分母。", "",
       "| 方法 | terminal | 请求 | 合法1–48 | 合法5–48 | unique合法 | copy同端基/任意train | train MMD² | dev MMD² | 解释状态 |",
       "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in facts["generation_observations"]:
        g = row["metrics"]
        lines.append("| " + " | ".join(fmt(x) for x in (row["method"], row["terminal"], g.get("requests"), g.get("valid_1_48_count"),
          g.get("valid_5_48_count"), dig(g, "valid_1_48", "unique_count"),
          fmt(g.get("exact_copy_train_count_all_requests")) + "/" + fmt(g.get("exact_sequence_copy_any_train_terminal_count")),
          dig(g, "valid_1_48_vs_train", "mmd2_linear_biased_fixed_features"),
          dig(g, "valid_1_48_vs_development", "mmd2_linear_biased_fixed_features"), row["interpretation_state"])) + " |")
    if not facts["generation_observations"]:
        lines += ["", "当前没有已计算的生成评价记录：NOT_EVALUABLE。"]
    lines += ["", "以下只列已有 dev posterior mean 重建。macro 为target支持AA类别的recall均值，PAD不入分母。", "",
       "| 方法 | dev对象 | micro | AA类macro | sequence mean | 长度MAE | 整串exact | 解释状态 |",
       "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    dev_rows = [r for r in facts["reconstruction_observations"] if r["split"] == "development"]
    for row in dev_rows:
        g = row["metrics"]
        lines.append("| " + " | ".join(fmt(x) for x in (row["method"], g.get("rows"), g.get("rendered_nonpad_micro"),
          g.get("rendered_nonpad_macro"), g.get("per_sequence_mean_nonpad_accuracy"), g.get("length_mae"),
          g.get("whole_rendered_exact_count"), row["interpretation_state"])) + " |")
    if not dev_rows:
        lines += ["", "当前没有已计算的dev mean重建评价记录：NOT_EVALUABLE。"]
    lines += ["", "| 方法 | 记录工程状态 | 完成/计划更新 | 训练秒 | 生成秒 | 进程秒 |",
       "| --- | --- | --- | ---: | ---: | ---: |"]
    for m in facts["methods"]:
        update = "; ".join(f"{k} {fmt(v['completed'])}/{fmt(v['requested'])}" for k, v in m["updates"].items())
        c = m["costs"]
        lines.append("| " + " | ".join(fmt(x) for x in (m["method"], m["engineering_record_state"], update,
           c.get("training_seconds"), c.get("generation_seconds"), c.get("process_seconds"))) + " |")
    lines += ["", "固定首末日志仅作描述，不据此新增充分性门或选终点：", ""]
    for m in facts["methods"]:
        for name, curve in m["curves"].items():
            if not curve.get("records"):
                continue
            first, last = curve.get("first"), curve.get("last")
            if name == "ar_fixed_monitor":
                first = {"step": first.get("step"), "train_nll": dig(first, "train", "mean_nll"), "dev_nll": dig(first, "development", "mean_nll")}
                last = {"step": last.get("step"), "train_nll": dig(last, "train", "mean_nll"), "dev_nll": dig(last, "development", "mean_nll")}
            lines.append(f"- {m['method']} / {name}：{curve['records']} 条；首 {json.dumps(first, ensure_ascii=False)}；末 {json.dumps(last, ensure_ascii=False)}。")
    lines += ["", "既有评价包的原始门记录完整保留于 EXECUTION_FACTS.json。partial包的门不能合并成完整比较；只有单个完整、绑定全部来源的评价包才能提供四臂比较门。", "",
       "已有全部分布字段、AR teacher-forced NLL、显式路径绑定和缺失清单见 [EXECUTION_FACTS.json](EXECUTION_FACTS.json)；哈希清单见 [READOUT_MANIFEST.json](READOUT_MANIFEST.json)。", ""]
    lines += ["- " + value for value in facts["limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("ce", "mse", "scalar", "ar"):
        parser.add_argument("--" + arm + "-run", required=True)
    parser.add_argument("--evaluation", action="append", default=[], help="existing metric directory; repeat for disjoint partial packages")
    parser.add_argument("--output", required=True, help="new readout directory, separate from every source run")
    args = parser.parse_args()
    paths = dict(zip(METHODS, (args.ce_run, args.mse_run, args.scalar_run, args.ar_run)))
    output = Path(args.output).resolve()
    sources = [Path(p).resolve() for p in paths.values()] + [Path(p).resolve() for p in args.evaluation]
    if any(output == p or p in output.parents or output in p.parents for p in sources):
        raise ValueError("output must be separate from source trees")
    if output.exists():
        raise ValueError("fresh readout directory required; no overwrite")
    facts = build(paths, args.evaluation)
    output.mkdir(parents=True)
    (output / "AUTHOR_READOUT.zh.md").write_text(render(facts))
    (output / "EXECUTION_FACTS.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": v1.record(__file__),
       "frozen_summary_dependency": v1.record(V1), "path_mapping": facts["path_mapping"],
       "outputs": [v1.record(output / n) for n in ("AUTHOR_READOUT.zh.md", "EXECUTION_FACTS.json")],
       "inputs": facts["consumed_artifacts"], "input_snapshot_stable": facts["engineering_E"]["input_snapshot_stable"],
       "independent_review": "PENDING", "PROMOTE": "HOLD"}
    (output / "READOUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "observations": facts["observation_counts"],
                      "engineering_E": facts["engineering_E"]["state"], "PROMOTE": "HOLD"}))


if __name__ == "__main__":
    main()
