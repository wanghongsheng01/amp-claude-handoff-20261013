#!/usr/bin/env python3
"""Add native termination flags and transcribe frozen metrics/costs; no model/data evaluation."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "feasibility/ce_recovery_ar_addendum_v1"
PRIOR = ROOT / "feasibility/CE_RECOVERY_AR_PARTIAL_DELIVERY_v1.json"
PRIOR_SHA = "2d896527331eed302e83afaa051137b68579e5430e0395df1c4e9db7769eff3e"
POST = ROOT / "review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json"
POST_SHA = "2450c62a2553d854685fee7dde300a2ee794746ad8af36bc4db814b9a472c484"
AR = ROOT / "runs/formal_v1/terminal_lstm"
CE = ROOT / "runs/ce_checkpoint_recovery_v1"
EVAL = ROOT / "feasibility/ce_recovery_ar_partial_evaluation_v1"
STEM = "CAP_HIT_AND_FULL_METRICS_ADDENDUM_v1"


def record(path):
    payload = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class BoundReads:
    def __init__(self):
        self.records = {}

    def read(self, path, binding=None, kind="json"):
        path = Path(path).resolve()
        payload = path.read_bytes()
        actual = {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        if binding is not None:
            assert actual["sha256"] == binding["sha256"], str(path)
            if "bytes" in binding:
                assert actual["bytes"] == binding["bytes"], str(path)
        self.records[str(path)] = actual
        text = payload.decode("utf-8-sig")
        if kind == "jsonl": return [json.loads(line) for line in text.splitlines() if line]
        if kind == "csv": return list(csv.DictReader(io.StringIO(text)))
        if kind == "source": return text
        return json.loads(text)


def dump(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    assert not OUT.exists(), "fresh addendum output required"
    reads = BoundReads()
    prior = reads.read(PRIOR, {"sha256": PRIOR_SHA})
    post = reads.read(POST, {"sha256": POST_SHA})
    assert post["status"] == "PASS" and post["scope"] == "RECOVERY_ENGINEERING_ONLY"
    prior_records = {e["path"]: e for p in prior["packages"].values() for e in p}
    manifest = reads.read(AR / "manifest.json")
    def ar(name, kind="json"):
        matches = [r for r in manifest["files"] if r["path"] == name]
        assert len(matches) == 1
        return reads.read(AR / name, matches[0], kind)
    native = ar("generation/samples.jsonl", "jsonl")
    standardized = ar("generated.csv", "csv")
    training = ar("training/RUN.json")
    monitors = ar("training/MONITORS.json")
    generation = ar("generation/GENERATION.json")
    status = ar("status.json")
    reads.read(ROOT / "implementation/ar_pipeline.py", training["code"], "source")
    ce_binding = next(r for r in post["source_bindings"] if r["path"] == str(CE / "generated.csv"))
    ce_rows = reads.read(CE / "generated.csv", ce_binding, "csv")
    ce_config_binding = next(r for r in post["source_bindings"] if r["path"] == str(CE / "config.json"))
    ce_config = reads.read(CE / "config.json", ce_config_binding)
    reads.read(ROOT / "implementation/vae_pipeline_formal_v2.py", {"sha256": ce_config["implementation_sha256"]}, "source")
    def evaluation(name):
        p = EVAL / name
        return reads.read(p, prior_records[str(p)])
    metrics, gates = evaluation("generation_metrics.json"), evaluation("exploratory_gates.json")
    nll = evaluation("ar_teacher_forced_nll.json")
    expected = {f"terminal_LSTM_{t}_{i:03d}" for t in ("FREE", "AMD") for i in range(16)}
    assert len(native) == len(standardized) == 32
    assert {r["sample_id"] for r in native} == {r["sample_id"] for r in standardized} == expected
    standard_by_id = {r["sample_id"]: r for r in standardized}
    native_audit = []
    for r in native:
        counterpart = standard_by_id[r["sample_id"]]
        assert all(str(r[k]) == counterpart[k] for k in ("sample_id", "method", "terminal", "sequence", "status", "length", "seed"))
        assert type(r["terminated"]) is bool and type(r["max_length_hit"]) is bool and r["complete_attempt"] is True
        assert len(r["sequence"]) == r["length"] and r["max_length_hit"] == (r["length"] == 48)
        assert r["terminated"] != r["max_length_hit"]
        native_audit.append({k: r[k] for k in ("sample_id", "terminal", "length", "status", "terminated", "max_length_hit", "complete_attempt", "seed")})
    termination = []
    for method, rows in (("CE21_DDPM", ce_rows), ("terminal_LSTM", native)):
        for terminal in ("FREE", "AMD"):
            selected = [r for r in rows if r["terminal"] == terminal]
            assert len(selected) == 16
            if method == "terminal_LSTM":
                ended = sum(r["terminated"] for r in selected)
                cap = [r for r in selected if r["max_length_hit"]]
                semantics = "native sampled EOS; terminated=true"
            else:
                assert all((int(r["length"]) == len(r["sequence"]) and
                    r["status"] == ("empty_sequence" if len(r["sequence"]) == 0 else "ok" if len(r["sequence"]) < 48 else "no_eos_at_limit")) for r in selected)
                ended = sum(int(r["length"]) < 48 for r in selected)
                cap = [r for r in selected if int(r["length"]) == 48]
                semantics = "J-selected length below48 / PAD suffix; not an autoregressive sampled EOS event"
            termination.append({"method": method, "terminal": terminal, "requests": 16,
                "ended_below_length_limit_count": ended, "length_cap_hit_count": len(cap),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "termination_semantics": semantics, "cap_hit_sample_ids": [r["sample_id"] for r in cap]})
    assert [r["step"] for r in monitors] == list(range(100, 1001, 100))
    monitor_rows = [{"step": r["step"], "train_nll": r["train"]["mean_nll"], "dev_nll": r["development"]["mean_nll"], "used_for_selection": r["used_for_selection"]} for r in monitors]
    assert all(r["used_for_selection"] is False for r in monitor_rows)
    summaries = []
    for group in metrics["groups"]:
        summary = {k: group[k] for k in ("method", "terminal", "requests", "valid_1_48_count", "valid_5_48_count", "exact_copy_train_count_all_requests", "exact_sequence_copy_any_train_terminal_count", "pooled_aa_entropy_bits_absolute_gap_to_train", "pooled_dipeptide_entropy_bits_absolute_gap_to_train")}
        summary.update(unique_valid=group["valid_1_48"]["unique_count"], length=group["valid_1_48"]["length"],
            entropy={k: v for k, v in group["valid_1_48"].items() if "entropy" in k},
            train=group["valid_1_48_vs_train"], development=group["valid_1_48_vs_development"])
        summaries.append(summary)
    costs = {"AR": {"train_updates_seconds": training["train_update_seconds"], "monitoring_seconds": training["monitoring_seconds"],
        "training_pipeline_wall_before_manifest_seconds": training["total_pipeline_seconds_before_manifest"],
        "generation_wall_before_manifest_seconds": generation["wall_seconds_before_manifest"], "entry_process_seconds": status["process_seconds"],
        "scope": "training pipeline includes updates, monitoring and overhead; do not add nested durations; process includes the pipeline/generation"},
        "CE": post["costs"], "equal_actual_compute_claim": False}
    preserved = [{**r, "unchanged": record(r["path"])["sha256"] == r["sha256"]} for r in reads.records.values()]
    assert all(r["unchanged"] for r in preserved)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "additive presentation correction; no metric recomputation or gate change",
        "original_protocol_E_L_G": "NOT_EVALUABLE", "recovery_L": "NOT_EVALUABLE", "recovery_G": "FAIL", "PROMOTE": "HOLD", "AR_ADEQUACY": "NOT_ESTABLISHED",
        "termination": termination, "AR_native_flag_audit": native_audit,
        "AR_native_time_budget_capped_by_terminal": generation["capped_by_terminal"],
        "status_ok_is_not_a_comparable_natural_termination_indicator": True,
        "AR_standardized_rows_match_native_rows": True,
        "flag_scope_note": "AR EOS termination and CE legal-prefix J/PAD termination are different mechanisms; cap denominator is comparable at16 requests per terminal. capped_by_terminal is a time-budget flag, not length-cap absence.",
        "full_requested_distribution_summary": summaries, "frozen_original_gate_record": gates,
        "AR_fixed_checkpoint_monitor": monitor_rows, "AR_final_teacher_forced_nll": nll,
        "AR_overfitting_interpretation": "Development teacher NLL rises from recorded step200 to fixed final1000 while training NLL decreases; observed overfitting evidence, not a new selection or adequacy gate.",
        "costs": costs, "source_bindings_and_preservation": preserved, "source": record(__file__),
        "operations": {"model_calls": 0, "metric_recomputations": 0, "cohort_reads": 0, "test_reads": 0, "checkpoint_or_array_reads": 0},
        "MSE_authorization_affected": False, "prior_delivery": record(PRIOR)}
    OUT.mkdir()
    dump(OUT / (STEM + ".json"), payload)
    lines = ["# AMP H5：终止/触顶与完整指标补充 v1", "",
        "**补充后结论不变：原正式 E/L/G 永久 NOT_EVALUABLE；恢复描述性 G=FAIL、L=NOT_EVALUABLE；AR_ADEQUACY=NOT_ESTABLISHED，PROMOTE=HOLD。** 本件仅核对既有标记与转录既存指标，不重算指标、不调用模型、不影响当前 MSE。", "",
        "AR 的 `status=ok` 包括长度48触顶而未抽到EOS的请求。因此不能用 CE/AR 的 `ok` 数横比自然终止率。AR 原生 `generation/samples.jsonl` 已逐行对齐标准CSV；原生flag没有进入旧partial标准样本审计，本件补充保留，旧包不改。", "",
        "| 方法 | 端基 | 请求 | 界内结束 | 长度48触顶 | 状态计数 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in termination:
        lines.append(f"| {r['method']} | {r['terminal']} | 16 | {r['ended_below_length_limit_count']}/16 | {r['length_cap_hit_count']}/16 | {r['status_counts']} |")
    lines += ["", "AR 界内结束指原生 `terminated=true` 的EOS采样事件；CE 界内结束指 J 选择长度<48并使用PAD后缀，不能伪称同一采样机制。CE FREE_009 为 `length=48,status=no_eos_at_limit`；AR FREE_001 同为48，原生 `terminated=false,max_length_hit=true,status=ok,complete_attempt=true`。两方法FREE均触顶1/16、AMD均0/16。它们仍保留在全部16请求及合法1–48分母中，不删样本、不补采。", "",
        "AR 的 `capped_by_terminal=false` 指没有触发时间预算截断，不表示没有长度触顶。两个方法均未观察到空串；这里不新增终止质量门。", "",
        "以下分布指标均沿冻结定义：合法1–48集合；same-terminal train/dev参照；MMD为421维固定特征的有偏线性MMD²；JS以bit计，KS无p值。每组请求16、合法1–48为16、unique16、同端及任意端train exact-copy均0。CE合法5–48为FREE14/16、AMD15/16，AR均16/16。", "",
        "| 方法 | 端基 | 参照 | MMD² | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        for ref in ("train", "development"):
            d=r[ref]
            values=[d[k] for k in ("mmd2_linear_biased_fixed_features","aa_jensen_shannon_bits","aa_l1_distance","dipeptide_jensen_shannon_bits","dipeptide_l1_distance","length_ks_statistic_no_pvalue")]
            lines.append("| " + " | ".join([r["method"],r["terminal"],ref]+[f"{v:.6f}" for v in values]) + " |")
    lines += ["", "| 方法 | 端基 | pooled AA熵 | pooled二肽熵 | AA熵距train | 二肽熵距train | 单序列AA熵均值 | 长度均值[min,max] |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in summaries:
        e=r['entropy']; l=r['length']
        values=[e['pooled_aa_entropy_bits'],e['pooled_dipeptide_entropy_bits'],r['pooled_aa_entropy_bits_absolute_gap_to_train'],r['pooled_dipeptide_entropy_bits_absolute_gap_to_train'],e['mean_per_sequence_aa_entropy_bits']]
        lines.append("| " + " | ".join([r['method'],r['terminal']]+[f"{v:.6f}" for v in values]+[f"{l['mean']:.4f} [{l['min']},{l['max']}]"]) + " |")
    lines += ["", "四组sequence entropy均4 bit（16条均unique）。CE双端train/dev MMD、AA/二肽JS与L1、长度KS均高于AR；双端AA/二肽熵距train也更大。原恢复G的8个失败子项保持：双端dev MMD未严格降低、双端合法5–48较少、双端各两项熵差保护不满足。上述触顶披露不改变任何指标或门值。", "",
        "| AR固定step | train teacher NLL | dev teacher NLL |", "| ---: | ---: | ---: |"]
    for r in monitor_rows: lines.append(f"| {r['step']} | {r['train_nll']:.6f} | {r['dev_nll']:.6f} |")
    lines += ["", "固定1000终点未更换。dev teacher NLL从200步约2.615253上升至1000步约3.169793，同时train NLL下降至约1.643124，支持已观察到的过拟合迹象；不据此选择checkpoint或声称AR已充分优化。NLL含AA及观察到的EOS目标，只作AR诊断，不与VAE目标横比。", "",
        "| 实际成本范围 | 秒数 |", "| --- | ---: |"]
    ar_cost=costs['AR']
    for key,label in (("train_updates_seconds","AR纯训练更新"),("monitoring_seconds","AR固定检查点监测"),("training_pipeline_wall_before_manifest_seconds","AR训练pipeline wall（含前两项及其他开销）"),("generation_wall_before_manifest_seconds","AR生成wall（manifest前）"),("entry_process_seconds","AR入口process（含pipeline/生成等）")):
        lines.append(f"| {label} | {ar_cost[key]:.9f} |")
    cc=costs['CE']
    for key,label in (("original_training_stage_subtotal","原CE VAE+DDPM阶段小计"),("original_generation_lower_bound","原CE中断生成下界 ≥"),("recovery_setup","新CE恢复setup"),("recovery_generation","新CE恢复生成"),("recovery_export","新CE恢复重建导出"),("recovery_process_before_final_io","新CE恢复process（final I/O前）"),("recovery_supervisor_outer","新CE恢复supervisor outer")):
        lines.append(f"| {label} | {cc[key]:.9f} |")
    lines += ["", "AR约14.293秒是训练pipeline wall，纯训练更新为12.268秒；不能把14.293写成纯训练时间。上述时间范围互相嵌套，不把更新、监测、pipeline、process重复相加。原CE总wall未知，原partial生成下界不当作0；恢复是额外attempt，不能宣称实际等计算。完整研究成本还包括此前smoke、诊断、入口失败及审查，未在此伪造总额。", "",
        "只有现有32条CE恢复与32条AR观测；旧CE中断32槽位不是另一组已完成样本。MSE/scalar没有被本件创建任何结果。无显著性、活性、安全性、新颖性、强基线充分性或可投稿断言；旧湿实验不验证H5。", "",
        f"全部原生flag（不重复序列正文）、精确数值、成本字段、输入hash及前后保全见 [{STEM}.json]({STEM}.json)。原partial包和冻结代码保持不变。"]
    with (OUT / (STEM + ".zh.md")).open("x") as f: f.write("\n".join(lines)+"\n")
    dump(OUT / "MANIFEST.json", {"source":record(__file__),"files":[record(p) for p in sorted(OUT.iterdir()) if p.is_file()],"inputs":list(reads.records.values()),"original_protocol_E_L_G":"NOT_EVALUABLE","PROMOTE":"HOLD"})
    index=ROOT/"feasibility/CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json"
    dump(index,{"created_utc":datetime.now(timezone.utc).isoformat(),"prior_unchanged_delivery":record(PRIOR),"addendum":[record(p) for p in sorted(OUT.iterdir()) if p.is_file()],"source":record(__file__),"original_protocol_E_L_G":"NOT_EVALUABLE","recovery_G":"FAIL","recovery_L":"NOT_EVALUABLE","PROMOTE":"HOLD","MSE_authorization_affected":False})
    print(json.dumps({"output":str(OUT),"index":record(index),"termination":termination,"inputs_unchanged":True},ensure_ascii=False))


if __name__ == "__main__":
    main()
