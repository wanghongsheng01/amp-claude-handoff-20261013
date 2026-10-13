#!/usr/bin/env python3
"""Summarize completed CE recovery/MSE/AR artifacts; never inspect scalar output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "feasibility/ce_recovery_mse_ar_evaluation_v1"
EXEC = ROOT / "feasibility/ce_recovery_mse_ar_execution_v1"
OUT = ROOT / "feasibility/ce_recovery_mse_ar_author_readout_v1"
EXPECTED = {
    "implementation/evaluate_h5.py": "b894a7c56cd5c59c5fb77c046baa3c9873ae390d244082dd1f09998459e85417",
    "implementation/frozen_metrics_v1.py": "f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480",
    "review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json": "2450c62a2553d854685fee7dde300a2ee794746ad8af36bc4db814b9a472c484",
    "review/independent/MSE_V2_POSTFLIGHT_v1.json": "a0a410526b6a252ba1f73fd3957dea5ee20413bb6e5c264fca5d34a4eec97b71",
    "feasibility/CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json": "bb0a2ec8a18742702d57f20fe559ece92fe9a0a81ec0734c7af0d084e58f0714",
}


def record(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def dump(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    assert not OUT.exists(), "fresh author output required"
    inputs = {}
    def read(path, expected=None, parse=True):
        r = record(path)
        if expected:
            assert r["sha256"] == expected["sha256"]
            if "bytes" in expected: assert r["bytes"] == expected["bytes"]
        inputs[r["path"]] = r
        return json.loads(Path(path).read_text()) if parse else None
    for path, sha in EXPECTED.items(): read(ROOT/path, {"sha256":sha}, False)
    ce_post = read(ROOT/"review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json")
    mse_post = read(ROOT/"review/independent/MSE_V2_POSTFLIGHT_v1.json")
    assert ce_post["status"] == mse_post["status"] == "PASS"
    assert ce_post["scope"] == "RECOVERY_ENGINEERING_ONLY" and mse_post["scope"] == "SINGLE_MSE_ENGINEERING_ONLY"
    decision = read(EXEC/"READOUT_DECISION_UPDATE.json")
    assert decision["three_arm_v4_executed"] is False and decision["scalar_output_reads"] == 0
    manifest = read(EVAL/"MANIFEST.json")
    files = ("generation_metrics.json", "generation_sample_audit.json", "reconstruction_metrics.json", "reconstruction_row_audit.json", "ar_teacher_forced_nll.json", "exploratory_gates.json", "run.json")
    assert len(manifest["files"]) == len(files)
    artifacts = {}
    for name in files:
        path = EVAL/name
        matches = [r for r in manifest["files"] if r["path"] == str(path)]
        assert len(matches) == 1, name
        artifacts[name] = read(path, matches[0], name not in ("generation_sample_audit.json","reconstruction_row_audit.json"))
    run = artifacts["run.json"]
    assert run["methods"] == ["CE21_DDPM","MSE21_DDPM","terminal_LSTM"] and run["rows_retained"] == 96
    assert run["test_accessed"] is False and run["model_calls"] == 0
    for method, root, receipt in (("CE21_DDPM", ROOT/"runs/ce_checkpoint_recovery_v1", ce_post),
                                   ("MSE21_DDPM", ROOT/"runs/formal_v2/raw_onehot_mse", mse_post)):
        for name in ("generated.csv","train_reconstruction.csv","dev_reconstruction.csv"):
            path = str(root/name)
            receipt_bindings = [r for r in receipt["source_bindings"] if r["path"] == path]
            evaluation_bindings = [r for r in run["sample_inputs"]+run["reconstruction_inputs"] if r["path"] == path]
            assert len(receipt_bindings) == len(evaluation_bindings) == 1
            assert receipt_bindings[0]["sha256"] == evaluation_bindings[0]["sha256"]
    generation, reconstruction, gates = artifacts["generation_metrics.json"], artifacts["reconstruction_metrics.json"], artifacts["exploratory_gates.json"]
    assert len(generation["groups"]) == 6 and len(reconstruction["groups"]) == 4
    for method in ("CE21_DDPM","MSE21_DDPM","terminal_LSTM"):
        selected = [g for g in generation["groups"] if g["method"] == method]
        assert {g["terminal"] for g in selected} == {"FREE","AMD"} and all(g["requests"] == g["execution_completed_count"] == 16 for g in selected)
    historical_index = read(ROOT/"feasibility/CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json")
    addendum_path = ROOT/"feasibility/ce_recovery_ar_addendum_v1/CAP_HIT_AND_FULL_METRICS_ADDENDUM_v1.json"
    addendum_binding = next(r for r in historical_index["addendum"] if r["path"] == str(addendum_path))
    addendum = read(addendum_path, addendum_binding)
    original = read(ROOT/"feasibility/ce_recovery_ar_partial_evaluation_v1/generation_metrics.json", {"sha256":"cbbd4c672bf2b1de6ddeec0f8202822985f9cfc2ffb79c2caf695bd96b8644e5"})
    old_groups = {(r["method"],r["terminal"]):r for r in original["groups"]}
    carried_equal = {"/".join((r["method"],r["terminal"])):r == old_groups[(r["method"],r["terminal"])] for r in generation["groups"] if r["method"] != "MSE21_DDPM"}
    assert all(carried_equal.values())
    def gate_state(g): return "PASS" if g["pass"] else "FAIL"
    assert gates["reconstruction_ce_vs_matched_mse"]["status"] == gates["generation_ce_vs_terminal_ar"]["status"] == "EVALUATED"
    L, G = gate_state(gates["reconstruction_ce_vs_matched_mse"]), gate_state(gates["generation_ce_vs_terminal_ar"])
    preserved = [{**r,"unchanged":record(r["path"])["sha256"] == r["sha256"]} for r in inputs.values()]
    assert all(r["unchanged"] for r in preserved)
    facts = {"created_utc":datetime.now(timezone.utc).isoformat(),"scope":"author description of three completed arms only; no v4 three-arm readout executed",
        "original_protocol_E_L_G":"NOT_EVALUABLE","recovery_descriptive_L":L,"recovery_descriptive_G":G,"PROMOTE":"HOLD","AR_ADEQUACY":"NOT_ESTABLISHED",
        "three_arm_v4_contract_created":False,"three_arm_v4_executed":False,"readout_decision":decision,
        "scalar_availability":{"state":"RUNNING_REPORTED_BY_ROOT_NOT_INSPECTED","reported_start":"19:10:17","output_reads":0,"observations_created":0},
        "generation_metrics_full":generation,"reconstruction_metrics_full":reconstruction,"all_gate_subchecks":gates,"AR_teacher_forced_nll":artifacts["ar_teacher_forced_nll.json"],
        "engineering_receipts":[inputs[str(ROOT/"review/independent/CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json")],inputs[str(ROOT/"review/independent/MSE_V2_POSTFLIGHT_v1.json")]],
        "CE_MSE_coupling_review":mse_post["CE_MSE_coupling"],"MSE_costs":mse_post["costs"],
        "native_cap_semantics_from_bound_addendum": {k:addendum[k] for k in ("termination","AR_native_flag_audit","flag_scope_note","AR_fixed_checkpoint_monitor","costs")},
        "prior_CE_AR_full_groups_identical":carried_equal,"evaluation_manifest_verified_members":len(files),"inputs_unchanged":preserved,
        "evaluation_model_calls":0,"evaluation_test_accessed":False,"author_model_calls":0,"author_metric_recomputations":0,"source":record(__file__),
        "quality_changes_execution_authorization":False,"limits":["One fitted seed per method and reused development41; no significance or independent generalization claim.","CE uses a separately approved additional generation restart; original CE interruption remains and its total wall is unknown.","L supports the fixed-budget matched CE-versus-onehot-MSE comparison only; it does not establish mature-baseline superiority or novelty.","G remains a descriptive failure against the existing AR, whose adequacy remains unestablished and overfitting evidence remains disclosed.","Scalar outputs have not been inspected; its running status comes only from root. No full four-arm result or v4 readout is claimed.","No antimicrobial activity, safety or submission claim; old wetlab does not validate H5."]}
    OUT.mkdir()
    dump(OUT/"AUTHOR_FACTS.json",facts)
    lines=["# AMP H5：CE恢复、MSE与AR三臂作者读出", "",
        f"**恢复描述性 L={L}、G={G}；原正式 E/L/G 永久 NOT_EVALUABLE；AR_ADEQUACY=NOT_ESTABLISHED，PROMOTE=HOLD。**", "",
        "本次仅执行冻结评价器的三臂评价，96个请求全部保留。三臂v4合同未创建、v4读出未执行；等待scalar终态后才使用全部真实root。scalar状态仅引用主线程“19:10:17已启动”的陈述，没有stat/读取其运行输出，也没有创建scalar观测。预备的withheld路径映射没有采用、没有目录或假run。", "",
        "CE恢复与MSE各自独立工程回执均为PASS，MSE回执报告57检查及CE/MSE随机批次耦合核查通过；本件只转录这些独审结论，不把工程通过解释为质量成功。单seed、反复使用dev41及CE额外恢复attempt仍是限制；当前scalar许可在这些质量数值之前已确定。", "",
        "| 方法 | split | 对象数 | 内容micro | AA支持类macro | sequence mean | 长度MAE | exact整串 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in reconstruction['groups']:
        lines.append(f"| {r['method']} | {r['split']} | {r['rows']} | {r['rendered_nonpad_micro']:.6f} | {r['rendered_nonpad_macro']:.6f} | {r['per_sequence_mean_nonpad_accuracy']:.6f} | {r['length_mae']:.6f} | {r['whole_rendered_exact_count']} |")
    lines += ["", "| L子项（dev mean，CE对MSE） | CE | MSE | 固定方向 | 结果 |", "| --- | ---: | ---: | --- | --- |"]
    for k,c in gates['reconstruction_ce_vs_matched_mse']['checks'].items(): lines.append(f"| {k} | {c['left']:.6f} | {c['right']:.6f} | {c['operator']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", "L的4项均通过，但它只支持本固定预算匹配MSE比较；不是新颖性、成熟VAE最强基线优势或活性证明。生成方面，CE也未对MSE所有项占优：FREE合法5–48数量CE14/16低于MSE16/16。", "",
        "| 方法 | 端基 | 请求/合法1–48 | 合法5–48 | unique | 同端/任意train-copy | train MMD² | dev MMD² |", "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |"]
    for r in generation['groups']:
        lines.append(f"| {r['method']} | {r['terminal']} | {r['requests']}/{r['valid_1_48_count']} | {r['valid_5_48_count']} | {r['valid_1_48']['unique_count']} | {r['exact_copy_train_count_all_requests']}/{r['exact_sequence_copy_any_train_terminal_count']} | {r['valid_1_48_vs_train']['mmd2_linear_biased_fixed_features']:.6f} | {r['valid_1_48_vs_development']['mmd2_linear_biased_fixed_features']:.6f} |")
    lines += ["", "所有分布指标以合法1–48集合为分母，对应端基的train/dev为参照。MMD²采用冻结421维固定特征；JS单位bit；KS无p值。", "",
        "| 方法 | 端基 | 参照 | AA JS | AA L1 | 二肽JS | 二肽L1 | 长度KS |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in generation['groups']:
        for ref,key in (('train','valid_1_48_vs_train'),('dev','valid_1_48_vs_development')):
            d=r[key]; numbers=[d[k] for k in ('aa_jensen_shannon_bits','aa_l1_distance','dipeptide_jensen_shannon_bits','dipeptide_l1_distance','length_ks_statistic_no_pvalue')]
            lines.append('| '+' | '.join([r['method'],r['terminal'],ref]+[f'{x:.6f}' for x in numbers])+' |')
    lines += ["", "| 方法 | 端基 | pooled AA熵 | pooled二肽熵 | AA熵距train | 二肽熵距train | 长度均值[min,max] |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in generation['groups']:
        s=r['valid_1_48'];l=s['length']
        nums=[s['pooled_aa_entropy_bits'],s['pooled_dipeptide_entropy_bits'],r['pooled_aa_entropy_bits_absolute_gap_to_train'],r['pooled_dipeptide_entropy_bits_absolute_gap_to_train']]
        lines.append('| '+' | '.join([r['method'],r['terminal']]+[f'{x:.6f}' for x in nums]+[f"{l['mean']:.4f} [{l['min']},{l['max']}]"])+' |')
    lines += ["", "| G子项（CE对AR） | 端基 | CE | AR | 固定方向 | 结果 |", "| --- | --- | ---: | ---: | --- | --- |"]
    for terminal,checks in gates['generation_ce_vs_terminal_ar']['checks'].items():
        for k,c in checks.items(): lines.append(f"| {k} | {terminal} | {c['left']:.6f} | {c['right']:.6f} | {c['operator']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", "G仍有8项失败：两个端基dev MMD均未低于AR、合法5–48数量较少、AA及二肽熵与train的距离更大。所有CE/AR完整group与此前二臂结果逐值一致；本次没有为改善结果修改指标或门槛。", "",
        "AR原生cap语义继续有效：FREE_001为length48、terminated=false、max_length_hit=true但status=ok；CE FREE_009为length48/no_eos_at_limit。CE/AR均FREE触顶1/16、AMD0/16，不能横比status_ok为自然终止率；CE J/PAD与AR EOS机制不同。MSE本次两端最大长度22/17，没有48长度触顶。", "",
        "AR纯训练更新12.268115932秒、监测0.569787959秒、训练pipeline wall14.292998417秒、生成0.020189667秒、入口process15.109931750秒，范围嵌套不相加。AR dev NLL从200步2.615253升至固定1000步3.169793、train降至1.643124，保留过拟合证据与AR_ADEQUACY=NOT_ESTABLISHED。", "",
        f"MSE记录训练{mse_post['costs']['training_seconds']:.6f}秒、生成{mse_post['costs']['generation_seconds']:.6f}秒、process（final I/O前）{mse_post['costs']['process_seconds_before_final_io']:.6f}秒、supervisor outer{mse_post['costs']['supervisor_outer_seconds']:.6f}秒。CE原中断生成≥451.935743秒、原总wall未知、额外恢复生成1115.496170秒分别保留，不声称实际等计算。", "",
        "完整原生flag、AR10点曲线及成本语义来自已封口的[补充链](../CE_RECOVERY_AR_PARTIAL_WITH_ADDENDUM_INDEX_v1.json)。旧补充件当时只有两臂，故其历史L不可判；本三臂新计算得到的L=PASS不会覆盖旧包。", "",
        "完整所有指标、最近邻分布、全部门子项和来源hash见[AUTHOR_FACTS.json](AUTHOR_FACTS.json)；逐样本与逐重建行审计保留在[冻结评价包](../ce_recovery_mse_ar_evaluation_v1/MANIFEST.json)。"]
    lines += ['','- '+'\n- '.join(facts['limits'])]
    with (OUT/'AUTHOR_READOUT.zh.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    dump(OUT/'MANIFEST.json',{'source':record(__file__),'files':[record(p) for p in sorted(OUT.iterdir()) if p.is_file()], 'inputs':list(inputs.values()),'v4_executed':False,'original_protocol_E_L_G':'NOT_EVALUABLE','recovery_L':L,'recovery_G':G,'PROMOTE':'HOLD'})
    print(json.dumps({'output':str(OUT),'recovery_L':L,'recovery_G':G,'original_protocol_E_L_G':'NOT_EVALUABLE','v4_executed':False,'scalar_output_reads':0}))


if __name__ == '__main__':main()
