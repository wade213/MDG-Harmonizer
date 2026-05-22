"""MDG-Harmonizer ablation runner（毕设论文消融实验编排脚本）。

围绕 baseline ``config/harmonization_day2night_mdg.json`` 派生 5 组变体，依次跑
``train -> test -> compute_baseline_metrics`` 并汇总到 ``ablation_summary.{json,md}``。

Ablation 矩阵
-------------
| 实验 ID    | 关键差异（相对 full MDG）          | 目的                |
|------------|------------------------------------|---------------------|
| A_full     | CDP + AFM + FB-Loss 全开           | 主结果              |
| B_no_cdp   | ``network.cdp_zero_vec=true``      | 验证 CDPNet 贡献    |
| C_no_afm   | ``network.disable_afm=true``       | 验证 AFM 贡献       |
| D_no_fb    | ``network.loss_weights.fb=0``      | 验证 FB-Loss 贡献   |
| E_baseline | inference-only，读已有 metrics     | 起点对照            |

A/B/C/D 从 baseline 770-epoch 预训练权重起步训 30 epoch；E 直接读
``experiments/test_harmonization_allinone_260506_123100/baseline_metrics.json``。

用法
----
::

    # 先 dry-run 确认派生命令正确（强烈推荐）
    .\\.venv\\Scripts\\python.exe -W ignore scripts/run_ablations.py --dry-run
    # 内置 self-test（不调子进程）
    .\\.venv\\Scripts\\python.exe -W ignore scripts/run_ablations.py --self-test
    # 真正跑（需 GPU 空闲）
    .\\.venv\\Scripts\\python.exe -W ignore scripts/run_ablations.py
    # 只跑某几组
    .\\.venv\\Scripts\\python.exe -W ignore scripts/run_ablations.py --only A_full,D_no_fb

设计要点
--------
- 不修改 baseline (``model_rihd.py`` / ``network_modified_backup.py``)。
- 每组 train config 用 ``name=mdg_ablation_<exp_id>``，``experiments/`` 目录隔离。
- 训练后扫 ``train_<name>_*`` 取最新目录，再从 ``checkpoint/`` 找最大 epoch。
- test config 把 ``pretrained_label`` 移除（默认走 ``MDGNetwork``），关闭 AMP/累积。
- 全过程 try/except：某组挂掉标 ``status=failed``，不影响其他组继续。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根目录：scripts/.. == D:/HCDM-master/HCDM-master
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
BASELINE_CFG = REPO_ROOT / "config" / "harmonization_day2night_mdg.json"
BASELINE_METRICS = (
    REPO_ROOT / "experiments" / "test_harmonization_allinone_260506_123100"
    / "baseline_metrics.json"
)
SUMMARY_DIR = REPO_ROOT / "experiments" / "ablation_summary"

# Ablation 表：dict[exp_id] = {description, type, ...}
ABLATIONS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("A_full",     {"description": "Full MDG (CDP + AFM + FB-Loss)", "type": "train"}),
    ("B_no_cdp",   {"description": "Disable CDPNet (zero deg_vec)", "type": "train"}),
    ("C_no_afm",   {"description": "Disable AFM (deg_vec=None)",   "type": "train"}),
    ("D_no_fb",    {"description": "FB-Loss weight = 0",            "type": "train"}),
    ("E_baseline", {"description": "Baseline (read precomputed metrics)", "type": "inference_only"}),
])


# ----------------------------------------------------------------------
# JSONC（带 // 注释的 JSON）解析；与 core/praser.py 同款逻辑
# ----------------------------------------------------------------------
def _load_jsonc(path: Path) -> Dict[str, Any]:
    """读取允许 ``//`` 行尾注释的 JSON 文件，返回 dict。"""
    txt = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            txt += line.split("//")[0] + "\n"
    return json.loads(txt)


def _dump_json(cfg: Dict[str, Any], path: Path) -> None:
    """格式化写出 JSON。派生 config 不再保留注释（无信息损失）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ----------------------------------------------------------------------
# config 派生
# ----------------------------------------------------------------------
def derive_train_config(baseline: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
    """从 baseline mdg config 派生某组 ablation 的 train config。"""
    cfg = copy.deepcopy(baseline)
    cfg["name"] = f"mdg_ablation_{exp_id}"
    network_args = cfg["model"]["which_networks"][0]["args"]

    if exp_id == "A_full":
        # 主结果：完全使用 baseline mdg 配置，不改任何字段
        pass
    elif exp_id == "B_no_cdp":
        # CDPNet 禁用：deg_vec 强制为 (B, deg_dim) 全零，AFM 仅剩 token + FiLM
        network_args["cdp_zero_vec"] = True
    elif exp_id == "C_no_afm":
        # AFM 禁用：deg_vec=None，UNet bottleneck 跳过 AFM，等价 baseline UNet
        network_args["disable_afm"] = True
    elif exp_id == "D_no_fb":
        # FB-Loss 禁用：仅留多尺度噪声 L1
        if "loss_weights" not in network_args:
            network_args["loss_weights"] = {}
        network_args["loss_weights"]["fb"] = 0.0
    else:
        raise ValueError(f"Unknown ablation id: {exp_id}")
    return cfg


def derive_test_config(
    train_cfg: Dict[str, Any],
    train_run_dir: Path,
    latest_epoch: int,
) -> Dict[str, Any]:
    """从某组 ablation 的 train config 派生其 test config。

    关键改动：
    - ``path.resume_state`` 指向训练产物 ``<train_run>/checkpoint/<epoch>``。
    - ``pretrained_label`` 移除，让 ``load_network`` 用 ``MDGNetwork`` 默认 label，
      与 ``save_network`` 写出的 ``<epoch>_MDGNetwork.pth`` 对得上。
    - 测试不需要 AMP / 梯度累积，关掉省心。
    """
    cfg = copy.deepcopy(train_cfg)
    # 注意：praser 会拼 "{resume_state}_MDGNetwork.pth"；此处只给前缀
    rel_resume = (train_run_dir / "checkpoint" / str(latest_epoch)).relative_to(REPO_ROOT)
    cfg["path"]["resume_state"] = str(rel_resume).replace("\\", "/")

    trainer_args = cfg["model"]["which_model"]["args"]
    trainer_args.pop("pretrained_label", None)
    trainer_args["use_amp"] = False
    trainer_args["gradient_accumulation_steps"] = 1
    return cfg


# ----------------------------------------------------------------------
# 训练 / 测试 / 指标 编排
# ----------------------------------------------------------------------
def find_latest_train_run(name: str, mtime_floor: float = 0.0) -> Optional[Path]:
    """在 ``experiments/`` 中找 ``train_<name>_<ts>`` 命名最新的目录。

    ``mtime_floor`` 让我们只选「在脚本开始之后才创建」的目录，避免拿到旧的同名 run。
    """
    pat = re.compile(rf"^train_{re.escape(name)}_\d{{6}}_\d{{6}}$")
    candidates: List[Path] = []
    exp_root = REPO_ROOT / "experiments"
    if not exp_root.exists():
        return None
    for d in exp_root.iterdir():
        if not d.is_dir():
            continue
        if pat.match(d.name) and d.stat().st_mtime >= mtime_floor:
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_checkpoint(train_run_dir: Path) -> Optional[int]:
    """扫 ``train_run/checkpoint/`` 下 ``<epoch>_MDGNetwork.pth``，返回最大 epoch。"""
    ckpt_dir = train_run_dir / "checkpoint"
    if not ckpt_dir.exists():
        return None
    pat = re.compile(r"^(\d+)_MDGNetwork\.pth$")
    epochs: List[int] = []
    for f in ckpt_dir.iterdir():
        m = pat.match(f.name)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _build_run_cmd(config_path: Path, phase: str) -> List[str]:
    """构造 ``run.py`` 的标准调用命令。

    所有 Python 调用必须走项目内 venv，不能走系统 Python（会找不到 torch）。
    """
    return [
        str(PYTHON_EXE), "-W", "ignore", "run.py",
        "-p", phase,
        "-c", str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]


def _build_metrics_cmd(test_run_dir: Path, out_json: Path) -> List[str]:
    return [
        str(PYTHON_EXE), "-W", "ignore", "scripts/compute_baseline_metrics.py",
        "--run", str(test_run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "--out-json", str(out_json),
    ]


def run_subprocess(
    cmd: List[str],
    log_file: Path,
    dry_run: bool,
    label: str,
) -> int:
    """在 REPO_ROOT 下执行子进程，stdout/stderr 重定向到 log_file。"""
    print(f"  [{label}] $ {' '.join(cmd)}")
    print(f"  [{label}]   log -> {log_file}")
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as logf:
        # 直接 cwd=REPO_ROOT；run.py 内有大量相对路径假设
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=logf)
    return int(proc.returncode)


# ----------------------------------------------------------------------
# 单次 ablation 执行
# ----------------------------------------------------------------------
def run_one_ablation(
    exp_id: str,
    baseline_cfg: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    """跑一组 ablation：派生 config -> train -> test -> 算指标。"""
    info: Dict[str, Any] = {
        "exp_id": exp_id,
        "description": ABLATIONS[exp_id]["description"],
        "status": "pending",
    }

    # E_baseline 走 inference-only 分支：直接读已落盘的 baseline_metrics.json
    if ABLATIONS[exp_id]["type"] == "inference_only":
        if dry_run:
            print(f"  [E] read precomputed metrics from {BASELINE_METRICS}")
            info.update(status="ok", metrics={"_dry_run": True})
            return info
        if not BASELINE_METRICS.exists():
            info.update(status="failed", error=f"baseline metrics not found: {BASELINE_METRICS}")
            return info
        info["metrics"] = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
        info.update(status="ok", source=str(BASELINE_METRICS))
        return info

    # A/B/C/D：训练流程
    train_cfg = derive_train_config(baseline_cfg, exp_id)
    train_cfg_path = REPO_ROOT / "config" / f"ablation_{exp_id}_train.json"
    print(f"  derive train config -> {train_cfg_path}")
    if not dry_run:
        _dump_json(train_cfg, train_cfg_path)
    info["train_config"] = str(train_cfg_path.relative_to(REPO_ROOT))

    log_dir = REPO_ROOT / "experiments" / f"ablation_{exp_id}"
    train_log = log_dir / "train.log"
    t0 = time.time()
    rc = run_subprocess(_build_run_cmd(train_cfg_path, "train"), train_log, dry_run, f"{exp_id}/train")
    if rc != 0:
        info.update(status="failed", error=f"training rc={rc}, see {train_log}")
        return info

    if dry_run:
        train_run_dir = REPO_ROOT / "experiments" / f"train_mdg_ablation_{exp_id}_<ts>"
        latest_epoch: Any = "<epoch>"
    else:
        train_run_dir = find_latest_train_run(f"mdg_ablation_{exp_id}", mtime_floor=t0)
        if train_run_dir is None:
            info.update(status="failed", error=f"no train_run_dir found for {exp_id}")
            return info
        latest_epoch = find_latest_checkpoint(train_run_dir)
        if latest_epoch is None:
            info.update(status="failed", error=f"no checkpoint in {train_run_dir}")
            return info
    info.update(train_run_dir=str(train_run_dir), latest_epoch=latest_epoch)

    # 派生 test config
    test_cfg_path = REPO_ROOT / "config" / f"ablation_{exp_id}_test.json"
    print(f"  derive test config  -> {test_cfg_path} (resume @ epoch {latest_epoch})")
    if not dry_run:
        test_cfg = derive_test_config(train_cfg, train_run_dir, int(latest_epoch))
        _dump_json(test_cfg, test_cfg_path)
    info["test_config"] = str(test_cfg_path.relative_to(REPO_ROOT))

    test_log = log_dir / "test.log"
    t1 = time.time()
    rc = run_subprocess(_build_run_cmd(test_cfg_path, "test"), test_log, dry_run, f"{exp_id}/test")
    if rc != 0:
        info.update(status="failed", error=f"test rc={rc}, see {test_log}")
        return info

    if dry_run:
        test_run_dir = REPO_ROOT / "experiments" / f"test_mdg_ablation_{exp_id}_<ts>"
    else:
        pat = re.compile(rf"^test_mdg_ablation_{re.escape(exp_id)}_\d{{6}}_\d{{6}}$")
        cands = [d for d in (REPO_ROOT / "experiments").iterdir()
                 if d.is_dir() and pat.match(d.name) and d.stat().st_mtime >= t1]
        if not cands:
            info.update(status="failed", error=f"no test_run_dir found for {exp_id}")
            return info
        test_run_dir = max(cands, key=lambda p: p.stat().st_mtime)
    info["test_run_dir"] = str(test_run_dir)

    metrics_out = test_run_dir / "ablation_metrics.json"
    rc = run_subprocess(
        _build_metrics_cmd(test_run_dir, metrics_out), log_dir / "metrics.log",
        dry_run, f"{exp_id}/metrics",
    )
    if rc != 0:
        info.update(status="failed", error=f"metrics rc={rc}, see {log_dir / 'metrics.log'}")
        return info
    info["metrics"] = ({"_dry_run": True} if dry_run
                       else json.loads(metrics_out.read_text(encoding="utf-8")))
    info["status"] = "ok"
    return info


# ----------------------------------------------------------------------
# 汇总输出
# ----------------------------------------------------------------------
_METRIC_KEYS = ["MAE", "MSE", "PSNR", "SSIM", "fMAE", "fPSNR"]


def write_summary(results: List[Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
    """把所有 ablation 结果写到 ``ablation_summary.{json,md}``。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_summary.json"
    md_path = out_dir / "ablation_summary.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = ["exp_id", "description", "status"] + _METRIC_KEYS
    lines = [
        "# MDG-Harmonizer Ablation Summary",
        "",
        f"_generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in results:
        m = r.get("metrics") or {}
        row = [r["exp_id"], r["description"], r["status"]] + [_fmt(m.get(k)) for k in _METRIC_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MDG-Harmonizer ablation runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不真的跑训练/测试")
    parser.add_argument("--only", default="",
                        help="逗号分隔的实验 ID 子集，如 'A_full,D_no_fb'")
    parser.add_argument("--baseline-config", default=str(BASELINE_CFG),
                        help="baseline mdg config 路径")
    parser.add_argument("--summary-out", default=str(SUMMARY_DIR),
                        help="汇总输出目录")
    parser.add_argument("--self-test", action="store_true",
                        help="运行内置 dry-run + assert 自检，不调子进程")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    exp_ids = [eid for eid in ABLATIONS.keys() if (not only or eid in only)]
    if not exp_ids:
        print(f"!! --only={args.only} 没匹配到任何 ablation；可用 IDs: {list(ABLATIONS)}")
        return 2

    baseline_cfg = _load_jsonc(Path(args.baseline_config))
    print("=" * 70)
    print(f"MDG Ablation Runner ({'DRY-RUN' if args.dry_run else 'LIVE'})")
    print(f"  baseline: {args.baseline_config}")
    print(f"  exp_ids:  {exp_ids}")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    for exp_id in exp_ids:
        print(f"\n--- {exp_id} : {ABLATIONS[exp_id]['description']} ---")
        try:
            r = run_one_ablation(exp_id, baseline_cfg, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001 — 单组失败不影响其他组
            r = {"exp_id": exp_id, "description": ABLATIONS[exp_id]["description"],
                 "status": "failed", "error": f"unhandled exception: {e!r}"}
        print(f"  -> status={r['status']}", end="")
        if "metrics" in r and isinstance(r["metrics"], dict):
            mae = r["metrics"].get("MAE")
            psnr = r["metrics"].get("PSNR")
            if mae is not None and psnr is not None:
                print(f"  MAE={mae:.4f}  PSNR={psnr:.4f}", end="")
        print()
        results.append(r)

    out_dir = Path(args.summary_out)
    if not args.dry_run:
        json_path, md_path = write_summary(results, out_dir)
        print(f"\nWrote summary: {json_path}\n               {md_path}")
    else:
        print("\n(dry-run: 跳过 summary 落盘)")

    return 0 if all(r["status"] == "ok" for r in results) else 1


# ----------------------------------------------------------------------
# 自检：脚本可 import + 五组 dry-run 全部派生成功
# ----------------------------------------------------------------------
def _self_test() -> int:
    """内置 self-test：纯逻辑校验，不调任何 subprocess、不写永久文件。"""
    print("[self-test] loading baseline config ...")
    baseline = _load_jsonc(BASELINE_CFG)
    assert baseline["name"] == "mdg_harmonizer_day2night"

    print("[self-test] derive_train_config for each exp_id ...")
    for exp_id in ["A_full", "B_no_cdp", "C_no_afm", "D_no_fb"]:
        cfg = derive_train_config(baseline, exp_id)
        net_args = cfg["model"]["which_networks"][0]["args"]
        assert cfg["name"] == f"mdg_ablation_{exp_id}"
        if exp_id == "A_full":
            assert "cdp_zero_vec" not in net_args and "disable_afm" not in net_args
        elif exp_id == "B_no_cdp":
            assert net_args["cdp_zero_vec"] is True
        elif exp_id == "C_no_afm":
            assert net_args["disable_afm"] is True
        elif exp_id == "D_no_fb":
            assert net_args["loss_weights"]["fb"] == 0.0
        print(f"  {exp_id} ok")

    print("[self-test] derive_test_config with fake train_run/epoch ...")
    fake_train = REPO_ROOT / "experiments" / "train_mdg_ablation_A_full_260507_010203"
    test_cfg = derive_test_config(derive_train_config(baseline, "A_full"), fake_train, 30)
    trainer_args = test_cfg["model"]["which_model"]["args"]
    assert "pretrained_label" not in trainer_args and trainer_args["use_amp"] is False
    assert test_cfg["path"]["resume_state"].endswith(
        "experiments/train_mdg_ablation_A_full_260507_010203/checkpoint/30"
    )

    print("[self-test] summary writer roundtrip ...")
    fake_results = [
        {"exp_id": eid, "description": ABLATIONS[eid]["description"],
         "status": "ok", "metrics": {"MAE": 1.23, "PSNR": 35.0}}
        for eid in ABLATIONS
    ]
    tmp_dir = REPO_ROOT / "experiments" / "_ablation_summary_selftest"
    json_path, md_path = write_summary(fake_results, tmp_dir)
    md_text = md_path.read_text(encoding="utf-8")
    assert "| A_full" in md_text and "| E_baseline" in md_text
    json_path.unlink(); md_path.unlink(); tmp_dir.rmdir()

    print("\n[self-test] PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
