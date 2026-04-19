#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_pipeline import run_modular
from tema.validation.ablation import build_pair_report, collect_run_summary


def _parse_json_dict(raw: str, arg_name: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return payload


def _safe_label(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-")
    return token or "challenger"


def _enforce_strict_independent(cfg: dict[str, Any], strict_independent: bool) -> dict[str, Any]:
    out = dict(cfg)
    out.pop("run_id", None)
    out.pop("out_root", None)
    if strict_independent:
        if out.get("parity_metrics_bridge") is True:
            raise ValueError("strict-independent ablation forbids parity_metrics_bridge")
        if out.get("ml_meta_comparator_use_benchmark_csv") is True:
            raise ValueError("strict-independent ablation forbids benchmark comparator injection")
        if "strict_independent_mode" in out and out["strict_independent_mode"] is False:
            raise ValueError("strict-independent ablation cannot disable strict_independent_mode")
        out["strict_independent_mode"] = True
    return out


def _run_single(run_id: str, out_root: str, cfg: dict[str, Any], label: str) -> dict[str, Any]:
    result = run_modular(run_id=run_id, out_root=out_root, **cfg)
    run_dir = Path(result["out_dir"])
    summary = collect_run_summary(
        run_dir,
        run_id=run_id,
        config_summary={"label": label, "config": cfg},
    )
    return summary


def run(
    *,
    run_id: str,
    out_root: str,
    champion_cfg: dict[str, Any],
    challenger_cfgs: list[dict[str, Any]],
    strict_independent: bool = True,
    decision_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_dir = Path(out_root) / run_id
    root_dir.mkdir(parents=True, exist_ok=True)

    champion_cfg = _enforce_strict_independent(champion_cfg, strict_independent)
    champion_run_id = f"{run_id}__champion"
    champion_summary = _run_single(champion_run_id, out_root, champion_cfg, label="champion")

    pair_reports: list[dict[str, Any]] = []
    for idx, raw_cfg in enumerate(challenger_cfgs):
        cfg = dict(raw_cfg)
        name = _safe_label(str(cfg.pop("name", f"challenger_{idx+1}")))
        cfg = _enforce_strict_independent(cfg, strict_independent)
        challenger_run_id = f"{run_id}__{name}"
        challenger_summary = _run_single(challenger_run_id, out_root, cfg, label=name)
        pair_reports.append(
            build_pair_report(
                champion_summary=champion_summary,
                challenger_summary=challenger_summary,
                decision_policy=decision_policy,
            )
        )

    report = {
        "run_id": run_id,
        "out_root": out_root,
        "strict_independent_mode": bool(strict_independent),
        "champion_run_id": champion_summary["run_id"],
        "n_challengers": len(pair_reports),
        "pairs": pair_reports,
    }
    report_path = root_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run champion-vs-challenger ablations with structured decisions")
    parser.add_argument("--run-id", default="ablation")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--champion-config-json", default="{}", help="JSON dict kwargs for champion run_modular")
    parser.add_argument(
        "--challenger-config-json",
        action="append",
        default=[],
        help="JSON dict kwargs for challenger run_modular (repeat for multiple challengers, optional key 'name')",
    )
    parser.add_argument("--strict-independent", dest="strict_independent", action="store_true")
    parser.add_argument("--no-strict-independent", dest="strict_independent", action="store_false")
    parser.set_defaults(strict_independent=True)
    parser.add_argument("--min-sharpe-delta", type=float, default=0.0)
    parser.add_argument("--min-annual-return-delta", type=float, default=0.0)
    parser.add_argument("--max-annual-volatility-increase", type=float, default=0.0)
    parser.add_argument("--max-drawdown-worsening", type=float, default=0.0)
    parser.add_argument("--max-turnover-increase", type=float, default=0.0)
    parser.add_argument("--require-available-hard-gates-to-pass", action="store_true", default=True)
    parser.add_argument("--allow-missing-hard-gates", action="store_true")
    parser.add_argument("--enforce-keep", action="store_true", help="Exit non-zero if any challenger is rejected")
    args = parser.parse_args(argv)

    if not args.challenger_config_json:
        raise SystemExit("At least one --challenger-config-json is required")

    champion_cfg = _parse_json_dict(args.champion_config_json, "--champion-config-json")
    challengers = [_parse_json_dict(raw, "--challenger-config-json") for raw in args.challenger_config_json]
    decision_policy = {
        "min_sharpe_delta": float(args.min_sharpe_delta),
        "min_annual_return_delta": float(args.min_annual_return_delta),
        "max_annual_volatility_increase": float(args.max_annual_volatility_increase),
        "max_drawdown_worsening": float(args.max_drawdown_worsening),
        "max_turnover_increase": float(args.max_turnover_increase),
        "require_available_hard_gates_to_pass": (
            False if bool(args.allow_missing_hard_gates) else bool(args.require_available_hard_gates_to_pass)
        ),
    }

    report = run(
        run_id=str(args.run_id),
        out_root=str(args.out_root),
        champion_cfg=champion_cfg,
        challenger_cfgs=challengers,
        strict_independent=bool(args.strict_independent),
        decision_policy=decision_policy,
    )
    print(json.dumps(report))

    if bool(args.enforce_keep):
        any_rejected = any(not bool(pair["decision"]["keep_challenger"]) for pair in report.get("pairs", []))
        if any_rejected:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

