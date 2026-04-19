from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DECISION_POLICY: dict[str, Any] = {
    "min_sharpe_delta": 0.0,
    "min_annual_return_delta": 0.0,
    "max_annual_volatility_increase": 0.0,
    "max_drawdown_worsening": 0.0,
    "max_turnover_increase": 0.0,
    "require_available_hard_gates_to_pass": True,
}

_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "sharpe": ("sharpe", "sharpe_ratio", "annualized_sharpe", "test_sharpe"),
    "annual_return": ("annual_return", "annualized_return", "cagr", "test_annual_return"),
    "annual_volatility": ("annual_volatility", "annualized_volatility", "annual_vol"),
    "max_drawdown": ("max_drawdown", "mdd", "max_dd"),
    "turnover": ("annualized_turnover", "turnover", "annual_turnover", "turnover_proxy"),
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _pick_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
    return {name: _pick_metric(payload, aliases) for name, aliases in _METRIC_KEYS.items()}


def _first_metrics_payload(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", [])
    candidates: list[Path] = []
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, str):
                candidates.append(run_dir / f"{artifact}.json")
    for file_name in ("performance.json", "metrics.json", "stats.json"):
        p = run_dir / file_name
        if p not in candidates:
            candidates.append(p)
    for candidate in candidates:
        if candidate.exists():
            payload = _load_json(candidate)
            metrics = _extract_metrics_from_payload(payload)
            if any(v is not None for v in metrics.values()):
                return payload
    return {}


def collect_validation_gates(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "validation_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        hard_gate = summary.get("hard_gate", {}) if isinstance(summary, dict) else {}
        checks = hard_gate.get("checks", {}) if isinstance(hard_gate, dict) else {}
        thresholds = hard_gate.get("thresholds", {}) if isinstance(hard_gate, dict) else {}
        values = hard_gate.get("values", {}) if isinstance(hard_gate, dict) else {}
        return {
            "oos": {
                "available": "oos_passed" in checks,
                "passed": checks.get("oos_passed"),
                "value": checks.get("oos_passed"),
                "threshold": True,
            },
            "psr": {
                "available": values.get("target_psr") is not None,
                "passed": checks.get("psr_threshold"),
                "value": values.get("target_psr"),
                "threshold": thresholds.get("psr_threshold"),
            },
            "dsr": {
                "available": values.get("target_dsr") is not None,
                "passed": checks.get("dsr_threshold"),
                "value": values.get("target_dsr"),
                "threshold": thresholds.get("dsr_threshold"),
            },
            "pbo": {
                "available": values.get("pbo") is not None,
                "passed": checks.get("pbo_max"),
                "value": values.get("pbo"),
                "threshold": thresholds.get("pbo_max"),
            },
        }

    oos_report = _load_json(run_dir / "oos_report.json") if (run_dir / "oos_report.json").exists() else {}
    psr_report = (
        _load_json(run_dir / "probabilistic_sharpe_ml.json")
        if (run_dir / "probabilistic_sharpe_ml.json").exists()
        else _load_json(run_dir / "probabilistic_sharpe_baseline.json")
        if (run_dir / "probabilistic_sharpe_baseline.json").exists()
        else {}
    )
    cpcv_report = _load_json(run_dir / "cpcv_report.json") if (run_dir / "cpcv_report.json").exists() else {}
    pbo_payload = cpcv_report.get("pbo") if isinstance(cpcv_report, dict) else None
    pbo_value = pbo_payload.get("pbo") if isinstance(pbo_payload, dict) else None
    return {
        "oos": {
            "available": "passed" in oos_report,
            "passed": oos_report.get("passed"),
            "value": oos_report.get("passed"),
            "threshold": True,
        },
        "psr": {
            "available": isinstance(psr_report.get("psr"), dict) and psr_report["psr"].get("psr") is not None,
            "passed": None,
            "value": psr_report.get("psr", {}).get("psr") if isinstance(psr_report.get("psr"), dict) else None,
            "threshold": None,
        },
        "dsr": {
            "available": isinstance(psr_report.get("dsr"), dict) and psr_report["dsr"].get("dsr") is not None,
            "passed": None,
            "value": psr_report.get("dsr", {}).get("dsr") if isinstance(psr_report.get("dsr"), dict) else None,
            "threshold": None,
        },
        "pbo": {
            "available": isinstance(pbo_value, (int, float)),
            "passed": None,
            "value": pbo_value,
            "threshold": None,
        },
    }


def collect_run_summary(run_dir: str | Path, *, run_id: str, config_summary: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _load_json(run_path / "manifest.json") if (run_path / "manifest.json").exists() else {}
    perf_payload = _first_metrics_payload(run_path, manifest)
    metrics = _extract_metrics_from_payload(perf_payload)
    benchmark_injection = bool(perf_payload.get("benchmark_injection_detected", False))
    strict_independent_mode = perf_payload.get("strict_independent_mode")
    return {
        "run_id": str(run_id),
        "run_dir": str(run_path),
        "config_summary": config_summary,
        "metrics": metrics,
        "validation_hard_gates": collect_validation_gates(run_path),
        "strict_independent_mode": strict_independent_mode,
        "benchmark_injection_detected": benchmark_injection,
    }


def compute_metric_deltas(
    champion_metrics: dict[str, float | None], challenger_metrics: dict[str, float | None]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in _METRIC_KEYS:
        champion_value = champion_metrics.get(metric)
        challenger_value = challenger_metrics.get(metric)
        if champion_value is None or challenger_value is None:
            out[metric] = None
        else:
            out[metric] = float(challenger_value) - float(champion_value)
    return out


def _as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def decide_keep_challenger(
    *,
    champion_summary: dict[str, Any],
    challenger_summary: dict[str, Any],
    decision_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(DEFAULT_DECISION_POLICY)
    if decision_policy:
        policy.update(decision_policy)

    champion_metrics = champion_summary.get("metrics", {})
    challenger_metrics = challenger_summary.get("metrics", {})
    deltas = compute_metric_deltas(champion_metrics, challenger_metrics)
    reasons: list[str] = []

    sharpe_delta = deltas.get("sharpe")
    if sharpe_delta is None:
        reasons.append("missing sharpe delta")
    elif sharpe_delta < float(policy["min_sharpe_delta"]):
        reasons.append(f"sharpe delta {sharpe_delta:.6f} below minimum {float(policy['min_sharpe_delta']):.6f}")

    annual_return_delta = deltas.get("annual_return")
    if annual_return_delta is not None and annual_return_delta < float(policy["min_annual_return_delta"]):
        reasons.append(
            f"annual_return delta {annual_return_delta:.6f} below minimum {float(policy['min_annual_return_delta']):.6f}"
        )

    annual_vol_delta = deltas.get("annual_volatility")
    if annual_vol_delta is not None and annual_vol_delta > float(policy["max_annual_volatility_increase"]):
        reasons.append(
            f"annual_volatility increase {annual_vol_delta:.6f} above maximum {float(policy['max_annual_volatility_increase']):.6f}"
        )

    champion_mdd = champion_metrics.get("max_drawdown")
    challenger_mdd = challenger_metrics.get("max_drawdown")
    if champion_mdd is not None and challenger_mdd is not None:
        mdd_worsening = abs(float(challenger_mdd)) - abs(float(champion_mdd))
        if mdd_worsening > float(policy["max_drawdown_worsening"]):
            reasons.append(
                f"max_drawdown worsening {mdd_worsening:.6f} above maximum {float(policy['max_drawdown_worsening']):.6f}"
            )

    turnover_delta = deltas.get("turnover")
    if turnover_delta is not None and turnover_delta > float(policy["max_turnover_increase"]):
        reasons.append(
            f"turnover increase {turnover_delta:.6f} above maximum {float(policy['max_turnover_increase']):.6f}"
        )

    gates = challenger_summary.get("validation_hard_gates", {})
    if bool(policy["require_available_hard_gates_to_pass"]):
        for gate_name in ("oos", "psr", "dsr", "pbo"):
            gate_payload = gates.get(gate_name, {})
            available = bool(gate_payload.get("available", False))
            passed = _as_bool_or_none(gate_payload.get("passed"))
            if available and passed is False:
                reasons.append(f"{gate_name} hard gate failed")

    if challenger_summary.get("benchmark_injection_detected") is True:
        reasons.append("benchmark injection detected in challenger run")

    keep = len(reasons) == 0
    if keep:
        reasons.append("challenger accepted by configured ablation policy")

    return {
        "keep_challenger": keep,
        "rollback_to_champion": (not keep),
        "reasons": reasons,
        "policy": policy,
        "metric_deltas": deltas,
    }


def build_pair_report(
    *,
    champion_summary: dict[str, Any],
    challenger_summary: dict[str, Any],
    decision_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = decide_keep_challenger(
        champion_summary=champion_summary,
        challenger_summary=challenger_summary,
        decision_policy=decision_policy,
    )
    return {
        "champion": champion_summary,
        "challenger": challenger_summary,
        "metric_deltas": decision["metric_deltas"],
        "validation_hard_gates": {
            "champion": champion_summary.get("validation_hard_gates", {}),
            "challenger": challenger_summary.get("validation_hard_gates", {}),
        },
        "decision": {
            "keep_challenger": bool(decision["keep_challenger"]),
            "rollback_to_champion": bool(decision["rollback_to_champion"]),
            "reasons": list(decision["reasons"]),
            "policy": decision["policy"],
        },
    }

