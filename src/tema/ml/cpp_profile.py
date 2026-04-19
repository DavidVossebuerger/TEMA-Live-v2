from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CPP_HMM_PROFILE_PRESETS: dict[str, dict[str, float | int]] = {
    # Derived from cpp-param-sweep-run best candidate.
    "sweep-optimized-v1": {
        "hmm_n_states": 2,
        "hmm_n_iter": 30,
        "hmm_var_floor": 1e-8,
        "hmm_trans_sticky": 0.95,
        "rf_n_estimators": 400,
        "rf_max_depth": 4,
    }
}

CPP_HMM_SAFE_DEFAULT_PARAMS: dict[str, float | int] = {
    "hmm_n_states": 2,
    "hmm_n_iter": 30,
    "hmm_var_floor": 1e-8,
    "hmm_trans_sticky": 0.92,
    "rf_n_estimators": 400,
    "rf_max_depth": 4,
}

HONEST_SWEEP_DEFAULTS: dict[str, bool] = {
    "template_default_universe": True,
    "template_use_precomputed_artifacts": False,
    "strict_independent_mode": True,
    "ml_meta_comparator_use_benchmark_csv": False,
    "parity_metrics_bridge": False,
}


def available_cpp_hmm_profiles() -> tuple[str, ...]:
    return tuple(sorted(CPP_HMM_PROFILE_PRESETS))


def resolve_cpp_hmm_profile(profile_name: str) -> dict[str, float | int]:
    key = str(profile_name).strip()
    if key not in CPP_HMM_PROFILE_PRESETS:
        allowed = ", ".join(available_cpp_hmm_profiles())
        raise ValueError(f"Unknown cpp_hmm_profile '{profile_name}'. Available: {allowed}")
    return dict(CPP_HMM_PROFILE_PRESETS[key])


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_candidate_metrics_from_runs(sweep_runs_csv_path: str | Path, best: dict[str, Any]) -> dict[str, float | None]:
    path = Path(sweep_runs_csv_path)
    if not path.exists() or not best:
        return {}

    keys = ("hmm_n_states", "hmm_n_iter", "hmm_var_floor", "hmm_trans_sticky", "rf_n_estimators", "rf_max_depth")
    best_values = {k: _to_float(best.get(k)) for k in keys}
    if any(v is None for v in best_values.values()):
        return {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        matched = True
        for k in keys:
            row_v = _to_float(row.get(k))
            ref_v = best_values[k]
            if row_v is None or ref_v is None or abs(row_v - ref_v) > 1e-12:
                matched = False
                break
        if matched:
            return {
                "sharpe": _to_float(row.get("sharpe")),
                "annual_return": _to_float(row.get("annual_return")),
                "annual_volatility": _to_float(row.get("annual_volatility")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
                "ml_test_sharpe": _to_float(row.get("ml_test_sharpe")),
            }
    return {}


def build_cpp_hmm_profile_decision(
    *,
    baseline_summary: dict[str, Any],
    sweep_report: dict[str, Any],
    sweep_runs_csv_path: str | Path | None = None,
    candidate_profile_name: str = "sweep-optimized-v1",
) -> dict[str, Any]:
    baseline_metrics = baseline_summary.get("metrics", {}) if isinstance(baseline_summary, dict) else {}
    baseline_sharpe = _to_float(baseline_metrics.get("sharpe"))

    best = sweep_report.get("best", {}) if isinstance(sweep_report, dict) else {}
    candidate_metrics = {
        "sharpe": _to_float(best.get("ml_test_sharpe_mean")),
        "annual_return": _to_float(best.get("annual_return_mean")),
        "annual_volatility": _to_float(best.get("annual_volatility_mean")),
        "max_drawdown": _to_float(best.get("max_drawdown_mean")),
    }
    if sweep_runs_csv_path:
        candidate_metrics = {**candidate_metrics, **_find_candidate_metrics_from_runs(sweep_runs_csv_path, best)}
        # keep best-derived sharpe as fallback only.
        if candidate_metrics.get("sharpe") is None:
            candidate_metrics["sharpe"] = _to_float(best.get("ml_test_sharpe_mean"))

    candidate_sharpe = _to_float(candidate_metrics.get("sharpe"))
    sharpe_delta = (
        (candidate_sharpe - baseline_sharpe)
        if (baseline_sharpe is not None and candidate_sharpe is not None)
        else None
    )
    sharpe_improvement = bool(sharpe_delta is not None and sharpe_delta > 0.0)

    baseline_hard_gate_passed = bool(
        baseline_summary.get("hard_gate", {}).get("passed", False)
        if isinstance(baseline_summary, dict)
        else False
    )
    sweep_hard_gates_passed = bool(
        sweep_report.get("hard_gates", {}).get("passed", False) if isinstance(sweep_report, dict) else False
    )
    honest_defaults = sweep_report.get("honest_defaults", {}) if isinstance(sweep_report, dict) else {}
    sweep_honest_defaults_passed = bool(
        isinstance(honest_defaults, dict)
        and all(honest_defaults.get(k) is v for k, v in HONEST_SWEEP_DEFAULTS.items())
    )
    baseline_flags = baseline_summary.get("injection_flags", {}) if isinstance(baseline_summary, dict) else {}
    baseline_honest_passed = bool(
        baseline_flags.get("strict_independent_mode", False) and not baseline_flags.get("benchmark_injection_detected", True)
    )

    gate_status = {
        "baseline_hard_gate_passed": baseline_hard_gate_passed,
        "sweep_hard_gates_passed": sweep_hard_gates_passed,
        "sweep_honest_defaults_passed": sweep_honest_defaults_passed,
        "baseline_honest_passed": baseline_honest_passed,
        "sharpe_improvement": sharpe_improvement,
    }
    apply_profile = bool(all(gate_status.values()))

    if apply_profile:
        reason = "Candidate beats baseline Sharpe and all hard/honest gates passed."
        recommended_params = resolve_cpp_hmm_profile(candidate_profile_name)
    else:
        reason = "Keep baseline/defaults: no honest Sharpe improvement and/or gate failure."
        recommended_params = dict(CPP_HMM_SAFE_DEFAULT_PARAMS)

    return {
        "baseline_artifact": str(baseline_summary.get("run_id", "cpp-baseline")),
        "candidate_profile": str(candidate_profile_name),
        "baseline_metrics": {
            "sharpe": baseline_sharpe,
            "annual_return": _to_float(baseline_metrics.get("annual_return")),
            "annual_volatility": _to_float(baseline_metrics.get("annual_volatility")),
            "max_drawdown": _to_float(baseline_metrics.get("max_drawdown")),
        },
        "candidate_metrics": candidate_metrics,
        "sharpe_delta": sharpe_delta,
        "gate_status": gate_status,
        "apply_profile": apply_profile,
        "recommended_params": recommended_params,
        "reason": reason,
    }


def write_cpp_hmm_profile_decision(
    *,
    baseline_summary_path: str | Path,
    sweep_report_path: str | Path,
    sweep_runs_csv_path: str | Path,
    output_path: str | Path,
    candidate_profile_name: str = "sweep-optimized-v1",
) -> dict[str, Any]:
    baseline_path = Path(baseline_summary_path)
    sweep_path = Path(sweep_report_path)
    runs_path = Path(sweep_runs_csv_path)
    out_path = Path(output_path)

    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    sweep_report = json.loads(sweep_path.read_text(encoding="utf-8"))
    decision = build_cpp_hmm_profile_decision(
        baseline_summary=baseline_summary,
        sweep_report=sweep_report,
        sweep_runs_csv_path=runs_path,
        candidate_profile_name=candidate_profile_name,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision
