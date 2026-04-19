from tema.ml.cpp_profile import (
    build_cpp_hmm_profile_decision,
    resolve_cpp_hmm_profile,
)


def test_decision_keeps_defaults_without_real_improvement():
    baseline = {
        "run_id": "cpp-baseline",
        "metrics": {"sharpe": 1.14, "annual_return": 0.10, "annual_volatility": 0.09, "max_drawdown": -0.13},
        "hard_gate": {"passed": True},
        "injection_flags": {"strict_independent_mode": True, "benchmark_injection_detected": False},
    }
    sweep = {
        "best": {"ml_test_sharpe_mean": 0.71},
        "hard_gates": {"passed": True},
        "honest_defaults": {
            "template_default_universe": True,
            "template_use_precomputed_artifacts": False,
            "strict_independent_mode": True,
            "ml_meta_comparator_use_benchmark_csv": False,
            "parity_metrics_bridge": False,
        },
    }
    decision = build_cpp_hmm_profile_decision(baseline_summary=baseline, sweep_report=sweep)
    assert decision["apply_profile"] is False
    assert decision["gate_status"]["sharpe_improvement"] is False
    assert decision["recommended_params"]["hmm_trans_sticky"] == 0.92


def test_decision_applies_profile_when_improvement_and_gates_pass():
    baseline = {
        "run_id": "cpp-baseline",
        "metrics": {"sharpe": 0.50},
        "hard_gate": {"passed": True},
        "injection_flags": {"strict_independent_mode": True, "benchmark_injection_detected": False},
    }
    sweep = {
        "best": {"ml_test_sharpe_mean": 0.80},
        "hard_gates": {"passed": True},
        "honest_defaults": {
            "template_default_universe": True,
            "template_use_precomputed_artifacts": False,
            "strict_independent_mode": True,
            "ml_meta_comparator_use_benchmark_csv": False,
            "parity_metrics_bridge": False,
        },
    }
    decision = build_cpp_hmm_profile_decision(baseline_summary=baseline, sweep_report=sweep)
    assert decision["apply_profile"] is True
    assert decision["recommended_params"]["hmm_trans_sticky"] == 0.95


def test_unknown_cpp_profile_raises():
    try:
        resolve_cpp_hmm_profile("unknown-profile")
    except ValueError as exc:
        assert "Unknown cpp_hmm_profile" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown profile")
