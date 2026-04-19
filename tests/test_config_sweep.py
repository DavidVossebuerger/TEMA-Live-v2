import pandas as pd

from tema.validation.sweep import build_heatmap, pick_best_config, pick_best_config_with_hard_gate, summarize_sweep


def test_summarize_sweep_and_heatmap_and_best():
    per_run = pd.DataFrame(
        [
            {
                "seed": 42,
                "vol_target_annual": 0.10,
                "ml_position_scalar_max": 10.0,
                "base_train_sharpe": 0.5,
                "base_test_sharpe": 0.4,
                "ml_train_sharpe": 0.7,
                "ml_test_sharpe": 0.6,
                "runtime_s": 1.0,
                "base_overfit_gap": 0.1,
                "ml_overfit_gap": 0.1,
            },
            {
                "seed": 43,
                "vol_target_annual": 0.10,
                "ml_position_scalar_max": 10.0,
                "base_train_sharpe": 0.6,
                "base_test_sharpe": 0.5,
                "ml_train_sharpe": 0.8,
                "ml_test_sharpe": 0.65,
                "runtime_s": 2.0,
                "base_overfit_gap": 0.1,
                "ml_overfit_gap": 0.15,
            },
            {
                "seed": 42,
                "vol_target_annual": 0.12,
                "ml_position_scalar_max": 10.0,
                "base_train_sharpe": 0.55,
                "base_test_sharpe": 0.45,
                "ml_train_sharpe": 0.75,
                "ml_test_sharpe": 0.62,
                "runtime_s": 1.5,
                "base_overfit_gap": 0.1,
                "ml_overfit_gap": 0.13,
            },
        ]
    )

    summary = summarize_sweep(per_run, group_cols=("vol_target_annual", "ml_position_scalar_max"), sharpe_threshold=0.5)
    assert not summary.empty
    assert "ml_test_sharpe_mean" in summary.columns
    assert "ml_win_vs_base_mean" in summary.columns

    heatmap = build_heatmap(summary, index="ml_position_scalar_max", columns="vol_target_annual", values="ml_test_sharpe_mean")
    assert not heatmap.empty

    best = pick_best_config(summary, score_col="ml_test_sharpe_mean")
    assert best
    assert "ml_test_sharpe_mean" in best


def test_pick_best_config_with_hard_gate_prefers_passing_row():
    summary = pd.DataFrame(
        [
            {
                "hmm_n_states": 2,
                "ml_test_sharpe_mean": 1.20,
                "hard_gate_passed_mean": 0.0,
            },
            {
                "hmm_n_states": 3,
                "ml_test_sharpe_mean": 1.00,
                "hard_gate_passed_mean": 1.0,
            },
        ]
    )
    best = pick_best_config_with_hard_gate(
        summary,
        score_col="ml_test_sharpe_mean",
        hard_gate_col="hard_gate_passed_mean",
    )
    assert best
    assert best["hmm_n_states"] == 3


def test_pick_best_config_with_hard_gate_falls_back_to_sharpe_without_gate_col():
    summary = pd.DataFrame(
        [
            {"hmm_n_states": 2, "ml_test_sharpe_mean": 1.20},
            {"hmm_n_states": 3, "ml_test_sharpe_mean": 1.00},
        ]
    )
    best = pick_best_config_with_hard_gate(summary, score_col="ml_test_sharpe_mean", hard_gate_col="missing_col")
    assert best
    assert best["hmm_n_states"] == 2
