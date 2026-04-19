import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_pipeline


def test_run_modular_wires_new_knobs(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["run_id"] = run_id
        captured["cfg"] = cfg
        captured["out_root"] = out_root
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="wired",
        out_root="outputs",
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        ml_modular_path_enabled=True,
        ml_template_overlay=True,
        ml_meta_overlay=True,
        ml_meta_use_triple_barrier=True,
        ml_meta_tb_horizon=7,
        ml_meta_tb_upper=0.02,
        ml_meta_tb_lower=0.015,
        ml_probability_threshold=0.55,
        ml_feature_fracdiff_enabled=True,
        ml_feature_fracdiff_order=0.35,
        ml_feature_fracdiff_threshold=1e-4,
        ml_feature_fracdiff_max_terms=128,
        ml_feature_har_rv_enabled=True,
        ml_feature_har_rv_windows=(1, 5, 22),
        ml_feature_har_rv_use_log=False,
        data_max_assets=11,
        data_full_universe_for_parity=False,
        portfolio_method="bl",
        portfolio_risk_aversion=3.0,
        portfolio_cov_shrinkage=0.2,
        portfolio_covariance_backend="correlation",
        portfolio_correlation_backend="gerber",
        portfolio_gerber_threshold=0.4,
        portfolio_bl_tau=0.07,
        portfolio_bl_view_confidence=0.70,
        portfolio_bl_omega_scale=0.30,
        portfolio_bl_max_weight=0.20,
        portfolio_regime_mapping_enabled=True,
        portfolio_regime_mapping_mode="stepwise",
        portfolio_regime_mapping_min_multiplier=0.6,
        portfolio_regime_mapping_max_multiplier=1.4,
        portfolio_regime_mapping_step_thresholds=(0.2, 0.7),
        portfolio_regime_mapping_step_multipliers=(0.7, 1.0, 1.3),
        portfolio_regime_mapping_kelly_gamma=2.2,
        ml_hmm_scalar_floor=0.25,
        ml_hmm_scalar_ceiling=1.10,
        vol_target_apply_to_ml=True,
        fee_rate=0.0012,
        slippage_rate=0.0013,
        cost_model="extended",
        spread_bps=2.0,
        impact_coeff=0.02,
        borrow_bps=12.0,
        dynamic_trading_enabled=True,
        dynamic_trading_lambda=0.4,
        dynamic_trading_aim_multiplier=0.2,
        dynamic_trading_min_trade_rate=0.15,
        dynamic_trading_max_trade_rate=0.9,
        execution_backend="almgren_chriss",
        execution_ac_n_slices=5,
        execution_ac_risk_aversion=1.2,
        execution_ac_temporary_impact=0.08,
        execution_ac_permanent_impact=0.02,
        execution_ac_volatility_lookback=15,
        experimental_multi_horizon_blend_enabled=True,
        experimental_conformal_sizing_enabled=True,
        experimental_futuretesting_enabled=True,
        experimental_futuretesting_n_paths=321,
        experimental_futuretesting_horizon=77,
        cle_enabled=True,
        cle_use_external_proxies=True,
        cle_mode="confluence_blend",
        cle_mapping_mode="stepwise",
        cle_mapping_min_multiplier=0.70,
        cle_mapping_max_multiplier=1.80,
        cle_mapping_step_thresholds=(0.20, 0.60),
        cle_mapping_step_multipliers=(0.60, 1.10, 1.80),
        cle_mapping_kelly_gamma=2.5,
        cle_gate_event_blackout_cap=0.45,
        cle_gate_liquidity_spread_z_threshold=1.5,
        cle_gate_liquidity_depth_threshold=0.15,
        cle_gate_liquidity_reduction_factor=0.40,
        cle_gate_correlation_alert_cap=0.9,
        cle_leverage_floor=0.05,
        cle_leverage_cap=3.5,
        cle_policy_seed=7,
        cle_online_calibration_enabled=True,
        cle_online_calibration_window=3,
        cle_online_calibration_learning_rate=0.15,
        cle_online_calibration_l2=2e-4,
        validation_oos_min_calmar=0.8,
        validation_psr_threshold=0.9,
        validation_dsr_threshold=0.75,
        validation_pbo_max=0.4,
        validation_cpcv_n_groups=8,
        validation_cpcv_n_test_groups=2,
        validation_cpcv_purge_groups=1,
        validation_cpcv_embargo_groups=1,
        validation_cpcv_max_splits=64,
        validation_hard_fail=True,
    )

    cfg = captured["cfg"]
    assert cfg.modular_data_signals_enabled is True
    assert cfg.portfolio_modular_enabled is True
    assert cfg.ml_modular_path_enabled is True
    assert cfg.ml_template_overlay_enabled is True
    assert cfg.ml_meta_overlay_enabled is True
    assert cfg.ml_meta_use_triple_barrier is True
    assert cfg.ml_meta_tb_horizon == 7
    assert cfg.ml_meta_tb_upper == 0.02
    assert cfg.ml_meta_tb_lower == 0.015
    assert cfg.ml_probability_threshold == 0.55
    assert cfg.ml_feature_fracdiff_enabled is True
    assert cfg.ml_feature_fracdiff_order == 0.35
    assert cfg.ml_feature_fracdiff_threshold == 1e-4
    assert cfg.ml_feature_fracdiff_max_terms == 128
    assert cfg.ml_feature_har_rv_enabled is True
    assert cfg.ml_feature_har_rv_windows == (1, 5, 22)
    assert cfg.ml_feature_har_rv_use_log is False
    assert cfg.data_max_assets == 11
    assert cfg.data_full_universe_for_parity is False
    assert cfg.portfolio_risk_aversion == 3.0
    assert cfg.portfolio_cov_shrinkage == 0.2
    assert cfg.portfolio_covariance_backend == "correlation"
    assert cfg.portfolio_correlation_backend == "gerber"
    assert cfg.portfolio_gerber_threshold == 0.4
    assert cfg.portfolio_bl_tau == 0.07
    assert cfg.portfolio_bl_view_confidence == 0.70
    assert cfg.portfolio_bl_omega_scale == 0.30
    assert cfg.portfolio_bl_max_weight == 0.20
    assert cfg.portfolio_regime_mapping_enabled is True
    assert cfg.portfolio_regime_mapping_mode == "stepwise"
    assert cfg.portfolio_regime_mapping_min_multiplier == 0.6
    assert cfg.portfolio_regime_mapping_max_multiplier == 1.4
    assert cfg.portfolio_regime_mapping_step_thresholds == (0.2, 0.7)
    assert cfg.portfolio_regime_mapping_step_multipliers == (0.7, 1.0, 1.3)
    assert cfg.portfolio_regime_mapping_kelly_gamma == 2.2
    assert cfg.ml_hmm_scalar_floor == 0.25
    assert cfg.ml_hmm_scalar_ceiling == 1.10
    assert cfg.vol_target_apply_to_ml is True
    assert cfg.fee_rate == 0.0012
    assert cfg.slippage_rate == 0.0013
    assert cfg.cost_model == "extended"
    assert cfg.spread_bps == 2.0
    assert cfg.impact_coeff == 0.02
    assert cfg.borrow_bps == 12.0
    assert cfg.dynamic_trading_enabled is True
    assert cfg.dynamic_trading_lambda == 0.4
    assert cfg.dynamic_trading_aim_multiplier == 0.2
    assert cfg.dynamic_trading_min_trade_rate == 0.15
    assert cfg.dynamic_trading_max_trade_rate == 0.9
    assert cfg.execution_backend == "almgren_chriss"
    assert cfg.execution_ac_n_slices == 5
    assert cfg.execution_ac_risk_aversion == 1.2
    assert cfg.execution_ac_temporary_impact == 0.08
    assert cfg.execution_ac_permanent_impact == 0.02
    assert cfg.execution_ac_volatility_lookback == 15
    assert cfg.experimental_multi_horizon_blend_enabled is True
    assert cfg.experimental_conformal_sizing_enabled is True
    assert cfg.experimental_futuretesting_enabled is True
    assert cfg.experimental_futuretesting_n_paths == 321
    assert cfg.experimental_futuretesting_horizon == 77
    assert cfg.cle_enabled is True
    assert cfg.cle_use_external_proxies is True
    assert cfg.cle_mode == "confluence_blend"
    assert cfg.cle_mapping_mode == "stepwise"
    assert cfg.cle_mapping_min_multiplier == 0.70
    assert cfg.cle_mapping_max_multiplier == 1.80
    assert cfg.cle_mapping_step_thresholds == (0.20, 0.60)
    assert cfg.cle_mapping_step_multipliers == (0.60, 1.10, 1.80)
    assert cfg.cle_mapping_kelly_gamma == 2.5
    assert cfg.cle_gate_event_blackout_cap == 0.45
    assert cfg.cle_gate_liquidity_spread_z_threshold == 1.5
    assert cfg.cle_gate_liquidity_depth_threshold == 0.15
    assert cfg.cle_gate_liquidity_reduction_factor == 0.40
    assert cfg.cle_gate_correlation_alert_cap == 0.9
    assert cfg.cle_leverage_floor == 0.05
    assert cfg.cle_leverage_cap == 3.5
    assert cfg.cle_policy_seed == 7
    assert cfg.cle_online_calibration_enabled is True
    assert cfg.cle_online_calibration_window == 3
    assert cfg.cle_online_calibration_learning_rate == 0.15
    assert cfg.cle_online_calibration_l2 == 2e-4


def test_run_modular_defaults_keep_cle_disabled(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(run_id="cle-defaults", out_root="outputs")

    cfg = captured["cfg"]
    assert cfg.cle_enabled is False
    assert cfg.cle_use_external_proxies is False


def test_run_modular_template_default_universe_sets_profile_defaults(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="template-universe",
        out_root="outputs",
        template_default_universe=True,
        data_path=None,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.data_path == "merged_d1"
    assert cfg.signal_fast_period == 3
    assert cfg.signal_slow_period == 20
    # When template_default_universe is requested and ML is enabled by default,
    # the wiring should default ml_template_overlay to True unless the caller
    # explicitly set it. The meta overlay should remain off by default.
    assert cfg.ml_template_overlay_enabled is True
    assert cfg.ml_meta_overlay_enabled is False
    assert cfg.template_rebalance_enabled is False


def test_run_modular_wires_template_rebalance_opt_in(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="template-rebalance",
        out_root="outputs",
        template_default_universe=True,
        template_rebalance_enabled=True,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.template_rebalance_enabled is True
    assert cfg.modular_data_signals_enabled is True


def test_run_modular_wires_template_precomputed_toggle(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="template-toggle",
        out_root="outputs",
        template_default_universe=True,
        template_use_precomputed_artifacts=False,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.template_use_precomputed_artifacts is False
    assert cfg.template_grid_signal_logic == "or"
    assert cfg.ml_meta_comparator_use_benchmark_csv is False


def test_run_modular_wires_ml_meta_comparator_mode(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="template-meta-comparator",
        out_root="outputs",
        template_default_universe=True,
        template_use_precomputed_artifacts=False,
        ml_meta_comparator_use_benchmark_csv=True,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.template_use_precomputed_artifacts is False
    assert cfg.ml_meta_comparator_use_benchmark_csv is True


def test_run_modular_wires_strict_independent_mode(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="strict-independent",
        out_root="outputs",
        strict_independent_mode=True,
    )

    cfg = captured["cfg"]
    assert cfg.strict_independent_mode is True


def test_run_modular_template_default_universe_keeps_explicit_data_path(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="template-universe-explicit",
        out_root="outputs",
        template_default_universe=True,
        data_path="custom_data",
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.data_path == "custom_data"
    assert cfg.signal_fast_period == 3
    assert cfg.signal_slow_period == 20


def test_run_modular_cpp_hmm_profile_is_opt_in_and_applies_overrides(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(run_id="cpp-profile-on", out_root="outputs", cpp_hmm_profile="sweep-optimized-v1")
    cfg = captured["cfg"]
    assert cfg.cpp_hmm_profile == "sweep-optimized-v1"
    assert cfg.hmm_n_states == 2
    assert cfg.hmm_n_iter == 30
    assert cfg.hmm_var_floor == 1e-8
    assert cfg.hmm_trans_sticky == 0.95

    run_pipeline.run_modular(run_id="cpp-profile-off", out_root="outputs")
    cfg = captured["cfg"]
    assert cfg.cpp_hmm_profile is None
    assert cfg.hmm_trans_sticky == 0.92


def test_main_passes_out_root_to_modular(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["run_id"] = run_id
        captured["out_root"] = out_root
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(["--run-id", "mod-main", "--out-root", "custom-out"])

    assert captured["run_id"] == "mod-main"
    assert captured["out_root"] == "custom-out"


def test_main_passes_template_precomputed_toggle(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-template-toggle",
            "--template-default-universe",
            "--no-template-precomputed-artifacts",
        ]
    )

    assert captured["kwargs"]["template_default_universe"] is True
    assert captured["kwargs"]["template_use_precomputed_artifacts"] is False
    assert captured["kwargs"]["ml_meta_comparator_use_benchmark_csv"] is False


def test_main_passes_template_rebalance_toggle(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-template-rebalance",
            "--template-default-universe",
            "--template-rebalance",
        ]
    )

    assert captured["kwargs"]["template_default_universe"] is True
    assert captured["kwargs"]["template_rebalance_enabled"] is True


def test_main_passes_ml_meta_comparator_toggle(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-template-meta-comparator",
            "--template-default-universe",
            "--no-template-precomputed-artifacts",
            "--ml-meta-comparator-benchmark-csv",
        ]
    )

    assert captured["kwargs"]["template_default_universe"] is True
    assert captured["kwargs"]["template_use_precomputed_artifacts"] is False
    assert captured["kwargs"]["ml_meta_comparator_use_benchmark_csv"] is True


def test_main_passes_strict_independent_toggle(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-strict-independent",
            "--strict-independent",
        ]
    )

    assert captured["kwargs"]["strict_independent_mode"] is True


def test_main_passes_cpp_hmm_profile_toggle(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(["--run-id", "mod-cpp-profile", "--cpp-hmm-profile", "sweep-optimized-v1"])

    assert captured["kwargs"]["cpp_hmm_profile"] == "sweep-optimized-v1"


def test_main_passes_ml_advanced_feature_toggles(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-ml-advanced-features",
            "--ml-feature-fracdiff",
            "--ml-feature-fracdiff-order",
            "0.35",
            "--ml-feature-fracdiff-threshold",
            "0.0001",
            "--ml-feature-fracdiff-max-terms",
            "128",
            "--ml-feature-har-rv",
            "--ml-feature-har-rv-windows",
            "1,5,22",
            "--ml-feature-har-rv-no-log",
        ]
    )

    assert captured["kwargs"]["ml_feature_fracdiff_enabled"] is True
    assert captured["kwargs"]["ml_feature_fracdiff_order"] == 0.35
    assert captured["kwargs"]["ml_feature_fracdiff_threshold"] == 0.0001
    assert captured["kwargs"]["ml_feature_fracdiff_max_terms"] == 128
    assert captured["kwargs"]["ml_feature_har_rv_enabled"] is True
    assert captured["kwargs"]["ml_feature_har_rv_windows"] == (1, 5, 22)
    assert captured["kwargs"]["ml_feature_har_rv_use_log"] is False


def test_main_passes_execution_and_experimental_knobs(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-exec-exp",
            "--fee-rate",
            "0.0012",
            "--slippage-rate",
            "0.0013",
            "--cost-model",
            "extended",
            "--spread-bps",
            "2.0",
            "--impact-coeff",
            "0.02",
            "--borrow-bps",
            "12.0",
            "--dynamic-trading",
            "--dynamic-trading-lambda",
            "0.4",
            "--dynamic-trading-aim-multiplier",
            "0.2",
            "--dynamic-trading-min-trade-rate",
            "0.15",
            "--dynamic-trading-max-trade-rate",
            "0.9",
            "--execution-backend",
            "almgren_chriss",
            "--execution-ac-n-slices",
            "5",
            "--execution-ac-risk-aversion",
            "1.2",
            "--execution-ac-temporary-impact",
            "0.08",
            "--execution-ac-permanent-impact",
            "0.02",
            "--execution-ac-volatility-lookback",
            "15",
            "--experimental-multi-horizon-blend",
            "--experimental-conformal-sizing",
            "--experimental-futuretesting",
            "--experimental-futuretesting-n-paths",
            "321",
            "--experimental-futuretesting-horizon",
            "77",
        ]
    )

    assert captured["kwargs"]["fee_rate"] == 0.0012
    assert captured["kwargs"]["slippage_rate"] == 0.0013
    assert captured["kwargs"]["cost_model"] == "extended"
    assert captured["kwargs"]["spread_bps"] == 2.0
    assert captured["kwargs"]["impact_coeff"] == 0.02
    assert captured["kwargs"]["borrow_bps"] == 12.0
    assert captured["kwargs"]["dynamic_trading_enabled"] is True
    assert captured["kwargs"]["dynamic_trading_lambda"] == 0.4
    assert captured["kwargs"]["dynamic_trading_aim_multiplier"] == 0.2
    assert captured["kwargs"]["dynamic_trading_min_trade_rate"] == 0.15
    assert captured["kwargs"]["dynamic_trading_max_trade_rate"] == 0.9
    assert captured["kwargs"]["execution_backend"] == "almgren_chriss"
    assert captured["kwargs"]["execution_ac_n_slices"] == 5
    assert captured["kwargs"]["execution_ac_risk_aversion"] == 1.2
    assert captured["kwargs"]["execution_ac_temporary_impact"] == 0.08
    assert captured["kwargs"]["execution_ac_permanent_impact"] == 0.02
    assert captured["kwargs"]["execution_ac_volatility_lookback"] == 15
    assert captured["kwargs"]["experimental_multi_horizon_blend_enabled"] is True
    assert captured["kwargs"]["experimental_conformal_sizing_enabled"] is True
    assert captured["kwargs"]["experimental_futuretesting_enabled"] is True
    assert captured["kwargs"]["experimental_futuretesting_n_paths"] == 321
    assert captured["kwargs"]["experimental_futuretesting_horizon"] == 77


def test_main_passes_validation_probabilistic_and_cpcv_knobs(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-validation-prob",
            "--validation-oos-min-calmar",
            "0.8",
            "--validation-psr-threshold",
            "0.9",
            "--validation-dsr-threshold",
            "0.75",
            "--validation-pbo-max",
            "0.4",
            "--validation-cpcv-n-groups",
            "8",
            "--validation-cpcv-max-splits",
            "64",
            "--validation-hard-fail",
            "--ml-meta-triple-barrier",
            "--ml-meta-tb-horizon",
            "7",
            "--ml-meta-tb-upper",
            "0.02",
            "--ml-meta-tb-lower",
            "0.015",
        ]
    )

    assert captured["kwargs"]["validation_oos_min_calmar"] == 0.8
    assert captured["kwargs"]["validation_psr_threshold"] == 0.9
    assert captured["kwargs"]["validation_dsr_threshold"] == 0.75
    assert captured["kwargs"]["validation_pbo_max"] == 0.4
    assert captured["kwargs"]["validation_cpcv_n_groups"] == 8
    assert captured["kwargs"]["validation_cpcv_max_splits"] == 64
    assert captured["kwargs"]["validation_hard_fail"] is True
    assert captured["kwargs"]["ml_meta_use_triple_barrier"] is True
    assert captured["kwargs"]["ml_meta_tb_horizon"] == 7
    assert captured["kwargs"]["ml_meta_tb_upper"] == 0.02
    assert captured["kwargs"]["ml_meta_tb_lower"] == 0.015


def test_main_passes_template_bl_knobs(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-template-bl",
            "--portfolio-bl-omega-scale",
            "0.35",
            "--portfolio-bl-max-weight",
            "0.12",
        ]
    )

    assert captured["kwargs"]["portfolio_bl_omega_scale"] == 0.35
    assert captured["kwargs"]["portfolio_bl_max_weight"] == 0.12


def test_main_passes_cle_knobs_to_modular(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["run_id"] = run_id
        captured["out_root"] = out_root
        captured["kwargs"] = kwargs
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(
        [
            "--run-id",
            "mod-cle-main",
            "--cle-enabled",
            "--cle-use-external-proxies",
            "--cle-mode",
            "confluence_blend",
            "--cle-mapping-mode",
            "stepwise",
            "--cle-mapping-min-multiplier",
            "0.7",
            "--cle-mapping-max-multiplier",
            "1.8",
            "--cle-mapping-step-thresholds",
            "0.2,0.6",
            "--cle-mapping-step-multipliers",
            "0.6,1.1,1.8",
            "--cle-mapping-kelly-gamma",
            "2.5",
            "--cle-gate-event-blackout-cap",
            "0.45",
            "--cle-gate-liquidity-spread-z-threshold",
            "1.5",
            "--cle-gate-liquidity-depth-threshold",
            "0.15",
            "--cle-gate-liquidity-reduction-factor",
            "0.4",
            "--cle-gate-correlation-alert-cap",
            "0.9",
            "--cle-leverage-floor",
            "0.05",
            "--cle-leverage-cap",
            "3.5",
            "--cle-policy-seed",
            "7",
            "--cle-online-calibration-enabled",
            "--cle-online-calibration-window",
            "3",
            "--cle-online-calibration-learning-rate",
            "0.15",
            "--cle-online-calibration-l2",
            "0.0002",
        ]
    )

    assert captured["run_id"] == "mod-cle-main"
    assert captured["kwargs"]["cle_enabled"] is True
    assert captured["kwargs"]["cle_use_external_proxies"] is True
    assert captured["kwargs"]["cle_mode"] == "confluence_blend"
    assert captured["kwargs"]["cle_mapping_mode"] == "stepwise"
    assert captured["kwargs"]["cle_mapping_min_multiplier"] == 0.7
    assert captured["kwargs"]["cle_mapping_max_multiplier"] == 1.8
    assert captured["kwargs"]["cle_mapping_step_thresholds"] == (0.2, 0.6)
    assert captured["kwargs"]["cle_mapping_step_multipliers"] == (0.6, 1.1, 1.8)
    assert captured["kwargs"]["cle_mapping_kelly_gamma"] == 2.5
    assert captured["kwargs"]["cle_gate_event_blackout_cap"] == 0.45
    assert captured["kwargs"]["cle_gate_liquidity_spread_z_threshold"] == 1.5
    assert captured["kwargs"]["cle_gate_liquidity_depth_threshold"] == 0.15
    assert captured["kwargs"]["cle_gate_liquidity_reduction_factor"] == 0.4
    assert captured["kwargs"]["cle_gate_correlation_alert_cap"] == 0.9
    assert captured["kwargs"]["cle_leverage_floor"] == 0.05
    assert captured["kwargs"]["cle_leverage_cap"] == 3.5
    assert captured["kwargs"]["cle_policy_seed"] == 7
    assert captured["kwargs"]["cle_online_calibration_enabled"] is True
    assert captured["kwargs"]["cle_online_calibration_window"] == 3
    assert captured["kwargs"]["cle_online_calibration_learning_rate"] == 0.15
    assert captured["kwargs"]["cle_online_calibration_l2"] == 0.0002


def test_main_passes_out_root_to_legacy(monkeypatch):
    captured = {}

    def _fake_run_legacy(run_id, out_root="outputs", legacy_metrics_dataset=None):
        captured["run_id"] = run_id
        captured["out_root"] = out_root
        captured["legacy_metrics_dataset"] = legacy_metrics_dataset
        return {"run_id": run_id}

    monkeypatch.setattr(run_pipeline, "run_legacy", _fake_run_legacy)
    run_pipeline.main(["--run-id", "legacy-main", "--legacy", "--out-root", "custom-out", "--legacy-metrics-dataset", "test"])

    assert captured["run_id"] == "legacy-main"
    assert captured["out_root"] == "custom-out"
    assert captured["legacy_metrics_dataset"] == "test"


def test_run_legacy_writes_performance_manifest(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "TEMA-TEMPLATE(NEW_).py").write_text("print('ok')", encoding="utf-8")

    def _fake_run_path(path, run_name=None, init_globals=None):
        csv_data = (
            "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
            "train,0,0.20,0.10,2.0,-0.10\n"
            "test,0,0.12,0.08,1.5,-0.12\n"
        )
        (template_dir / "bl_portfolio_metrics.csv").write_text(csv_data, encoding="utf-8")
        return {}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    monkeypatch.setattr(run_pipeline.runpy, "run_path", _fake_run_path)
    monkeypatch.setenv("TEMA_RUN_LEGACY_EXECUTE", "1")

    out_root = tmp_path / "outputs"
    res = run_pipeline.run_legacy(run_id="legacy-test", out_root=str(out_root))

    perf_path = Path(res["out_dir"]) / "performance.json"
    manifest_path = Path(res["manifest_path"])
    assert perf_path.exists()
    assert manifest_path.exists()

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert perf["sharpe"] == 1.5
    assert perf["annual_return"] == 0.12
    assert manifest["legacy_executed"] is True
    assert manifest["artifacts"] == ["performance"]


def test_run_legacy_respects_explicit_legacy_metrics_dataset(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "TEMA-TEMPLATE(NEW_).py").write_text("print('ok')", encoding="utf-8")

    def _fake_run_path(path, run_name=None, init_globals=None):
        csv_data = (
            "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
            "train,0,0.20,0.10,2.0,-0.10\n"
            "test,0,0.12,0.08,1.5,-0.12\n"
        )
        (template_dir / "bl_portfolio_metrics.csv").write_text(csv_data, encoding="utf-8")
        return {}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    monkeypatch.setattr(run_pipeline.runpy, "run_path", _fake_run_path)
    monkeypatch.setenv("TEMA_RUN_LEGACY_EXECUTE", "1")

    out_root = tmp_path / "outputs"
    res = run_pipeline.run_legacy(run_id="legacy-train", out_root=str(out_root), legacy_metrics_dataset="train")
    perf = json.loads((Path(res["out_dir"]) / "performance.json").read_text(encoding="utf-8"))
    assert perf["sharpe"] == 2.0
    assert perf["annual_return"] == 0.20


def test_run_modular_parity_metrics_bridge_overrides_performance(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "bl_portfolio_metrics.csv").write_text(
        "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
        "test,0,0.12,0.08,1.5,-0.12\n",
        encoding="utf-8",
    )

    def _fake_pipeline(run_id, cfg, out_root):
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        perf = {
            "sharpe": 0.1,
            "annual_return": 0.01,
            "annual_volatility": 0.3,
            "annual_vol": 0.3,
            "max_drawdown": -0.5,
            "source": {},
        }
        (out_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
        (out_dir / "returns_csv_info.json").write_text(json.dumps({"baseline_written": True}), encoding="utf-8")
        manifest = {"run_id": run_id, "artifacts": ["performance"]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"manifest_path": str(out_dir / "manifest.json"), "out_dir": str(out_dir), "manifest": manifest}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    out_root = tmp_path / "outputs"
    res = run_pipeline.run_modular(
        run_id="mod-bridge",
        out_root=str(out_root),
        parity_metrics_bridge=True,
        parity_metrics_dataset="test",
    )

    perf = json.loads((Path(res["out_dir"]) / "performance.json").read_text(encoding="utf-8"))
    assert perf["sharpe"] == 1.5
    assert perf["annual_return"] == 0.12
    assert perf["annual_volatility"] == 0.08
    assert perf["annual_vol"] == 0.08
    assert perf["max_drawdown"] == -0.12
    assert perf["parity_metrics_bridge_applied"] is True
    assert perf["benchmark_injection_detected"] is True
    assert perf["source"]["benchmark_injection_detected"] is True
    returns_csv_info = json.loads((Path(res["out_dir"]) / "returns_csv_info.json").read_text(encoding="utf-8"))
    assert returns_csv_info["benchmark_injection_detected"] is True
    assert "parity_metrics_bridge.legacy_metrics_csv" in returns_csv_info["benchmark_injection_sources"]


def test_run_modular_parity_metrics_bridge_blocked_in_strict_independent_mode(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "bl_portfolio_metrics.csv").write_text(
        "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
        "test,0,0.12,0.08,1.5,-0.12\n",
        encoding="utf-8",
    )

    def _fake_pipeline(run_id, cfg, out_root):
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        perf = {"sharpe": 0.1, "annual_return": 0.01, "annual_volatility": 0.3, "annual_vol": 0.3, "max_drawdown": -0.5}
        (out_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
        manifest = {"run_id": run_id, "artifacts": ["performance"]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"manifest_path": str(out_dir / "manifest.json"), "out_dir": str(out_dir), "manifest": manifest}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    out_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="strict_independent_mode violation"):
        run_pipeline.run_modular(
            run_id="mod-bridge-strict",
            out_root=str(out_root),
            parity_metrics_bridge=True,
            parity_metrics_dataset="test",
            strict_independent_mode=True,
        )


def test_run_modular_parity_metrics_bridge_requires_complete_metrics(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "bl_portfolio_metrics.csv").write_text(
        "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
        "test,0,0.12,0.08,,-0.12\n",
        encoding="utf-8",
    )

    def _fake_pipeline(run_id, cfg, out_root):
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        perf = {
            "sharpe": 0.1,
            "annual_return": 0.01,
            "annual_volatility": 0.3,
            "annual_vol": 0.3,
            "max_drawdown": -0.5,
        }
        (out_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
        manifest = {"run_id": run_id, "artifacts": ["performance"]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"manifest_path": str(out_dir / "manifest.json"), "out_dir": str(out_dir), "manifest": manifest}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    out_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="sharpe"):
        run_pipeline.run_modular(
            run_id="mod-bridge",
            out_root=str(out_root),
            parity_metrics_bridge=True,
            parity_metrics_dataset="test",
        )


def test_run_modular_parity_metrics_bridge_requires_matching_dataset(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "bl_portfolio_metrics.csv").write_text(
        "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
        "train,0,0.20,0.10,2.0,-0.10\n",
        encoding="utf-8",
    )

    def _fake_pipeline(run_id, cfg, out_root):
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        perf = {
            "sharpe": 0.1,
            "annual_return": 0.01,
            "annual_volatility": 0.3,
            "annual_vol": 0.3,
            "max_drawdown": -0.5,
        }
        (out_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
        manifest = {"run_id": run_id, "artifacts": ["performance"]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"manifest_path": str(out_dir / "manifest.json"), "out_dir": str(out_dir), "manifest": manifest}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    out_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="dataset 'test' not found"):
        run_pipeline.run_modular(
            run_id="mod-bridge",
            out_root=str(out_root),
            parity_metrics_bridge=True,
            parity_metrics_dataset="test",
        )
