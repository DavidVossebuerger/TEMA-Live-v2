import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_pipeline


def test_template_default_universe_enables_modular_portfolio_by_default(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    # call without specifying modular_portfolio_enabled (should be None in signature)
    run_pipeline.run_modular(
        run_id="template-mod-default",
        out_root="outputs",
        template_default_universe=True,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.portfolio_modular_enabled is True


def test_template_default_universe_preserves_explicit_disable(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["cfg"] = cfg
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    # explicitly disable modular portfolio even though template mode is on
    run_pipeline.run_modular(
        run_id="template-mod-explicit-disable",
        out_root="outputs",
        template_default_universe=True,
        modular_portfolio_enabled=False,
    )

    cfg = captured["cfg"]
    assert cfg.template_default_universe is True
    assert cfg.portfolio_modular_enabled is False
