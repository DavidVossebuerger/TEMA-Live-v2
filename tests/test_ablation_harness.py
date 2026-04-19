from tema.validation.ablation import build_pair_report, decide_keep_challenger


def _summary(
    *,
    run_id: str,
    sharpe: float | None,
    annual_return: float | None,
    annual_volatility: float | None,
    max_drawdown: float | None,
    turnover: float | None,
    oos_passed: bool | None = True,
    psr_passed: bool | None = True,
    dsr_passed: bool | None = True,
    pbo_passed: bool | None = True,
) -> dict:
    return {
        "run_id": run_id,
        "config_summary": {"label": run_id},
        "metrics": {
            "sharpe": sharpe,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "turnover": turnover,
        },
        "validation_hard_gates": {
            "oos": {"available": oos_passed is not None, "passed": oos_passed, "value": oos_passed, "threshold": True},
            "psr": {"available": psr_passed is not None, "passed": psr_passed, "value": 0.95, "threshold": 0.95},
            "dsr": {"available": dsr_passed is not None, "passed": dsr_passed, "value": 0.80, "threshold": 0.80},
            "pbo": {"available": pbo_passed is not None, "passed": pbo_passed, "value": 0.40, "threshold": 0.50},
        },
        "benchmark_injection_detected": False,
    }


def test_decision_blocks_failed_hard_gate():
    champion = _summary(
        run_id="champion",
        sharpe=1.0,
        annual_return=0.10,
        annual_volatility=0.15,
        max_drawdown=-0.20,
        turnover=1.0,
    )
    challenger = _summary(
        run_id="challenger",
        sharpe=1.1,
        annual_return=0.11,
        annual_volatility=0.15,
        max_drawdown=-0.20,
        turnover=1.0,
        pbo_passed=False,
    )
    decision = decide_keep_challenger(champion_summary=champion, challenger_summary=challenger)
    assert decision["keep_challenger"] is False
    assert any("pbo hard gate failed" in reason for reason in decision["reasons"])


def test_decision_accepts_non_degrading_challenger():
    champion = _summary(
        run_id="champion",
        sharpe=1.0,
        annual_return=0.10,
        annual_volatility=0.15,
        max_drawdown=-0.20,
        turnover=1.0,
    )
    challenger = _summary(
        run_id="challenger",
        sharpe=1.1,
        annual_return=0.11,
        annual_volatility=0.14,
        max_drawdown=-0.18,
        turnover=0.9,
    )
    decision = decide_keep_challenger(champion_summary=champion, challenger_summary=challenger)
    assert decision["keep_challenger"] is True
    assert decision["rollback_to_champion"] is False


def test_pair_report_schema_contains_required_fields():
    champion = _summary(
        run_id="champion",
        sharpe=1.0,
        annual_return=0.10,
        annual_volatility=0.15,
        max_drawdown=-0.20,
        turnover=None,
    )
    challenger = _summary(
        run_id="challenger",
        sharpe=1.1,
        annual_return=0.12,
        annual_volatility=0.16,
        max_drawdown=-0.22,
        turnover=None,
    )
    report = build_pair_report(champion_summary=champion, challenger_summary=challenger)
    assert report["champion"]["run_id"] == "champion"
    assert report["challenger"]["run_id"] == "challenger"
    assert "metric_deltas" in report
    assert set(("sharpe", "annual_return", "annual_volatility", "max_drawdown", "turnover")).issubset(
        set(report["metric_deltas"].keys())
    )
    assert isinstance(report["decision"]["keep_challenger"], bool)
    assert isinstance(report["decision"]["reasons"], list)

