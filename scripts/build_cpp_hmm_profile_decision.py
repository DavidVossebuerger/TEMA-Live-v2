#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tema.ml.cpp_profile import write_cpp_hmm_profile_decision


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("build_cpp_hmm_profile_decision")
    p.add_argument("--baseline-summary", default="outputs/cpp-baseline/cpp-baseline-summary.json")
    p.add_argument("--sweep-report", default="outputs/cpp-param-sweep-run/cpp_hmm_sweep_report.json")
    p.add_argument("--sweep-runs-csv", default="outputs/cpp-param-sweep-run/cpp_hmm_sweep_runs.csv")
    p.add_argument("--output", default="outputs/cpp-profile-decision.json")
    p.add_argument("--candidate-profile", default="sweep-optimized-v1")
    args = p.parse_args(argv)

    decision = write_cpp_hmm_profile_decision(
        baseline_summary_path=args.baseline_summary,
        sweep_report_path=args.sweep_report,
        sweep_runs_csv_path=args.sweep_runs_csv,
        output_path=args.output,
        candidate_profile_name=args.candidate_profile,
    )
    print(json.dumps({"output": args.output, "apply_profile": decision.get("apply_profile")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
