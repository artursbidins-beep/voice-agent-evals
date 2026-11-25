"""Command line entry point.

    voice-evals run config/liepa-en.yaml
    voice-evals run config/liepa-en.yaml --provider vapi --save-fixtures
    voice-evals show reports/liepa-en.json

Exit code 1 means a threshold was breached, which is what makes this usable
as a build step rather than a dashboard nobody opens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voice_evals.loader import SuiteConfig, load_scenarios
from voice_evals.report import ScenarioResult, build_report, evaluate
from voice_evals.runners import get_runner
from voice_evals.runners.base import RunnerError


def _run(args: argparse.Namespace) -> int:
    config = SuiteConfig.load(args.config)
    provider = args.provider or config.provider
    scenarios = load_scenarios(config.scenarios_dir)

    if args.only:
        wanted = set(args.only)
        scenarios = [s for s in scenarios if s.id in wanted]
        if not scenarios:
            print(f"no scenario matched {', '.join(sorted(wanted))}", file=sys.stderr)
            return 2

    if provider == "vapi":
        runner = get_runner(
            provider,
            assistant_id=config.assistant_id,
            save_to=config.fixtures_dir if args.save_fixtures else None,
        )
    else:
        runner = get_runner(provider, fixtures_dir=config.fixtures_dir)

    results: list[ScenarioResult] = []
    errors = 0

    try:
        for scenario in scenarios:
            try:
                record = runner.run(scenario)
            except RunnerError as error:
                errors += 1
                print(f"  {scenario.id}: could not run: {error}", file=sys.stderr)
                continue
            results.append(evaluate(scenario, record, config.thresholds))
    finally:
        runner.close()

    if not results:
        print("no scenario produced a call record", file=sys.stderr)
        return 2

    report = build_report(config.name, provider, config.assistant_name, results)

    reports_dir = Path(args.out or "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{config.name}.json").write_text(report.to_json(), encoding="utf-8")
    (reports_dir / f"{config.name}.md").write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    print()
    print(f"report written to {reports_dir / (config.name + '.json')}")

    if errors:
        return 2
    return 0 if report.passed else 1


def _show(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    for result in payload["results"]:
        mark = "ok  " if result["passed"] else "FAIL"
        print(f"{mark} {result['scenario_id']}: {result['title']}")
        for failure in result["failures"]:
            print(f"       {failure}")
    return 0 if payload["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-evals", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and score it")
    run.add_argument("config", help="path to a suite config, for example config/liepa-en.yaml")
    run.add_argument("--provider", help="override the provider in the config")
    run.add_argument("--only", nargs="*", help="run only these scenario ids")
    run.add_argument("--out", help="directory for the report files (default: reports)")
    run.add_argument(
        "--save-fixtures",
        action="store_true",
        help="store each live call as a fixture so offline runs can replay it",
    )
    run.set_defaults(func=_run)

    show = sub.add_parser("show", help="print the failures from a saved report")
    show.add_argument("report", help="path to a report JSON file")
    show.set_defaults(func=_show)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
