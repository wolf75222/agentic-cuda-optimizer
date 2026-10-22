#!/usr/bin/env python3
"""Plan, execute or resume the full §16 study. Planning submits zero jobs."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import SuiteOrchestrator, read_json, write_json
from optimizer_agent.suite_report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status", "report", "recover-final", "recover-bootstrap", "resume-bootstrap-preparation", "recover-unscored", "resume-unscored-preparation", "recover-evaluation", "resume-evaluation-preparation", "reconcile-completed-generation"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1729, 2718, 3141))
    parser.add_argument("--include-greedy", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--requirement-evidence", type=Path,
                        help="Explicit per-requirement implementation/local/GPU proof catalog")
    parser.add_argument("--task", help="Exact incomplete task ID for explicit infrastructure recovery")
    parser.add_argument("--reason", help="Concrete audited infrastructure incident; required for recovery")
    parser.add_argument("--request", help="Exact unfinished canonical evaluation request SHA for recover-evaluation")
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    if args.command == "plan":
        if not args.config:
            parser.error("plan requires --config")
        plan = plan_suite(CampaignConfig.load(args.config), args.seeds, include_greedy=args.include_greedy)
        target = directory / "plan.json"
        if target.exists() and read_json(target) != json.loads(json.dumps(plan)):
            raise ValueError("Refusing to overwrite a different suite plan")
        write_json(target, plan)
    else:
        plan = read_json(directory / "plan.json")
    orchestrator = SuiteOrchestrator(plan, directory, concurrency=args.concurrency)
    recovery = None
    if args.command == "recover-final":
        if not args.task or not args.reason:
            parser.error("recover-final requires --task and --reason")
        recovery = orchestrator.recover_final(args.task, args.reason)
    elif args.command == "recover-bootstrap":
        if not args.task or not args.reason:
            parser.error("recover-bootstrap requires --task and --reason")
        recovery = orchestrator.recover_bootstrap(args.task, args.reason)
    elif args.command == "resume-bootstrap-preparation":
        if not args.task:
            parser.error("resume-bootstrap-preparation requires --task")
        recovery = orchestrator.resume_bootstrap_preparation(args.task)
    elif args.command == "recover-unscored":
        if not args.task or not args.reason:
            parser.error("recover-unscored requires --task and --reason")
        recovery = orchestrator.recover_unscored(args.task,args.reason)
    elif args.command == "resume-unscored-preparation":
        if not args.task:
            parser.error("resume-unscored-preparation requires --task")
        recovery = orchestrator.resume_unscored_preparation(args.task)
    elif args.command == "recover-evaluation":
        if not args.task or not args.request or not args.reason:
            parser.error("recover-evaluation requires --task, --request and --reason")
        recovery = orchestrator.recover_evaluation(args.task, args.request, args.reason)
    elif args.command == "resume-evaluation-preparation":
        if not args.task:
            parser.error("resume-evaluation-preparation requires --task")
        recovery = orchestrator.resume_evaluation_preparation(args.task)
    elif args.command == "reconcile-completed-generation":
        if not args.task or not args.reason:
            parser.error("reconcile-completed-generation requires --task and --reason")
        from optimizer_agent.terminal_recovery import reconcile_completed_generation
        recovery = reconcile_completed_generation(orchestrator, args.task, args.reason)
    status = orchestrator.run() if args.command == "run" else orchestrator.status()
    evidence = read_json(args.requirement_evidence) if args.requirement_evidence else None
    write_report(directory, status, evidence=evidence)
    print(json.dumps({"directory": str(directory), "planned_campaigns": len(plan["entries"]),
        "state_counts": {state: sum(row["state"] == state for row in status["tasks"])
                         for state in ("pending", "running", "completed", "recovery_required")},
        "complete": status["summary"]["complete"], "scientific_qualified": status["summary"]["scientific_qualified"],
        "jobs_submitted_by_this_command": args.command == "run", "recovery": recovery}, indent=2))
    return 0 if args.command != "run" or status["summary"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
