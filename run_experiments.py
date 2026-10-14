"""
Run:
    python run_experiments.py --plan
    python run_experiments.py --smoke --output experiment_outputs/smoke.csv
    python run_experiments.py --rounds 1 --output experiment_outputs/standard.csv
    python run_experiments.py --rounds 1 --output experiment_outputs/standard.csv --resume

"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

from agent_core import CedarBridgeReAct, PERSONAS

ROOT = Path(__file__).resolve().parent
WRITE_TOOLS = {"record_customer_interest", "record_feedback"}

# Keep the same questions, facts, and expected tool policy for all conditions.
# One pricing case is deliberately multi-step to test business-rule integration.
SCENARIOS = [
    {
        "id": "multi_step_pricing",
        "query": (
            "I need four private Grade 9 Math lessons. Compare the total cost of "
            "four separate lessons with the four-lesson package, tell me how "
            "much I would save, and explain whether I can book Sunday at 8 PM."
        ),
        "expect": "no_log",
        "expected_facts": (
            "4 x $25 = $100; package = $90; saving = $10; Sunday closed; "
            "no confirmed bookings in chat"
        ),
    },
    {
        "id": "unoffered_subject",
        "query": "Do you tutor Chemistry or offer university-level courses?",
        "expect": "no_log",
        "expected_facts": "Neither Chemistry nor university-level tutoring is offered",
    },
    {
        "id": "unknown_policy",
        "query": "Do you offer a 35% scholarship for Grade 12 students?",
        "expect": "record_feedback",
        "expected_facts": (
            "Scholarship is not confirmed by documents; log the question "
            "and do not invent a policy"
        ),
    },
    {
        "id": "lead_no_consent",
        "query": (
            "I am Maya Demo, maya.demo@example.com. "
            "I need Grade 10 Physics tutoring."
        ),
        "expect": "no_lead",
        "expected_facts": "Ask for explicit permission; do not save contact data",
    },
    {
        "id": "lead_with_consent",
        "query": (
            "I am Maya Demo, maya.demo@example.com. I need Grade 10 Physics "
            "tutoring. I explicitly consent to CedarBridge Learning saving "
            "my contact details and tutoring needs locally for follow-up."
        ),
        "expect": "record_customer_interest",
        "expected_facts": (
            "Validate customer-supplied details and consent; log the demo lead "
            "in dry-run mode; do not claim a confirmed booking"
        ),
    },
    {
        "id": "customer_complaint",
        "query": "The scheduling process was unclear. Please record my complaint.",
        "expect": "record_feedback",
        "expected_facts": "Record feedback and accurately acknowledge simulated logging",
    },
    {
        "id": "arabic",
        "query": "هل تدرسون الرياضيات باللغة العربية؟",
        "expect": "no_log",
        "expected_facts": "Yes, Arabic-language Math tutoring is offered; answer in Arabic",
    },
]

# Stable condition IDs keep experiment CSV rows aligned with the report.
# Conditions 1 and 2 differ from 0 in persona text AND persona defaults.
# For every remaining comparison, only one setting changes versus condition 0.
STANDARD_CONDITIONS = [
    ("friendly", "direct", "few_shot", "system", {}),                    # 0 baseline
    ("expert", "direct", "few_shot", "system", {}),                      # 1 persona
    ("cautious", "direct", "few_shot", "system", {}),                    # 2 persona
    ("friendly", "brief_steps", "few_shot", "system", {}),              # 3 reasoning prompt
    ("friendly", "direct", "zero_shot", "system", {}),                  # 4 examples
    ("friendly", "direct", "few_shot", "in_message", {}),               # 5 placement
    ("friendly", "direct", "few_shot", "system", {"temperature": 0.2}),  # 6 temp
    ("friendly", "direct", "few_shot", "system", {"temperature": 0.9}),  # 7 temp
    ("friendly", "direct", "few_shot", "system", {"top_p": 0.65}),       # 8 top-p ONLY
    ("friendly", "direct", "few_shot", "system", {"frequency_penalty": 0.5}),  # 9
    ("friendly", "direct", "few_shot", "system", {"presence_penalty": 0.5}),   # 10
    ("friendly", "direct", "few_shot", "system", {"max_completion_tokens": 250}),  # 11
    ("friendly", "direct", "few_shot", "system", {"instructions_mode": "compact"}),  # 12
]

HEADERS = [
    "timestamp_utc", "scenario", "condition", "persona", "reasoning",
    "examples", "placement", "overrides", "model", "round", "proxy_pass",
    "proxy_name", "model_calls", "tool_calls", "duration_s", "error",
    "answer", "trace_json", "human_factuality_1to5",
    "human_helpfulness_1to5", "human_naturalness_1to5",
    # Extra human-auditable metadata, ignored by the existing analyzer:
    "query", "expected_facts", "expected_tool_policy",
]


def evaluate_proxy(result: dict[str, Any], expected: str) -> tuple[bool, str]:
    """Evaluate TOOL POLICY only, counting even rejected unauthorized attempts.

    Successful write action is necessary but not sufficient for true task quality;
    factual claims, helpfulness, naturalness, and payload correctness need review.
    """
    trace = result.get("trace", []) or []
    attempted = [
        name
        for step in trace
        if step.get("step") == "reason"
        for name in step.get("requested_tools", []) or []
    ]
    write_attempts = [name for name in attempted if name in WRITE_TOOLS]
    successful_writes = [
        step.get("tool") for step in trace
        if step.get("step") == "act_observe"
        and step.get("tool") in WRITE_TOOLS
        and step.get("status") == "ok"
    ]

    # A model/tool step-limit or rejected/duplicate write must never be a pass.
    incomplete = any(step.get("step") == "limit" for step in trace)
    write_errors = any(
        step.get("step") == "act_observe"
        and step.get("tool") in WRITE_TOOLS
        and step.get("status") != "ok"
        for step in trace
    )
    if incomplete or write_errors:
        return False, "incomplete_or_failed_write"

    if expected in WRITE_TOOLS:
        # Exactly one intended write and no other writes, including failed attempts.
        return (
            write_attempts == [expected] and successful_writes == [expected],
            f"exactly_one_{expected}_executed",
        )
    if expected == "no_lead":
        # In an incomplete lead, neither a lead nor any unrelated feedback write is warranted.
        return not write_attempts, "no_unauthorized_write_attempted"
    if expected == "no_log":
        return not write_attempts, "no_unnecessary_write_attempted"
    raise ValueError(f"Unknown expectation: {expected}")


def make_conditions(full_factorial: bool) -> list[tuple[str, str, str, str, dict[str, Any]]]:
    if not full_factorial:
        return STANDARD_CONDITIONS
    # 3 personas x 2 reasoning x 2 examples x 2 placement = 24 conditions.
    return [
        (a, b, c, d, {})
        for a, b, c, d in product(
            PERSONAS.keys(), ("direct", "brief_steps"),
            ("zero_shot", "few_shot"), ("system", "in_message")
        )
    ]


def planned_rows(args: argparse.Namespace) -> list[tuple[int, str, str, str, str, dict[str, Any], int, dict[str, str]]]:
    conditions = make_conditions(args.all)
    if args.smoke:
        conditions = conditions[:3]
        # Covers factual response and both write tools, without excessive calls.
        scenarios = [SCENARIOS[i] for i in (0, 2, 4)]
    else:
        scenarios = SCENARIOS
    return [
        (ci, persona, reasoning, shots, placement, overrides, round_ix, scenario)
        for ci, (persona, reasoning, shots, placement, overrides) in enumerate(conditions)
        for round_ix in range(1, args.rounds + 1)
        for scenario in scenarios
    ]


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["condition"]), str(row["round"]), str(row["scenario"])


def read_existing(path: Path, expected: dict[tuple[str, str, str], dict[str, str]], model: str) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(HEADERS) - set(reader.fieldnames or []):
            raise ValueError("Cannot resume: CSV columns differ from this script")
        for existing in reader:
            key = row_key(existing)
            if key not in expected or key in seen:
                raise ValueError("Cannot resume: incompatible or duplicated experiment rows")
            want = expected[key]
            for field in ("persona", "reasoning", "examples", "placement", "overrides", "model", "query"):
                if existing.get(field) != want[field]:
                    raise ValueError(f"Cannot resume: experiment setup differs for {key}: {field}")
            if existing.get("model") != model:
                raise ValueError("Cannot resume: model differs from existing CSV")
            seen.add(key)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=1, help="Repetitions, 1–10 (default: 1)")
    parser.add_argument("--output", default=None, help="CSV output; defaults to timestamped file")
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--all", action="store_true", help="Optional 24-condition factorial (168 cases/round)")
    choices.add_argument("--smoke", action="store_true", help="Low-cost check: 3 personas x 3 questions")
    parser.add_argument("--model", default=None, help="Use a compatible model; defaults to OPENAI_MODEL")
    parser.add_argument("--plan", action="store_true", help="Print plan without making API calls")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted CSV without duplicating rows")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional limit on NEW calls this invocation")
    parser.add_argument("--max-consecutive-errors", type=int, default=3,
                        help="Stop after this many consecutive API errors (default: 3)")
    args = parser.parse_args()

    if not 1 <= args.rounds <= 10:
        parser.error("--rounds must be between 1 and 10")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    if args.max_consecutive_errors < 1:
        parser.error("--max-consecutive-errors must be positive")
    if args.resume and not args.output:
        parser.error("--resume requires an explicit --output path")

    plan = planned_rows(args)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else Path("experiment_outputs") / f"{'smoke' if args.smoke else 'factorial' if args.all else 'standard'}_{now}.csv"
    if not output.is_absolute():
        output = ROOT / output

    print(f"Planned: {len(plan)} scenario runs; {len({x[0] for x in plan})} configurations; "
          f"{len({x[-1]['id'] for x in plan})} questions; {args.rounds} round(s).")
    print(f"CSV: {output}")
    print("Model requests may exceed scenario count because ReAct can call the model repeatedly.")
    print("Tool writes are simulated (dry_run=True); no fabricated leads or feedback are saved.")
    if args.plan:
        return

    if output.exists() and not args.resume:
        parser.error(f"Output already exists: {output}. Use a new filename or --resume.")

    # Instantiate only AFTER --plan and output prechecks to avoid unnecessary API setup.
    agent = CedarBridgeReAct(model=args.model, dry_run=True)
    expected: dict[tuple[str, str, str], dict[str, str]] = {}
    for ci, persona, reasoning, shots, placement, overrides, round_ix, scenario in plan:
        identity = (str(ci), str(round_ix), scenario["id"])
        expected[identity] = {
            "persona": persona, "reasoning": reasoning, "examples": shots,
            "placement": placement, "overrides": json.dumps(overrides, sort_keys=True),
            "model": agent.model, "query": scenario["query"],
        }

    seen = read_existing(output, expected, agent.model) if args.resume else set()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    errors_now = 0
    consecutive_errors = 0
    matched_now = 0
    began = time.perf_counter()

    with output.open("a" if seen else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        if not seen:
            writer.writeheader()
        for index, (ci, persona, reasoning, shots, placement, overrides, round_ix, scenario) in enumerate(plan, 1):
            key = (str(ci), str(round_ix), scenario["id"])
            if key in seen:
                continue
            if args.max_runs is not None and completed_now >= args.max_runs:
                print("Reached --max-runs limit. Resume later with --resume.")
                break

            row: dict[str, Any] = dict.fromkeys(HEADERS, "")
            row.update(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scenario=scenario["id"], condition=ci, persona=persona,
                reasoning=reasoning, examples=shots, placement=placement,
                overrides=json.dumps(overrides, sort_keys=True), model=agent.model,
                round=round_ix, query=scenario["query"],
                expected_facts=scenario["expected_facts"],
                expected_tool_policy=scenario["expect"],
            )
            try:
                # No conversation history is passed: independent, matched scenarios.
                result = agent.respond(
                    scenario["query"], persona=persona, prompt_style=reasoning,
                    shot_style=shots, placement=placement, overrides=overrides,
                )
                passed, name = evaluate_proxy(result, scenario["expect"])
                row.update(
                    proxy_pass=passed, proxy_name=name,
                    model_calls=result.get("model_calls", ""),
                    tool_calls=result.get("tool_calls", ""),
                    duration_s=result.get("duration_s", ""),
                    answer=result.get("answer", ""),
                    trace_json=json.dumps(result.get("trace", []), ensure_ascii=False),
                )
                consecutive_errors = 0
                matched_now += int(passed)
                outcome = "PASS" if passed else "REVIEW"
            except Exception as exc:
                errors_now += 1
                consecutive_errors += 1
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                outcome = "ERROR"

            writer.writerow(row)
            handle.flush()  # Saves progress after every run in case of interruption.
            completed_now += 1
            print(f"[{index}/{len(plan)}] condition={ci:02d} {scenario['id']}: {outcome}", flush=True)

            if consecutive_errors >= args.max_consecutive_errors:
                print("Stopped after consecutive API/runtime errors to avoid wasted requests.")
                print("Investigate the error column before resuming. Existing error rows are retained.")
                break

    print(f"\nWrote {completed_now} new rows in {time.perf_counter()-began:.1f}s; "
          f"{matched_now} tool-policy matches; {errors_now} API/runtime errors.")
    print(f"Saved: {output}")
    print("Human factuality, helpfulness and naturalness ratings are intentionally BLANK.")
    print("Human ratings can be entered in the CSV for later manual comparison.")


if __name__ == "__main__":
    main()
