"""C4 matched prompting ablation: direct answers versus structured (CoT-style) reasoning.

Run from the CedarBridge project root:
    python run_cot_ablation.py --plan
    python run_cot_ablation.py --model gpt-4.1-mini --output experiment_outputs/cot_ablation.csv

The main agent_core.py is NOT modified. Both conditions use the same underlying
LangGraph agent and settings; only one system-level reasoning instruction differs.
Customer-data tools always run in dry-run mode.

The structured condition requests a brief evidence-based DECISION SUMMARY, not
private/internal chain-of-thought. A prompt cannot establish what private
reasoning the model actually used; evaluate observable actions and answers.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from agent_core import CedarBridgeReAct
from run_experiments import SCENARIOS, evaluate_proxy

ROOT = Path(__file__).resolve().parent

# Isolated independent variable. The normal system prompt, tools, persona,
# safety rules, worked example, model and decoding settings remain identical.
STRUCTURED_REASONING_INSTRUCTION = """
EXPERIMENTAL THINKING MECHANISM — STRUCTURED REASONING (CoT-STYLE):
Before answering or taking any tool action, think through this short sequence:
(1) Identify the customer's actual question and any missing information.
(2) Determine which business source or tool observation would verify the facts.
(3) Check any arithmetic, source support, consent, and action constraints.
(4) Choose the tool action or final answer based on what was verified.
Keep this thinking private. Do NOT reproduce private chain-of-thought.
In your final customer-facing reply, answer the question directly and, where
meaningful, add one short sentence starting 'Decision basis:' that summarizes
only the verified facts or observable tool outcomes supporting the decision.
Do not invent evidence, add unneeded tools, or mistake a simulated write for
real storage or staff follow-up. All original CedarBridge rules still apply.
""".strip()


class CoTComparisonAgent(CedarBridgeReAct):
    """Inject only the reasoning treatment; preserve the production agent."""

    structured_reasoning: bool = False

    def _prompt_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        messages = super()._prompt_messages(state)
        if self.structured_reasoning:
            # Copy the message object, rather than mutating state/history.
            messages[0] = dict(messages[0])
            messages[0]["content"] = (
                str(messages[0]["content"]) + "\n\n" + STRUCTURED_REASONING_INSTRUCTION
            )
        return messages


HEADERS = [
    "timestamp_utc", "scenario", "variant", "persona", "prompt_style", "shot_style",
    "placement", "model", "proxy_pass", "proxy_name", "model_calls",
    "tool_calls", "duration_s", "decision_basis_present", "error", "answer",
    "trace_json", "query", "expected_facts", "expected_tool_policy",
    "human_factuality_1to5", "human_helpfulness_1to5", "human_naturalness_1to5",
]


def planned() -> list[tuple[dict[str, str], str]]:
    # Alternating order reduces a systematic direct-first ordering effect.
    result = []
    for index, scenario in enumerate(SCENARIOS):
        variants = ("direct", "structured_cot") if index % 2 == 0 else ("structured_cot", "direct")
        for variant in variants:
            result.append((scenario, variant))
    return result


def read_existing(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if set(HEADERS) - set(reader.fieldnames or []):
            raise ValueError("Cannot resume: incompatible CSV columns")
        rows = list(reader)
    keys = [(r["scenario"], r["variant"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Cannot resume: duplicate scenario/variant rows")
    if not set(keys).issubset({(s["id"], v) for s, v in planned()}):
        raise ValueError("Cannot resume: unexpected scenario/variant")
    return set(keys)


def summarize(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print("\nObserved results (single trial per matched scenario):")
    for variant in ("direct", "structured_cot"):
        group = [r for r in rows if r["variant"] == variant]
        good = [r for r in group if not r["error"]]
        passes = sum(r["proxy_pass"].lower() == "true" for r in good)
        mean_duration = mean(float(r["duration_s"]) for r in good) if good else float("nan")
        mean_calls = mean(float(r["model_calls"]) for r in good) if good else float("nan")
        basis = sum(r["decision_basis_present"].lower() == "true" for r in good)
        print(
            f"  {variant:14s}: {passes}/{len(good)} tool-policy passes, "
            f"{mean_duration:.2f}s mean latency, {mean_calls:.2f} mean model calls, "
            f"{basis}/{len(good)} final responses with a 'Decision basis' phrase, "
            f"{len(group) - len(good)} errors"
        )
    print("Reminder: tool-policy matches and visible summaries are not proof of factual quality or private reasoning.")
    print("For the homework, manually compare pricing accuracy, scholarship escalation, and response helpfulness.")



def write_markdown_report(path: Path) -> Path:
    """Create a report-ready addendum from OBSERVED CSV values only."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(r["scenario"], r["variant"]): r for r in rows}
    outcomes = []
    for scenario in SCENARIOS:
        scores = []
        for v in ("direct", "structured_cot"):
            r = indexed.get((scenario["id"], v))
            scores.append("NOT RUN" if r is None else "ERROR" if r["error"] else "PASS" if r["proxy_pass"].lower() == "true" else "REVIEW")
        outcomes.append(f"| `{scenario['id']}` | {scores[0]} | {scores[1]} |")

    aggregates = {}
    for variant in ("direct", "structured_cot"):
        group = [r for r in rows if r["variant"] == variant and not r["error"]]
        aggregates[variant] = {
            "n": len(group),
            "passed": sum(r["proxy_pass"].lower() == "true" for r in group),
            "duration": mean(float(r["duration_s"]) for r in group) if group else float("nan"),
            "calls": mean(float(r["model_calls"]) for r in group) if group else float("nan"),
            "tools": mean(float(r["tool_calls"]) for r in group) if group else float("nan"),
            "summary": sum(r["decision_basis_present"].lower() == "true" for r in group),
        }
    complete = len(rows) == 14 and not any(r["error"] for r in rows)
    a, b = aggregates["direct"], aggregates["structured_cot"]
    text = f"""# C4 — Direct versus structured-reasoning prompt ablation

**Status:** {'Complete (14/14)' if complete else f'Incomplete ({len(rows)}/14 records; inspect errors before reporting final results)'}.

## Method

- Same `CedarBridgeReAct` LangGraph agent, tools, business sources, **Friendly Advisor** persona, `gpt-4.1-mini`, persona-default sampling settings, few-shot worked example, and system-level instructions.
- **Direct:** original `prompt_style='direct'`, no extra reasoning scaffold.
- **Structured (CoT-style):** exactly the same base prompt plus a checklist to identify the question, verify sources/tools, check arithmetic/consent, and select the supported action. Requests a short, observable `Decision basis:` summary; **no private chain-of-thought is displayed**.
- Seven predeclared scenarios run once per variant (14 fresh calls); order alternates by scenario; tool writes run only as simulation.
- Metrics: tool-policy proxy success (not factual accuracy), mean duration, model calls, tool observations, and occurrence of a visible decision-basis label. Responses require separate qualitative review.

## Observed summary

| Metric | Direct | Structured CoT-style |
|---|---:|---:|
| Tool-policy passes | {a['passed']}/{a['n']} | {b['passed']}/{b['n']} |
| Mean latency (s) | {a['duration']:.2f} | {b['duration']:.2f} |
| Mean model calls | {a['calls']:.2f} | {b['calls']:.2f} |
| Mean tool observations | {a['tools']:.2f} | {b['tools']:.2f} |
| Responses containing `Decision basis:` | {a['summary']}/{a['n']} | {b['summary']}/{b['n']} |

## Per-scenario tool-policy results

| Scenario | Direct | Structured CoT-style |
|---|---|---|
""" + "\n".join(outcomes) + """

## Interpretation and limitations

This is an explicit **prompt intervention** testing a CoT-style stepwise verification instruction against no additional reasoning instruction. Only externally observable model behavior is measured. A brief evidence summary must not be mistaken for access to private internal reasoning. This single-round, seven-scenario comparison is exploratory; it is not a statistically powered estimate of improvement. The instruction can change wording without improving tool correctness, and enforced dry-run acknowledgments can hide the intended summary in some responses.

**Human review before submission:** Inspect both variants' pricing arithmetic, unsupported-policy handling and escalation, consent compliance, Arabic response, and response naturalness. Do not fill human-rating columns without conducting the evaluation.

## Reproduce

```powershell
python run_cot_ablation.py --model gpt-4.1-mini --output experiment_outputs/cot_ablation.csv
```

The experimental prompt and full traces are preserved in `run_cot_ablation.py` and `experiment_outputs/cot_ablation.csv`. The existing `agent_core.py` and final 91-case evaluation remain unchanged.
"""
    target = ROOT / "COT_ABLATION_RESULTS.md"
    target.write_text(text, encoding="utf-8")
    print(f"Report addendum saved: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--output", default="experiment_outputs/cot_ablation.csv")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue incomplete results without duplicate API calls")
    args = parser.parse_args()

    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    print(f"Planned: 14 runs, 7 identical scenarios per condition; model={args.model}; dry_run=True")
    print("Compared: baseline direct system prompt versus identical prompt plus CoT-style evidence checklist.")
    print(f"Output: {path}")
    if args.plan:
        return
    if path.exists() and not args.resume:
        parser.error("Output file already exists. Use --resume for a partial run or choose another filename.")
    seen = read_existing(path) if args.resume else set()
    path.parent.mkdir(parents=True, exist_ok=True)

    # A single agent is reused. Switching a flag only affects _prompt_messages
    # during the subsequent synchronous respond() call.
    agent = CoTComparisonAgent(model=args.model, dry_run=True)
    with path.open("a" if seen else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        if not seen:
            writer.writeheader()
        for i, (scenario, variant) in enumerate(planned(), 1):
            if (scenario["id"], variant) in seen:
                continue
            agent.structured_reasoning = variant == "structured_cot"
            row: dict[str, Any] = dict.fromkeys(HEADERS, "")
            row.update(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scenario=scenario["id"], variant=variant, persona="friendly",
                prompt_style="direct", shot_style="few_shot", placement="system",
                model=args.model, query=scenario["query"],
                expected_facts=scenario["expected_facts"],
                expected_tool_policy=scenario["expect"],
            )
            try:
                result = agent.respond(
                    scenario["query"], persona="friendly", prompt_style="direct",
                    shot_style="few_shot", placement="system",
                )
                passed, proxy_name = evaluate_proxy(result, scenario["expect"])
                answer = result.get("answer", "") or ""
                row.update(
                    proxy_pass=passed, proxy_name=proxy_name,
                    model_calls=result.get("model_calls", ""),
                    tool_calls=result.get("tool_calls", ""),
                    duration_s=result.get("duration_s", ""),
                    answer=answer,
                    decision_basis_present="decision basis:" in answer.lower(),
                    trace_json=json.dumps(result.get("trace", []), ensure_ascii=False),
                )
                outcome = "PASS" if passed else "REVIEW"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"[:300]
                outcome = "ERROR"
            writer.writerow(row)
            stream.flush()
            print(f"[{i:02d}/14] {variant:14s} {scenario['id']}: {outcome}", flush=True)
            if outcome == "ERROR":
                print("Stopping on first runtime/API error. Inspect CSV and restart with --resume.")
                break
    summarize(path)
    write_markdown_report(path)


if __name__ == "__main__":
    main()
