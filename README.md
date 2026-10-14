# C4 — CedarBridge Learning: Configurable Persona ReAct Agent
**EECE 503P/798S · Agent Personas & Thinking Mechanisms · Fall 2026**

## What this project actually is
This is a Chapter 4 continuation of the Chapter 3 **CedarBridge Learning** project (fictional online tutoring business). It retains Noura's business identity, official PDF/TXT grounding, a lead-collection function, feedback recording, English/Arabic intent, and a Gradio chat application. It adds a **MANUALLY IMPLEMENTED** ReAct loop inside **LangGraph**, multiple prompt-plus-configuration personas, parameter experiments, transparent action traces, testing, and an evidence-based reflection worksheet.


## Getting started

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill `OPENAI_API_KEY=...`. Do not commit `.env` or real customer logs. The default model is `gpt-4.1-mini`; choose one that supports Chat Completions function tools and the selected generation arguments. API runs incur provider charges.

```bash
python -m pytest -q
python app.py
```

Open the local URL Gradio prints. Change persona, prompt variant and temperature. The trace panel shows model decisions and tool observations **without tool arguments/customer data**.

Notebook option:

```bash
jupyter notebook C4_CedarBridge_ReAct.ipynb
```

Execute it from this project folder. The notebook imports `agent_core.py`, so keep the files together.

## Manual ReAct implementation — inspect these 4 methods
| Source | Where to look | Why it meets requirement |
|---|---|---|
| `agent_core.py` | `decide_node()` | Sends system instructions, chat history and **documented tools** to LLM; explicitly detects `message.tool_calls` |
| `agent_core.py` | `execute_node()` | Parses model-suggested JSON args; **our code** calls the tools and appends the observation as `role=tool` |
| `agent_core.py` | `route_after_decide()` | Manually chooses `execute` or END; no prebuilt agent/tool executor |
| `agent_core.py` | `build_graph()` | `StateGraph` nodes/conditional edges implement `reason → act+observe → reason` |

The prompt and tool descriptions are in `INSTRUCTIONS`, `WORKED_EXAMPLE`, and `TOOL_SCHEMAS`. `max_model_calls` and `max_tool_calls` bound loops. The code deliberately avoids `AgentExecutor`, `ToolNode`, and prebuilt ReAct agents.

### Tools
- `lookup_business_info(question)`: read-only access to the approved PDF and TXT contents for semantic interpretation.
- `record_customer_interest(email, name, message)`: locally saves a lead, but **only** if the customer's messages contain name, email, tutoring needs, and clear consent. `dry_run=True` validates without writing.
- `record_feedback(question)`: locally records a complaint/unanswered business question; does not email any actual staff.

The tool layer validates inputs and rejects unknown tools. Saving a lead also requires customer-provided details, a valid email, and a consent verdict; the code blocks storage whenever those checks fail.

## Personas and experimental configurations
| Persona | Role | Temperature | Top-p | Frequency penalty | Presence penalty | Max output tokens |
|---|---|---:|---:|---:|---:|---:|
| Noura — Friendly Advisor | Explaining lessons, guiding inquiries | 0.6 | 1 | 0.1 | 0 | 450 |
| Noura — Strict Academic Expert | Price and policy precision | 0.2 | 1 | 0 | 0 | 350 |
| Noura — Cautious Helper | Consent and ambiguous requests | 0.1 | 1 | 0 | 0 | 480 |

All three use exactly the same business facts and tools. Configuration defaults are justified hypotheses, **not** empirical findings. The actual model may produce different behavior.

### Experimental dimensions
- Prompt placement and style (91-run study): `direct` vs. `brief_steps`; the latter requests a short action plan, not hidden model reasoning.
- Separate CoT-style ablation (14-run study): `run_cot_ablation.py` compares the unchanged direct prompt with the same prompt plus an explicit four-step verification checklist. It requests only a brief, user-visible `Decision basis:` when useful; it does not expose private chain-of-thought.
- Demonstration: `zero_shot` vs. `few_shot`. `WORKED_EXAMPLE` is included in the normal instructions and removed **only** for the explicit zero-shot ablation.
- Placement: persona/task instructions in `system` vs. wrapped within the current `user` turn (`in_message`). The core privacy rules and approved business knowledge remain system-level across both conditions; thus this isolates persona/task placement, not removal of safety rules.
- Model configuration: temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens and optionally model type. A dedicated `instructions_mode=compact` ablation isolates the detailed instruction block, while core safety protections remain enforced. **Top-k is not exposed by the OpenAI Chat Completions endpoint in use, and is recorded as unsupported rather than silently faked.**

The top-p experiment changes `top_p` from 1.0 to 0.65 while holding the Friendly Advisor's temperature at 0.6, so only top-p differs from the baseline. Both sampling controls remain non-neutral.

### Direct versus structured CoT-style evaluation

The dedicated reasoning-prompt ablation uses the same seven questions in two matched conditions, with **Friendly Advisor**, `gpt-4.1-mini`, the same sampling settings, tools, business sources, few-shot example, and system-message placement. Only the extra structured reasoning instruction changes. All customer-data writes are simulated.

| Recorded metric | Direct | Structured CoT-style |
|---|---:|---:|
| Tool-policy checks passed | **7/7** | **7/7** |
| Mean latency | 7.66 s | 4.74 s |
| Median latency | 4.43 s | 4.42 s |
| Mean model calls per scenario | 4.14 | 4.14 |
| Mean tool calls per scenario | 1.00 | 1.00 |
| Responses containing `Decision basis:` | 0/7 | 2/7 |
| Errors | 0 | 0 |

Both conditions made the expected tool decisions in all seven scenarios. Neither established a tool-policy advantage. The structured prompt added a visible evidence summary to the pricing and unsupported-subject answers, but the application’s dry-run response handling and other reply paths sometimes omitted the requested label. One **25.95-second direct run** heavily affected its mean latency; nearly equal medians are a better indication of the observed typical latency. These results represent one run per condition/question, not a statistically reliable speed or accuracy comparison. The human-rating columns remain blank.

The recorded `unknown_policy` responses also illustrate a qualitative difference: the direct variant returned only the simulated-feedback acknowledgment, while the structured variant explicitly said the documents do not confirm the policy before acknowledging the simulated feedback. Both called the correct tools. This is an observed example, not a general proof of improved helpfulness.

Source and reproducibility: [`run_cot_ablation.py`](run_cot_ablation.py) and [`experiment_outputs/cot_ablation.csv`](experiment_outputs/cot_ablation.csv). The primary 91-case data and production agent were not changed by this ablation.

View the committed results directly; a fresh run incurs model usage. To run a new comparison without overwriting the submitted evidence:

```bash
python run_cot_ablation.py --plan
python run_cot_ablation.py --model gpt-4.1-mini --output experiment_outputs/cot_ablation_new.csv
```

### Running matched experiments
**Low-cost check** (3 persona conditions × 3 matched questions = 9 scenarios, may involve more than one model call each):
```bash
python run_experiments.py --smoke
```
**Prespecified set** (13 conditions × 7 matched questions = 91 scenarios):
```bash
python run_experiments.py --rounds 1
```
**Full prompt factorial** (3 × 2 × 2 × 2 = 24 conditions × 7 questions = 168 scenarios):
```bash
python run_experiments.py --all --rounds 1
```
To compare a second compatible model, pass `--model MODEL_NAME` and save to a separate `--output FILE` so you do not overwrite the first results. All experiment runs use `dry_run=True`, so they cannot create synthetic lead/feedback records. **Cost is not fixed**; each scenario can make several model requests due to ReAct loops.


