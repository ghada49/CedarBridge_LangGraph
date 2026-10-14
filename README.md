# C4 — CedarBridge Learning: Configurable Persona ReAct Agent
**EECE 503P/798S · Agent Personas & Thinking Mechanisms · Fall 2026**

## What this project actually is
This is a Chapter 4 continuation of the Chapter 3 **CedarBridge Learning** project (fictional online tutoring business). It retains Noura's business identity, official PDF/TXT grounding, a lead-collection function, feedback recording, English/Arabic intent, and a Gradio chat application. It adds a **MANUALLY IMPLEMENTED** ReAct loop inside **LangGraph**, multiple prompt-plus-configuration personas, parameter experiments, transparent action traces, testing, and an evidence-based reflection worksheet.

> **Provenance note:** The original C3 ZIP could be located in the user's Library but its raw bytes could not be copied into the working container. The included two-page `me/about_business.pdf` is a *readable reconstruction* of the textual information in the accessible C3 PDF, not an exact binary or visual copy. `me/business_summary.txt` is a new summary derived from those original facts. If you have your C3 ZIP locally, replace the two files with your exact C3 originals; no code changes are needed.

## Getting started
Use a terminal in the `CedarBridge_C4_LangGraph` directory, Python 3.10+.

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
- `lookup_business_info(question)`: read-only local keyword lookup over the approved PDF and TXT.
- `record_customer_interest(email, name, message)`: locally saves a lead, but **only** if the customer's messages contain name, email, tutoring needs, and clear consent. `dry_run=True` validates without writing.
- `record_feedback(question)`: locally records a complaint/unanswered business question; does not email any actual staff.

The deterministic protection layer rejects fabricated name/email, malformed email, missing consent, invalid tool arguments, and unknown tools. These are deliberately enforced in code rather than entrusted solely to an LLM instruction.

## Personas and experimental configurations
| Persona | Role | Temperature | Top-p | Frequency penalty | Presence penalty | Max output tokens |
|---|---|---:|---:|---:|---:|---:|
| Noura — Friendly Advisor | Explaining lessons, guiding inquiries | 0.6 | 1 | 0.1 | 0 | 450 |
| Noura — Strict Academic Expert | Price and policy precision | 0.2 | 1 | 0 | 0 | 350 |
| Noura — Cautious Helper | Consent and ambiguous requests | 0.1 | 1 | 0 | 0 | 480 |

All three use exactly the same business facts and tools. Configuration defaults are justified hypotheses, **not** empirical findings. The actual model may produce different behavior.

### Experimental dimensions
- Prompt: `direct` vs. `brief_steps` (the latter asks for a concise, user-visible action rationale; it is **not** disclosure of private model reasoning).
- Demonstration: `zero_shot` vs. `few_shot`. `WORKED_EXAMPLE` is included in the normal instructions and removed **only** for the explicit zero-shot ablation.
- Placement: persona/task instructions in `system` vs. wrapped within the current `user` turn (`in_message`). The core privacy rules and approved business knowledge remain system-level across both conditions; thus this isolates persona/task placement, not removal of safety rules.
- Model configuration: temperature, top_p, frequency_penalty, presence_penalty, max_completion_tokens and optionally model type. A dedicated `instructions_mode=compact` ablation isolates the detailed instruction block, while core safety protections remain enforced. **Top-k is not exposed by the OpenAI Chat Completions endpoint in use, and is recorded as unsupported rather than silently faked.**

OpenAI recommends normally adjusting **temperature OR top-p**, not both simultaneously. The experiment at `temperature=1.0, top_p=0.65` holds temperature at its neutral setting.

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

The experiment CSV records: scenario ID, condition, persona, mode, token configuration, number of model/tool calls, wall time, tool-use proxy score, sanitized trace, and model response. Columns for **human factuality, helpfulness and naturalness** must be completed by actually reading the outputs. Heuristic tool correctness is not a valid substitute for human evaluation or source-level factuality checks. The user should not report performance conclusions until actual live runs and manual ratings are finished.

## Suggested demonstration (3 minutes)
1. Explain CedarBridge and the three personas. Show PDF grounding and the user's question about Chemistry.
2. Ask an unknown scholarship policy question. Show `reason → record_feedback → observation → final` in the trace.
3. Ask for tutoring without permission; verify no lead is logged. Then give name, valid email, needs **and consent** and show successful record creation in the fictional local log.
4. Change from Friendly to Strict Expert, repeat the same pricing question, and discuss observed differences (do not claim changes unless the output demonstrates them).
5. Show CSV experiment conditions, tests, and a reflection conclusion supported by results.

## Evidence mapping to the homework rubric
| Rubric | File / location | Points |
|---|---|---:|
| Use case and audience | Business PDF/TXT and notebook introduction | 10 |
| ReAct manually implemented | `agent_core.py`: `decide_node`, `execute_node`, `route_after_decide`, `build_graph`; trace | 30 |
| Persona prompts + configurations | `PERSONAS` dict, Gradio persona selection | 20 |
| Config experimentation | `run_experiments.py`, output CSV and notebook | 15 |
| Clear instructions + worked example | `INSTRUCTIONS`, `WORKED_EXAMPLE`, `TOOL_SCHEMAS` | 10 |
| Creativity, clarity, reflection | Agent demo, UI, `REFLECTION_TEMPLATE.md` with actual evidence | 15 |

## Known limitations (disclose in reflection)
- Exact visual/binary C3 PDF reconstruction was not possible; facts were preserved from its accessible text.
- This environment did **not** have outbound pip network access and LangGraph was not installed, so the real LangGraph compilation/integration test and hosted API inference could not be executed here. Install dependencies in your own environment and run `python -m pytest -q`, including `tests/test_langgraph_integration.py`.
- Simple lexical business lookup cannot guarantee semantic completeness, especially for Arabic queries. The full source text is ALSO included in the model prompt.
- Consent matching is intentionally conservative, but is heuristic and not legal-grade consent management.
- Some newer OpenAI models reject certain generation parameters; use a compatible model or adjust settings explicitly, and log unsupported conditions.
- No real website search, booking, payment, email, or customer-management integration is performed.
- `brief_steps` evaluates prompting for public intermediate explanation; it does not measure inaccessible private model chain-of-thought.

## Sources
- EECE503P/798S homework: `HW C4 - Agent Personas Thinking Mechanisms(2).pdf` (uploaded by user).
- Chapter 4 notes and slides (uploaded by user).
- LangGraph StateGraph API: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- LangGraph conditional edges: https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges
- OpenAI Chat Completions / function tools: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
