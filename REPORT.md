# Chapter 4 — CedarBridge Learning: Final Experiment Report

**Course:** EECE 503P/798S — Agentic Systems  
**Assignment:** Build a Custom ReAct Agent with Configurable Personas   
**Evidence:** `experiment_outputs/final_standard.csv` (91 rows; answers and traces retained); earlier `standard.csv` is the pre-redesign baseline.

## 1. Use case and system

CedarBridge Learning is a fictional online STEM tutoring business for students in Grades 7–12 in Lebanon. Its Noura agent provides grounded answers about Math, Physics and beginner Python tutoring, prices, supported languages and service hours. It can execute `lookup_business_info`, `record_customer_interest` (after context-aware permission to store contact details), and `record_feedback` (complaints and unanswered business questions). The agent must not invent business policies or book appointments without human confirmation.

The ReAct loop is manually implemented through LangGraph nodes and conditional routing. The model proposes an action, Python executes the corresponding tool, the observation is appended to state, and the model continues until a final answer. The evolved version adds semantic consent verification and answer/source-grounding audits. These are **additional model calls** and must be distinguished from ReAct tool calls. In all experiments, `dry_run=True` simulated write actions without creating lead/feedback log entries.

## 2. Experimental design

- **Model:** `gpt-4.1-mini` fixed for all final runs, to isolate persona, prompting, and sampling settings.
- **13 conditions × 7 identical independent scenarios × 1 repetition = 91 scenario runs.** Condition 0 is Friendly Advisor with direct instructions, few-shot example and system placement. Conditions 1–2 compare personas (their prompt text and default decoding settings both change); conditions 3–12 each change one named element relative to condition 0.
- **Prompt/config dimensions:** persona; direct vs brief-step guidance (an instruction-level reasoning comparison, **not inspection of private chain-of-thought**); zero-shot vs few-shot; system vs in-message instructions; temperature 0.2/0.9; top-p 0.65; frequency/presence penalties 0.5; max completion tokens 250; compact vs full instructions.
- **Scenarios:** combined pricing arithmetic and Sunday restrictions, unsupported Chemistry/university subject, undocumented scholarship question, lead without consent, lead with explicit consent, customer complaint, and Arabic-language Math question.
- **Primary measured metric:** `proxy_pass` — whether expected write tools executed exactly once, or unauthorized/unnecessary write attempts were absent. This does **not** score the truthfulness, completeness or naturalness of the response. Other recorded metrics are model calls, tool calls, latency, final answer, and ReAct/audit trace.
- **Study boundaries:** no multiple random-seed trials; seven cases are a deliberately small set; no separate model-family comparison. The experiment compares within the specified model rather than claiming broad generalization.

## 3. Overall results

| Metric | Final result |
|---|---:|
| Completed scenario runs | 91/91 |
| Tool-policy checks passed | 84/91 (**92.3%**) |
| Tool-policy checks failed | 7/91 |
| API or runtime errors | 0 |
| Total model requests, including verification calls | 370 |
| Tool observations | 84 |
| Mean scenario latency | 4.60 s |
| Median scenario latency | 4.43 s |
| Total measured scenario time | 418.95 s |

### 3.1 By scenario

| Scenario | Tool-policy passes |
|---|---:|
| multi_step_pricing | 13/13 |
| unoffered_subject | 13/13 |
| unknown_policy | 6/13 |
| lead_no_consent | 13/13 |
| lead_with_consent | 13/13 |
| customer_complaint | 13/13 |
| arabic | 13/13 |

All seven failed checks were the **unresolved scholarship policy** scenario. They all omitted `record_feedback`. Conversely, all 13 cases with explicit lead consent successfully invoked `record_customer_interest` in simulation, and all 13 cases without consent refrained from saving it. Pricing, subject boundaries, complaints, and Arabic responses also passed the tool-policy proxy in every condition.

### 3.2 By configuration

| ID | Configuration relative to Friendly baseline | Passes | Rate | Mean latency (s) | Model calls |
|---:|---|---:|---:|---:|---:|
| 00 | Friendly baseline: direct / few-shot / system | 7/7 | 100.0% | 5.11 | 30 |
| 01 | Strict Academic Expert (persona defaults) | 6/7 | 85.7% | 4.18 | 28 |
| 02 | Cautious Helper (persona defaults) | 6/7 | 85.7% | 4.77 | 28 |
| 03 | Brief-step reasoning instructions | 7/7 | 100.0% | 4.88 | 29 |
| 04 | Zero-shot instructions | 6/7 | 85.7% | 4.85 | 27 |
| 05 | Persona instructions in message | 6/7 | 85.7% | 4.75 | 28 |
| 06 | Temperature 0.2 | 7/7 | 100.0% | 4.43 | 29 |
| 07 | Temperature 0.9 | 6/7 | 85.7% | 4.52 | 28 |
| 08 | Top-p 0.65 | 6/7 | 85.7% | 4.52 | 28 |
| 09 | Frequency penalty 0.5 | 7/7 | 100.0% | 4.49 | 29 |
| 10 | Presence penalty 0.5 | 7/7 | 100.0% | 4.44 | 29 |
| 11 | Max completion tokens 250 | 7/7 | 100.0% | 4.55 | 29 |
| 12 | Compact instructions | 6/7 | 85.7% | 4.35 | 28 |

**Interpretation:** Condition 0 (Friendly Advisor, system instructions, few-shot, direct guidance) obtained 7/7. Brief-step instructions, temperature 0.2, frequency penalty 0.5, presence penalty 0.5, and a 250-token output cap also obtained 7/7. Zero-shot, in-message instructions, higher temperature, top-p 0.65, and compact instructions each obtained 6/7 because of the same policy-follow-up case. The Expert and Cautious personas obtained 6/7. Because there is only one run per matched scenario and several conditions tie, these outcomes do **not** demonstrate statistically reliable superiority of any particular parameter.

## 4. Representative ReAct observations

**Successful fact lookup:** For a question comparing four single lessons with the package and asking about Sunday at 8 PM, the agent used `lookup_business_info` and correctly calculated $100 versus $90, a $10 saving. It rejected Sunday because CedarBridge is closed that day. The final answer did not assert a confirmed booking.

**Lead authorization:** When a user provided a name, email and tutoring need without explicit consent, the agent requested permission and made no write call. With explicit consent, the model requested `record_customer_interest`, and the validator returned `ok` in dry-run mode. This corrects the original version's brittle consent validation without requiring the customer to recite an exact phrase.

**Feedback tool selection:** On the undocumented 35% scholarship question, six conditions chose `lookup_business_info → record_feedback → final_answer`; seven chose `lookup_business_info → final_answer`. In the seven failed cases, the grounding audit recorded `coverage=resolved` and `independent_coverage=resolved`, even though the sources did not establish whether the scholarship exists. The two semantic checks thus exhibited the **same failure mode**, rather than reliably correcting it.

**Simulation and response-quality limitation:** In several successful write-tool tests, the final answer was only a simulation acknowledgment, e.g., "Your feedback was validated in simulation only; it was not stored or sent to staff." That is honest about `dry_run=True`, but in the scholarship scenario it omits the substantive answer: the business documents do not confirm the scholarship. **A tool-policy PASS is not equivalent to a fully helpful response.**

## 5. Persona and prompting comparison

**Friendly Advisor:** The selected default for student-facing interactions. In normal factual responses it offered clear, conversational follow-up questions; its final tool-policy score was 7/7. The Strict Academic Expert produced more concise, fact-oriented answers and cited the business summary, while the Cautious Helper emphasized careful permission and follow-up. Because the three human-rating columns in the CSV are blank, describing Friendly as *most natural* is an **informed qualitative preference**, not a measured human preference or validated numerical score.

**Reasoning instructions:** Direct and brief-step prompting both achieved 7/7 for the Friendly persona. The arithmetic answers were correct under both; brief-step guidance did not establish a measurable tool-policy benefit on this small set. No private CoT was inspected.

**Examples and placement:** Few-shot/system achieved 7/7, zero-shot/system 6/7, and few-shot/in-message 6/7. The difference is localized to recording the undocumented policy question; remaining answers were similar. This is limited evidence favoring the default configuration, not a proof that few-shot or system placement is generally superior.

**Generation parameters:** Lowering temperature to 0.2 retained 7/7; raising it to 0.9 returned 6/7. The top-p, penalties and output-token variants generally preserved factual-service answers, with some differences in recording the unanswered policy case. Differences among 6/7 and 7/7 in a seven-item, single-round experiment are too small to justify strong claims about decoding configurations.

**Instructions:** Full instructions (7/7) vs compact instructions (6/7) suggests the longer operational guidance may have helped with the unresolved-policy workflow in this instance. The result is limited by sample size and stochastic variation.

## 6. Development-stage comparison

| Version / stage | Tool-policy passes | Mean scenario latency | Model requests |
|---|---:|---:|---:|
| Pre-redesign 91-run baseline | 65/91 (71.4%) | 2.21 s | 168 |
| Final frozen version, matched 91 cases | 84/91 (92.3%) | 4.60 s | 370 |

The final version recorded **19 more passes**, a **20.9-percentage-point absolute increase**. The improvement coincided with several architecture changes—semantic consent validation, source-grounding checks, and revised instructions—so it cannot be causally credited to any single addition. It required roughly **2.2× as many model requests** and **2.1× the mean latency**. Dollar spending was **not** recorded in the CSV, so no measured API cost is reported.

## 7. Evaluation limitations and future work

1. The primary score measures tool-policy behavior only; factuality, usefulness and naturalness must be evaluated from answers. The stored `human_*` fields are blank, so no fabricated human evaluation figures are included.
2. The source-grounding audits can agree incorrectly: both labeled the missing scholarship policy "resolved" in failed cases. Agreement between LLM verifiers is not proof of correctness.
3. Some `dry_run` response rewrites prioritized avoiding false claims about data persistence but failed to answer the user's substantive question. Real application behavior and simulated evaluation mode should be distinguished.
4. Tests are one-turn, independent and limited to seven situations, not a production benchmark or statistically representative sample. Natural multi-turn consent and unforeseen policies require separate future evaluation.
5. Model type was held fixed to isolate settings. The assignment's "model type" dimension can be acknowledged as a possible extension rather than claiming it was tested.

## 8. Conclusion

The agent implements a manual, traceable ReAct workflow with configurable personas and controlled prompt/model-parameter comparisons. The final tool-policy pass rate was 84/91 (92.3%), up from the earlier 65/91 (71.4%) baseline, with reliable handling of documented service questions, complaints and explicit lead consent in these tests. The persistent failure concerned detecting that an undocumented policy question still required follow-up, including when independent LLM audits agreed on an incorrect resolution. The principal lesson is that **tool validation, grounded uncertainty management and honest execution reporting matter at least as much as persona styling or generation settings**. The results are suitable for course-level evaluation when their scope and limitations are reported transparently.
