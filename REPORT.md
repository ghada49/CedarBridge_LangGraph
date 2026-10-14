# Chapter 4 — CedarBridge Learning: Final Experiment Report

**Course:** EECE 503P/798S — Agentic Systems  
**Assignment:** Build a Custom ReAct Agent with Configurable Personas  
**Implementation evaluated:** final frozen LangGraph + `agent_core.py` version used for `final_standard.csv`, 14 October 2026.  
**Evidence:** `experiment_outputs/final_standard.csv` (91 runs), `experiment_outputs/standard.csv` (earlier 91-run implementation), and the separate matched reasoning-prompt evaluation `experiment_outputs/cot_ablation.csv` (14 runs). All results contain responses and tool traces.

## 1. Use case and system

CedarBridge Learning is a fictional online STEM tutoring business for students in Grades 7–12 in Lebanon. Its Noura agent provides grounded answers about Math, Physics and beginner Python tutoring, prices, supported languages and service hours. It can execute `lookup_business_info`, `record_customer_interest` (after context-aware permission to store contact details), and `record_feedback` (complaints and unanswered business questions). The agent must not invent business policies or book appointments without human confirmation.

The ReAct loop is manually implemented through LangGraph nodes and conditional routing. The model proposes an action, Python executes the corresponding tool, the observation is appended to state, and the model continues until a final answer. The evolved version adds semantic consent verification and answer/source-grounding audits. These are **additional model calls** and must be distinguished from ReAct tool calls. In all experiments, `dry_run=True` simulated write actions without creating lead/feedback log entries.

## 2. Experimental design

- **Model:** `gpt-4.1-mini` fixed for all final runs, to isolate persona, prompting, and sampling settings.
- **13 conditions × 7 identical independent scenarios × 1 repetition = 91 scenario runs.** Condition 0 is Friendly Advisor with direct instructions, few-shot example and system placement. Conditions 1–2 compare personas (their prompt text and default decoding settings both change); conditions 3–12 each change one named element relative to condition 0.
- **Prompt/config dimensions:** persona; direct vs brief-step guidance (an instruction-level reasoning comparison, **not inspection of private chain-of-thought**); zero-shot vs few-shot; system vs in-message instructions; temperature 0.2/0.9; top-p 0.65; frequency/presence penalties 0.5; max completion tokens 250; compact vs full instructions.
- **Scenarios:** combined pricing arithmetic and Sunday restrictions, unsupported Chemistry/university subject, undocumented scholarship question, lead without consent, lead with explicit consent, customer complaint, and Arabic-language Math question.
- **Primary measured metric:** `proxy_pass` — whether expected write tools executed exactly once, or unauthorized/unnecessary write attempts were absent. This does **not** score the truthfulness, completeness or naturalness of the response. Other recorded metrics are model calls, tool calls, latency, final answer, and ReAct/audit trace.
- **Additional reasoning-prompt comparison:** a separate, matched 14-run experiment compares the standard direct prompt with the same prompt plus a four-step CoT-style verification instruction. It keeps the Friendly Advisor, model, generation settings, documents, tools, examples and prompt placement fixed. This experiment is reported independently in Section 7.
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

**Reasoning instructions:** In the 91-run comparison, direct and `brief_steps` prompting both achieved 7/7 tool-policy checks for the Friendly persona. The `brief_steps` mode prompts a short, user-visible action plan, rather than explicitly intervening on private reasoning. A **separate controlled direct-versus-structured CoT-style ablation** (Section 7) adds an explicit evidence-verification checklist to an otherwise unchanged prompt. Neither study reveals the model's private chain-of-thought.

**Examples and placement:** Few-shot/system achieved 7/7, zero-shot/system 6/7, and few-shot/in-message 6/7. The difference is localized to recording the undocumented policy question; remaining answers were similar. This is limited evidence favoring the default configuration, not a proof that few-shot or system placement is generally superior.

**Generation parameters:** Lowering temperature to 0.2 retained 7/7; raising it to 0.9 returned 6/7. The top-p, penalties and output-token variants generally preserved factual-service answers, with some differences in recording the unanswered policy case. Differences among 6/7 and 7/7 in a seven-item, single-round experiment are too small to justify strong claims about decoding configurations.

**Instructions:** Full instructions (7/7) vs compact instructions (6/7) suggests the longer operational guidance may have helped with the unresolved-policy workflow in this instance. The result is limited by sample size and stochastic variation.

## 6. Comparison with the earlier committed experiment

The numbers below compare the **two 91-run implementation-stage CSVs**, which are separate from the 14-run reasoning-prompt ablation.

| Committed experiment file | Tool-policy passes | Mean scenario latency | Model requests |
|---|---:|---:|---:|
| `experiment_outputs/standard.csv` (earlier implementation) | 77/91 (84.6%) | 6.45 s | 305 |
| `experiment_outputs/final_standard.csv` (submitted implementation) | 84/91 (92.3%) | 4.60 s | 370 |

The submitted version recorded **7 additional matches**, an increase of **7.7 percentage points**, while making 65 more model requests (about 21% more). Its observed mean latency was **1.85 seconds lower** in this run. Latency is influenced by network and service conditions, so the difference should not be attributed to a particular code change. The architecture and prompts changed between runs; this is a **development-stage comparison**, not an isolated causal experiment. Dollar spending was not recorded, so no measured API cost is asserted.

### 6.1 Qualitative observations from representative outputs

The stored final answers provide a useful *qualitative* cross-check, but no human 1–5 scores were entered in the evaluation CSV:

| Matched situation | Friendly Advisor | Strict Academic Expert | Cautious Helper |
|---|---|---|---|
| Four-lesson pricing and Sunday hours | Correct arithmetic and clear follow-up invitation | Correct arithmetic with more formal source-oriented wording | Correct arithmetic with cautious scheduling language |
| Inquiry without storage consent | Requests scoped permission instead of writing a lead | Requests explicit authorization | Requests explicit authorization |
| Undocumented scholarship | Invokes feedback in the baseline condition, but its simulation acknowledgment does not fully answer the question | States that scholarship is undocumented but fails to log the unanswered question | Same unresolved feedback-logging failure |

These are **reviewer observations**, not human rating measurements. Tool-policy compliance and customer-facing answer completeness should be reported separately. A grader can inspect the corresponding `answer` and `trace_json` entries in `final_standard.csv` to reproduce the interpretations.

## 7. Direct versus structured CoT-style prompting

The separate evaluation in `experiment_outputs/cot_ablation.csv` contains **14 completed runs without runtime errors**: seven matched business scenarios tested once with the existing direct prompt and once with an otherwise identical prompt augmented by the checklist in `run_cot_ablation.py`. The variant order alternates by scenario. Both use the Friendly Advisor, `gpt-4.1-mini`, persona-default settings, a few-shot example and system-level placement. `agent_core.py` is unchanged, and write actions remain in `dry_run=True` mode.

The added prompt asks the model to identify the real question, look for supporting sources or tool observations, check arithmetic and consent/action constraints, and choose a supported action or final answer. It requests a short optional `Decision basis:` sentence about observable evidence. **It does not expose or measure private internal chain-of-thought**; the experiment tests the behavioral effect of an explicit CoT-style reasoning instruction.

| Metric | Direct | Structured CoT-style |
|---|---:|---:|
| Completed runs | 7/7 | 7/7 |
| Tool-policy passes | **7/7** | **7/7** |
| Mean latency | 7.66 s | 4.74 s |
| Median latency | 4.43 s | 4.42 s |
| Mean model requests per run | 4.14 | 4.14 |
| Mean tool observations per run | 1.00 | 1.00 |
| Answers containing `Decision basis:` | 0/7 | 2/7 |
| Runtime/API errors | 0 | 0 |

The tool-policy results were identical for all seven scenarios. Both variants performed the pricing arithmetic correctly ($100 for four separate lessons, $90 for the package, $10 saved), observed Sunday closure, withheld lead storage without consent, validated the consented lead in simulation, recorded complaints and the unanswered scholarship question in simulation, and answered the Arabic service question in Arabic.

The structured variant included an explicit evidence-basis label for the pricing and unsupported-subject responses. It did not consistently produce that label elsewhere; `dry_run` response replacement is one reason the requested presentation can be absent. On the scholarship scenario, both variants called the correct logging tool; only the structured final reply explicitly said the documents did not confirm the policy before acknowledging the simulated feedback. This illustrates a potential answer-completeness gain in one case, **not** a scored helpfulness improvement.

The direct condition contained a 25.95-second pricing run. It inflates the direct mean relative to the structured mean, while the medians (4.43 s vs 4.42 s) were nearly identical. Accordingly, the results **do not establish lower latency or improved tool correctness from CoT prompting**. The small single-round sample, blank human-rating fields, and fixed model limit broader conclusions.

## 8. Evaluation limitations and future work

1. The primary score measures tool-policy behavior only; factuality, usefulness and naturalness must be evaluated from answers. The stored `human_*` fields are blank, so no fabricated human evaluation figures are included.
2. The source-grounding audits can agree incorrectly: both labeled the missing scholarship policy "resolved" in failed cases. Agreement between LLM verifiers is not proof of correctness. The `lookup_business_info` tool description also says it retrieves short relevant excerpts, whereas this version supplies the entire approved source texts; this documented mismatch remains in the frozen evaluated code.
3. Some `dry_run` response rewrites prioritized avoiding false claims about data persistence but failed to answer the user's substantive question. Real application behavior and simulated evaluation mode should be distinguished.
4. Each experimental comparison uses a small set of independent one-turn scenarios, with one observation per matched question/configuration. The 14-run CoT-style prompt ablation does not measure hidden reasoning, and its latency means are sensitive to an outlier. Multi-turn consent and unforeseen policies require separate future evaluation.
5. Model type was held fixed to isolate settings. The assignment's "model type" dimension can be acknowledged as a possible extension rather than claiming it was tested.

## 9. Conclusion

The agent implements a manual, traceable ReAct workflow with configurable personas and controlled prompt/model-parameter comparisons. The final tool-policy pass rate was 84/91 (92.3%), compared with 77/91 (84.6%) in the earlier implementation-stage CSV, with strong handling of documented service questions, complaints and explicit lead consent in these tests. A separate matched CoT-style prompt evaluation achieved 7/7 tool-policy checks under both direct and structured conditions, without establishing a measurable advantage in tool correctness. The persistent failure concerned detecting that an undocumented policy question still required follow-up, including when independent LLM audits agreed on an incorrect resolution. The principal lesson is that **tool validation, grounded uncertainty management and honest execution reporting matter at least as much as persona styling or generation settings**. The results are suitable for course-level evaluation when their scope and limitations are reported transparently.
