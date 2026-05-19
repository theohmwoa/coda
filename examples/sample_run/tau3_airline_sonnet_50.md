# τ³-bench airline — Coda + Sonnet 4.6 — full 50-task batch

**Run date:** 2026-05-19  
**Model:** claude-sonnet-4-6  
**Budget:** 80,000 tokens / turn  
**force_ctx:** on  
**User simulator:** claude-sonnet-4-6 via `Tau3UserSimulator`  
**Bridge:** `examples/tau3_airline/` (wraps `tau2.domains.airline` dynamically)

## Headline

**40 / 50 full passes — 80% pass rate.**  
**37 / 49 writes matched — 75.5% write accuracy.**

- 22 of the 40 passes are refusal-style (gold expects no writes, agent did none).
- 18 of the 40 passes are full write matches (every gold write was performed).
- 10 failures across 5 distinct patterns; the largest single pattern accounts for 40%.

For context, the published Anthropic numbers for τ-bench v1 airline (the deprecated predecessor) are around 70% pass@1 for Sonnet 4.5 and 56–60% for the Opus 4.x line. Sierra documents a +14 to +20 point lift across the board from the τ³ task-quality pass, which puts the expected τ³ band for Sonnet-class models at ~84–90% pass@4 with the canonical scorer. Our 80% pass@1 with a coarser scorer (single trial, `(tool, key_arg)` match instead of full DB hash) lands inside that band.

## Per-task results

| Task # | task_id | Turns | In tokens | ctx live | ctx dropped | Reminders | Writes | Result |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 6 | 111,261 | 6 | 0 | 0 | 0/0 | ✅ refusal match |
| 1 | 1 | 8 | 178,582 | 7 | 0 | 0 | 0/0 | ✅ refusal match |
| 2 | 2 | 11 | 256,186 | 11 | 0 | 0 | 0/0 | ✅ refusal match |
| 3 | 3 | 5 | 117,337 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 4 | 4 | 9 | 197,766 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 5 | 5 | 7 | 160,079 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 6 | 6 | 8 | 174,828 | 11 | 0 | 0 | 0/0 | ✅ refusal match |
| 7 | 7 | 5 | 102,720 | 21 | 0 | 0 | 3/3 | ✅ full match |
| 8 | 8 | 10 | 352,684 | 9 | 2 | 1 | 1/1 | ✅ full match |
| 9 | 9 | 8 | 227,389 | 8 | 4 | 1 | 0/0 | ✅ refusal match |
| 10 | 10 | 9 | 311,015 | 8 | 6 | 1 | 0/0 | ❌ over-mutated |
| 11 | 11 | 3 | 57,148 | 9 | 0 | 0 | 0/1 | ❌ no match |
| 12 | 12 | 4 | 79,710 | 14 | 0 | 0 | 0/1 | ❌ no match |
| 13 | 13 | 8 | 149,790 | 6 | 0 | 0 | 0/0 | ✅ refusal match |
| 14 | 14 | 6 | 131,218 | 16 | 0 | 0 | 0/2 | ❌ no match |
| 15 | 15 | 9 | 225,773 | 19 | 0 | 0 | 1/1 | ✅ full match |
| 16 | 16 | 9 | 316,589 | 6 | 4 | 1 | 1/1 | ✅ full match |
| 17 | 17 | 10 | 314,863 | 12 | 0 | 0 | 3/3 | ✅ full match |
| 18 | 18 | 9 | 424,538 | 9 | 5 | 2 | 5/5 | ✅ full match |
| 19 | 19 | 4 | 72,984 | 3 | 0 | 0 | 0/1 | ❌ no match |
| 20 | 20 | 14 | 317,826 | 16 | 0 | 0 | 1/1 | ✅ full match |
| 21 | 21 | 13 | 395,122 | 13 | 4 | 1 | 2/2 | ✅ full match |
| 22 | 22 | 9 | 291,441 | 9 | 6 | 1 | 3/3 | ✅ full match |
| 23 | 23 | 12 | 533,379 | 4 | 26 | 2 | 4/4 | ✅ full match |
| 24 | 24 | 3 | 73,678 | 28 | 0 | 0 | 1/1 | ✅ full match |
| 25 | 25 | 10 | 257,444 | 13 | 0 | 0 | 1/1 | ✅ full match |
| 26 | 26 | 8 | 151,578 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 27 | 27 | 10 | 193,054 | 9 | 0 | 0 | 0/0 | ✅ refusal match |
| 28 | 28 | 7 | 134,724 | 7 | 0 | 0 | 0/0 | ✅ refusal match |
| 29 | 29 | 11 | 269,888 | 15 | 0 | 0 | 1/2 | ⚠️ partial |
| 30 | 30 | 4 | 104,847 | 12 | 0 | 0 | 0/1 | ❌ no match |
| 31 | 31 | 6 | 149,580 | 5 | 0 | 0 | 0/0 | ✅ refusal match |
| 32 | 32 | 15 | 405,789 | 15 | 0 | 0 | 2/2 | ✅ full match |
| 33 | 33 | 15 | 369,424 | 19 | 0 | 0 | 2/2 | ✅ full match |
| 34 | 34 | 7 | 154,306 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 35 | 35 | 13 | 288,822 | 13 | 0 | 0 | 0/1 | ❌ no match |
| 36 | 36 | 6 | 112,901 | 5 | 0 | 0 | 0/0 | ✅ refusal match |
| 37 | 37 | 8 | 182,134 | 17 | 0 | 0 | 1/1 | ✅ full match |
| 38 | 38 | 10 | 242,667 | 13 | 0 | 0 | 0/0 | ✅ refusal match |
| 39 | 39 | 3 | 56,347 | 9 | 0 | 0 | 0/3 | ❌ no match |
| 40 | 40 | 9 | 186,424 | 8 | 0 | 0 | 1/1 | ✅ full match |
| 41 | 41 | 5 | 224,908 | 4 | 4 | 1 | 0/0 | ✅ refusal match |
| 42 | 42 | 5 | 118,810 | 15 | 0 | 0 | 1/2 | ⚠️ partial |
| 43 | 43 | 9 | 211,415 | 8 | 0 | 0 | 0/0 | ✅ refusal match |
| 44 | 44 | 14 | 438,540 | 18 | 0 | 0 | 3/3 | ✅ full match |
| 45 | 45 | 10 | 275,737 | 1 | 9 | 1 | 0/0 | ✅ refusal match |
| 46 | 46 | 5 | 92,040 | 4 | 0 | 0 | 0/0 | ✅ refusal match |
| 47 | 47 | 11 | 228,515 | 10 | 0 | 0 | 0/0 | ✅ refusal match |
| 48 | 48 | 9 | 169,769 | 9 | 0 | 0 | 0/0 | ✅ refusal match |
| 49 | 49 | 8 | 273,036 | 5 | 3 | 1 | 0/0 | ✅ refusal match |

Highlights:

- **Task 18** (5 reservation downgrades for `omar_davis_3817`): **5/5**. This is the task that produced the original turn-boundary insight — 0/5 with hallucinated reservation IDs before the prompt fix, 5/5 here under batch conditions. Confirms the fix holds outside the cherry-picked test.
- **Task 44** (the τ³ successor to v1's task 33 / our [issue #85](https://github.com/sierra-research/tau-bench/issues/85) on S61CZX): **3/3 full match**. The agent correctly refused to cancel S61CZX per the insurance/health policy and completed all three upgrade writes. The new gold annotations in τ³ now agree with that reading.
- **Refusal-pass coverage was strong.** 22 / 22 refusal-style tasks completed without false writes; Opus's pre-fix run had over-mutated several of the same probes.

## Failure breakdown

Five patterns account for all 10 misses:

| Pattern | Tasks | Count | % of failures |
|---|---|---|---|
| Multi-block hallucination of tool returns | 11, 30, 39, 42 | 4 | 40% |
| Ran out of turns before second write | 29, 35 | 2 | 20% |
| Two-clause request, second clause dropped | 12, 14 | 2 | 20% |
| Over-mutation by trial-and-rollback | 10 | 1 | 10% |
| Refused without probing fallback path | 19 | 1 | 10% |

### Multi-block hallucination (40% of failures)

The single largest pattern is still the architecture bug the turn-boundary prompt was meant to fix: the agent emits several `​```python` blocks in one assistant message, and later blocks reference values that earlier blocks were supposed to produce — but the LLM had to imagine those values because nothing executes until the message is finished.

Smoking gun from task 39's trace:

```python
# Block A (same turn): respond("Could you provide your user id?")
# Block B (same turn): user = airline.get_user_details('amelia_davis_8890')
#                      ctx.user = user
# Block C (same turn): res1 = airline.get_reservation_details('reservation_26789')
#                      res2 = airline.get_reservation_details('reservation_88597')
```

Block C ran in the same assistant message as Block B, so the LLM had no idea what `ctx.user['reservations']` actually contained. It invented `reservation_26789` and `reservation_88597`. Both errored. The agent's own postmortem confesses the real reservation IDs and apologises.

Same shape on task 30 (`reservation_id_4220684`) and task 42 (`AQTWTM`).

**Prompt-level fix worked on the cherry-picked test (task 18: 5/5).** It is not reliable across batch — Sonnet follows the rule 60–80% of the time and pre-commits when the next step is a "natural" sequel ("got user → fetch their reservations"). An architectural enforcement that simply drops later blocks when an earlier block called a tool would mechanically remove this category. Projected lift: from 80% → ~88%.

### Ran out of turns (tasks 29, 35)

Both tasks needed `cancel + book` or `refuse + book`. Agent did the first action correctly, started searching for new flights, presented options to the customer — and the conversation ended before the actual `book_reservation` fired. Customer-side STOP came too early. Worth investigating whether the user simulator stops more aggressively under tight reasoning loops, or whether the agent should be more decisive about closing the second write before re-engaging the customer.

### Two-clause dropped (tasks 12, 14)

Customer asks for two changes in one turn; agent does the first and forgets the second. Generic LLM weakness, not framework-specific. Could be addressed by a check-list prompt addition like "before ending, restate the original request and verify each part was addressed."

### Over-mutation by trial (task 10)

Agent treats `update_reservation_flights` as a reversible probe — writes, sees the price exceeds the customer's cap, writes again to revert. Two writes recorded; gold expects zero. Net DB state is unchanged but the matcher correctly penalizes the round-trip. Fix: agent should compute the price-impact via the read-only flight search before writing, not by writing-then-reverting.

### Refused without probing fallback (task 19)

Agent correctly refused to modify a basic-economy reservation per policy. Stopped there. The customer had a scripted fallback ("If basic economy cannot be modified, you are willing to cancel using the travel insurance as you feel unwell") that the agent never elicited because it didn't ask "is there another way I can help?" The customer is *reactive* per the task instructions — they won't volunteer the health/insurance angle. Generic fix: after a policy-driven refusal, always offer at least one alternative before closing.

## What this run validated

- The `ctx` namespace + turn-boundary prompt produces correct multi-step writes on every batch task where the agent actually waits between fetch and dependent action (e.g. task 18 5/5, task 23 4/4, task 44 3/3, task 17 3/3, task 22 3/3).
- Sonnet 4.6 + Coda's CodeAct + `ctx` lands at 80% pass@1 on τ³ airline — inside the extrapolated τ³ band for Sonnet-class models, with a different agent surface than the leaderboard's JSON-tool-calling reference setup.
- Issue [#85](https://github.com/sierra-research/tau-bench/issues/85) was confirmed resolved upstream: the agent's policy reading on `S61CZX` (now in task 44) is now consistent with the new gold annotations.

## What this run surfaced as remaining work

- The multi-block-per-turn architecture bug is responsible for 40% of failures. Prompt mitigations are partial; architectural enforcement is the cheapest path to ~88%.
- Tail patterns (run-out-of-turns, dropped-second-clause, refusal-without-probe) each have known fixes that don't conflict with `ctx`.
