# assay

Benchmark figures for coding agents, stored so their qualifiers cannot be dropped — and a
harness for producing the figures that do not exist yet.

## Why

"Claude scores 72% on aider polyglot" is the sentence people repeat. It silently drops the
attempt count, the edit format, the thinking budget and the model date. Every one of those
moves the number, and one of them — the date — makes that particular sentence useless for
judging a 2026 system, because the leaderboard's best Claude row is an Opus 4 from 2025-05.

The same leaderboard shows what the qualifiers are worth: the same model, the same day,
**37.3%** on one attempt and **72.0%** on two attempts with the test output fed back. The
attempt budget is worth more than a model generation, and prose about "72%" hides that.

This project began from a mistake of exactly that kind. A ~13-point gap was reconstructed by
comparing one vendor's per-component maxima against composites normalized over a different
set of components. The retraction is in [emet](https://github.com/O6lvl4/emet)'s DESIGN.md.
`assay` exists so the same arithmetic cannot be done again by accident.

## The two rules

1. **A figure belongs here only if it was read off the benchmark's own leaderboard.** Numbers
   reconstructed from vendor announcements are what produced the founding mistake.
2. **Where a model has no published run, the absence is recorded as an absence** — as data,
   queryable, in `published.ABSENCES`. A reader who cannot see a gap will fill it in from
   vendor figures, which is the mistake again.

`score.incomparable(a, b, varying)` reads the stored harness and refuses a comparison that
differs in something other than what you said you were varying. `score.coverage` computes
what a composite averages over, and `composite_delta` **refuses to subtract** two composites
whose component sets differ.

## What the data currently says

Verified against primary leaderboards on 2026-09-13.

Terminal-Bench 2.0, which is 30% of BenchLM's agentic composite:

| model | score | |
|---|---|---|
| Kimi K3 *(open weights)* | 88.3 | leads every listed Claude |
| Claude Mythos 5 | 88.0 | |
| Claude Fable 5 | 84.3 | |
| **GLM-5.2** *(open weights)* | **81.0** | |
| Claude Sonnet 5 | 80.4 | |
| Claude Opus 4.8 | 74.6 | |
| GLM-5.1 *(open weights)* | 63.5 | |

Two findings that only exist because absences are recorded:

- **The composite cannot be decomposed.** Claude Fable 5.1 holds the published agentic
  composite of 80.2 and appears on *none* of its own component leaderboards. So 80.2 cannot
  be reproduced from any published component value, in either direction.
- **The leaderboard maximum is not a plan.** `available.best_reachable` answers a different
  question: the best published score among models this project can actually call. Kimi K3
  leads Terminal-Bench 2.0 and is not in the Cloudflare Workers AI catalogue; GLM-5.3 is in
  the catalogue and has no published Terminal-Bench 2.0 run at all. That intersection —
  runnable and unmeasured — is what `bench/` is for.

## Measuring

`bench/tb2.sh` runs [golemide](https://github.com/O6lvl4/golemide) against Terminal-Bench 2.0
through [harbor](https://github.com/laude-institute/harbor), via the adapter in
`adapters/golemide_agent.py`. `claude-code` is a built-in harbor agent, so the same 89 tasks
and the same harness give a head-to-head with no undisclosed scaffold on either side.

```sh
HB=/path/to/hb GOLEMIDE_BINARY=/path/to/golemide bench/tb2.sh pilot   # 10 tasks
HB=... GOLEMIDE_BINARY=... bench/tb2.sh all                           # all 89
```

A pilot first, because 89 tasks at each task's own `[agent] timeout_sec` (600s to 12000s) is
41 hours of wall clock if every task runs to its wall, and a systematic adapter fault costs
the whole run to discover.

`bench/tb2-score.py` reads each trial's `result.json` rather than the job summary, because a
cancelled or errored trial records no reward **and** no cost while the summary averages it in
as a zero. A zero earned by failing and a zero from a trial that never finished are different
measurements. Cost is reported as a floor for the same reason: a trial killed before its
agent's summary line spent money the score cannot see.

## State

**No Terminal-Bench 2.0 score has been measured yet.** One task has been attempted —
`circuit-fibsqrt`, `difficulty = "hard"`, `expert_time_estimate_min = 960` — which was a poor
first choice for judging an agent, and it produced no reward. The measurement path is built
and validated (harbor 0.23.0, 89 tasks cached, oracle solution verified at reward 1.000); the
figures in `src/published.almd` are other people's, and the column for this project is empty.

`almide test` — 14 tests across `src/`.
