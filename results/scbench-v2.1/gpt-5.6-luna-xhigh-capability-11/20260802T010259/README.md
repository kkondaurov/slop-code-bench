# GPT-5.6 Luna xhigh capability-11 v2.1 run

This is the first valid GPT-5.6 Luna xhigh observation of the repaired SCBench
v2.1 capability-11 panel. Earlier runs used superseded problem inputs and are
not included in the comparison set.

## Configuration

- Model: `gpt-5.6-luna`
- Reasoning effort: `xhigh`
- Agent harness: Codex CLI `0.146.0`
- Prompt: `just-solve`
- Seed: `42`
- Workers: `2`
- Runner commit: `736f88284fa4a500f3bf5fadbe49ee9f86bd918f`
- Runner tag: `scbench-v2.1-repro.1`
- Problem catalog: `kkondaurov/scb-problems` `v1.0.1`
- Problem catalog commit: `9b4864d6bdefd8cd0f2d66d3eb0d1972914aefd3`
- Problem catalog tree SHA256: `263199721db8c873aef0643224d244712fd5f56f4565198bf4cefd0e65211f8c`
- Evaluator: `scb-check==0.1.3`
- Started: `2026-08-01T23:03:10Z`
- Finished: `2026-08-02T05:59:22Z`
- Wall time: `6:56:12`

The checked-in configuration is
[`configs/runs/gpt-5.6-luna-xhigh-capability-11.yaml`](../../../../configs/runs/gpt-5.6-luna-xhigh-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `10/66` (`15.15%`)
- Current-checkpoint behavior clean: `24/66` (`36.36%`)
- Essential behavior clean: `45/66` (`68.18%`)
- Tests passed: `7,201/8,113` (`88.76%`)
- Accumulated regression tests passed: `5,168/5,733` (`90.14%`)
- API-equivalent retail cost estimate: `$27.47`
- Recorded checkpoint work: `10:38:34.759`
- Evaluator time: `2:00:26.959`
- Aggregate per-problem overhead: `0:20:15.564`
- Steps: `2,468`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The overall result is **mixed**. Luna delivered strong essential behavior at a
low API-equivalent cost, but only ten checkpoints were completely clean.
Forge, SheetEval, TextDrop, Execution Server, and most of DataGate were the
strongest areas. Trajectory API, EVE Market Tools, and Mocked HTTP account for
most of the missing behavior, so the modest strict score is not merely a few
threshold-edge regressions.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 2/5 | 411/442 | $2.26 | 0:55:47 | mixed |
| `datagate` | 0/7 | 3/7 | 5/7 | 1,570/1,613 | $1.58 | 0:55:54 | strong partial, regression-limited |
| `dynamic_config_service_api` | 0/4 | 0/4 | 2/4 | 247/303 | $2.69 | 1:02:23 | mixed-to-weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 66/179 | $2.64 | 2:06:05 | weak |
| `execution_server` | 0/6 | 2/6 | 5/6 | 593/608 | $1.55 | 0:43:32 | strong-to-mixed |
| `forge` | 5/8 | 5/8 | 8/8 | 1,231/1,279 | $2.28 | 1:12:52 | strong |
| `textdrop` | 1/6 | 4/6 | 6/6 | 619/635 | $1.83 | 0:49:12 | strong, regression-limited |
| `trajectory_api` | 0/5 | 0/5 | 0/5 | 825/1,200 | $2.50 | 1:02:47 | weak |
| `mocked_http` | 0/8 | 1/8 | 5/8 | 703/894 | $5.08 | 1:57:42 | mixed-to-weak |
| `sheeteval` | 2/7 | 5/7 | 7/7 | 644/653 | $2.69 | 1:09:13 | strong, small regression tail |
| `eve_industry` | 1/6 | 2/6 | 5/6 | 292/307 | $2.38 | 1:03:52 | strong-to-mixed |

SheetEval is a useful validation of the v2.1 repair: all seven checkpoints kept
their essential behavior clean, and the final checkpoint passed all of its
current core, functionality, and error tests. Its three final misses were
accumulated regressions rather than a repaired-suite or infrastructure failure.

## Failure clusters

- `trajectory_api` is the largest capability gap. No checkpoint reached a
  clean current or core threshold, and the later environment, grammar,
  tool-extraction, and toolpack behavior remained substantially incomplete.
- `eve_market_tools` missed broad market-statistics, reprocessing,
  compression, hauling, and trading behavior. Its long evaluator time reflects
  the workload; failing quickly would not be a useful speed advantage here.
- `mocked_http` implemented substantial HTTP behavior, but accumulated config
  regressions plus missing AMQP/Kafka and checkpoint-8 evaluation behavior
  prevented every strict solve.
- `dynamic_config_service_api` preserved two core checkpoints but did not
  complete checkpoint-4 policy evaluation, versioning, review, and merge
  behavior.
- In the stronger problems, accumulated regressions still mattered: DataGate,
  Forge, TextDrop, SheetEval, and Execution Server often passed current core
  behavior while missing the fully clean threshold.

No security-classifier or refusal evidence was found in the run log or
inference summaries. All 11 problem jobs completed normally; all 66 evaluator
records were present, and postprocessing reported zero errors and full
`scb-check` coverage.

## Comparison status

This is `n=1` for the v2.1 comparison set, so the profile judgment is
preliminary. Pre-v2.1 runs are intentionally excluded: they used superseded
problem inputs and cannot be treated as additional samples. Same-profile
flips, spreads, and variance-oriented summaries begin with samples two and
three.

## Timing notes

The observed wall time was `6:56:12` with two workers. Recorded checkpoint work
sums checkpoint durations and therefore exceeds wall time when workers overlap.
Across the 11 problem jobs, recorded checkpoint work was `10:38:34.759`,
evaluator work was `2:00:26.959`, and supported per-problem overhead was
`0:20:15.564`. `problem_timings.json` preserves the corresponding problem-level
measurements.

## Files

- `result.json`: aggregate metrics, copied byte-for-byte from the run.
- `checkpoint_results.jsonl`: 66 checkpoint records with only the authorized
  machine-specific path fields removed.
- `problem_timings.json`: compact per-problem wall, checkpoint, evaluator,
  step, and cost measurements.
- `SHA256SUMS`: checksums for the published files.

The full local run remains the provenance archive. Generated code, prompts,
snapshots, evaluator internals, and agent rollouts are intentionally excluded
from Git.
