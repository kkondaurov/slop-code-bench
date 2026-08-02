# GPT-5.6 Terra high capability-11 v2.1 run

This is the first valid GPT-5.6 Terra high observation of the repaired SCBench
v2.1 capability-11 panel. Earlier runs used superseded problem inputs and are
not included in the comparison set.

## Configuration

- Model: `gpt-5.6-terra`
- Reasoning effort: `high`
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
- Started: `2026-08-02T06:12:51Z`
- Finished: `2026-08-02T09:48:20Z`
- Wall time: `3:35:29`

The checked-in configuration is
[`configs/runs/gpt-5.6-terra-high-capability-11.yaml`](../../../../configs/runs/gpt-5.6-terra-high-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `9/66` (`13.64%`)
- Current-checkpoint behavior clean: `23/66` (`34.85%`)
- Essential behavior clean: `43/66` (`65.15%`)
- Tests passed: `7,110/8,113` (`87.64%`)
- Accumulated regression tests passed: `5,126/5,733` (`89.41%`)
- API-equivalent retail cost estimate: `$32.26`
- Recorded checkpoint work: `5:08:45.763`
- Evaluator time: `1:01:02.052`
- Aggregate per-problem overhead: `0:21:01.758`
- Steps: `1,678`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The overall result is **mixed**. Terra high delivered useful capability in
roughly half the wall time of Luna xhigh, but it was slightly weaker at all
three checkpoint thresholds and cost more at API-equivalent retail rates.
Forge, EVE Industry, TextDrop, Execution Server, and DataGate were the strongest
areas. Trajectory API, EVE Market Tools, and Mocked HTTP remained substantial
capability gaps rather than narrow threshold misses.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 3/5 | 405/442 | $2.60 | 0:27:47 | mixed |
| `datagate` | 0/7 | 4/7 | 5/7 | 1,576/1,613 | $2.39 | 0:39:04 | strong partial, regression-limited |
| `dynamic_config_service_api` | 0/4 | 0/4 | 2/4 | 230/303 | $2.81 | 0:27:14 | mixed-to-weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 30/179 | $2.77 | 0:32:31 | weak |
| `execution_server` | 0/6 | 4/6 | 5/6 | 599/608 | $2.22 | 0:24:14 | strong-to-mixed |
| `forge` | 5/8 | 5/8 | 7/8 | 1,219/1,279 | $2.06 | 0:31:13 | strong |
| `textdrop` | 1/6 | 4/6 | 6/6 | 614/635 | $2.38 | 0:30:19 | strong, regression-limited |
| `trajectory_api` | 0/5 | 0/5 | 0/5 | 833/1,200 | $3.25 | 0:36:13 | weak |
| `mocked_http` | 0/8 | 0/8 | 3/8 | 681/894 | $5.30 | 1:05:41 | weak |
| `sheeteval` | 1/7 | 2/7 | 6/7 | 628/653 | $3.05 | 0:36:23 | strong-to-mixed |
| `eve_industry` | 1/6 | 2/6 | 6/6 | 295/307 | $3.44 | 0:40:09 | strong-to-mixed |

SheetEval again exercised the repaired v2.1 evaluator normally: six of seven
checkpoints kept their essential behavior clean, all seven evaluator records
were valid, and no infrastructure failure occurred. Its 25 misses are model
behavior, including the final core miss and accumulated regressions, rather
than evidence that the repaired suite failed.

## Failure clusters

- `trajectory_api` remained the largest capability gap. No checkpoint reached
  a clean current or essential threshold, although its `833/1,200` tests were
  eight more than Luna xhigh.
- `eve_market_tools` completed only `30/179` tests, with broad market-statistics,
  reprocessing, compression, hauling, and trading behavior still absent.
- `mocked_http` completed substantial baseline HTTP behavior, but later config,
  AMQP/Kafka, and checkpoint-8 behavior left every checkpoint short of the
  current and strict thresholds.
- `dynamic_config_service_api` preserved two essential checkpoints but did not
  complete the later policy, versioning, review, and merge surface.
- In stronger problems, accumulated regressions still limited strict scores.
  DataGate, Execution Server, TextDrop, SheetEval, and EVE Industry often kept
  current or essential behavior while missing the fully clean threshold.

No security-classifier or refusal evidence was found in the run log or
inference summaries. All 11 problem jobs completed normally; all 66 evaluator
records were present, and postprocessing reported zero errors and full
`scb-check` coverage.

## Preliminary comparison

This is `n=1` for Terra high and the comparison is preliminary. Against the
only other v2.1 profile so far, Luna xhigh, Terra high was lower by one strict
checkpoint (`9` versus `10`), one current checkpoint (`23` versus `24`), two
essential checkpoints (`43` versus `45`), and 91 passed tests (`7,110` versus
`7,201`). Terra's wall time was `3:35:29`, about 48% shorter than Luna's
`6:56:12`, but its `$32.26` API-equivalent estimate was `$4.79` higher than
Luna's `$27.47`.

That makes Terra high a strong throughput result but a mixed overall tradeoff:
the wall-time gain is real and capability stayed close, yet it was not a cost
win. This comparison also changes both model and reasoning effort, so it does
not isolate either factor. Same-profile flips, spreads, and variance-oriented
summaries begin with samples two and three.

## Timing notes

The observed wall time was `3:35:29` with two workers. Recorded checkpoint work
sums checkpoint durations and therefore exceeds wall time when workers overlap.
Across the 11 problem jobs, recorded checkpoint work was `5:08:45.763`,
evaluator work was `1:01:02.052`, and supported per-problem overhead was
`0:21:01.758`. `problem_timings.json` preserves the corresponding
problem-level measurements.

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
