# GPT-5.6 Luna xhigh capability-11 v2.2 run

This is sample 1 for GPT-5.6 Luna xhigh on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs used superseded problem inputs and are not
part of this comparison set.

## Configuration

- Model: `gpt-5.6-luna`
- Reasoning effort: `xhigh`
- Agent harness: Codex CLI `0.146.0`
- Prompt: `just-solve`
- Seed: `42`
- Workers: `4`
- Runner commit: `79341bdd53d16129669eaef28871fbff49dd49b2`
- Runner tag: `scbench-v2.2-workers4`
- Problem catalog: `kkondaurov/scb-problems` `v1.0.2`
- Problem catalog commit: `88e9666f9529c97a30951fca17cb38656ea0a5f1`
- Problem catalog tree SHA256: `42a5a447856e057d67716edb83aec17ff72b98d50d83b500eb439fdc947c89e5`
- Evaluator: `scb-check==0.1.3`
- Started: `2026-08-03T14:16:00+02:00`
- Finished: `2026-08-03T18:12:00+02:00`
- Wall time: `3:55:59.359`

The checked-in configuration is
[`configs/runs/gpt-5.6-luna-xhigh-capability-11.yaml`](../../../../configs/runs/gpt-5.6-luna-xhigh-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `7/66` (`10.61%`)
- Current-checkpoint behavior clean: `22/66` (`33.33%`)
- Essential behavior clean: `47/66` (`71.21%`)
- Tests passed: `7,483/8,113` (`92.23%`)
- Accumulated regression tests passed: `5,325/5,733` (`92.88%`)
- API-equivalent retail cost estimate: `$28.29`
- Recorded agent time: `12:29:03.623`
- Evaluator time: `1:08:41.249`
- Aggregate per-problem overhead: `0:22:20.314`
- Steps: `2,524`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **mixed with strong essential coverage**. Luna preserved the
essential behavior of 47 checkpoints, but accumulated regressions and broader
functionality gaps reduced the strict score to seven. EVE Market Tools is the
clearest broad capability miss; EVE Industry, TextDrop, Forge, and Execution
Server retained the strongest essential coverage.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 3/5 | 420/442 | $2.33 | 0:54:54 | mixed |
| `datagate` | 0/7 | 4/7 | 4/7 | 1,569/1,613 | $2.07 | 1:07:14 | mixed, regression-limited |
| `dynamic_config_service_api` | 0/4 | 0/4 | 2/4 | 263/303 | $3.25 | 1:12:23 | mixed-to-weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 87/179 | $2.38 | 1:06:20 | weak |
| `execution_server` | 0/6 | 3/6 | 5/6 | 597/608 | $1.70 | 0:42:29 | strong-to-mixed |
| `forge` | 2/8 | 4/8 | 8/8 | 1,204/1,279 | $2.03 | 2:49:38 | strong essential, regression-limited |
| `textdrop` | 1/6 | 3/6 | 6/6 | 610/635 | $1.66 | 0:46:53 | strong, regression-limited |
| `trajectory_api` | 0/5 | 0/5 | 4/5 | 1,120/1,200 | $2.22 | 1:04:11 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 1/8 | 4/8 | 746/894 | $4.75 | 1:46:30 | mixed-to-weak |
| `sheeteval` | 1/7 | 2/7 | 5/7 | 570/653 | $3.08 | 1:17:28 | mixed, alternate-answer gaps |
| `eve_industry` | 2/6 | 3/6 | 6/6 | 297/307 | $2.81 | 1:12:03 | strong |

## Failure clusters

- `eve_market_tools` missed core behavior in every checkpoint, spanning market
  statistics, reprocessing, compression, hauling, and trading.
- `mocked_http` implemented substantial HTTP behavior, but template helpers,
  persistence, message-transport behavior, and checkpoint-8 evaluation left a
  large current and regression tail.
- `dynamic_config_service_api` preserved the first two core checkpoints but
  remained incomplete in proposal/review flows and policy evaluation.
- `sheeteval` was limited mainly by alternate-answer and expression-matching
  behavior; checkpoints 3 and 7 also missed essential cases.
- The repaired `trajectory_api` timestamp contract behaved as intended: core
  behavior was clean through checkpoint 4. Its remaining misses concern input
  boundaries, lifecycle edge cases, environment/grammar behavior, and the
  final tool-extraction/toolpack checkpoint rather than timestamp formatting.
- In the stronger problems, strict solves were mostly lost to accumulated
  regression tails. EVE Industry kept all six core checkpoints clean;
  TextDrop did the same, and Forge kept all eight.

No security-classifier or refusal evidence was found in the run log or
inference summaries. All 11 problem jobs completed normally, all 66 evaluator
records were present, and postprocessing reported zero errors with full
`scb-check` coverage. Forge was a long-running problem job, but it completed
normally without a timeout or infrastructure failure.

## Comparison status

This is `n=1` for the v2.2 comparison set, so the profile judgment is
preliminary. Pre-v2.2 runs are intentionally excluded because they used
superseded catalog inputs. Same-profile flips, spreads, and variance summaries
begin with samples 2 and 3.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `12:29:03.623`, evaluator work was
`1:08:41.249`, and supported aggregate overhead was `0:22:20.314`.
`problem_timings.json` preserves the corresponding problem-level values.

## Known panel limitations

This run does not turn the v2.2 evaluator into a complete conformance proof.
The published catalog notes known areas where tests provide incomplete evidence:
the Dynamic Config timeout behavior is not deterministically distinguished,
EVE compression optimality is not exhaustively established, TextDrop does not
verify object-store credentials against a live service, and Trajectory API does
not stress concurrent mutation atomicity. These are documented limitations,
not inferred passes or post-run score adjustments.

## Files

- `result.json`: aggregate metrics, copied byte-for-byte from the run.
- `checkpoint_results.jsonl`: 66 checkpoint records with only the authorized
  machine-specific path fields removed.
- `problem_timings.json`: compact per-problem wall, agent, evaluator, step, and
  cost measurements.
- `SHA256SUMS`: checksums for the published files.

The full local run remains the provenance archive. Generated code, prompts,
snapshots, evaluator internals, and agent rollouts are intentionally excluded
from Git.
