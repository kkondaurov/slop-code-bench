# GPT-5.6 Sol medium capability-11 v2.2 run

This is sample 1 for GPT-5.6 Sol medium on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs and repair canaries are not part of this
comparison set.

## Configuration

- Model: `gpt-5.6-sol`
- Reasoning effort: `medium`
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
- Started: `2026-08-03T23:52:24+02:00`
- Finished: `2026-08-04T02:13:04+02:00`
- Wall time: `2:20:39.558`

The checked-in configuration is
[`configs/runs/gpt-5.6-sol-medium-capability-11.yaml`](../../../../configs/runs/gpt-5.6-sol-medium-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `12/66` (`18.18%`)
- Current-checkpoint behavior clean: `24/66` (`36.36%`)
- Essential behavior clean: `50/66` (`75.76%`)
- Tests passed: `7,508/8,113` (`92.54%`)
- Accumulated regression tests passed: `5,365/5,733` (`93.58%`)
- API-equivalent retail cost estimate: `$60.73`
- Recorded agent time: `5:10:40.867`
- Evaluator time: `2:16:39.678`
- Aggregate per-problem overhead: `0:22:13.604`
- Steps: `1,524`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **mixed with strong essential coverage**. Sol medium kept the
core behavior of 50 checkpoints, the highest sample-1 total so far, and was
completely clean on 12. Forge, Execution Server, TextDrop, SheetEval, and EVE
Industry were strongest. EVE Market Tools remained entirely unsolved at the
core checkpoint level, while Mocked HTTP and Dynamic Config retained broad
functional gaps.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 2/5 | 390/442 | $3.92 | 0:25:41 | mixed-to-weak |
| `datagate` | 0/7 | 3/7 | 5/7 | 1,578/1,613 | $4.54 | 0:39:41 | strong aggregate, core gaps |
| `dynamic_config_service_api` | 1/4 | 1/4 | 3/4 | 235/303 | $6.66 | 0:37:54 | mixed, final checkpoint weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 72/179 | $4.65 | 1:37:20 | weak and evaluator-heavy |
| `execution_server` | 3/6 | 5/6 | 6/6 | 605/608 | $3.86 | 0:22:48 | strong |
| `forge` | 2/8 | 4/8 | 8/8 | 1,249/1,279 | $4.15 | 0:32:05 | strong essential, regression-limited |
| `textdrop` | 1/6 | 4/6 | 6/6 | 625/635 | $3.63 | 0:26:54 | strong, regression-limited |
| `trajectory_api` | 1/5 | 1/5 | 4/5 | 1,143/1,200 | $7.63 | 0:43:10 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 0/8 | 4/8 | 737/894 | $10.63 | 1:14:02 | weak-to-mixed and expensive |
| `sheeteval` | 0/7 | 1/7 | 7/7 | 591/653 | $5.85 | 0:35:47 | core-strong, functionality-weak |
| `eve_industry` | 3/6 | 3/6 | 5/6 | 283/307 | $5.22 | 0:34:10 | strong-to-mixed |

## Failure clusters

- `eve_market_tools` again missed broad market statistics, reprocessing,
  compression, hauling, and trading behavior. Its evaluator consumed about 69
  minutes, so the zero essential score was neither an infrastructure artifact
  nor a fast failure.
- `mocked_http` retained gaps in template and regex helpers, persistence,
  command push behavior, AMQP/Kafka cases, and merged evaluation contexts. It
  was also the costliest problem job at $10.63.
- `dynamic_config_service_api` passed its first checkpoint cleanly but missed
  input-validation edges, proposal constraints, and much of the final policy,
  review, merge, and bulk-evaluation flow.
- `sheeteval` kept every core checkpoint clean but lost broad current behavior
  around incomplete annotations, missing references, expression warnings,
  penalty accounting, and scenario placeholders.
- `database_migration` still missed backfills, rollback behavior, foreign-key
  handling, and dependency semantics. DataGate retained core gaps around
  Latin-1 autodetection, row identifiers, and spreadsheet query controls.
- `trajectory_api` again shows the timestamp repair did not make the problem
  trivially clean: checkpoint 1 was completely clean and checkpoints 1–4 kept
  their essential behavior, while later lifecycle, boundary, environment, and
  tool-execution cases still failed.
- Execution Server kept all core checkpoints and three completely clean
  checkpoints. Forge kept all eight core checkpoints; TextDrop did the same
  across its six checkpoints.

No security-classifier or refusal evidence was found in supported logs or
inference summaries. All 66 inference runs completed without error, all 66
evaluator records were present, and postprocessing reported zero errors with
full `scb-check` coverage.

## Comparison status

This is `n=1` for Sol medium and the fourth completed profile in the v2.2
sample-1 breadth pass. Relative to Terra high sample 1, Sol medium recorded two
more strict solves (`12` versus `10`), the same 24 current solves, and one more
core solve (`50` versus `49`), but passed eight fewer tests and cost `$29.14`
more. Agent time was nearly identical (`5:10:40.867` versus `5:10:00.298`),
while Sol's wall time was 19 minutes longer because evaluator work more than
doubled. This is a small capability trade rather than a clear overall lead.

Against Terra xhigh, Sol medium recorded three more strict solves, three fewer
current solves, and one more core solve, while costing `$10.41` more but using
2 hours 50 minutes less agent time. Bidirectional flips versus Terra xhigh were
`4/1` strict, `4/7` current, and `2/1` core. With only one observation per
profile, these are preliminary outcomes rather than reliability or variance
estimates.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `5:10:40.867`, evaluator work was
`2:16:39.678`, and supported aggregate overhead was `0:22:13.604`.
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
