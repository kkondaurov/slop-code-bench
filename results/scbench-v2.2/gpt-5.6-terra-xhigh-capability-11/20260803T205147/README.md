# GPT-5.6 Terra xhigh capability-11 v2.2 run

This is sample 1 for GPT-5.6 Terra xhigh on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs and repair canaries are not part of this
comparison set.

## Configuration

- Model: `gpt-5.6-terra`
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
- Started: `2026-08-03T20:52:00+02:00`
- Finished: `2026-08-03T23:37:12+02:00`
- Wall time: `2:45:11.583`

The checked-in configuration is
[`configs/runs/gpt-5.6-terra-xhigh-capability-11.yaml`](../../../../configs/runs/gpt-5.6-terra-xhigh-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `9/66` (`13.64%`)
- Current-checkpoint behavior clean: `27/66` (`40.91%`)
- Essential behavior clean: `49/66` (`74.24%`)
- Tests passed: `7,562/8,113` (`93.21%`)
- Accumulated regression tests passed: `5,412/5,733` (`94.40%`)
- API-equivalent retail cost estimate: `$50.33`
- Recorded agent time: `8:00:55.432`
- Evaluator time: `1:09:02.881`
- Aggregate per-problem overhead: `0:22:04.348`
- Steps: `2,048`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **mixed with strong essential coverage**. Terra xhigh retained
the core behavior of 49 checkpoints and improved current-checkpoint coverage
over both earlier v2.2 sample-1 profiles, but did not improve aggregate core
coverage over Terra high. EVE Market Tools remained entirely unsolved at the
core checkpoint level, while Execution Server, SheetEval, TextDrop, Forge,
and the repaired Trajectory API were the strongest areas.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 2/5 | 386/442 | $3.74 | 0:40:55 | mixed-to-weak |
| `datagate` | 0/7 | 3/7 | 5/7 | 1,570/1,613 | $3.35 | 0:52:12 | strong aggregate, core gaps |
| `dynamic_config_service_api` | 0/4 | 0/4 | 3/4 | 234/303 | $5.63 | 0:51:56 | mixed, final checkpoint weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 70/179 | $3.87 | 0:50:23 | weak |
| `execution_server` | 0/6 | 5/6 | 6/6 | 602/608 | $3.42 | 0:36:58 | strong, regression-limited |
| `forge` | 2/8 | 4/8 | 7/8 | 1,199/1,279 | $2.93 | 0:42:37 | strong essential, regression-limited |
| `textdrop` | 1/6 | 3/6 | 6/6 | 618/635 | $3.81 | 0:44:05 | strong, regression-limited |
| `trajectory_api` | 2/5 | 2/5 | 4/5 | 1,177/1,200 | $5.22 | 0:50:09 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 1/8 | 5/8 | 774/894 | $8.54 | 1:27:17 | mixed, expensive |
| `sheeteval` | 0/7 | 4/7 | 7/7 | 641/653 | $4.68 | 0:56:01 | strong, regression-limited |
| `eve_industry` | 3/6 | 3/6 | 4/6 | 291/307 | $5.15 | 0:59:28 | mixed |

## Failure clusters

- `eve_market_tools` again missed broad market statistics, reprocessing,
  compression, hauling, and trading behavior. Its zero essential solves are
  substantive rather than an evaluator failure.
- `dynamic_config_service_api` retained an explicit-version resolution miss,
  then lost broad policy, proposal/review, bulk-evaluation, and final-checkpoint
  behavior. The repaired bootstrap path did not manufacture a clean score.
- `mocked_http` kept core gaps in template normalization, persistence, command
  push behavior, AMQP/Kafka cases, and merged evaluation contexts. At 87 minutes
  and $8.54, this was the slowest and most expensive problem job.
- `database_migration` still missed backfills, rollback behavior, foreign-key
  handling, and dependency semantics.
- `datagate` passed 97.33% of tests but retained core gaps around Latin-1
  autodetection, row identifiers, spreadsheet query controls, and charset
  handling; the aggregate score should not be mistaken for full conformance.
- `trajectory_api` confirms the repaired timestamp contract without becoming
  trivially clean: checkpoints 1 and 2 were completely clean, checkpoints 1–4
  retained their essential behavior, and remaining misses concern fork/search
  details, environment activation, and tool-execution/toolpack behavior.
- Execution Server and SheetEval kept every core checkpoint clean. TextDrop did
  the same; their remaining misses are narrow functionality or accumulated
  regressions rather than infrastructure artifacts.

No security-classifier or refusal evidence was found in supported logs or
inference summaries. All 66 inference runs completed without error, all 66
evaluator records were present, and postprocessing reported zero errors with
full `scb-check` coverage.

## Comparison status

This is `n=1` for Terra xhigh and the third completed profile in the v2.2
sample-1 breadth pass. Relative to Terra high sample 1, Terra xhigh recorded
one fewer strict solve (`9` versus `10`), three more current solves (`27`
versus `24`), and the same 49 core solves. It passed 46 more tests but cost
`$18.73` more. Agent time was `8:00:55.432` versus `5:10:00.298`, and wall
time was `2:45:11.583` versus `2:01:33.945`; the extra reasoning effort did
not produce a clear aggregate capability win on this sample.

The aggregates conceal bidirectional flips against Terra high: Terra xhigh
gained/lost `1/2` strict, `7/4` current, and `4/4` core solves. Against Luna
xhigh it gained/lost `4/2` strict, `10/5` current, and `7/5` core solves.
With only one observation per profile, these are preliminary outcomes rather
than reliability or variance estimates.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `8:00:55.432`, evaluator work was
`1:09:02.881`, and supported aggregate overhead was `0:22:04.348`.
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
