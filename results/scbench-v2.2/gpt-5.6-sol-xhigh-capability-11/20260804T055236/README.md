# GPT-5.6 Sol xhigh capability-11 v2.2 run

This is sample 1 for GPT-5.6 Sol xhigh on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs and repair canaries are not part of this
comparison set.

## Configuration

- Model: `gpt-5.6-sol`
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
- Started: `2026-08-04T05:52:49+02:00`
- Finished: `2026-08-04T09:59:11+02:00`
- Wall time: `4:06:21.631`

The checked-in configuration is
[`configs/runs/gpt-5.6-sol-xhigh-capability-11.yaml`](../../../../configs/runs/gpt-5.6-sol-xhigh-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `18/66` (`27.27%`)
- Current-checkpoint behavior clean: `31/66` (`46.97%`)
- Essential behavior clean: `52/66` (`78.79%`)
- Tests passed: `7,645/8,113` (`94.23%`)
- Accumulated regression tests passed: `5,464/5,733` (`95.31%`)
- API-equivalent retail cost estimate: `$116.46`
- Recorded agent time: `11:40:34.407`
- Evaluator time: `1:08:03.176`
- Aggregate per-problem overhead: `0:21:56.450`
- Steps: `2,061`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **the strongest sample-1 capability observation and the most
expensive**. Sol xhigh led all six profiles on strict, current, core, and total
tests passed. Forge, TextDrop, SheetEval, and EVE Industry were strongest;
Mocked HTTP improved materially over every lower-effort Sol run but remained
regression-heavy. EVE Market Tools still had no solved core checkpoint.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 2/5 | 384/442 | $8.10 | 0:59:37 | mixed-to-weak |
| `datagate` | 1/7 | 5/7 | 5/7 | 1,591/1,613 | $8.07 | 1:08:25 | strong aggregate, regression-limited |
| `dynamic_config_service_api` | 1/4 | 1/4 | 3/4 | 231/303 | $10.67 | 1:08:49 | mixed-to-weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 106/179 | $7.68 | 0:54:43 | weak despite improved test coverage |
| `execution_server` | 0/6 | 3/6 | 5/6 | 597/608 | $7.52 | 0:52:31 | strong aggregate, no clean checkpoint |
| `forge` | 5/8 | 5/8 | 8/8 | 1,225/1,279 | $6.82 | 0:53:44 | core-complete and strong |
| `textdrop` | 3/6 | 4/6 | 6/6 | 631/635 | $7.82 | 0:55:20 | core-complete and strong |
| `trajectory_api` | 1/5 | 1/5 | 4/5 | 1,141/1,200 | $12.40 | 1:15:58 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 1/8 | 6/8 | 802/894 | $24.72 | 2:19:04 | improved, expensive, regression-heavy |
| `sheeteval` | 5/7 | 6/7 | 7/7 | 649/653 | $12.11 | 1:23:35 | exceptionally strong |
| `eve_industry` | 1/6 | 3/6 | 6/6 | 288/307 | $10.55 | 1:18:43 | core-complete with functionality gaps |

## Failure clusters

- `eve_market_tools` passed more tests than the other Sol profiles but still
  missed core behavior across market statistics, reprocessing, compression,
  hauling, and trading. The zero essential score is substantive rather than an
  infrastructure artifact.
- `mocked_http` improved to six core and one current checkpoint, but retained
  template/helper, persistence, command, merged-context, and accumulated
  regression gaps. At `$24.72` and more than two hours of problem wall time,
  its remaining failures are emphatically not a fast-failure speed win.
- `dynamic_config_service_api` again passed checkpoint 1 cleanly while later
  validation, proposal, policy, merge, and bulk-evaluation behavior remained
  broad failure clusters.
- `database_migration` regressed relative to Sol high on aggregate tests and
  essential checkpoints, retaining backfill, rollback, constraint, and
  dependency gaps. Execution Server passed 597 of 608 tests but accumulated
  enough current and regression failures to leave no checkpoint fully clean.
- `trajectory_api` exactly matched Sol high at `1/1/4` and `1,141/1,200`,
  retaining lifecycle, environment, grammar, file-snapshot, and accumulated
  regression gaps. The timestamp repair therefore did not trivialize it.
- Forge kept every core checkpoint and five clean checkpoints. TextDrop kept
  all six core checkpoints with only four failed tests. SheetEval and EVE
  Industry also kept every core checkpoint, with their remaining misses in
  functionality and accumulated regressions.

No security-classifier or refusal evidence was found in supported logs or
inference summaries. All 66 inference runs completed without error, all 66
evaluator records were present, and postprocessing reported zero errors with
full `scb-check` coverage.

## Comparison status

This is `n=1` for Sol xhigh and completes the six-profile v2.2 sample-1 breadth
pass. Sol xhigh leads that pass at `18` strict, `31` current, `52` core, and
`7,645` passing tests. This is a preliminary capability ordering, not a
reliability or variance estimate.

Relative to Sol high sample 1, Sol xhigh gained four strict solves, three
current solves, one core solve, and 83 passing tests. It also cost `$25.28`
more, used about 2 hours 27 minutes more recorded agent time, and took about 55
minutes longer wall time. Bidirectional checkpoint flips were `6/2` strict,
`5/2` current, and `2/1` core. The additional reasoning produced a real but
expensive gain.

Against Terra xhigh, Sol xhigh gained nine strict solves, four current solves,
three core solves, and 83 passing tests while costing `$66.13` more and using
about 3 hours 40 minutes more recorded agent time. Bidirectional flips were
`12/3` strict, `9/5` current, and `5/2` core. This is the best preliminary
capability result, but not the best useful-speed or cost profile.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `11:40:34.407`, evaluator work was
`1:08:03.176`, and supported aggregate overhead was `0:21:56.450`.
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
