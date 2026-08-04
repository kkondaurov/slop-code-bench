# GPT-5.6 Sol high capability-11 v2.2 run

This is sample 1 for GPT-5.6 Sol high on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs and repair canaries are not part of this
comparison set.

## Configuration

- Model: `gpt-5.6-sol`
- Reasoning effort: `high`
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
- Started: `2026-08-04T02:21:34+02:00`
- Finished: `2026-08-04T05:33:25+02:00`
- Wall time: `3:11:50.209`

The checked-in configuration is
[`configs/runs/gpt-5.6-sol-high-capability-11.yaml`](../../../../configs/runs/gpt-5.6-sol-high-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `14/66` (`21.21%`)
- Current-checkpoint behavior clean: `28/66` (`42.42%`)
- Essential behavior clean: `51/66` (`77.27%`)
- Tests passed: `7,562/8,113` (`93.21%`)
- Accumulated regression tests passed: `5,400/5,733` (`94.19%`)
- API-equivalent retail cost estimate: `$91.18`
- Recorded agent time: `9:13:19.207`
- Evaluator time: `1:35:18.241`
- Aggregate per-problem overhead: `0:21:44.298`
- Steps: `1,866`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **strong but expensive**. Sol high recorded the highest
strict, current, and core solve totals in the sample-1 breadth pass so far,
while tying Terra xhigh for the most tests passed. SheetEval was exceptionally
clean; Forge and EVE Industry retained every core checkpoint. EVE Market Tools
remained entirely unsolved at the core level, and Mocked HTTP retained broad
gaps despite being the longest and costliest problem job.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 2/5 | 3/5 | 420/442 | $7.50 | 0:47:10 | mixed |
| `datagate` | 0/7 | 4/7 | 5/7 | 1,584/1,613 | $5.38 | 0:51:39 | strong aggregate, regression-limited |
| `dynamic_config_service_api` | 1/4 | 1/4 | 3/4 | 232/303 | $9.07 | 0:56:33 | mixed-to-weak |
| `eve_market_tools` | 0/4 | 0/4 | 0/4 | 75/179 | $7.26 | 1:24:32 | weak and evaluator-heavy |
| `execution_server` | 2/6 | 4/6 | 5/6 | 603/608 | $5.75 | 0:49:37 | strong with a final-checkpoint gap |
| `forge` | 2/8 | 5/8 | 8/8 | 1,202/1,279 | $5.70 | 0:52:52 | core-complete, regression-limited |
| `textdrop` | 1/6 | 3/6 | 6/6 | 621/635 | $6.79 | 0:55:56 | core-complete, regression-limited |
| `trajectory_api` | 1/5 | 1/5 | 4/5 | 1,141/1,200 | $10.42 | 1:06:38 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 0/8 | 4/8 | 759/894 | $17.56 | 1:34:59 | mixed, expensive, and regression-heavy |
| `sheeteval` | 5/7 | 6/7 | 7/7 | 651/653 | $7.98 | 0:52:12 | exceptionally strong |
| `eve_industry` | 1/6 | 2/6 | 6/6 | 274/307 | $7.76 | 0:58:09 | core-complete with functionality gaps |

## Failure clusters

- `eve_market_tools` again missed broad market statistics, reprocessing,
  compression, hauling, and trading behavior. The result is substantive rather
  than an infrastructure artifact: all four checkpoints were evaluated, none
  kept every core case, and the problem consumed about 37 minutes of evaluator
  time.
- `mocked_http` retained gaps in template normalization and helpers,
  persistence across restart, command push behavior, and merged evaluation
  contexts. It was the most expensive problem at `$17.56`, so its zero current
  solves are not a fast-failure speed win.
- `dynamic_config_service_api` passed checkpoint 1 cleanly but retained broad
  validation, proposal, policy, merge, and bulk-evaluation gaps in later
  checkpoints.
- `database_migration` still missed default-value backfills, rollback behavior,
  foreign-key handling, and dependency semantics. DataGate retained isolated
  core gaps around Latin-1 autodetection, row identifiers, and spreadsheet
  query controls.
- `trajectory_api` kept the essential behavior of checkpoints 1–4 but still
  missed lifecycle defaults, environment reparsing, grammar validation, file
  snapshots, and later accumulated regressions. The timestamp repair therefore
  did not make the problem trivially clean.
- Execution Server missed only five tests overall, concentrated in compressed
  structured files and duplicate-environment behavior. Forge and TextDrop kept
  every core checkpoint but lost current or accumulated regression behavior.
- SheetEval passed 651 of 653 tests and kept all seven core checkpoints; its
  remaining misses were a missing-placeholder warning and its accumulated
  regression. EVE Industry also kept every core checkpoint but retained
  invention and build-calculation functionality gaps.

No security-classifier or refusal evidence was found in supported logs or
inference summaries. All 66 inference runs completed without error, all 66
evaluator records were present, and postprocessing reported zero errors with
full `scb-check` coverage.

## Comparison status

This is `n=1` for Sol high and the fifth completed profile in the v2.2
sample-1 breadth pass. It currently leads the pass on strict (`14`), current
(`28`), and core (`51`) solves, while tying Terra xhigh at `7,562` tests.
That is a preliminary capability lead, not a reliability estimate.

Relative to Sol medium sample 1, Sol high recorded two more strict solves,
four more current solves, one more core solve, and 54 more passing tests. It
also cost `$30.45` more and used about 4 hours 3 minutes more recorded agent
time. Bidirectional checkpoint flips were `5/3` strict, `7/3` current, and
`2/1` core. The gain is real in this observation, but modest relative to its
additional compute.

Against Terra xhigh, Sol high recorded five more strict solves, one more
current solve, and two more core solves while tying the total test count. It
cost `$40.85` more and used about 1 hour 12 minutes more recorded agent time;
bidirectional flips were `8/3` strict, `6/5` current, and `4/2` core. Sol high
therefore has the best preliminary capability result so far, but not the best
cost or speed profile.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `9:13:19.207`, evaluator work was
`1:35:18.241`, and supported aggregate overhead was `0:21:44.298`.
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
