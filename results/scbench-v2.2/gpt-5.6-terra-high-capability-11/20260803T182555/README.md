# GPT-5.6 Terra high capability-11 v2.2 run

This is sample 1 for GPT-5.6 Terra high on the frozen SCBench v2.2
capability-11 panel. Pre-v2.2 runs and repair canaries are not part of this
comparison set.

## Configuration

- Model: `gpt-5.6-terra`
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
- Started: `2026-08-03T18:26:08+02:00`
- Finished: `2026-08-03T20:27:42+02:00`
- Wall time: `2:01:33.945`

The checked-in configuration is
[`configs/runs/gpt-5.6-terra-high-capability-11.yaml`](../../../../configs/runs/gpt-5.6-terra-high-capability-11.yaml).

## Headline results

- Completely clean checkpoints: `10/66` (`15.15%`)
- Current-checkpoint behavior clean: `24/66` (`36.36%`)
- Essential behavior clean: `49/66` (`74.24%`)
- Tests passed: `7,516/8,113` (`92.64%`)
- Accumulated regression tests passed: `5,364/5,733` (`93.56%`)
- API-equivalent retail cost estimate: `$31.59`
- Recorded agent time: `5:10:00.298`
- Evaluator time: `1:37:40.340`
- Aggregate per-problem overhead: `0:22:45.428`
- Steps: `1,611`
- Evaluator coverage: `66/66` (`100%`)
- Infrastructure/evaluator failures: `0`

"Completely clean" requires every current and accumulated regression test to
pass. "Current-checkpoint behavior clean" excludes accumulated regressions.
"Essential behavior clean" considers only each checkpoint's core tests.

The result is **mixed with strong essential coverage**. Terra kept the core
behavior of 49 checkpoints and was completely clean on ten. DataGate,
SheetEval, TextDrop, and most of Execution Server were the strongest areas.
Dynamic Config, EVE Market Tools, and Mocked HTTP account for the broadest
remaining capability gaps.

## Problem results

| Problem | Clean | Current | Essential | Tests | Cost | Problem wall time | Judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `database_migration` | 1/5 | 1/5 | 2/5 | 375/442 | $2.79 | 0:30:26 | mixed-to-weak |
| `datagate` | 0/7 | 5/7 | 7/7 | 1,594/1,613 | $2.48 | 0:41:28 | strong, regression-limited |
| `dynamic_config_service_api` | 0/4 | 0/4 | 2/4 | 236/303 | $2.86 | 0:29:52 | weak |
| `eve_market_tools` | 0/4 | 0/4 | 1/4 | 88/179 | $2.14 | 1:06:03 | weak |
| `execution_server` | 0/6 | 3/6 | 5/6 | 594/608 | $1.97 | 0:23:39 | strong-to-mixed |
| `forge` | 2/8 | 4/8 | 7/8 | 1,175/1,279 | $1.89 | 0:30:48 | strong essential, regression-limited |
| `textdrop` | 1/6 | 3/6 | 6/6 | 622/635 | $2.48 | 0:35:13 | strong, regression-limited |
| `trajectory_api` | 1/5 | 1/5 | 4/5 | 1,145/1,200 | $2.81 | 0:32:51 | strong essential through checkpoint 4 |
| `mocked_http` | 0/8 | 1/8 | 4/8 | 761/894 | $5.35 | 1:00:16 | mixed-to-weak |
| `sheeteval` | 2/7 | 3/7 | 7/7 | 632/653 | $3.12 | 0:38:02 | strong, regression-limited |
| `eve_industry` | 3/6 | 3/6 | 4/6 | 294/307 | $3.70 | 0:41:41 | mixed |

## Failure clusters

- `dynamic_config_service_api` remained incomplete across policy evaluation,
  proposal/review flows, bulk policy handling, and associated regressions.
- `eve_market_tools` missed broad market statistics, reprocessing,
  compression, hauling, and trading behavior. Its evaluator alone consumed
  about 42 minutes, so this is neither a useful nor a fast failure.
- `mocked_http` retained gaps in template helpers, persistence, command-line
  push behavior, and AMQP/Kafka handling.
- `database_migration` missed data backfills, rollback behavior, foreign-key
  handling, and dependency semantics.
- `trajectory_api` again showed that the repaired timestamp contract behaves
  as intended: checkpoint 1 was completely clean and checkpoints 1–4 kept
  their essential behavior. Remaining misses concern boundaries, lifecycle,
  environment activation, and the final tool-execution/toolpack behavior.
- DataGate and SheetEval kept all core checkpoints clean. TextDrop did the
  same; its remaining misses concern ordered-list behavior, error/debug pages,
  and object-store failure handling.

No security-classifier or refusal evidence was found in supported logs or
inference summaries. All 66 inference runs completed without error, all 66
evaluator records were present, and postprocessing reported zero errors with
full `scb-check` coverage.

## Comparison status

This is `n=1` for Terra high and the second completed profile in the v2.2
sample-1 breadth pass. Against Luna xhigh sample 1, Terra recorded three more
strict solves (`10` versus `7`), two more current solves (`24` versus `22`),
and two more core solves (`49` versus `47`), while costing `$3.31` more.
Its agent time was `5:10:00.298` versus Luna's `12:29:03.623`, and wall time
was `2:01:33.945` versus `3:55:59.359`; because capability also increased,
this is useful speed rather than merely faster failure.

The aggregate lead conceals bidirectional checkpoint flips: Terra gained/lost
`4/1` strict, `7/5` current, and `8/6` core solves relative to Luna. With only
one observation per profile, these differences are preliminary profile
outcomes, not reliability or variance estimates.

## Timing notes

Agent time is the primary model-speed measure; wall time is the four-worker
operational throughput measure. Recorded agent work sums checkpoint durations
and therefore exceeds wall time when workers overlap. Across the 11 problem
jobs, recorded agent work was `5:10:00.298`, evaluator work was
`1:37:40.340`, and supported aggregate overhead was `0:22:45.428`.
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
