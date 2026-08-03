# SCBench v2.2 capability-panel repair

SCBench v2.2 is the measurement surface for the eleven-problem,
66-checkpoint capability panel after a narrow catalog repair. It does not
rebalance the panel, lower its intended difficulty, or reinterpret ordinary
model failures as benchmark defects. Pre-v2.2 outputs are diagnostic evidence
only and must not be published or compared as v2.2 samples.

## Release identity

- Runner: `kkondaurov/slop-code-bench`, tag `scbench-v2.2-workers4`.
- Catalog: `kkondaurov/scb-problems`, tag `v1.0.2`, commit
  `88e9666f9529c97a30951fca17cb38656ea0a5f1`.
- Catalog content: 36 problems, 196 checkpoints, 6,184 locked files, tree
  SHA-256 `42a5a447856e057d67716edb83aec17ff72b98d50d83b500eb439fdc947c89e5`.
- Capability panel: exactly 11 problems and 66 checkpoints.

The immutable catalog identity and content digest are bound by
`configs/scbench-v2/manifest.yaml` and
`configs/scbench-v2/content-lock.json`. The catalog release contains the
canonical detailed record in `SCBENCH_V2_2_REPAIR_NOTES.md`.

## Audit method and repair boundary

The review adapted the cross-artifact questions from
[BenchGuard](https://arxiv.org/abs/2604.24955): compare the public contract,
evaluator, reference, runtime assumptions, and observed executions. A change
was retained only for a demonstrated contradiction, an evaluator rejection of
publicly compliant behavior, an unstated interpretation enforced by scoring,
or a demonstrated false success against an existing public requirement.

The review did not change requirements because models found them difficult,
relax tests to improve solve rates, retain speculative mutant-driven
hardening, or relabel older results.

## Retained catalog changes

| Problem | Retained change | Scoring effect |
| --- | --- | --- |
| `eve_market_tools` | Clarify that response `yields` use the existing 0–1 multiplier scale. | None; fixtures and scoring are unchanged. |
| `trajectory_api` | Accept an optional decimal fraction in otherwise constrained ISO-8601 UTC timestamps ending in `Z`. | Removes rejection of timestamps allowed by the public contract. |
| `textdrop` | Recognize a direct-child `code` element with optional attributes and whitespace. | Removes serialization-only rejection while preserving structure, content, and escaping checks. |
| `sheeteval` | Treat literal apostrophes and their standard HTML character references as equivalent. | Removes serialization-only rejection; other HTML expectations remain. |
| `database_migration` | Run the inherited drop-column/foreign-key regression cumulatively and verify schema, data, metadata, and `foreign_key_check`. | Prevents later-checkpoint credit after corrupting an inherited foreign-key relationship. |
| `forge` | State that a relative config `rules_file` resolves from the process working directory. | None; this publishes the evaluator's existing rule. |
| `eve_industry` | Validate the documented final canonical block, table structure, and unique row keys. | Prevents malformed, trailing, or duplicate-key output from normalizing to a pass while retaining arbitrary preamble and harmless Markdown spacing. |

Additional coverage proposals—including a MinIO/SigV4 rewrite, deterministic
Dynamic timeout seam, Trajectory concurrency probe, and real-broker rewrite—
were not retained in this release. Established limitations among those areas
are documented below rather than silently treated as solved.

## Known unresolved scoring issues

These are documented limitations, not changes in v2.2:

- `dynamic_config_service_api`: the required 500 ms evaluation timeout is not
  deterministically distinguished from an ordinary successful response.
- `eve_market_tools`: compression feasibility is checked, but global purchase
  cost optimality is not independently proven by the evaluator.
- `textdrop`: the controlled object-store server does not verify the required
  credentials' observable authentication behavior.
- `trajectory_api`: activation atomicity is stated, but no concurrent read
  probes for an observable intermediate state.

Controlled Kafka and RabbitMQ client fakes in `mocked_http` remain legitimate
test isolation; no real-broker requirement or runtime dependency was added.

## Validation and runtime

The three scoring-hardening changes added in the final repair commit were
validated through the sanctioned snapshot evaluator:

- Database Migration checkpoints 1–5: `39/39`, `62/62`, `87/87`, `117/117`,
  and `137/137`.
- Forge checkpoint 8: `295/295`.
- EVE Industry checkpoints 1–6: `12/12`, `33/33`, `50/50`, `59/59`, `73/73`,
  and `80/80`.

Targeted EVE Industry mutants with trailing junk and duplicate canonical rows
failed the intended assertions without evaluator or infrastructure failure.
The cumulative Database Migration test also failed the unreleased checkpoint
3–5 references before their incoming-foreign-key drop behavior was corrected.

The catalog-only repair adds no runtime dependency. The v2.2 environment
configuration therefore reuses the exact published v2.1 base image and archive.
The breadth-first queue permits up to four concurrent problem workers. Worker
count is an operational protocol choice, not part of the catalog repair; agent
and evaluator time remain separate from end-to-end wall time in result
reporting.

No pre-v2.2, abandoned, duplicate, or incomplete output counts toward the new
breadth-first queue.
