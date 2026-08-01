# SCBench v2 harness review record

This document records the review gate used before spending model credits on
the SCBench v2 profiles. It is part of the experiment provenance: the benchmark
was not launched from the first working implementation.

## Review method

The fork was reviewed in rotating scopes, with each repair re-read by someone
other than its implementer:

1. Catalog/profile identity, preflight, provenance, resume, and artifact
   integrity.
2. Checkpoint execution, evaluator failure classification, cancellation,
   telemetry, and durable reporting.
3. Docker parent identity, agent-image construction, snapshot capture and
   extraction, and cross-platform behavior.
4. Adversarial post-repair passes aimed at crash windows, filesystem races,
   contradictory artifacts, and compound failures.

Paid diagnostics and the full run are gated on the final full-suite tests,
profile verifier, dry-runs, real Docker builds, and archive-loader smoke test.

## Material findings and resolutions

The review found issues that could otherwise have changed a score or mislabeled
an infrastructure failure as model behavior:

- Named-profile runs originally had mutation windows between integrity
  validation and provenance initialization. Crashes during preflight, catalog
  staging, or image preparation could wedge a new or resumed output.
- Incomplete provenance did not structurally reject symlink traps before
  config/environment rewrites, and the artifact manifest excluded too many
  recursively nested log files.
- Complete checkpoint artifacts could disagree with, or outlive, missing or
  corrupt `run_info.yaml`; several compound solve/evaluator/cleanup failures
  could also produce false durable metadata.
- Evaluator infrastructure errors, cancellation, and secondary cleanup errors
  were not consistently kept distinct from model solution failures.
- Docker provenance labels named immutable parents while some builds still
  consumed mutable names in `FROM`; archive extraction and Docker build
  contexts also had no-follow and mutation-detection gaps.
- Snapshot carry-forward lost file mtimes, real and empty directories,
  directory metadata, and could miss a same-size rewrite with restored mtime.
- The first frozen `linux/arm64` base accidentally bundled an amd64 MinIO
  binary. It worked locally only through implicit emulation.
- Several published manifest claims were descriptive rather than verified
  against the actual environment, evaluator invocation, profile configuration,
  and pinned tool recipe.
- The first public harness tag, `scbench-v2-repro.1`, left two ISO timestamps
  unquoted in YAML. The first real reference diagnostic stopped before model
  work when provenance refused to serialize the resulting `datetime` values.
  Tag `scbench-v2-repro.2` quotes those values and makes JSON compatibility a
  verifier invariant. The base-image bytes and checksum were unaffected.
- The first `.2` reference diagnostic then exposed an incomplete test
  collection hash: pytest node IDs containing spaces inside parameter values
  were discarded even though those tests executed and contributed to the
  pass counts. The diagnostic was stopped after two durable checkpoints.
  `scbench-v2-repro.3` preserves those node IDs and regression-tests the exact
  observed cases; no `.2` result is treated as a benchmark baseline.
- The first `.3` current-profile diagnostic attempts stopped before model work
  because the evaluator subprocess scrubbed the explicitly configured
  `UV_CACHE_DIR` and fell back to a protected host cache. The cache path cannot
  change the locked dependency graph under `uv run --frozen`.
  `scbench-v2-repro.4` therefore preserves only `UV_CACHE_DIR` while continuing
  to scrub semantic uv overrides; the two failed attempts are retained as
  preflight evidence and incurred no model spend.

## v2.1 catalog repair gate

The first capability-panel runs exposed defects in the upstream `v1.0`
catalog. Those runs are not benchmark baselines. The repaired catalog is the
fork release
[`v1.0.1`](https://github.com/kkondaurov/scb-problems/releases/tag/v1.0.1)
at commit `9b4864d6bdefd8cd0f2d66d3eb0d1972914aefd3`, based on current
upstream `main` commit `ef6a9dd`.

The repair scope is limited to reproduced contradictions and infrastructure
failures in the capability-11 panel: writable database seeds, statements that
disagreed with their tests, skipped lifecycle behavior in the dynamic-config
tests, invalid trajectory fixtures, brittle SheetEval diagnostic matching,
SheetEval fatal-prerequisite and HTML-output prose that contradicted the
reference and tests, and the Mocked HTTP test harness losing already-buffered
stderr lines. Reference
solution changes are confined to Dynamic Config, whose corrected lifecycle
tests exposed a genuine version-zero inconsistency in its reference. Dynamic
Buffer and Test Translator remain byte-for-byte upstream. No checkpoint intent
or expected outcome was changed.

All repair validation used the benchmark's sanctioned `tools run-case`
evaluator. Cumulative reference results were Database Migration 136 passed with
one pre-existing platform skip, DataGate 405/405, Dynamic Config 84/84, EVE
Market Tools 76/76, Forge 295/295, SheetEval 164/164, and Trajectory API
375/375; the two Mocked HTTP cases exercising the repaired reader passed 2/2.
Problem tests were never invoked directly with pytest.

The v2.1 base adds a lock-installed Node 22.21.1, `tsx` 4.23.1, and TypeScript
7.0.2 toolchain so generated TypeScript is compiled strictly without on-demand
`npx` downloads. Its verified local image is:

- image ID:
  `sha256:f92550022dbc45c417e0c5bfcab706411b7407881ffd2d9b74d4e2049bbce985`
- platform: `linux/arm64`
- release archive SHA256:
  `eac1f5965f0c563529861a8b7b62fd2e2c0de6f726ffcdf38bcd8538eb145a44`
- catalog tree SHA256:
  `263199721db8c873aef0643224d244712fd5f56f4565198bf4cefd0e65211f8c`

Repairs are covered by adversarial tests for the reproduced failure modes.
The final snapshot format preserves safe relative symlinks, executable modes,
file/directory/symlink mtimes, and empty directories; unsafe archive paths,
workspace-escaping links, path swaps, unsupported types, and concurrent source
mutation fail closed. Resume reconciliation treats complete config-bound
checkpoint artifacts as primary evidence and repairs metadata without rerunning
or repaying a completed solve.

The corrected base is:

- image ID:
  `sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14`
- platform: `linux/arm64`
- MinIO: `RELEASE.2025-09-07T16-13-09Z`, native `linux/arm64`
- MinIO SHA256:
  `5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d`
- release archive SHA256:
  `dff52ff24d1d7e7d88525ef403b374d6464d3331c2b3df1ce72e4d56a3ec5df9`

The rejected candidate was image
`sha256:495ea3d315fc7daaa9f81415f2d429bf0eb22970a5294551dee171cbe80bfd15`
with archive SHA256
`91add59fc4aa4647881f7d562f6dee36f479ee9cb0770f9f8f9046d257973283`.
Its local archive and obsolete Docker images were removed after the native
replacement passed its loader smoke test; no benchmark run output was removed.

## Verification evidence

Confirmed release gates:

- `DOCKER_HOST=unix:///Users/kkonstant/.colima/default/docker.sock
  TMPDIR=<checkout>/tmp/pytest-host UV_NO_CONFIG=1
  UV_CACHE_DIR=/private/tmp/slopcodebench-uv-cache uv run --frozen pytest -q`:
  2,058 passed, 18 warnings.
- `UV_NO_CONFIG=1 uv run --frozen python scripts/verify_paper_v2.py`: 36
  problems, 196 checkpoints, 6,184 files, catalog tree
  `263199721db8c873aef0643224d244712fd5f56f4565198bf4cefd0e65211f8c`.
- Runner regression tests passed 173/173 with one intentional skip. The wider
  non-integration suite passed 2,030 tests in the restricted sandbox; its two
  uvx-dependent cases passed separately once the temporary tool cache was
  populated with network access.
- Both named diagnostic profiles passed non-mutating `--dry-run` preflight:
  `paper-v2-reference-diagnostic.yaml` and
  `gpt-5.5-current-xhigh-diagnostic.yaml`.
- `scripts/load_scbench_v2_base.sh` loaded exact image
  `sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14`
  as `linux/arm64`; the loader verified MinIO
  `RELEASE.2025-09-07T16-13-09Z`, checksum
  `5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d`,
  and native `linux/arm64` runtime.
- The Codex 0.124.0 agent image was built as
  `sha256:c22f8e82355b732c523c44ee116437da3f9e9f11f37c2a691b3d5a40b8f348fc`.
  It reported `arm64`, the exact pinned parent-image label
  `sha256:d2b862aad2bf40fe80573d0facc462608ce2a2fe76b56927a36050dc02a44f14`,
  Codex CLI 0.124.0, Git 2.47.3, and ripgrep 14.1.1.
- The Codex 0.146.0 agent image was built as
  `sha256:81f369d5267fc4c03799cbe47b1bfaabf2b666d374216b1c5cae7c539b467d05`.
  It reported `arm64`, the same exact pinned parent-image label, Codex CLI
  0.146.0, Git 2.47.3, and ripgrep 14.1.1.

## Reproduction boundaries

- The catalog bytes are exact for fork release `v1.0.1`, based on catalog
  upstream `main` commit `ef6a9dd`. The runner is based on public upstream
  commit `13de1a7` plus
  reviewed fixes in this fork, published as
  [`scbench-v2.1-repro.1`](https://github.com/kkondaurov/slop-code-bench/releases/tag/scbench-v2.1-repro.1).
  It is not the unpublished byte-exact paper runner, and old `v1.0` capability
  runs are not v2.1 benchmark baselines.
- `scb-check==0.1.3` is a pinned reconstruction choice because the paper did
  not publish the evaluator version.
- The published base artifact is arm64-only. An amd64 result requires a
  separately built and locked base and must be labeled as a distinct runtime.
- The upstream evaluation protocol resolves problem-specified and pytest
  dependencies through `uvx`; those package versions were not published by the
  paper. Run artifacts and timestamps must therefore be preserved, and this
  remaining upstream dependency-resolution boundary must not be described as
  bit-for-bit hermetic reproduction.
