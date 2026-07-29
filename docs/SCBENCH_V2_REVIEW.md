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
  UV_CACHE_DIR=/private/tmp/slopcodebench-uv-cache .venv/bin/pytest -q`:
  2,054 passed, 3 skipped.
- `UV_NO_CONFIG=1 uv run --frozen python scripts/verify_paper_v2.py`: 36
  problems, 196 checkpoints, catalog tree
  `199ae38f7dc07b5bbaba3683ca98e783c3a800f27ee4b1075fa2fa14d32d1249`.
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

- The catalog bytes are exact for `scb-problems` release `v1.0`; the runner is
  based on public upstream commit `13de1a7` plus the reviewed fixes in this
  fork, published as
  [`scbench-v2-repro.1`](https://github.com/kkondaurov/slop-code-bench/releases/tag/scbench-v2-repro.1).
  It is not the unpublished byte-exact paper runner.
- `scb-check==0.1.3` is a pinned reconstruction choice because the paper did
  not publish the evaluator version.
- The published base artifact is arm64-only. An amd64 result requires a
  separately built and locked base and must be labeled as a distinct runtime.
- The upstream evaluation protocol resolves problem-specified and pytest
  dependencies through `uvx`; those package versions were not published by the
  paper. Run artifacts and timestamps must therefore be preserved, and this
  remaining upstream dependency-resolution boundary must not be described as
  bit-for-bit hermetic reproduction.
