# SCBench v2 experiment profiles

This branch locks a repaired revision of the 36-problem, 196-checkpoint catalog
used by revision v2 of
[arXiv:2603.24755](https://arxiv.org/html/2603.24755v2). The fork repair release
`v1.0.1` is based on upstream `main` commit `ef6a9dd`. The pre-run adversarial
review and its reproduced defects are recorded in
[the harness review record](SCBENCH_V2_REVIEW.md).

The exact catalog source is `kkondaurov/scb-problems` release `v1.0.1`, commit
`9b4864d6bdefd8cd0f2d66d3eb0d1972914aefd3`. Install and verify it before a
diagnostic or experiment:

```bash
UV_NO_CONFIG=1 uv sync --frozen
UV_NO_CONFIG=1 uv run --frozen slop-code sync v1.0.1
UV_NO_CONFIG=1 uv run --frozen python scripts/verify_paper_v2.py
```

The verifier checks the managed release metadata, all 36 declared problem
names, all 196 checkpoint declarations, the 6,184-file set, and a deterministic
content hash. It does not update or download the catalog. A byte mismatch is a
failed preflight, not an invitation to regenerate the lock in place.

The experiment profiles use the dedicated
`docker-python3.12-uv-scb-v2.1` environment. Two different Docker identities are
locked and should not be conflated:

- The upstream Astral `uv` manifest digest is the immutable parent input to the
  base-image setup recipe. The mutable tag is recorded only as a human-readable
  source reference.
- `sha256:f92550022dbc45c417e0c5bfcab706411b7407881ffd2d9b74d4e2049bbce985`
  is the fully built `linux/arm64` Docker image after that recipe ran. Named
  profiles use this prebuilt image directly and verify its image ID and
  architecture instead of rerunning setup.

The v2.1 harness is published as
[`scbench-v2.1-repro.1`](https://github.com/kkondaurov/slop-code-bench/releases/tag/scbench-v2.1-repro.1).
Its `linux/arm64` base image and
[`release-assets-v2.1.sha256`](https://github.com/kkondaurov/slop-code-bench/releases/download/scbench-v2.1-repro.1/release-assets-v2.1.sha256)
are release assets. Download the image to the loader's default path, or pass an
existing local copy explicitly:

```bash
mkdir -p outputs/reproducibility-images
curl -fL \
  https://github.com/kkondaurov/slop-code-bench/releases/download/scbench-v2.1-repro.1/slopcodebench-base-scb-v2.1-linux-arm64-image-f92550022dbc.tar.zst \
  -o outputs/reproducibility-images/slopcodebench-base-scb-v2.1-linux-arm64-image-f92550022dbc.tar.zst
```

```bash
scripts/load_scbench_v2_1_base.sh

# Or:
scripts/load_scbench_v2_1_base.sh /absolute/path/to/the-image.tar.zst
```

The loader checks the archive SHA256, runs `zstd --test`, imports it through
`docker load`, and then verifies the loaded image ID and `linux/arm64`
platform. It also starts the image to verify the pinned MinIO checksum, release,
and native `linux/arm64` runtime. The harness independently repeats the image-ID
and architecture checks before use. The published base is arm64-only and fails
closed on another architecture; an amd64 run needs a separately built,
published, and locked artifact.

## Profiles

`paper-v2-reference` uses GPT-5.5 through local Codex subscription
authentication, Codex CLI 0.124.0, high reasoning, and the `just-solve` prompt.
These are the published GPT-5.5 settings. The tagged reproducibility fork begins
at upstream commit `13de1a7` and adds reviewed post-paper correctness fixes, so
this is a reference-aligned replication rather than a claim of byte-exact
reconstruction.

`gpt-5.5-current-xhigh` keeps the catalog and prompt fixed while changing Codex
CLI to 0.146.0 and reasoning to xhigh. Version 0.146.0 was the latest stable
`@openai/codex` release returned by the official npm registry at
2026-07-29T11:24:57Z. The registry publication timestamp and package and Linux
platform integrity values are frozen in the manifest. This profile is an
extension experiment and must not be labeled as a reproduction of the paper's
GPT-5.5 row. GPT-5.6 Luna, Sol, and Terra profiles are likewise extension
experiments on the repaired catalog.

Each profile has a full config and an 11-checkpoint diagnostic over `mvvault`
and `xjq`. Selected extension profiles also have a `capability-11` config over
11 problems and 66 checkpoints:

```bash
UV_NO_CONFIG=1 uv run --frozen slop-code run \
  --config configs/runs/paper-v2-reference-diagnostic.yaml \
  --num-workers 2

UV_NO_CONFIG=1 uv run --frozen slop-code run \
  --config configs/runs/gpt-5.5-current-xhigh-diagnostic.yaml \
  --num-workers 2
```

The first `uv sync --frozen` is a one-time environment setup. After that, each
experiment is a single `uv run --frozen ...` command; there is no recurring
sync ritual. `--frozen` prevents an experiment command from rewriting the
repository lockfile.

For a real named-profile run, the runner sets `SLOP_CODE_TMPDIR` to
`<checkout>/tmp/scbench-v2` when that variable is unset, before worker
processes start. This keeps host workspaces and snapshot archives under a path
that Docker VMs such as Colima can bind-mount. An explicit
`SLOP_CODE_TMPDIR` override is preserved, but it must be a non-empty absolute
directory path that the Docker daemon can access.

Named profiles are fail-closed. After all config overrides and problem
selection are resolved, the runner requires the exact diagnostic or full
problem list and verifies the model, provider, agent type and version, reasoning
level, prompt, seed, pass policy, timeout and cost limits, environment, source
image digest, prebuilt image lock, two-worker count, evaluation mode, and
catalog release and bytes.
Do not add `--problem`, change worker count, disable evaluation, enable
concurrent evaluation, or add semantic overrides to these commands.

Every real named-profile start and resume also resolves the frozen evaluator
before model work and appends expected-versus-actual evidence to
`scbench_v2_preflight.json`. Earlier attempts remain in the artifact and a
current-attempt pointer identifies the latest one. `--dry-run` performs only
read-only profile and catalog validation: it does not create the output
directory, persist preflight evidence, install the catalog, or resolve/install
the evaluator.

The reference diagnostic is a harness-alignment check, not a statistically
complete reproduction of the paper's 196-checkpoint aggregate. Before the full
extension run, inspect both diagnostic runs for model behavior and for harness
failures: unavailable expected tools, authentication or rate-limit errors,
Docker isolation failures, evaluator errors, missing metric coverage, stream
loss, or inconsistent token and cost accounting. Fix a harness defect and
repeat the affected diagnostic; do not reinterpret infrastructure failure as a
model miss.

Headline verbosity and erosion use `scb-check==0.1.3`. This is a pinned
paper-era reconstruction choice: the paper did not publish its exact evaluator
version. The evaluator has its own Python 3.12 project and complete `uv.lock`;
it runs with `UV_NO_CONFIG=1 uv run --frozen --project` and records the lock
hash with every checkpoint measurement. Evaluator output is collected through
bounded, disk-backed streams. Checkpoint carry-forward preserves executable
regular-file modes and safe relative symlinks; absolute or workspace-escaping
links are rejected while snapshotting. The primary quality evaluator still
treats every preserved symlink as unsupported and fails closed rather than
following it into host state. A separately labeled
`scb-check==0.2.0` pass may be run later against the preserved snapshots as a
sensitivity analysis, but it must not overwrite the primary quality artifacts
or headline metrics.

Only after that gate is clean should an extension config be launched. The full
Terra-high capability profile is:

```bash
UV_NO_CONFIG=1 uv run --frozen slop-code run \
  --config configs/runs/gpt-5.6-terra-high-capability-11.yaml \
  --num-workers 2
```

The post-repair SheetEval validation is deliberately narrower than the named
capability profile. It is a one-problem diagnostic and therefore omits the
named-profile key rather than weakening the named profile's fail-closed problem
set:

```bash
UV_NO_CONFIG=1 uv run --frozen slop-code run \
  --config configs/runs/gpt-5.6-terra-high-sheeteval-validation.yaml \
  --num-workers 2
```

## Published comparison surfaces

[Paper Table 1](https://arxiv.org/html/2603.24755v2) reports GPT-5.5
strict/isolated/core rates of 14.8%, 28.1%, and 66.8%. The
[live Snorkel leaderboard](https://snorkel.ai/leaderboard/slopcode-bench/)
currently displays 14.29%, 28.06%, and 65.31% for the row labeled GPT-5.5,
high reasoning, Codex 0.124.0. These are distinct published surfaces. Without
the underlying export and denominator history, their difference must not be
treated as experimental variance or silently reconciled. A result report must
name the one comparison surface it uses. The 11-checkpoint diagnostic is a
harness gate and is not directly comparable to either 196-checkpoint aggregate.

The immutable experiment facts and profile semantics live in
`configs/scbench-v2/manifest.yaml`; the catalog bytes are locked by
`configs/scbench-v2/content-lock.json`. Source and release assets are bound to
the public
[`scbench-v2.1-repro.1` release](https://github.com/kkondaurov/slop-code-bench/releases/tag/scbench-v2.1-repro.1).
