# Paper-v1 reproduction profile

This profile packages the benchmark described in
[arXiv:2603.24755v1](https://arxiv.org/html/2603.24755v1): 20 Python-track
problems, 93 sequential checkpoints, the baseline `just-solve` prompt, and
the paper's 137-rule verbosity metric.

## Source pin

The source baseline is commit
[`21dd1f58f408cd89b7acccdf5db969905aece51b`](https://github.com/SprocketLab/slop-code-bench/commit/21dd1f58f408cd89b7acccdf5db969905aece51b),
the last public commit before the paper's 25 March 2026 submission. The paper
does not name a source SHA. This commit is the closest public snapshot that
simultaneously has the reported 20 problems, 93 checkpoints, and consolidated
137-rule implementation. The older `v0.2` archive has the same specifications
but older tests, metric code, and 355 AST-grep rules.

The machine-readable record is
[`configs/paper-v1/manifest.yaml`](../configs/paper-v1/manifest.yaml). Verify it
before a run:

```bash
UV_NO_CONFIG=1 uv sync --frozen
uv run --frozen python scripts/verify_paper_v1.py
```

The verifier checks the declared source pin, expected corpus/checkpoint and
hidden-test inventory, prompt hash, rule count and IDs, model matrix, and the
paper-relevant fields in the resolved run presets. It also requires the pinned
AST-grep executable and smoke-scans all 137 rules so that verbosity cannot
silently degrade to clone-only. It does not resolve credentials or contact
model providers.

Metric collection is fail-closed as well: a missing executable, invalid or
empty ruleset, failed scan, or malformed AST-grep result aborts quality
measurement instead of emitting a misleading zero.

The paper does not report its AST-grep executable version. This profile pins
AST-grep `0.42.0`, the latest official release before the selected source
snapshot: it was released on 16 March 2026, while `0.42.1` followed after the
paper submission. The exact `ast-grep-cli==0.42.0` native CLI distribution is
part of `pyproject.toml` and `uv.lock`, including platform artifact hashes.
The lockfile also records the source snapshot timestamp as `exclude-newer`, so
later-uploaded wheel files cannot silently enter an otherwise paper-era lock.
This is a best-evidence environment pin, not a version stated by the authors.
Set `AST_GREP_BIN` only if you deliberately want to test another executable;
the verifier still requires it to report version `0.42.0`.

## Fresh-container alignment

The paper says every checkpoint starts in a fresh non-root container and only
the prior working-directory snapshot carries forward. Installed packages,
shell history, agent session state, and conversation context must reset. The
public source snapshot instead kept one execution session alive for an entire
problem.

This suite fixes that paper/artifact discrepancy: it tears down the inference
session after every checkpoint and creates the next session from the saved
workspace snapshot. It attempts to reconstruct excluded runtime state such as
`.venv` from `requirements.txt`; if that attempt fails, the fresh agent session
can repair its declared environment. Hidden tests are materialized only after
the agent session is gone. Thus an agent inherits its code, but not its
container or conversation.

## Locked benchmark inputs

[`configs/paper-v1/content-lock.json`](../configs/paper-v1/content-lock.json)
is the byte-level identity of this profile. It records a SHA-256 for every
non-ephemeral file under `problems/` (including hidden tests, test data,
solutions, and static assets), plus the exact paper presets, model/provider/
agent/environment configs, prompt, AST-grep rules, `pyproject.toml`, `uv.lock`,
and the complete `src/slop_code/` harness implementation. The verifier compares
both the complete path set and every file's bytes. Only named caches, virtual
environments, bytecode, and `.DS_Store` are excluded.

The selected upstream source commit must be an ancestor of the checked-out
fork commit. The dependency verifier also rejects relative
`exclude-newer-span` settings and any locked artifact whose recorded upload
time is later than the absolute paper-era cutoff.

Regenerate the content lock only after an intentional benchmark-input change,
then review the resulting diff and update the verifier's
`EXPECTED_CONTENT_TREE_SHA256` before committing:

```bash
UV_NO_CONFIG=1 uv run --frozen python scripts/verify_paper_v1.py \
  --write-content-lock
```

## Run and evaluate

Docker must be running. Credentials remain entirely user-supplied through the
runner's configured providers; no API keys or login files belong in this
repository. The base presets resolve `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`;
the GLM override resolves `ZHIPU_API_KEY`. Set them in the invoking environment
or use the runner's documented credential override.

On macOS, Colima is one supported headless Docker setup. Start it with:

```bash
colima start --runtime docker --cpu 4 --memory 8 --vm-type vz
docker version
docker run --rm hello-world
```

Colima can bind-mount paths under `/Users`, but not macOS's default
`/private/var` temporary tree. Set `SLOP_CODE_TMPDIR` to an absolute directory
under `/Users` and outside the Git checkout, so solution-side tools cannot
mistake the harness project for their own. Create a local `.env` from
[`.env.example`](../.env.example) when using this setup; `.env` is ignored and
must never be committed.

The two presets default to the newest paper-v1 row for each harness:

```bash
# GPT 5.4 through Codex CLI 0.110.0
uv run --frozen slop-code run --config configs/runs/paper-v1-codex.yaml

# Opus 4.6 through Claude Code 2.1.32
uv run --frozen slop-code run --config configs/runs/paper-v1-claude-code.yaml
```

Credential source and model identity are separate. To use an existing Codex
subscription login from `~/.codex/auth.json`, keep the GPT 5.4 model and change
only the credential provider:

```bash
uv run --frozen slop-code run \
  --config configs/runs/paper-v1-codex.yaml \
  model.provider=codex_auth
```

Claude Code's login state on the macOS host is not mounted wholesale into the
fresh benchmark containers. Subscription use requires an exported
`CLAUDE_CODE_OAUTH_TOKEN`; with that present, use
`model.provider=claude_code_oauth`. The default `anthropic` provider instead
reads `ANTHROPIC_API_KEY`.

Both explicitly select all 20 problems, `just-solve`, high reasoning, a
7,200-second checkpoint timeout, and zero turn/cost caps. Outputs go below
`outputs/paper-v1/`. `pass_policy: any-case` is intentional: it records
whether any case passed while the runner's explicit exemption keeps failed
checkpoints from truncating the iterative trajectory. Strict, isolated, and
core correctness are still calculated from the detailed test results.

Every actual run automatically writes `provenance.json` at its output root.
It records the sanitized invocation; model, credential-provider name, harness
and version; seed and problem list; fork/upstream commits; clean or dirty Git
state (with only a diff hash); input/config/lock hashes; host and Docker image
metadata; tool versions when available; invocation status; and SHA-256 hashes
for stable run artifacts. Credential values are never recorded. Live `.log`
files, symlinks, and `provenance.json` itself are excluded from artifact hashes.
Resuming appends a new invocation with its own complete Git/input/host/image
context, executed-problem list, and final artifact snapshot, so later work can
never rewrite the identity of earlier output in the same run directory.

Codex token telemetry uses the CLI's final cumulative usage record. Its
inclusive input is split into uncached input plus cached input; output remains
inclusive of reasoning, so neither cache nor reasoning is priced twice. Raw
rollout reasoning is accepted only when its thread ID and cumulative totals
exactly match stdout; missing, ambiguous, or mismatched raw telemetry warns and
records zero reasoning instead of inventing a value.

`run` evaluates checkpoints by default. To separate inference and evaluation:

```bash
uv run --frozen slop-code run \
  --config configs/runs/paper-v1-codex.yaml \
  --no-evaluate

uv run --frozen slop-code eval outputs/paper-v1/<model>/<run> \
  --pass-policy any-case
```

## Exact model and CLI overrides

Override only the model/provider and agent version shown below; the preset
supplies the remaining protocol. For example:

```bash
uv run --frozen slop-code run \
  --config configs/runs/paper-v1-codex.yaml \
  model.name=gpt-5.2-codex agent.version=0.80.0
```

| Paper model | Preset | Overrides |
| --- | --- | --- |
| Sonnet 4.5 | Claude Code | `model.name=sonnet-4.5 agent.version=2.0.65` |
| Sonnet 4.6 | Claude Code | `model.name=sonnet-4.6 agent.version=2.1.44` |
| Opus 4.5 | Claude Code | `model.name=opus-4.5 agent.version=2.0.51` |
| Opus 4.6 | Claude Code | `model.name=opus-4.6 agent.version=2.1.32` |
| GPT 5.1 Codex Max | Codex | `model.name=gpt-5.1-codex-max agent.version=0.65.0` |
| GPT 5.2 | Codex | `model.name=gpt-5.2 agent.version=0.71.0` |
| GPT 5.2 Codex | Codex | `model.name=gpt-5.2-codex agent.version=0.80.0` |
| GPT 5.3 Spark | Codex | `model.name=gpt-5.3-codex-spark agent.version=0.100.0` |
| GPT 5.3 Codex | Codex | `model.name=gpt-5.3-codex agent.version=0.98.0` |
| GPT 5.4 | Codex | `model.name=gpt-5.4 agent.version=0.110.0` |
| GLM 4.7 | Claude Code | `model.provider=zhipu-coding-plan model.name=glm-4.7 agent.version=2.0.76` |

The paper did not release its raw trajectories, complete run manifest, or
provider-side snapshots. These presets reproduce the published local
protocol and version matrix; hosted model behavior can still drift.

## Publication-ready workflow

For an experiment you intend to cite, start from the public release tag and a
clean worktree:

```bash
git clone https://github.com/kkondaurov/slop-code-bench.git
cd slop-code-bench
git fetch --tags
git switch --detach paper-v1-repro.2
git status --short
UV_NO_CONFIG=1 uv sync --frozen
UV_NO_CONFIG=1 uv run --frozen python scripts/verify_paper_v1.py \
  --publication-ready
```

`git status --short` must print nothing. Publication mode also requires HEAD to
be exactly at `paper-v1-repro.2`; the verifier must pass before the run. Then
invoke one of the exact commands above. Preserve and publish the complete run
directory together with the command you used and the tag. The run's
`provenance.json` binds the results to the suite inputs, resolved configuration,
Git state, Docker images, and stable artifact bytes; the tag makes the harness
implementation independently retrievable.
