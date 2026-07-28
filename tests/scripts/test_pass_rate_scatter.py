from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "graphing"
    / "pass_rate_scatter.py"
)
if not MODULE_PATH.exists():
    pytest.skip(
        "paper-v1 source snapshot omits scripts/graphing/pass_rate_scatter.py",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location(
    "pass_rate_scatter_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prepare_data_uses_saved_checkpoint_composites():
    runs = pd.DataFrame(
        [
            {
                "model": "model-a",
                "thinking": "none",
                "agent_version": "v1",
                "prompt": "baseline",
                "num_checkpoints": 93,
                "num_problems": 20,
            }
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "model": "model-a",
                "thinking": "none",
                "agent_version": "v1",
                "prompt": "baseline",
                "cost": 1.5,
                "isolated_pass_rate": 0.8,
                "verbosity": 0.95,
                "erosion": 0.6,
                "clone_lines": 20,
                "loc": 100,
            }
        ]
    )

    out = MODULE.prepare_data(runs, checkpoints)

    assert len(out) == 1
    assert out.iloc[0]["verbosity_mean"] == pytest.approx(0.95)
    assert out.iloc[0]["erosion_mean"] == pytest.approx(0.6)
    assert out.iloc[0]["isolated_pass_rate_mean"] == pytest.approx(80.0)
