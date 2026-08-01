from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import typer

from slop_code.entrypoints.commands.tools import run_case
from slop_code.evaluation.config import CheckpointConfig
from slop_code.evaluation.report import CorrectnessResults


@pytest.mark.parametrize("json_output", [False, True])
def test_run_case_exits_nonzero_when_tests_fail(
    tmp_path, *, json_output: bool
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    checkpoint = CheckpointConfig(
        name="checkpoint_1",
        version=1,
        order=1,
        timeout=10,
        env={},
    )
    problem = MagicMock()
    problem.iterate_checkpoint_items.return_value = [
        ("checkpoint_1", checkpoint)
    ]
    results = CorrectnessResults(
        problem_name="problem",
        problem_version=1,
        checkpoint_name="checkpoint_1",
        checkpoint_version=1,
        duration=1.0,
        entrypoint="python main.py",
        pytest_exit_code=1,
        pytest_collected=1,
        stdout="1 failed\n",
    )
    ctx = SimpleNamespace(obj=SimpleNamespace(scbench_home=tmp_path))

    with (
        patch(
            "slop_code.entrypoints.commands.tools.common.resolve_problem_catalog_root",
            return_value=tmp_path,
        ),
        patch(
            "slop_code.entrypoints.commands.tools.common.load_problem_config_or_exit",
            return_value=problem,
        ),
        patch(
            "slop_code.entrypoints.commands.tools.config_loader.resolve_environment",
            return_value=MagicMock(),
        ),
        patch(
            "slop_code.entrypoints.commands.tools.common.ensure_docker_ready",
            return_value=None,
        ),
        patch(
            "slop_code.entrypoints.commands.tools.run_checkpoint_pytest",
            return_value=results,
        ),
        pytest.raises(typer.Exit) as exc_info,
    ):
        run_case(
            ctx=ctx,
            snapshot_dir=snapshot_dir,
            problem_name="problem",
            checkpoint_num=1,
            env_config=tmp_path / "env.yaml",
            json_output=json_output,
        )

    assert exc_info.value.exit_code == 1
