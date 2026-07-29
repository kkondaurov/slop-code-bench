from __future__ import annotations

from types import SimpleNamespace

from slop_code import evaluation
from slop_code.common import WORKSPACE_TEST_DIR
from slop_code.evaluation.collection import _build_collect_cmd
from slop_code.evaluation.collection import _parse_collect_stdout
from slop_code.evaluation.collection import compute_tc_hash


def test_public_collection_functions_exported() -> None:
    assert callable(evaluation.collect_checkpoint_tc)
    assert callable(evaluation.compute_tc_hash)


def test_collect_cmd_excludes_agent_conftest() -> None:
    """The --collect-only passes must exclude an agent-authored root conftest.py.

    Pytest otherwise loads a conftest.py at the workspace root (an ancestor of
    the eval test dir); if it errors or conflicts, collection aborts (exit 4)
    and the checkpoint scores 0 as an infrastructure failure.
    """
    cmd = _build_collect_cmd(
        problem=SimpleNamespace(test_dependencies=[]),
        checkpoint_name="checkpoint_1",
        entrypoint="python main.py",
        marker="functionality",
        pytest_args=None,
    )
    assert f"--confcutdir={WORKSPACE_TEST_DIR}" in cmd


def test_compute_tc_hash_stable_for_reordered_input() -> None:
    hash_a = compute_tc_hash(
        {
            "checkpoint_2-Core": ["test_b", "test_a"],
            "checkpoint_2-Error": ["test_e"],
        }
    )
    hash_b = compute_tc_hash(
        {
            "checkpoint_2-Error": ["test_e"],
            "checkpoint_2-Core": ["test_a", "test_b"],
        }
    )

    assert hash_a == hash_b


def test_collect_stdout_keeps_spaces_inside_parameter_ids() -> None:
    stdout = """
============================= test session starts ==============================
tests/test_checkpoint_3.py::test_metadata[title-Episode One]
tests/test_checkpoint_3.py::test_metadata[description-Episode Description]
ERROR tests/test_checkpoint_3.py::test_noise collection failed
tests/test_checkpoint_3.py::test_noise PASSED
========================== 2 tests collected in 0.01s ==========================
"""

    nodeids = _parse_collect_stdout(stdout)
    expected_nodeids = [
        "tests/test_checkpoint_3.py::test_metadata[title-Episode One]",
        (
            "tests/test_checkpoint_3.py::"
            "test_metadata[description-Episode Description]"
        ),
    ]

    assert nodeids == expected_nodeids
    assert len(nodeids) == 2

    collected_test_ids = [nodeid.split("::", 1)[1] for nodeid in nodeids]
    expected_test_ids = [nodeid.split("::", 1)[1] for nodeid in expected_nodeids]
    collection_hash = compute_tc_hash(
        {"checkpoint_3-Core": collected_test_ids}
    )

    assert collection_hash == compute_tc_hash(
        {"checkpoint_3-Core": expected_test_ids}
    )
    assert collection_hash != compute_tc_hash(
        {"checkpoint_3-Core": expected_test_ids[:1]}
    )
