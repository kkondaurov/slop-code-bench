from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    module_path = (
        Path(__file__).parent.parent / "scripts" / "abstraction_roi.py"
    )
    if not module_path.exists():
        pytest.skip(
            "paper-v1 source snapshot omits scripts/abstraction_roi.py",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(
        "abstraction_roi", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load scripts/abstraction_roi.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module()
analyze_file = _MODULE.analyze_file
analyze_snapshot = _MODULE.analyze_snapshot


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_wrapper_function_has_negative_net_savings(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
def round_duration(seconds):
    return format_seconds(seconds)

def run():
    return round_duration(5)
""".strip(),
    )

    result = analyze_file(path)
    by_name = {f.name: f for f in result.functions}

    assert "round_duration" in by_name
    assert by_name["round_duration"].external_calls == 1
    assert by_name["round_duration"].net_savings < 0


def test_recursive_self_calls_are_not_external(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "recursive.py",
        """
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
""".strip(),
    )

    result = analyze_file(path)
    fact = next(f for f in result.functions if f.name == "fact")

    assert fact.is_recursive is True
    assert fact.external_calls == 0


def test_module_and_method_with_same_name_do_not_share_calls(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "same_name.py",
        """
def helper(x):
    return x + 1


class Worker:
    def helper(self, x):
        return x + 2


def run(worker, value):
    return helper(value) + worker.helper(value)
""".strip(),
    )

    result = analyze_file(path)
    module_helper = next(
        f
        for f in result.functions
        if f.name == "helper" and f.parent_class is None
    )
    method_helper = next(
        f
        for f in result.functions
        if f.name == "helper" and f.parent_class == "Worker"
    )

    assert module_helper.external_calls == 1
    assert method_helper.external_calls == 1


def test_chained_attribute_call_is_not_marked_recursive(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "chained.py",
        """
class Worker:
    def execute(self, x):
        return x + 1


class Runner:
    def run(self, x):
        return self.worker.execute(x)
""".strip(),
    )

    result = analyze_file(path)
    run_method = next(
        f
        for f in result.functions
        if f.name == "run" and f.parent_class == "Runner"
    )

    assert run_method.is_recursive is False
    assert run_method.external_calls == 0


def test_branch_dup_and_repeated_dict_detection(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "dup.py",
        """
def compile_pattern_for_language(lang, pattern):
    if lang == "python":
        return build(pattern)
    elif lang == "javascript":
        return build(pattern)
    else:
        return build(pattern)


def a():
    return {"code": 1, "line": 2, "message": "x"}


def b():
    return {"code": 5, "line": 9, "message": "y"}
""".strip(),
    )

    result = analyze_file(path)

    assert len(result.branch_dups) == 1
    assert result.branch_dups[0].similarity >= 0.7
    assert len(result.repeated_dicts) == 1
    assert result.repeated_dicts[0].sites == 2


def test_exception_roi_detects_single_site_overhead(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "exceptions.py",
        """
class MyError(Exception):
    pass


def boom():
    raise MyError("bad")


def run():
    try:
        boom()
    except MyError:
        return 1
""".strip(),
    )

    result = analyze_file(path)

    assert len(result.exceptions) == 1
    exc = result.exceptions[0]
    assert exc.name == "MyError"
    assert exc.raise_functions == 1
    assert exc.catch_sites == 1


def test_exception_roi_counts_raise_via_local_variable(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "exceptions_alias.py",
        """
class MyError(Exception):
    pass


def boom():
    err = MyError("bad")
    raise err
""".strip(),
    )

    result = analyze_file(path)
    exc = next(e for e in result.exceptions if e.name == "MyError")
    assert exc.raise_functions == 1


def test_snapshot_aggregates_method_and_closure_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "flags.py",
        """
class DataBag:
    def __init__(self, value):
        self.value = value


class Worker:
    def run(self, x):
        return x + 1


def outer(v):
    y = 3

    def inner(z):
        return z + 1

    return inner(v)
""".strip(),
    )

    summary = analyze_snapshot(tmp_path)

    assert summary.data_only_classes == 1
    assert summary.methods_no_self == 1
    assert summary.empty_closures == 1
