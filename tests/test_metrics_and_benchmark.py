from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_exact_solvers import (
    mean_ci95,
    paired_gurobi_copt,
    pending_matrix,
    summarize,
    upsert_result,
)
from unified_framework.io import write_error, write_outputs
from unified_framework.schedule import SchedulePlan


def test_write_outputs_persists_metrics_and_progress(tmp_path: Path) -> None:
    case = SimpleNamespace(
        case_dir=tmp_path,
        output_dir=tmp_path / "output",
        config=SimpleNamespace(case_id="metrics-case"),
    )
    plan = SchedulePlan(
        steps=[],
        rows=[
            {
                "day": 1,
                "No": 1,
                "task_id": 10,
                "action": "demo",
                "location": 2,
                "time": 3.0,
                "power": 4.0,
                "revenue": 5.0,
            }
        ],
        objective_value=5.0,
        metrics={"solver_name": "test", "solve_seconds": 1.25},
        progress=[
            {
                "event": "final",
                "elapsed_seconds": 1.25,
                "incumbent_objective": 5.0,
                "best_bound": 5.0,
                "absolute_gap": 0.0,
                "relative_gap": 0.0,
                "solution_source": "solver",
            }
        ],
    )

    result = write_outputs(case, plan)

    assert result.metrics is not None
    assert result.metrics["progress_path"] == "output/progress.jsonl"
    payload = json.loads((tmp_path / "output" / "result.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["solve_seconds"] == 1.25
    events = [
        json.loads(line)
        for line in (tmp_path / "output" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["solution_source"] == "solver"


def test_write_error_keeps_metrics(tmp_path: Path) -> None:
    result = write_error(
        tmp_path,
        "failed-case",
        RuntimeError("license unavailable"),
        metrics={"solver_name": "copt", "total_seconds": 0.2},
    )
    assert result.status == "solver_error"
    assert result.metrics == {"solver_name": "copt", "total_seconds": 0.2}


def test_mean_ci95_uses_student_interval() -> None:
    count, mean, low, high = mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert count == 5
    assert mean == 3.0
    assert low == pytest.approx(1.0368, rel=1e-3)
    assert high == pytest.approx(4.9632, rel=1e-3)


def _result_row(solver: str, repeat: int, solve_seconds: float) -> dict[str, object]:
    return {
        "case": "instance1",
        "profile": "core",
        "solver": solver,
        "objective": "maxRevenue",
        "repeat": repeat,
        "status": "success",
        "objective_value": 10.0,
        "solve_seconds": solve_seconds,
        "native_solution_found": True,
        "fallback_used": False,
        "profile_compliant": True,
    }


def test_summary_and_paired_difference_keep_all_repeats() -> None:
    rows = [
        _result_row("ea", 1, 1.0),
        _result_row("ea", 2, 2.0),
        _result_row("eac", 1, 1.5),
        _result_row("eac", 2, 3.0),
    ]
    summaries = summarize(rows)
    assert len(summaries) == 2
    gurobi = next(row for row in summaries if row["solver"] == "ea")
    assert gurobi["solve_seconds_n"] == 2
    assert gurobi["solve_seconds_mean"] == 1.5

    paired = paired_gurobi_copt(rows)
    assert len(paired) == 1
    assert paired[0]["solve_seconds_difference_n"] == 2
    assert paired[0]["solve_seconds_difference_mean"] == 0.75


def test_resume_skips_completed_and_retries_missing_result() -> None:
    matrix = [
        {
            "case": "362",
            "profile": "core",
            "solver": "eac",
            "objective": "maxRevenue",
            "repeat": repeat,
        }
        for repeat in (1, 2, 3)
    ]
    rows = [
        {**matrix[0], "status": "success"},
        {**matrix[1], "status": "missing_result"},
    ]

    pending = pending_matrix(matrix, rows)

    assert [item["repeat"] for item in pending] == [2, 3]
    replacement = {**matrix[1], "status": "success"}
    upsert_result(rows, replacement)
    assert len(rows) == 2
    assert rows[1]["status"] == "success"


def test_summary_reads_boolean_strings_from_resumed_csv() -> None:
    rows = [
        {
            **_result_row("ea", 1, 1.0),
            "native_solution_found": "False",
            "fallback_used": "False",
            "profile_compliant": "False",
        }
    ]

    summary = summarize(rows)[0]

    assert summary["native_solution_runs"] == 0
    assert summary["fallback_runs"] == 0
    assert summary["profile_compliant_runs"] == 0
