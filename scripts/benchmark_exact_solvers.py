from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable


REQUIRED_CASE_FILES = (
    "config.json",
    "task.csv",
    "package.csv",
    "point.csv",
    "distance.csv",
    "time.csv",
    "power.csv",
)

SOLVER_MODULES = {
    "ea": "gurobipy",
    "eac": "coptpy",
    "eao": "pyscipopt",
    "eah": "highspy",
}

SOLVER_LABELS = {
    "ea": "gurobi",
    "eac": "copt",
    "eao": "scip",
    "eah": "highs",
}

CORE_SOLVERS = {"ea", "eac"}
DEFAULT_SOLVERS = ("ea", "eac", "eao", "eah")
DEFAULT_OBJECTIVES = ("maxRevenue", "minTime", "minPower")
DEFAULT_CASES = ("362", "456", "instance1", "instance2", "instance3")
DEFAULT_PROFILES = ("core", "engineering")

NUMERIC_METRICS = (
    "objective_value",
    "model_build_seconds",
    "solve_seconds",
    "postprocess_seconds",
    "algorithm_total_seconds",
    "total_seconds",
    "first_feasible_seconds",
    "incumbent_objective",
    "best_bound",
    "absolute_gap",
    "relative_gap",
    "node_count",
    "model_state_count",
    "model_edge_count",
    "model_variable_count",
    "solver_reported_variable_count",
    "model_constraint_count",
    "model_general_constraint_count",
)

RAW_FIELDS = (
    "case",
    "profile",
    "solver",
    "solver_label",
    "objective",
    "repeat",
    "status",
    "return_code",
    "error_type",
    "error_message",
    "objective_value",
    "termination_status",
    "solver_version",
    "model_build_seconds",
    "solve_seconds",
    "postprocess_seconds",
    "algorithm_total_seconds",
    "total_seconds",
    "first_feasible_seconds",
    "first_feasible_observed",
    "incumbent_objective",
    "best_bound",
    "absolute_gap",
    "relative_gap",
    "node_count",
    "model_state_count",
    "model_edge_count",
    "model_variable_count",
    "solver_reported_variable_count",
    "model_constraint_count",
    "model_general_constraint_count",
    "time_limit_seconds",
    "threads",
    "warm_start_enabled",
    "warm_start_accepted",
    "warm_start_source",
    "fallback_used",
    "native_solution_found",
    "business_validation_passed",
    "profile_compliant",
    "run_dir",
    "stdout_path",
    "stderr_path",
)


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(df, 1.960)


def mean_ci95(values: Iterable[float]) -> tuple[int, float | None, float | None, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    count = len(clean)
    if not clean:
        return 0, None, None, None
    mean = statistics.fmean(clean)
    if count == 1:
        return 1, mean, None, None
    margin = _t_critical_95(count - 1) * statistics.stdev(clean) / math.sqrt(count)
    return count, mean, mean - margin, mean + margin


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _run_key(item: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(item["case"]),
        str(item["profile"]),
        str(item["solver"]),
        str(item["objective"]),
        int(item["repeat"]),
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pending_matrix(
    matrix: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    status_by_key = {_run_key(row): row.get("status") for row in existing_rows}
    retry_statuses = {"missing_result", "environment_unavailable"}
    return [
        item
        for item in matrix
        if _run_key(item) not in status_by_key
        or status_by_key[_run_key(item)] in retry_statuses
    ]


def upsert_result(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    key = _run_key(result)
    for index, row in enumerate(rows):
        if _run_key(row) == key:
            rows[index] = result
            return
    rows.append(result)


def _case_map(repo_root: Path) -> dict[str, Path]:
    return {
        "362": repo_root / "362",
        "456": repo_root / "456",
        "instance1": repo_root / "unified_framework" / "inputs" / "instance1",
        "instance2": repo_root / "unified_framework" / "inputs" / "instance2",
        "instance3": repo_root / "unified_framework" / "inputs" / "instance3",
    }


def _validate_case(case_dir: Path) -> None:
    missing = [name for name in REQUIRED_CASE_FILES if not (case_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Case {case_dir} is missing files: {missing}")


def _solver_available(solver: str) -> bool:
    module_name = SOLVER_MODULES[solver]
    return importlib.util.find_spec(module_name) is not None


def _repeat_count(solver: str, core_repeats: int, supplementary_repeats: int) -> int:
    return core_repeats if solver in CORE_SOLVERS else supplementary_repeats


def build_matrix(args: argparse.Namespace, repo_root: Path) -> list[dict[str, Any]]:
    cases = _case_map(repo_root)
    matrix: list[dict[str, Any]] = []
    for case_name in args.cases:
        if case_name not in cases:
            raise ValueError(f"Unknown case: {case_name}")
        _validate_case(cases[case_name])
        for profile in args.profiles:
            for solver in args.solvers:
                repeats = _repeat_count(solver, args.core_repeats, args.supplementary_repeats)
                for objective in args.objectives:
                    for repeat in range(1, repeats + 1):
                        matrix.append(
                            {
                                "case": case_name,
                                "case_dir": cases[case_name],
                                "profile": profile,
                                "solver": solver,
                                "objective": objective,
                                "repeat": repeat,
                            }
                        )
    return matrix


def _prepare_run_case(
    item: dict[str, Any],
    run_dir: Path,
    time_limit: int,
    engineering_threads: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    source = item["case_dir"]
    for name in REQUIRED_CASE_FILES:
        shutil.copy2(source / name, run_dir / name)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    algorithm = config.setdefault("algorithm", {})
    algorithm.update(
        {
            "name": item["solver"],
            "mode": "normal",
            "obj": item["objective"],
            "timeLimit": time_limit,
            "threads": 1 if item["profile"] == "core" else engineering_threads,
            "useHeuristicStart": item["profile"] == "engineering",
            "allowFallback": item["profile"] == "engineering",
        }
    )
    config["case_id"] = (
        f"{item['case']}-{item['profile']}-{item['solver']}-"
        f"{item['objective']}-r{item['repeat']}"
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_row(item: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    row = {field: None for field in RAW_FIELDS}
    row.update(
        {
            "case": item["case"],
            "profile": item["profile"],
            "solver": item["solver"],
            "solver_label": SOLVER_LABELS[item["solver"]],
            "objective": item["objective"],
            "repeat": item["repeat"],
            "run_dir": str(run_dir),
            "stdout_path": str(run_dir / "stdout.log"),
            "stderr_path": str(run_dir / "stderr.log"),
        }
    )
    return row


def run_item(
    item: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    time_limit: int,
    engineering_threads: int,
) -> dict[str, Any]:
    run_name = (
        f"{item['case']}__{item['profile']}__{item['solver']}__"
        f"{item['objective']}__r{item['repeat']:02d}"
    )
    run_dir = output_root / "runs" / run_name
    row = _base_row(item, run_dir)

    if not _solver_available(item["solver"]):
        row.update(
            {
                "status": "environment_unavailable",
                "error_type": "ModuleNotFoundError",
                "error_message": f"Missing Python module: {SOLVER_MODULES[item['solver']]}",
                "profile_compliant": False,
            }
        )
        return row

    _prepare_run_case(item, run_dir, time_limit, engineering_threads)
    completed = subprocess.run(
        [sys.executable, "-m", "unified_framework", str(run_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    row["return_code"] = completed.returncode

    result_path = run_dir / "output" / "result.json"
    if not result_path.exists():
        row.update(
            {
                "status": "missing_result",
                "error_type": "MissingResult",
                "error_message": "Solver process did not create output/result.json",
                "profile_compliant": False,
            }
        )
        return row

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    error = payload.get("error") or {}
    row.update(
        {
            "status": payload.get("status"),
            "error_type": error.get("type"),
            "error_message": error.get("message"),
            "objective_value": payload.get("objective_value"),
        }
    )
    for field in RAW_FIELDS:
        if field in metrics:
            row[field] = metrics[field]

    configuration_compliant = (
        metrics.get("threads") in {None, 1}
        and not metrics.get("warm_start_enabled", False)
        and not metrics.get("fallback_used", False)
        if item["profile"] == "core"
        else metrics.get("threads") == engineering_threads
        and metrics.get("warm_start_enabled") is True
        and metrics.get("warm_start_source") == "shared"
    )
    row["profile_compliant"] = (
        payload.get("status") == "success"
        and bool(metrics.get("business_validation_passed"))
        and configuration_compliant
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["case"], row["profile"], row["solver"], row["objective"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (case_name, profile, solver, objective), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "case": case_name,
            "profile": profile,
            "solver": solver,
            "solver_label": SOLVER_LABELS[solver],
            "objective": objective,
            "runs": len(group),
            "successes": sum(row.get("status") == "success" for row in group),
            "native_solution_runs": sum(_as_bool(row.get("native_solution_found")) for row in group),
            "fallback_runs": sum(_as_bool(row.get("fallback_used")) for row in group),
            "profile_compliant_runs": sum(_as_bool(row.get("profile_compliant")) for row in group),
        }
        for metric in NUMERIC_METRICS:
            count, mean, low, high = mean_ci95(_as_float(row.get(metric)) for row in group)
            summary[f"{metric}_n"] = count
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        output.append(summary)
    return output


def paired_gurobi_copt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["case"], row["profile"], row["objective"], int(row["repeat"]), row["solver"]): row
        for row in rows
    }
    pairs: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key, gurobi in indexed.items():
        case_name, profile, objective, repeat, solver = key
        if solver != "ea":
            continue
        copt = indexed.get((case_name, profile, objective, repeat, "eac"))
        if copt is not None:
            pairs.setdefault((case_name, profile, objective), []).append((gurobi, copt))

    output: list[dict[str, Any]] = []
    for (case_name, profile, objective), group in sorted(pairs.items()):
        summary: dict[str, Any] = {
            "case": case_name,
            "profile": profile,
            "objective": objective,
            "paired_runs": len(group),
            "difference_direction": "copt_minus_gurobi",
        }
        for metric in ("solve_seconds", "first_feasible_seconds", "objective_value", "relative_gap"):
            differences = []
            for gurobi, copt in group:
                left = _as_float(gurobi.get(metric))
                right = _as_float(copt.get(metric))
                if left is not None and right is not None:
                    differences.append(right - left)
            count, mean, low, high = mean_ci95(differences)
            summary[f"{metric}_difference_n"] = count
            summary[f"{metric}_difference_mean"] = mean
            summary[f"{metric}_difference_ci95_low"] = low
            summary[f"{metric}_difference_ci95_high"] = high
        output.append(summary)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-solver benchmark matrix.")
    parser.add_argument("--output", type=Path, default=Path("output/exact_solver_benchmark"))
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--profiles", nargs="+", choices=DEFAULT_PROFILES, default=list(DEFAULT_PROFILES))
    parser.add_argument("--solvers", nargs="+", choices=DEFAULT_SOLVERS, default=list(DEFAULT_SOLVERS))
    parser.add_argument("--objectives", nargs="+", choices=DEFAULT_OBJECTIVES, default=list(DEFAULT_OBJECTIVES))
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--engineering-threads", type=int, default=4)
    parser.add_argument("--core-repeats", type=int, default=5)
    parser.add_argument("--supplementary-repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output/raw_results.csv and retry incomplete rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output if args.output.is_absolute() else repo_root / args.output
    matrix = build_matrix(args, repo_root)
    availability = {solver: _solver_available(solver) for solver in args.solvers}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned_runs": len(matrix),
                    "availability": availability,
                    "cases": args.cases,
                    "profiles": args.profiles,
                    "solvers": args.solvers,
                    "objectives": args.objectives,
                    "time_limit_seconds": args.time_limit,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_root / "raw_results.csv"
    rows = _read_csv(raw_results_path) if args.resume else []
    pending = pending_matrix(matrix, rows) if args.resume else matrix
    matrix_positions = {_run_key(item): index for index, item in enumerate(matrix, start=1)}
    manifest = {
        "planned_runs": len(matrix),
        "resume": args.resume,
        "resume_existing_rows": len(rows),
        "resume_pending_runs": len(pending),
        "availability": availability,
        "time_limit_seconds": args.time_limit,
        "engineering_threads": args.engineering_threads,
        "core_repeats": args.core_repeats,
        "supplementary_repeats": args.supplementary_repeats,
        "cases": args.cases,
        "profiles": args.profiles,
        "solvers": args.solvers,
        "objectives": args.objectives,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.resume:
        print(
            f"RESUME existing={len(rows)} pending={len(pending)} total={len(matrix)}",
            flush=True,
        )
    for item in pending:
        index = matrix_positions[_run_key(item)]
        print(
            f"[{index}/{len(matrix)}] {item['case']} {item['profile']} "
            f"{item['solver']} {item['objective']} repeat={item['repeat']}",
            flush=True,
        )
        result = run_item(
            item,
            repo_root,
            output_root,
            args.time_limit,
            args.engineering_threads,
        )
        upsert_result(rows, result)
        _write_csv(raw_results_path, rows, RAW_FIELDS)

    summary_rows = summarize(rows)
    summary_fields = list(summary_rows[0]) if summary_rows else []
    _write_csv(output_root / "summary.csv", summary_rows, summary_fields)

    paired_rows = paired_gurobi_copt(rows)
    paired_fields = list(paired_rows[0]) if paired_rows else []
    if paired_rows:
        _write_csv(output_root / "paired_gurobi_copt.csv", paired_rows, paired_fields)
    print(f"Raw results: {output_root / 'raw_results.csv'}")
    print(f"Summary: {output_root / 'summary.csv'}")
    if paired_rows:
        print(f"Paired comparison: {output_root / 'paired_gurobi_copt.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
