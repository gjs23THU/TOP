from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import AlgorithmConfig, CaseConfig, Package, Point, SolveResult, Task, UnifiedCase
from .schedule import SchedulePlan


REQUIRED_FILES = (
    "config.json",
    "task.csv",
    "package.csv",
    "point.csv",
    "distance.csv",
    "time.csv",
    "power.csv",
)


def _to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return None
    if not isinstance(value, (list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
    return value


def _parse_bool(value: Any) -> bool:
    value = _to_none(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_num_list(value: Any, field_name: str) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    value = _to_none(value)
    if value is None:
        raise ValueError(f"Missing list value for {field_name}")
    text = str(value).strip()
    separator = ";" if ";" in text else ","
    if separator in text:
        return [float(item.strip()) for item in text.split(separator) if item.strip()]
    return [float(text)]


def _parse_int_tokens(value: Any) -> list[int] | None:
    value = _to_none(value)
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return [int(float(token)) for token in tokens]


def _read_csv(case_dir: Path, file_name: str) -> pd.DataFrame:
    path = case_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _load_config(case_dir: Path) -> CaseConfig:
    config_path = case_dir / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    algorithm_raw = raw.get("algorithm", {})
    algorithm = AlgorithmConfig(
        name=str(algorithm_raw.get("name", "")).strip().lower(),
        mode=str(algorithm_raw.get("mode", "normal")).strip().lower(),
        obj=str(algorithm_raw.get("obj", "maxRevenue")).strip(),
        time_limit=None
        if _to_none(algorithm_raw.get("timeLimit")) is None
        else int(algorithm_raw["timeLimit"]),
        decimal=int(algorithm_raw.get("decimal", 5)),
        random_seed=None
        if _to_none(algorithm_raw.get("random_seed")) is None
        else int(algorithm_raw["random_seed"]),
    )
    if algorithm.name not in {"ea", "eao", "ha", "ga", "sa", "pso", "ai"}:
        raise ValueError(f"Unsupported algorithm.name: {algorithm.name}")
    if algorithm.mode not in {"normal", "revisional", "back"}:
        raise ValueError(f"Unsupported algorithm.mode: {algorithm.mode}")
    if algorithm.obj not in {"maxRevenue", "minTime", "minPower"}:
        raise ValueError(f"Unsupported algorithm.obj: {algorithm.obj}")
    config = CaseConfig(
        case_id=str(raw.get("case_id", case_dir.name)),
        algorithm=algorithm,
        max_distance=float(raw["max-distance"]),
        total_time_day=_parse_num_list(raw["total-time/day"], "total-time/day"),
        total_power_day=_parse_num_list(raw["total-power/day"], "total-power/day"),
        min_continuous=int(raw["min-continuous"]),
        gap_12=float(raw["12-gap"]),
        gap_23=float(raw["23-gap"]),
        raw=raw,
    )
    if len(config.total_time_day) != 3 or len(config.total_power_day) != 3:
        raise ValueError("Only 3 planning days are supported.")
    return config


def _label_to_point_id(value: Any, name_to_id: dict[str, int]) -> int:
    text = str(value).strip()
    if text in name_to_id:
        return name_to_id[text]
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unknown point label: {value}") from exc


def _load_points(point_df: pd.DataFrame) -> tuple[list[Point], dict[int, str], dict[str, int]]:
    for column in ["point_name", "point_id", "X", "Y"]:
        if column not in point_df.columns:
            raise ValueError(f"point.csv missing required column: {column}")

    points: list[Point] = []
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for _, row in point_df.iterrows():
        point_id = int(row["point_id"])
        point_name = str(row["point_name"])
        if point_id in id_to_name:
            raise ValueError(f"Duplicate point_id in point.csv: {point_id}")
        if point_name in name_to_id:
            raise ValueError(f"Duplicate point_name in point.csv: {point_name}")
        id_to_name[point_id] = point_name
        name_to_id[point_name] = point_id
        points.append(
            Point(
                id=point_id,
                name=point_name,
                x=float(row["X"]),
                y=float(row["Y"]),
                comment=None if _to_none(row.get("comment")) is None else str(row.get("comment")),
            )
        )
    if 0 not in id_to_name:
        raise ValueError("point.csv must contain depot point_id=0")
    return points, id_to_name, name_to_id


def _load_tasks(task_df: pd.DataFrame, point_ids: set[int]) -> list[Task]:
    required_columns = [
        "task_id",
        "task_name",
        "revenue",
        "location",
        "day",
        "time",
        "power",
        "required",
        "continuous",
        "remote",
        "exceptO",
        "tag",
    ]
    for column in required_columns:
        if column not in task_df.columns:
            raise ValueError(f"task.csv missing required column: {column}")

    non_depot_ids = sorted(point_id for point_id in point_ids if point_id != 0)
    tasks: list[Task] = []
    for uid, (_, row) in enumerate(task_df.iterrows()):
        location_ids = _parse_int_tokens(row["location"])
        if location_ids is None:
            location_ids = non_depot_ids
        invalid = sorted(set(location_ids) - point_ids)
        if invalid:
            raise ValueError(f"Task {row['task_name']} references unknown point ids: {invalid}")
        day_options = _parse_int_tokens(row["day"])
        if day_options is not None:
            invalid_days = [day for day in day_options if day not in {1, 2, 3}]
            if invalid_days:
                raise ValueError(f"Task {row['task_name']} has invalid day values: {invalid_days}")
        tasks.append(
            Task(
                uid=uid,
                task_id=int(row["task_id"]),
                name=str(row["task_name"]),
                revenue=float(row["revenue"]),
                location_ids=location_ids,
                day_options=day_options,
                time=float(row["time"]),
                power=float(row["power"]),
                required=_parse_bool(row["required"]),
                continuous=_parse_bool(row["continuous"]),
                remote=_parse_bool(row["remote"]),
                except_o=_parse_bool(row["exceptO"]),
                tag=None if _to_none(row["tag"]) is None else str(row["tag"]).strip(),
            )
        )
    return tasks


def _load_packages(package_df: pd.DataFrame) -> list[Package]:
    for column in ["task_id", "task_name", "time", "power", "tag"]:
        if column not in package_df.columns:
            raise ValueError(f"package.csv missing required column: {column}")
    return [
        Package(
            task_id=int(row["task_id"]),
            name=str(row["task_name"]),
            time=float(row["time"]),
            power=float(row["power"]),
            tag=str(row["tag"]).strip(),
        )
        for _, row in package_df.iterrows()
    ]


def _parse_matrix(
    matrix_df: pd.DataFrame,
    point_ids: list[int],
    point_name_to_id: dict[str, int],
    matrix_name: str,
) -> pd.DataFrame:
    if matrix_df.shape[1] < 2:
        raise ValueError(f"{matrix_name} has invalid shape")

    raw = matrix_df.copy()
    index_column = raw.columns[0]
    raw[index_column] = raw[index_column].map(lambda value: _label_to_point_id(value, point_name_to_id))
    rename_map = {
        column: _label_to_point_id(column, point_name_to_id)
        for column in raw.columns[1:]
        if not str(column).startswith("Unnamed:")
    }
    raw = raw.rename(columns=rename_map).set_index(index_column)
    raw = raw[[column for column in raw.columns if column in set(point_ids)]]
    raw.index = raw.index.astype(int)
    raw.columns = [int(column) for column in raw.columns]

    missing_rows = sorted(set(point_ids) - set(raw.index))
    missing_columns = sorted(set(point_ids) - set(raw.columns))
    if missing_rows or missing_columns:
        raise ValueError(
            f"{matrix_name} missing rows {missing_rows} or columns {missing_columns}"
        )

    matrix = raw.loc[point_ids, point_ids].copy()
    for column in matrix.columns:
        matrix[column] = matrix[column].map(
            lambda value: float("inf")
            if isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity"}
            else float(value)
        )
    for point_id in point_ids:
        matrix.loc[point_id, point_id] = 0.0
    return matrix


def load_case(case_dir: str | Path) -> UnifiedCase:
    case_path = Path(case_dir).resolve()
    for file_name in REQUIRED_FILES:
        if not (case_path / file_name).exists():
            raise FileNotFoundError(f"Missing required input: {case_path / file_name}")

    config = _load_config(case_path)
    point_df = _read_csv(case_path, "point.csv")
    task_df = _read_csv(case_path, "task.csv")
    package_df = _read_csv(case_path, "package.csv")
    distance_df = _read_csv(case_path, "distance.csv")
    time_df = _read_csv(case_path, "time.csv")
    power_df = _read_csv(case_path, "power.csv")

    points, id_to_name, name_to_id = _load_points(point_df)
    point_ids = [point.id for point in points]
    point_id_set = set(point_ids)
    tasks = _load_tasks(task_df, point_id_set)
    packages = _load_packages(package_df)
    distance = _parse_matrix(distance_df, point_ids, name_to_id, "distance.csv")
    time = _parse_matrix(time_df, point_ids, name_to_id, "time.csv")
    power = _parse_matrix(power_df, point_ids, name_to_id, "power.csv")

    output_dir = case_path / "output"
    return UnifiedCase(
        config=config,
        case_dir=case_path,
        output_dir=output_dir,
        points=points,
        tasks=tasks,
        packages=packages,
        distance=distance,
        time=time,
        power=power,
        point_id_to_name=id_to_name,
        point_name_to_id=name_to_id,
    )


def status_from_exception(exc: Exception) -> str:
    name = exc.__class__.__name__
    if name in {"FileNotFoundError", "ValueError", "NotImplementedError"}:
        return "input_error"
    if name == "PermissionError":
        return "io_error"
    if name == "MaxDistanceError":
        return "max_distance_error"
    if name == "MIPError":
        return "infeasible"
    if name == "SubtourError":
        return "subtour_error"
    return "solver_error"


def write_result(case_dir: Path, result: SolveResult) -> SolveResult:
    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.status != "success":
        schedule_path = output_dir / "schedule.csv"
        if schedule_path.exists():
            schedule_path.unlink()
    result_path = output_dir / "result.json"
    payload = {
        "case_id": result.case_id,
        "status": result.status,
        "objective_value": result.objective_value,
        "schedule_path": result.schedule_path,
        "error": result.error,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_error(case_dir: str | Path, case_id: str, exc: Exception) -> SolveResult:
    result = SolveResult(
        case_id=case_id,
        status=status_from_exception(exc),
        objective_value=None,
        schedule_path=None,
        error={"type": exc.__class__.__name__, "message": str(exc)},
    )
    return write_result(Path(case_dir).resolve(), result)


def write_outputs(case: UnifiedCase, plan: SchedulePlan) -> SolveResult:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = case.output_dir / "schedule.csv"
    result_path = case.output_dir / "result.json"

    schedule_ref: str | None = None
    if plan.status == "success" and (plan.rows is not None or plan.steps):
        plan.to_frame().to_csv(schedule_path, index=False)
        schedule_ref = "output/schedule.csv"
    elif schedule_path.exists():
        schedule_path.unlink()

    result = SolveResult(
        case_id=case.config.case_id,
        status=plan.status,
        objective_value=plan.objective_value if plan.status == "success" else None,
        schedule_path=schedule_ref,
        error=None if plan.status == "success" else {"type": plan.status, "message": plan.message or ""},
    )
    return write_result(case.case_dir, result)
