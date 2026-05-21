from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import Package, Task, UnifiedCase


@dataclass(frozen=True)
class ScheduleStep:
    day: int
    sequence: int
    kind: str
    task_uid: int | None
    task_id: int | None
    name: str
    location_id: int
    location_name: str
    travel_time: float = 0.0
    service_time: float = 0.0
    travel_power: float = 0.0
    service_power: float = 0.0
    revenue: float = 0.0
    required: bool = False
    note: str | None = None


@dataclass
class SchedulePlan:
    steps: list[ScheduleStep]
    objective_value: float | None
    status: str = "success"
    message: str | None = None
    rows: list[dict[str, Any]] | None = None

    def to_frame(self) -> pd.DataFrame:
        if self.rows is not None:
            return pd.DataFrame(
                self.rows,
                columns=["day", "No", "task_id", "action", "location", "time", "power", "revenue"],
            )
        return pd.DataFrame(
            [
                {
                    "day": step.day,
                    "sequence": step.sequence,
                    "kind": step.kind,
                    "task_uid": step.task_uid,
                    "task_id": step.task_id,
                    "name": step.name,
                    "location_id": step.location_id,
                    "location_name": step.location_name,
                    "travel_time": step.travel_time,
                    "service_time": step.service_time,
                    "travel_power": step.travel_power,
                    "service_power": step.service_power,
                    "revenue": step.revenue,
                    "required": step.required,
                    "note": step.note,
                }
                for step in self.steps
            ]
        )


@dataclass
class LegacyBundle:
    case: UnifiedCase
    info: pd.DataFrame
    task: pd.DataFrame
    package: pd.DataFrame
    point: pd.DataFrame
    distance: pd.DataFrame
    time: pd.DataFrame
    power: pd.DataFrame


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str | None = None


def matrix_value(case: UnifiedCase, matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
    value = case.matrix_value(matrix, from_id, to_id)
    return value if isfinite(value) else float("inf")


def package_day(package: Package) -> int | None:
    if len(package.tag) >= 2 and package.tag[0].upper() == "D" and package.tag[1].isdigit():
        day = int(package.tag[1])
        if day in {1, 2, 3}:
            return day
    return None


def package_costs(case: UnifiedCase) -> dict[int, tuple[float, float]]:
    costs = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0]}
    for package in case.packages:
        day = package_day(package)
        if day is None:
            continue
        costs[day][0] += package.time
        costs[day][1] += package.power
    return {day: (cost[0], cost[1]) for day, cost in costs.items()}


def _day_value(task: Task) -> Any:
    if task.day_options is None:
        return np.nan
    if len(task.day_options) == 1:
        return task.day_options[0]
    return ",".join(str(day) for day in task.day_options)


def _task_location_value(case: UnifiedCase, task: Task) -> Any:
    non_depot = sorted(point_id for point_id in case.point_ids if point_id != case.depot_id)
    if sorted(task.location_ids) == non_depot:
        return np.nan
    names = [case.point_name(point_id) for point_id in task.location_ids]
    return ",".join(names)


def build_legacy_bundle(case: UnifiedCase) -> LegacyBundle:
    """Build Excel-like in-memory tables consumed by the migrated legacy algorithms."""
    info = pd.DataFrame(
        {
            "max-distance": [case.config.max_distance],
            "total-time/day": [";".join(str(value) for value in case.config.total_time_day)],
            "total-power/day": [";".join(str(value) for value in case.config.total_power_day)],
            "min-continuous": [case.config.min_continuous],
            "12-gap": [case.config.gap_12],
            "23-gap": [case.config.gap_23],
        }
    )
    task_rows = []
    for task in case.tasks:
        task_rows.append(
            {
                "No": task.uid,
                "name": task.name,
                "revenue": task.revenue,
                "location": _task_location_value(case, task),
                "day": _day_value(task),
                "time": task.time,
                "power": task.power,
                "required": task.required,
                "continuous": task.continuous,
                "remote": task.remote,
                "exceptO": task.except_o,
                "tag": np.nan if task.tag is None else task.tag,
                "task_id": task.task_id,
            }
        )
    task_df = pd.DataFrame(task_rows)

    package_df = pd.DataFrame(
        [
            {
                "task_id": package.task_id,
                "name": package.name,
                "time": package.time,
                "power": package.power,
                "tag": package.tag,
            }
            for package in case.packages
        ]
    )

    point_df = pd.DataFrame(
        [
            {
                "name": point.name,
                "point_id": point.id,
                "X": point.x,
                "Y": point.y,
                "备注": point.comment,
            }
            for point in case.points
        ]
    )

    id_to_name = case.point_id_to_name
    distance = case.distance.rename(index=id_to_name, columns=id_to_name).copy()
    time = case.time.rename(index=id_to_name, columns=id_to_name).copy()
    power = case.power.rename(index=id_to_name, columns=id_to_name).copy()
    return LegacyBundle(case, info, task_df, package_df, point_df, distance, time, power)


def write_legacy_workbook_inputs(bundle: LegacyBundle, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle.info.to_excel(work_dir / "info.xlsx", index=False)
    bundle.task.to_excel(work_dir / "task.xlsx", index=False)
    bundle.package[["name", "time", "power", "tag"]].to_excel(
        work_dir / "package.xlsx", index=False
    )
    bundle.point[["name", "X", "Y", "备注"]].to_excel(work_dir / "point.xlsx", index=False)
    bundle.distance.to_excel(work_dir / "distance.xlsx", index=True)
    bundle.time.to_excel(work_dir / "time.xlsx", index=True)
    bundle.power.to_excel(work_dir / "power.xlsx", index=True)


def _format_location(case: UnifiedCase, value: Any) -> Any:
    if pd.isna(value):
        return ""
    text = str(value)
    for origin in ["探测起点1", "探测起点2", "探测起点3", "探测起点4", "探测起点"]:
        text = text.replace(origin, "0")
    for name, point_id in sorted(case.point_name_to_id.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(name, str(point_id))
    return text


def _task_id_lookup(case: UnifiedCase) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for task in case.tasks:
        lookup.setdefault(task.name, task.task_id)
    for package in case.packages:
        lookup.setdefault(package.name, package.task_id)
    return lookup


def _revenue_lookup(case: UnifiedCase) -> dict[str, float]:
    return {task.name: task.revenue for task in case.tasks}


def legacy_schedule_to_rows(case: UnifiedCase, schedule_df: pd.DataFrame) -> list[dict[str, Any]]:
    task_ids = _task_id_lookup(case)
    revenues = _revenue_lookup(case)
    rows: list[dict[str, Any]] = []
    day = 1
    no = 1
    for _, row in schedule_df.iterrows():
        action = row.get("action")
        if pd.isna(action):
            continue
        action_text = str(action)
        if action_text == "Begin of Day1":
            day, no = 1, 1
            continue
        if action_text == "Break between Day1 and Day2":
            day, no = 2, 1
            continue
        if action_text == "Break between Day2 and Day3":
            day, no = 3, 1
            continue
        if action_text == "End of Day3" or action_text == "xxx":
            continue

        task_id = task_ids.get(action_text, "")
        rows.append(
            {
                "day": day,
                "No": no,
                "task_id": task_id,
                "action": action_text,
                "location": _format_location(case, row.get("location", "")),
                "time": row.get("time", ""),
                "power": row.get("power", ""),
                "revenue": revenues.get(action_text, 0),
            }
        )
        no += 1
    return rows


def package_steps(case: UnifiedCase, day: int, start_sequence: int = 1) -> list[ScheduleStep]:
    steps: list[ScheduleStep] = []
    sequence = start_sequence
    for package in case.packages:
        if package_day(package) != day:
            continue
        steps.append(
            ScheduleStep(
                day=day,
                sequence=sequence,
                kind="package",
                task_uid=None,
                task_id=package.task_id,
                name=package.name,
                location_id=case.depot_id,
                location_name=case.point_name(case.depot_id),
                service_time=package.time,
                service_power=package.power,
                note=package.tag,
            )
        )
        sequence += 1
    return steps


def task_allowed_on_day(task: Task, day: int) -> bool:
    if task.day_options is not None and day not in task.day_options:
        return False
    if task.tag in {"12s", "12e"}:
        return day in {1, 2}
    if task.tag in {"23s", "23e"}:
        return day in {2, 3}
    return True


def feasible_locations(case: UnifiedCase, task: Task) -> list[int]:
    locations: list[int] = []
    for location_id in task.location_ids:
        if location_id not in case.point_id_to_name:
            continue
        if not isfinite(matrix_value(case, case.time, case.depot_id, location_id)):
            continue
        if not isfinite(matrix_value(case, case.power, case.depot_id, location_id)):
            continue
        locations.append(location_id)
    return locations


def check_remote_requirement(case: UnifiedCase) -> ValidationResult:
    remote_tasks = [task for task in case.tasks if task.remote]
    if not remote_tasks:
        return ValidationResult(True)

    for task in remote_tasks:
        for location_id in task.location_ids:
            if location_id not in case.point_id_to_name:
                continue
            distance_from_depot = matrix_value(case, case.distance, case.depot_id, location_id)
            if distance_from_depot >= case.config.max_distance:
                return ValidationResult(True)

    return ValidationResult(
        False,
        "No candidate point in remote tasks meets max-distance requirement.",
    )


def selected_task_uids(plan: SchedulePlan) -> set[int]:
    return {
        step.task_uid
        for step in plan.steps
        if step.kind == "task" and step.task_uid is not None
    }


def objective_revenue(case: UnifiedCase, selected_uids: set[int]) -> float:
    return sum(task.revenue for task in case.tasks if task.uid in selected_uids)


def validate_plan(case: UnifiedCase, plan: SchedulePlan) -> ValidationResult:
    if plan.status != "success":
        return ValidationResult(False, plan.message)

    selected = selected_task_uids(plan)
    duplicate_count = sum(1 for step in plan.steps if step.kind == "task")
    if duplicate_count != len(selected):
        return ValidationResult(False, "A task appears more than once in the schedule.")

    missing_required = [
        task.name for task in case.tasks if task.required and task.uid not in selected
    ]
    if missing_required:
        return ValidationResult(False, f"Missing required tasks: {missing_required}")

    continuous_count = sum(
        1 for task in case.tasks if task.continuous and task.uid in selected
    )
    if continuous_count < case.config.min_continuous:
        return ValidationResult(
            False,
            f"Continuous task count {continuous_count} is below required {case.config.min_continuous}.",
        )

    task_by_uid = {task.uid: task for task in case.tasks}
    for day in {1, 2, 3}:
        day_steps = [step for step in plan.steps if step.day == day]
        total_time = sum(step.travel_time + step.service_time for step in day_steps)
        total_power = sum(step.travel_power + step.service_power for step in day_steps)
        if total_time > case.config.total_time_day[day - 1] + 1e-6:
            return ValidationResult(False, f"Day {day} exceeds time budget: {total_time}")
        if total_power > case.config.total_power_day[day - 1] + 1e-6:
            return ValidationResult(False, f"Day {day} exceeds power budget: {total_power}")

    for step in plan.steps:
        if step.kind != "task" or step.task_uid is None:
            continue
        task = task_by_uid[step.task_uid]
        if not task_allowed_on_day(task, step.day):
            return ValidationResult(False, f"Task {task.name} is not allowed on day {step.day}.")
        if step.location_id not in task.location_ids:
            return ValidationResult(False, f"Task {task.name} uses an invalid location.")
    return check_remote_requirement(case)
