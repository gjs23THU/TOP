from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def exact_big_m(
    resource_bound: float,
    extra_cost: float = 0.0,
    task_cost: float = 0.0,
    eps: float = 0.0,
) -> float:
    return max(
        1.0 + float(eps),
        float(resource_bound)
        + max(0.0, float(extra_cost))
        + max(0.0, -float(task_cost))
        + float(eps),
    )


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    mode: str = "normal"
    obj: str = "maxRevenue"
    time_limit: int | None = None
    decimal: int = 5
    random_seed: int | None = None


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    algorithm: AlgorithmConfig
    max_distance: float
    total_time_day: list[float]
    total_power_day: list[float]
    min_continuous: int
    gap_12: float
    gap_23: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class Point:
    id: int
    name: str
    x: float
    y: float
    comment: str | None = None


@dataclass(frozen=True)
class Task:
    uid: int
    task_id: int
    name: str
    revenue: float
    location_ids: list[int]
    day_options: list[int] | None
    time: float
    power: float
    required: bool
    continuous: bool
    remote: bool
    except_o: bool
    tag: str | None


@dataclass(frozen=True)
class Package:
    task_id: int
    name: str
    time: float
    power: float
    tag: str


@dataclass
class UnifiedCase:
    config: CaseConfig
    case_dir: Path
    output_dir: Path
    points: list[Point]
    tasks: list[Task]
    packages: list[Package]
    distance: pd.DataFrame
    time: pd.DataFrame
    power: pd.DataFrame
    point_id_to_name: dict[int, str]
    point_name_to_id: dict[str, int]

    @property
    def depot_id(self) -> int:
        return 0

    @property
    def point_ids(self) -> list[int]:
        return [point.id for point in self.points]

    def point_name(self, point_id: int) -> str:
        return self.point_id_to_name[point_id]

    def matrix_value(self, matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
        return float(matrix.loc[from_id, to_id])


class MaxDistanceError(Exception):
    """Raised when no remote task candidate satisfies max_distance."""


class MIPError(Exception):
    """Raised when a solver cannot produce a feasible plan."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class SubtourError(Exception):
    """Raised when subtour elimination fails."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class SolveResult:
    case_id: str
    status: str
    objective_value: float | None
    schedule_path: str | None
    error: dict[str, str] | None = None
