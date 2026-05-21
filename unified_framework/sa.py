# -*- coding: utf-8 -*-
"""
Simulated Annealing solver for multi-day task & path co-planning.

Unified entry point ``solve(case, mode)`` consumes in-memory DataFrames built
from the normalized CSV case. The ``task_optimize`` class still accepts legacy
workbook path parameters for compatibility, but DataFrame inputs bypass Excel
I/O and return the generated schedule as a DataFrame.

Provides:
  - task_optimize class with identical __init__ signature and run() method
  - CONST class with MAX_REVENUE / MIN_TIME / MIN_POWER
  - solve(case, mode) wrapper for the UnifiedCase integration layer

Date: 2026-05-05
"""

import copy
import math
import os
import random
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from typing import List, Tuple, Optional, Dict, Set
from openpyxl.styles import PatternFill

from .models import MIPError, SubtourError


# ============================================================================
#  Constants — identical to ea.py
# ============================================================================

class CONST(object):
    MAX_REVENUE = "maxRevenue"
    MIN_TIME = "minTime"
    MIN_POWER = "minPower"


# ============================================================================
#  Exceptions — identical to ea.py
# ============================================================================

class MaxDistanceError(Exception):
    def __init__(self, message, df, path):
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer)
        self.message = message

    def __str__(self):
        return self.message


# ============================================================================
#  Internal data structures (not exposed in the public API)
# ============================================================================

@dataclass
class _Task:
    idx: int
    name: str
    revenue: float
    locations: List[str]
    day: Optional[str]        # None / "1" / "2" / "3" / "1,2" / "2,3"
    time: float
    power: float
    required: bool
    continuous: bool
    remote: bool
    except_origin: bool
    tag: Optional[str]        # None / "12s" / "12e" / "23s" / "23e"


@dataclass
class _ScheduleItem:
    task_idx: int
    location: str


@dataclass
class _Solution:
    days: List[List[_ScheduleItem]] = field(
        default_factory=lambda: [[], [], []]
    )
    total_revenue: float = 0.0
    total_time: float = 0.0
    total_power: float = 0.0
    penalty: float = 0.0
    fitness: float = -math.inf


# ============================================================================
#  Main class — mirrors ea.py's task_optimize API exactly
# ============================================================================

class task_optimize(object):
    """
    SA-based task & path co-planning optimizer.

    Constructor signature and run() method are **identical** to ea.py's
    task_optimize so that calling code needs zero changes.
    """

    def __init__(
        self,
        obj=CONST.MAX_REVENUE,
        timeLimit=np.inf,
        solNum=np.inf,
        autoSave=True,
        MIPFocus=None,
        heuristics=None,
        decimal=5,
        lpPath=None,
        solPath=None,
        infoPath="info.xlsx",
        taskPath="task.xlsx",
        packPath="package.xlsx",
        pointPath="point.xlsx",
        distancePath="distance.xlsx",
        timePath="time.xlsx",
        powerPath="power.xlsx",
        outputPath="schedule.xlsx",
        dataFrames=None,
        writeOutput=True,
        # ----- SA-specific parameters (extra, non-breaking) -----
        T0=1000.0,
        T_min=0.01,
        alpha=0.995,
        iter_per_temp=200,
        penalty_weight=1000.0,
        reheat_every=500,
        reheat_factor=2.0,
        seed=42,
    ):
        # ---- store paths (same names as ea.py) ----
        self.__objective = obj
        self.__Obj = [CONST.MAX_REVENUE, CONST.MIN_TIME, CONST.MIN_POWER]
        self.__autosavestate = autoSave
        self.__decimal = decimal
        self.__lppath, self.__solpath = lpPath, solPath
        self.__infoPath = infoPath
        self.__taskPath = taskPath
        self.__pakpath = packPath
        self.__pointpath = pointPath
        self.__distancepath = distancePath
        self.__timepath = timePath
        self.__powerpath = powerPath
        self.__outputpath = outputPath
        self.__dataframes = dataFrames or {}
        self.__write_output = writeOutput
        self.__solcount = 1

        # timeLimit → convert to SA iteration budget (approximate)
        self.__timeLimit = timeLimit

        # ---- SA hyper-parameters ----
        self.__T0 = T0
        self.__T_min = T_min
        self.__alpha = alpha
        self.__iter_per_temp = iter_per_temp
        self.__penalty_weight = penalty_weight
        self.__reheat_every = reheat_every
        self.__reheat_factor = reheat_factor
        self.__seed = seed

    # ================================================================
    #  I/O validation — identical to ea.py
    # ================================================================

    def test_IO(self):
        if self.__dataframes:
            missing = [
                key for key in
                ["info", "task", "package", "point", "distance", "time", "power"]
                if key not in self.__dataframes
            ]
            if missing:
                raise ValueError(f"Missing DataFrame inputs: {missing}")
            return None

        FileNotFound = []
        if os.path.exists("autosave"):
            if len(os.listdir("autosave")) > 0:
                for f in os.listdir("autosave"):
                    os.remove(os.path.join("autosave", f))
        else:
            os.makedirs("autosave")
        for p in [
            self.__infoPath, self.__taskPath, self.__pakpath,
            self.__pointpath, self.__distancepath, self.__timepath,
            self.__powerpath,
        ]:
            if not os.path.exists(p):
                FileNotFound.append(p)
        if FileNotFound:
            raise FileNotFoundError(
                "No such file: {}".format(" ".join(FileNotFound))
            )
        notPermitted = False
        if self.__write_output and os.path.exists(self.__outputpath):
            try:
                pd.read_excel(self.__outputpath)
            except PermissionError:
                notPermitted = True
        if notPermitted:
            raise PermissionError(f"Permission denied: {self.__outputpath}")

    # ================================================================
    #  Data readers — same file formats as ea.py
    # ================================================================

    def read_info(self):
        info = self.__dataframes.get("info")
        info = info.copy() if info is not None else pd.read_excel(self.__infoPath)
        self.__MDistance = info["max-distance"][0]
        self.__TTime = list(map(float, info["total-time/day"][0].split(";")))
        self.__TPower = list(map(float, info["total-power/day"][0].split(";")))
        self.__Mincontinuous = info["min-continuous"][0]
        self.__12gap = info["12-gap"][0]
        self.__23gap = info["23-gap"][0]

    def __read_matrix(self, path, key):
        matrix = self.__dataframes.get(key)
        if matrix is not None:
            matrix = matrix.copy()
        else:
            matrix = pd.read_excel(path)
            matrix.set_index(matrix.columns[0], inplace=True)
            matrix.index.rename(None, inplace=True)
        return matrix

    def read_task(self):
        task = self.__dataframes.get("task")
        self.__task = task.copy() if task is not None else pd.read_excel(self.__taskPath)
        self.__dmatrix = self.__read_matrix(self.__distancepath, "distance")
        self.__dmatrix.replace(np.inf, self.__MDistance * 3, inplace=True)
        np.fill_diagonal(self.__dmatrix.values, 0)
        self.__tmatrix = self.__read_matrix(self.__timepath, "time")
        self.__tmatrix.replace(np.inf, max(self.__TTime), inplace=True)
        np.fill_diagonal(self.__tmatrix.values, 0)
        self.__pmatrix = self.__read_matrix(self.__powerpath, "power")
        self.__pmatrix.replace(np.inf, max(self.__TPower), inplace=True)
        np.fill_diagonal(self.__pmatrix.values, 0)

    def read_package(self):
        package = self.__dataframes.get("package")
        self.__package = package.copy() if package is not None else pd.read_excel(self.__pakpath)
        tags = [
            "D1ss", "D1se", "D1es", "D1ee",
            "D2ss", "D2se", "D2es", "D2ee",
            "D3ss", "D3se", "D3es", "D3ee",
        ]
        time_pkg = [
            sum(self.__package[self.__package["tag"] == t]["time"])
            for t in tags
        ]
        power_pkg = [
            sum(self.__package[self.__package["tag"] == t]["power"])
            for t in tags
        ]
        self.__PKGTime = [sum(time_pkg[d * 4 : d * 4 + 4]) for d in range(3)]
        self.__PKGPower = [sum(power_pkg[d * 4 : d * 4 + 4]) for d in range(3)]
        for d in range(3):
            self.__TTime[d] -= self.__PKGTime[d]
            self.__TPower[d] -= self.__PKGPower[d]

    def read_point(self):
        point = self.__dataframes.get("point")
        self.__pointdf = point.copy() if point is not None else pd.read_excel(self.__pointpath)
        self.__pointdf.set_index(self.__pointdf.columns[0], inplace=True)
        self.__pointdf.index.rename(None, inplace=True)

    def __add_void_matrix(self, matrix):
        """Mirror ea.py: replace the physical origin by four virtual day-boundary origins."""
        voidrow = [matrix.loc["探测起点", :].values.tolist() for _ in range(4)]
        voidrowdf = pd.DataFrame(
            voidrow,
            index=["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
            columns=matrix.columns,
        )
        matrix = pd.concat([voidrowdf, matrix])
        matrix.drop(["探测起点"], inplace=True)
        voidcol = np.array(
            [matrix.loc[:, "探测起点"].values.tolist() for _ in range(4)]
        ).T
        voidcoldf = pd.DataFrame(
            voidcol,
            columns=["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
            index=matrix.index,
        )
        matrix = pd.concat([voidcoldf, matrix], axis=1)
        matrix.drop(["探测起点"], axis=1, inplace=True)
        for i, j in product(
            ["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
            ["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
        ):
            if i != j:
                matrix.loc[i, j] = 0
        return matrix

    def gen_void_point(self):
        """Mirror ea.py preprocessing so SA uses the same virtual origin model."""
        void_names = ["探测起点1", "探测起点2", "探测起点3", "探测起点4"]
        voidpoint = pd.DataFrame(
            {
                "X": [self.__pointdf.loc["探测起点", "X"]] * 4,
                "Y": [self.__pointdf.loc["探测起点", "Y"]] * 4,
                "备注": [
                    "虚拟原点：第一天出发",
                    "虚拟原点：第一天返回第二天出发",
                    "虚拟原点：第二天返回第三天出发",
                    "虚拟原点：第三天返回",
                ],
            },
            index=void_names,
        )
        self.__pointdf = pd.concat([voidpoint, self.__pointdf])
        self.__pointdf.drop(["探测起点"], inplace=True)
        self.__pointdf["No"] = range(self.__pointdf.shape[0])
        self.__dmatrix = self.__add_void_matrix(self.__dmatrix)
        self.__tmatrix = self.__add_void_matrix(self.__tmatrix)
        self.__pmatrix = self.__add_void_matrix(self.__pmatrix)

        self.__task["location"] = self.__task["location"].replace(
            "探测起点", "探测起点1,探测起点2,探测起点3,探测起点4"
        )
        self.__task["location"] = self.__task["location"].replace(
            np.nan, ",".join(self.__pointdf.index.values)
        )
        parsed_locations = []
        for locs in self.__task["location"].values:
            if pd.isna(locs):
                parsed_locations.append(np.nan)
            elif isinstance(locs, list):
                parsed_locations.append(locs)
            else:
                parsed_locations.append([s.strip() for s in str(locs).split(",")])
        self.__task["location"] = pd.Series(parsed_locations, index=self.__task.index)

        base = self.__task.shape[0]
        voidtaskdf = pd.DataFrame(
            {
                "No": [base, base + 1, base + 2, base + 3],
                "name": ["void1", "void2", "void3", "void4"],
                "revenue": [0, 0, 0, 0],
                "location": [["探测起点1"], ["探测起点2"], ["探测起点3"], ["探测起点4"]],
                "day": [np.nan, np.nan, np.nan, np.nan],
                "time": [0, 0, 0, 0],
                "power": [0, 0, 0, 0],
                "required": [False, False, False, False],
                "continuous": [False, False, False, False],
                "remote": [False, False, False, False],
                "exceptO": [False, False, False, False],
                "tag": [np.nan, np.nan, np.nan, np.nan],
            },
            index=[base, base + 1, base + 2, base + 3],
        )
        self.__task = pd.concat([self.__task, voidtaskdf])
        self.__void_task_indices = set(voidtaskdf.index.tolist())
        self.__virtual_origins = void_names

    def drop_O(self):
        """Mirror ea.py exceptO handling after virtual origins have been generated."""
        origins = ["探测起点1", "探测起点2", "探测起点3", "探测起点4"]
        for idx in self.__task[self.__task["exceptO"] == True].index.to_list():
            locs = self.__task.loc[idx, "location"]
            if isinstance(locs, list):
                self.__task.at[idx, "location"] = [l for l in locs if l not in origins]

    # ================================================================
    #  Preprocessing — mirrors ea.py's check_remote (same exception)
    # ================================================================

    def check_remote(self):
        """Mirror ea.py: remote locations are filtered against 探测起点1 after virtual-origin expansion."""
        remoteindex = self.__task[self.__task["remote"] == True].index.to_list()
        origin = "探测起点1"
        points_found = []
        for index in remoteindex:
            locs = self.__task.loc[index, "location"]
            if pd.isna(locs).all() if isinstance(locs, list) else pd.isna(locs):
                continue
            loc_list = locs if isinstance(locs, list) else [s.strip() for s in str(locs).split(",")]
            pts = []
            for point in loc_list:
                if point in self.__dmatrix.index and self.__dmatrix.loc[origin, point] >= self.__MDistance:
                    points_found.append(point)
                    pts.append(point)
            self.__task.at[index, "location"] = pts
        if len(points_found) == 0:
            path = "new-point.xlsx"
            sortedseries = self.__dmatrix.sort_values(by=[origin], ascending=False).loc[origin, :]
            start = 0
            while start < len(sortedseries) and sortedseries.iloc[start] >= self.__MDistance:
                start += 1
            pts = self.__dmatrix.sort_values(by=[origin], ascending=False).iloc[start : start + 5, :].index.values
            ox, oy = self.__pointdf.loc[origin, ["X", "Y"]]
            rec = []
            for p in pts:
                px, py = self.__pointdf.loc[p, ["X", "Y"]]
                theta = np.arctan2(py - oy, px - ox)
                rec.append((self.__MDistance * np.cos(theta) + ox, self.__MDistance * np.sin(theta) + oy))
            newpt = pd.DataFrame(
                rec,
                columns=["X", "Y"],
                index=pd.Index([f"新最远探测点{i}" for i in range(1, len(rec) + 1)], name="name"),
            )
            raise MaxDistanceError(
                f"No point meets the max distance requirement. Recommended points have been generated in {path}",
                newpt,
                path,
            )

    # ================================================================
    #  Convert raw task DataFrame → internal _Task list
    # ================================================================

    def __build_task_list(self):
        origins = ["探测起点1", "探测起点2", "探测起点3", "探测起点4"]
        all_points = list(self.__pointdf.index)
        void_names = {"void1", "void2", "void3", "void4"}
        tasks: List[_Task] = []
        for i, row in self.__task.iterrows():
            if str(row["name"]) in void_names:
                continue
            locs = row["location"]
            if isinstance(locs, list):
                loc_list = list(locs)
            elif pd.isna(locs):
                loc_list = list(all_points)
            else:
                loc_list = [s.strip() for s in str(locs).split(",")]
            except_o = bool(row.get("exceptO", False))
            if except_o:
                loc_list = [l for l in loc_list if l not in origins]
            tasks.append(_Task(
                idx=int(i),
                name=str(row["name"]),
                revenue=float(row["revenue"]),
                locations=loc_list,
                day=None if pd.isna(row["day"]) else str(row["day"]).strip(),
                time=float(row["time"]),
                power=float(row["power"]),
                required=bool(row["required"]),
                continuous=bool(row["continuous"]),
                remote=bool(row["remote"]),
                except_origin=except_o,
                tag=None if pd.isna(row["tag"]) else str(row["tag"]).strip(),
            ))
        self.__tasks = tasks
        self.__req_idx = [t.idx for t in tasks if t.required]
        self.__opt_idx = [t.idx for t in tasks if not t.required]
        self.__rem_idx = [t.idx for t in tasks if t.remote]
        self.__con_idx = [t.idx for t in tasks if t.continuous]
        self.__tag_map: Dict[str, List[int]] = {"12s": [], "12e": [], "23s": [], "23e": []}
        for t in tasks:
            if t.tag in self.__tag_map:
                self.__tag_map[t.tag].append(t.idx)
        self.__task_by_idx: Dict[int, _Task] = {t.idx: t for t in tasks}
        self.__virtual_origins = origins

    # ================================================================
    #  Helpers
    # ================================================================

    def __day_start(self, day_idx: int) -> str:
        return self.__virtual_origins[day_idx]

    def __day_end(self, day_idx: int) -> str:
        return self.__virtual_origins[day_idx + 1]

    def __travel_time(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return float(self.__tmatrix.loc[a, b])

    def __travel_power(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return float(self.__pmatrix.loc[a, b])

    def __format_number(self, number):
        number = float(number)
        fmt = "{:." + str(self.__decimal) + "f}"
        s = fmt.format(number)
        while s.endswith("0"):
            s = s[:-1]
        if s.endswith("."):
            s = s[:-1]
        return s

    def __day_allowed(self, task: _Task, day_idx: int) -> bool:
        if task.day is None:
            return True
        allowed = {int(float(x.strip())) for x in task.day.split(",")}
        return (day_idx + 1) in allowed

    def __tag_allowed_days(self, tag: str) -> List[int]:
        if tag in ("12s", "12e"):
            return [0, 1]
        if tag in ("23s", "23e"):
            return [1, 2]
        return [0, 1, 2]

    def __placed_indices(self, sol: _Solution) -> Set[int]:
        s = set()
        for seq in sol.days:
            for item in seq:
                s.add(item.task_idx)
        return s

    def __tag_indices(self) -> Set[int]:
        tag_set = set()
        for k in self.__tag_map:
            tag_set.update(self.__tag_map[k])
        return tag_set

    def __free_positions(self, seq: List[_ScheduleItem]) -> List[int]:
        tag_set = self.__tag_indices()
        return [i for i, item in enumerate(seq) if item.task_idx not in tag_set]

    def __seq_time_power(self, seq: List[_ScheduleItem], day_idx: int) -> Tuple[float, float]:
        prev = self.__day_start(day_idx)
        total_t = 0.0
        total_p = 0.0
        for item in seq:
            task = self.__task_by_idx[item.task_idx]
            total_t += self.__travel_time(prev, item.location) + task.time
            total_p += self.__travel_power(prev, item.location) + task.power
            prev = item.location
        total_t += self.__travel_time(prev, self.__day_end(day_idx))
        total_p += self.__travel_power(prev, self.__day_end(day_idx))
        return total_t, total_p

    # ================================================================
    #  Evaluator
    # ================================================================

    def __evaluate(self, sol: _Solution) -> float:
        penalty = 0.0
        total_rev = 0.0
        total_time_all = 0.0
        total_power_all = 0.0
        selected: Set[int] = set()

        for day_idx in range(3):
            seq = sol.days[day_idx]
            day_time, day_power = self.__seq_time_power(seq, day_idx)

            if day_time > self.__TTime[day_idx]:
                penalty += day_time - self.__TTime[day_idx]
            if day_power > self.__TPower[day_idx]:
                penalty += day_power - self.__TPower[day_idx]

            cum_t, cum_p = 0.0, 0.0
            prev = self.__day_start(day_idx)
            origin = "探测起点1"  # physical origin, matching ea.py safe_constrs
            for item in seq:
                t = self.__task_by_idx[item.task_idx]
                selected.add(item.task_idx)
                total_rev += t.revenue
                cum_t += self.__travel_time(prev, item.location) + t.time
                cum_p += self.__travel_power(prev, item.location) + t.power
                # ea.py safe_constrs check return to *physical* origin "探测起点1":
                #   time:  tmatrix.loc[point_name, "探测起点1"]  (point → origin)
                #   power: pmatrix.loc["探测起点1", point_name]  (origin → point)
                # The formula is: (cum - task_time) + return_travel <= day_budget
                # which equals: cum + return_travel - task_time <= day_budget
                ret_t = float(self.__tmatrix.loc[item.location, origin])
                ret_p = float(self.__pmatrix.loc[origin, item.location])
                safe_t = cum_t - t.time + ret_t
                safe_p = cum_p - t.power + ret_p
                if safe_t > self.__TTime[day_idx]:
                    penalty += safe_t - self.__TTime[day_idx]
                if safe_p > self.__TPower[day_idx]:
                    penalty += safe_p - self.__TPower[day_idx]
                prev = item.location

            total_time_all += day_time
            total_power_all += day_power

        for idx in self.__req_idx:
            if idx not in selected:
                penalty += 10.0

        rc = sum(1 for idx in self.__rem_idx if idx in selected)
        if rc != 1:
            penalty += abs(rc - 1) * 10.0

        # ea.py uses equality: sum(continuous selected) == Mincontinuous.
        cc = sum(1 for idx in self.__con_idx if idx in selected)
        if cc != self.__Mincontinuous:
            penalty += abs(cc - self.__Mincontinuous) * 5.0

        for day_idx in range(3):
            for item in sol.days[day_idx]:
                t = self.__task_by_idx[item.task_idx]
                if not self.__day_allowed(t, day_idx):
                    penalty += 10.0
                if item.location not in t.locations:
                    penalty += 5.0

        penalty += self.__check_tag_penalty(sol)

        all_sel = []
        for seq in sol.days:
            all_sel.extend(item.task_idx for item in seq)
        if len(all_sel) != len(set(all_sel)):
            penalty += (len(all_sel) - len(set(all_sel))) * 20.0

        if self.__objective == CONST.MAX_REVENUE:
            obj = total_rev
        elif self.__objective == CONST.MIN_TIME:
            obj = -total_time_all
        elif self.__objective == CONST.MIN_POWER:
            obj = -total_power_all
        else:
            obj = total_rev

        sol.total_revenue = total_rev
        sol.total_time = total_time_all
        sol.total_power = total_power_all
        sol.penalty = penalty
        sol.fitness = obj - self.__penalty_weight * penalty
        return sol.fitness

    def __find_task_day_pos(self, sol: _Solution, task_idx: int):
        found = []
        for d in range(3):
            for pos, item in enumerate(sol.days[d]):
                if item.task_idx == task_idx:
                    found.append((d, pos))
        return found

    def __check_tag_penalty(self, sol: _Solution) -> float:
        """Penalty version of ea.py's day/tag boundary equalities.

        Zero penalty means every tag task is selected exactly once, start tags are
        day-start boundary tasks, end tags are day-end boundary tasks, and each
        start/end pair selects the same physical day. This is the SA equivalent
        of ea.py's 12s/12e/23s/23e constraints.
        """
        pen = 0.0
        pairs = [("12s", "12e", [0, 1]), ("23s", "23e", [1, 2])]
        pair_days = {}

        for ts, te, allowed_days in pairs:
            s_days = []
            e_days = []

            for idx in self.__tag_map[ts]:
                found = self.__find_task_day_pos(sol, idx)
                if len(found) != 1:
                    pen += abs(len(found) - 1) * 10.0
                    continue
                d, pos = found[0]
                s_days.append(d)
                if d not in allowed_days:
                    pen += 10.0
                if pos != 0:
                    pen += 5.0

            for idx in self.__tag_map[te]:
                found = self.__find_task_day_pos(sol, idx)
                if len(found) != 1:
                    pen += abs(len(found) - 1) * 10.0
                    continue
                d, pos = found[0]
                e_days.append(d)
                if d not in allowed_days:
                    pen += 10.0
                if pos != len(sol.days[d]) - 1:
                    pen += 5.0

            # If both groups exist, ea.py's sameday equalities require the chosen
            # start boundary and end boundary to describe the same day. Multiple
            # same-group tasks are effectively infeasible because only one task can
            # be the first/last item of a day. Penalize that explicitly.
            if self.__tag_map[ts] or self.__tag_map[te]:
                if len(s_days) != len(self.__tag_map[ts]):
                    pen += 10.0
                if len(e_days) != len(self.__tag_map[te]):
                    pen += 10.0
                if len(set(s_days)) > 1:
                    pen += (len(set(s_days)) - 1) * 10.0
                if len(set(e_days)) > 1:
                    pen += (len(set(e_days)) - 1) * 10.0
                if s_days and e_days and set(s_days) != set(e_days):
                    pen += 10.0
            pair_days[ts] = set(s_days)
            pair_days[te] = set(e_days)

        # ea.py's startnotsameday/endnotsameday constraints prevent both
        # transition starts or both transition ends from occupying Day2.
        day2_starts = int(1 in pair_days.get("12s", set())) + int(1 in pair_days.get("23s", set()))
        day2_ends = int(1 in pair_days.get("12e", set())) + int(1 in pair_days.get("23e", set()))
        if day2_starts > 1:
            pen += (day2_starts - 1) * 10.0
        if day2_ends > 1:
            pen += (day2_ends - 1) * 10.0
        return pen

    # ================================================================
    #  Initial solution builder (greedy)
    # ================================================================

    def __build_initial(self) -> _Solution:
        sol = _Solution()

        # 1) required tasks — greedy cheapest insertion
        req = [self.__task_by_idx[i] for i in self.__req_idx]
        random.shuffle(req)
        for t in req:
            d, pos, loc = self.__best_insertion(sol, t)
            if d is not None:
                sol.days[d].insert(pos, _ScheduleItem(t.idx, loc))

        # 2) tag tasks at boundaries
        for ts, te in [("12s", "12e"), ("23s", "23e")]:
            placed = self.__placed_indices(sol)
            s_unplaced = [self.__task_by_idx[i] for i in self.__tag_map[ts]
                          if i not in placed]
            e_unplaced = [self.__task_by_idx[i] for i in self.__tag_map[te]
                          if i not in placed]
            if s_unplaced or e_unplaced:
                days = self.__tag_allowed_days(ts)
                cd = random.choice(days)
                for t in s_unplaced:
                    loc = random.choice(t.locations) if t.locations else self.__day_start(cd)
                    sol.days[cd].insert(0, _ScheduleItem(t.idx, loc))
                for t in e_unplaced:
                    loc = random.choice(t.locations) if t.locations else self.__day_start(cd)
                    sol.days[cd].append(_ScheduleItem(t.idx, loc))

        # 3) optional tasks by revenue density
        opt = [self.__task_by_idx[i] for i in self.__opt_idx]
        opt.sort(key=lambda t: t.revenue / max(t.time, 0.01), reverse=True)
        placed = self.__placed_indices(sol)
        for t in opt:
            if t.idx in placed:
                continue
            d, pos, loc = self.__best_insertion(sol, t)
            if d is not None:
                sol.days[d].insert(pos, _ScheduleItem(t.idx, loc))

        # 4) fix remote = exactly 1
        self.__fix_remote(sol)
        return sol

    def __best_insertion(self, sol, task):
        best = (None, 0, task.locations[0] if task.locations else self.__day_start(0))
        best_cost = math.inf
        for d in range(3):
            if not self.__day_allowed(task, d):
                continue
            # tag tasks can only be inserted at ea.py-equivalent day boundaries.
            candidate_positions = None
            if task.tag in ("12s", "23s"):
                if d not in self.__tag_allowed_days(task.tag):
                    continue
                candidate_positions = [0]
            elif task.tag in ("12e", "23e"):
                if d not in self.__tag_allowed_days(task.tag):
                    continue
                candidate_positions = [len(sol.days[d])]
            seq = sol.days[d]
            positions = candidate_positions if candidate_positions is not None else list(range(len(seq) + 1))
            for loc in task.locations:
                for pos in positions:
                    prev = self.__day_start(d) if pos == 0 else seq[pos - 1].location
                    nxt = self.__day_end(d) if pos == len(seq) else seq[pos].location
                    old = self.__travel_time(prev, nxt)
                    new = self.__travel_time(prev, loc) + self.__travel_time(loc, nxt)
                    cost = new - old + task.time
                    if cost < best_cost:
                        best_cost = cost
                        best = (d, pos, loc)
        return best

    def __fix_remote(self, sol):
        placed = self.__placed_indices(sol)
        rp = [i for i in self.__rem_idx if i in placed]
        if len(rp) == 0 and self.__rem_idx:
            t = self.__task_by_idx[random.choice(self.__rem_idx)]
            d, pos, loc = self.__best_insertion(sol, t)
            if d is not None:
                sol.days[d].insert(pos, _ScheduleItem(t.idx, loc))
        elif len(rp) > 1:
            for idx in rp[1:]:
                for seq in sol.days:
                    seq[:] = [item for item in seq if item.task_idx != idx]

    # ================================================================
    #  Neighborhood operators
    # ================================================================

    def __neighbor(self, sol: _Solution, T: float) -> _Solution:
        ratio = T / self.__T0
        weights = {
            "intra_swap":      3.0 + 4.0 * (1 - ratio),
            "intra_2opt":      2.0 + 3.0 * (1 - ratio),
            "inter_move":      3.0 + 3.0 * ratio,
            "inter_swap":      2.0 + 2.0 * ratio,
            "insert_remove":   3.0 + 4.0 * ratio,
            "change_location": 2.0 + 2.0 * (1 - ratio),
        }
        ops = list(weights.keys())
        ws = [weights[k] for k in ops]
        chosen = random.choices(ops, weights=ws, k=1)[0]
        ns = copy.deepcopy(sol)

        if chosen == "intra_swap":
            self.__op_intra_swap(ns)
        elif chosen == "intra_2opt":
            self.__op_intra_2opt(ns)
        elif chosen == "inter_move":
            self.__op_inter_move(ns)
        elif chosen == "inter_swap":
            self.__op_inter_swap(ns)
        elif chosen == "insert_remove":
            self.__op_insert_remove(ns)
        elif chosen == "change_location":
            self.__op_change_location(ns)
        return ns

    def __op_intra_swap(self, sol):
        d = random.randint(0, 2)
        free = self.__free_positions(sol.days[d])
        if len(free) < 2:
            return
        i, j = random.sample(free, 2)
        sol.days[d][i], sol.days[d][j] = sol.days[d][j], sol.days[d][i]

    def __op_intra_2opt(self, sol):
        d = random.randint(0, 2)
        free = self.__free_positions(sol.days[d])
        if len(free) < 2:
            return
        i, j = sorted(random.sample(free, 2))
        seg = list(range(i, j + 1))
        if all(p in free for p in seg):
            items = [sol.days[d][p] for p in seg]
            items.reverse()
            for k, p in enumerate(seg):
                sol.days[d][p] = items[k]

    def __op_inter_move(self, sol):
        src = random.randint(0, 2)
        free = self.__free_positions(sol.days[src])
        if not free:
            return
        pos = random.choice(free)
        item = sol.days[src][pos]
        t = self.__task_by_idx[item.task_idx]
        dsts = [dd for dd in range(3) if dd != src and self.__day_allowed(t, dd)]
        if not dsts:
            return
        dst = random.choice(dsts)
        sol.days[src].pop(pos)
        sol.days[dst].insert(random.randint(0, len(sol.days[dst])), item)

    def __op_inter_swap(self, sol):
        d1, d2 = random.sample(range(3), 2)
        f1 = self.__free_positions(sol.days[d1])
        f2 = self.__free_positions(sol.days[d2])
        if not f1 or not f2:
            return
        p1 = random.choice(f1)
        p2 = random.choice(f2)
        sol.days[d1][p1], sol.days[d2][p2] = sol.days[d2][p2], sol.days[d1][p1]

    def __op_insert_remove(self, sol):
        placed = self.__placed_indices(sol)
        tag_set = self.__tag_indices()
        unplaced = [t for t in self.__tasks if (not t.required) and t.idx not in placed]
        removable = []
        for d in range(3):
            for pos in self.__free_positions(sol.days[d]):
                item = sol.days[d][pos]
                if not self.__task_by_idx[item.task_idx].required:
                    removable.append((d, pos))

        if random.random() < 0.5 and unplaced:
            t = random.choice(unplaced)
            d, pos, loc = self.__best_insertion(sol, t)
            if d is not None:
                sol.days[d].insert(pos, _ScheduleItem(t.idx, loc))
        elif removable:
            d, pos = random.choice(removable)
            # Do not remove tag tasks with the generic operator; tag feasibility is handled by paired boundary logic.
            if sol.days[d][pos].task_idx not in tag_set:
                sol.days[d].pop(pos)

    def __op_change_location(self, sol):
        d = random.randint(0, 2)
        if not sol.days[d]:
            return
        pos = random.randint(0, len(sol.days[d]) - 1)
        item = sol.days[d][pos]
        t = self.__task_by_idx[item.task_idx]
        if len(t.locations) <= 1:
            return
        others = [l for l in t.locations if l != item.location]
        if others:
            item.location = random.choice(others)

    # ================================================================
    #  SA main loop
    # ================================================================

    def __run_sa(self):
        random.seed(self.__seed)
        np.random.seed(self.__seed)

        current = self.__build_initial()
        self.__evaluate(current)
        best = copy.deepcopy(current)

        T = self.__T0
        no_improve = 0
        iteration = 0
        import time as _time
        start_time = _time.time()

        print(f"[SA] Initial fitness={current.fitness:.4f}  "
              f"rev={current.total_revenue:.2f}  pen={current.penalty:.4f}")

        while T > self.__T_min:
            # respect timeLimit if set
            if self.__timeLimit != np.inf:
                if _time.time() - start_time > self.__timeLimit:
                    print("[SA] Time limit reached.")
                    break

            improved = False
            for _ in range(self.__iter_per_temp):
                iteration += 1
                cand = self.__neighbor(current, T)
                self.__evaluate(cand)
                delta = cand.fitness - current.fitness
                if delta > 0 or random.random() < math.exp(
                    delta / T if T > 1e-12 else -1e12
                ):
                    current = cand
                    if cand.fitness > best.fitness:
                        best = copy.deepcopy(cand)
                        improved = True
                        # autosave
                        if self.__autosavestate:
                            self.__try_autosave(best)

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= self.__reheat_every:
                T = min(T * self.__reheat_factor, self.__T0)
                no_improve = 0

            T *= self.__alpha

        self.__best_sol = best
        # Store objective value so solve() wrapper can read it
        if self.__objective == CONST.MAX_REVENUE:
            self.__objvalue = best.total_revenue
        elif self.__objective == CONST.MIN_TIME:
            # Match ea.py solver objective: maximize negative total completion time.
            self.__objvalue = -best.total_time
        elif self.__objective == CONST.MIN_POWER:
            # Match ea.py solver objective: maximize negative total completion power.
            self.__objvalue = -best.total_power
        else:
            self.__objvalue = best.total_revenue

        print(f"[SA] Best fitness={best.fitness:.4f}  "
              f"rev={best.total_revenue:.2f}  "
              f"time={best.total_time:.2f}  "
              f"power={best.total_power:.2f}  "
              f"pen={best.penalty:.4f}")

    def __try_autosave(self, sol):
        try:
            if self.__outputpath.endswith(".xlsx"):
                p = self.__outputpath[:-5] + f"-sol{self.__solcount}.xlsx"
            elif self.__outputpath.endswith(".xls"):
                p = self.__outputpath[:-4] + f"-sol{self.__solcount}.xls"
            else:
                p = self.__outputpath + f".sol{self.__solcount}"
            p = os.path.join("autosave", os.path.basename(p))
            self.__write_schedule(sol, p)
            self.__solcount += 1
        except Exception:
            pass

    # ================================================================
    #  Output generation — same columns & structure as ea.py
    #  Columns: No, action, location, time, power, 1, 2
    # ================================================================

    def __gen_packagedf(self, tag):
        """Identical to ea.py's __gen_packagedf."""
        pak = self.__package[self.__package["tag"] == tag]
        n = pak.shape[0]
        origin = "探测起点1"
        ox = self.__pointdf.loc[origin, "X"]
        oy = self.__pointdf.loc[origin, "Y"]
        location = f"({self.__format_number(ox)},{self.__format_number(oy)})"
        player = ["√"] * n
        return pd.DataFrame({
            "No": list(range(n)),
            "action": pak["name"].values,
            "location": [location] * n,
            "time": pak["time"].values,
            "power": pak["power"].values,
            "1": player,
            "2": player,
        })

    def __build_plandf(self, sol: _Solution):
        """Build per-day rows using the same virtual-origin route semantics as ea.py.

        ea.py's cal_route() pops the last entry (void task at virtual end origin)
        and its preceding travel row.  We replicate that by only iterating over
        real tasks — no sentinel for the end virtual origin.
        """
        self.__plandf_days = []
        n = 1

        for day_idx in range(3):
            no, action, loc_col, t_col, p_col, p1, p2 = ([], [], [], [], [], [], [])
            seq = sol.days[day_idx]
            prev_loc = self.__day_start(day_idx)

            for item in seq:
                task = self.__task_by_idx[item.task_idx]
                pt_name = item.location

                px = self.__pointdf.loc[pt_name, "X"]
                py = self.__pointdf.loc[pt_name, "Y"]

                both_virtual = prev_loc in self.__virtual_origins and pt_name in self.__virtual_origins
                if pt_name != prev_loc and not both_virtual:
                    cx = self.__pointdf.loc[prev_loc, "X"]
                    cy = self.__pointdf.loc[prev_loc, "Y"]
                    no.append(n)
                    action.append(
                        f"Travel from ({self.__format_number(cx)},{self.__format_number(cy)}) to "
                        f"({self.__format_number(px)},{self.__format_number(py)})"
                    )
                    t_col.append(float(self.__format_number(self.__travel_time(prev_loc, pt_name))))
                    p_col.append(float(self.__format_number(self.__travel_power(prev_loc, pt_name))))
                    loc_col.append(
                        f"({self.__format_number(cx)},{self.__format_number(cy)})→"
                        f"({self.__format_number(px)},{self.__format_number(py)})"
                    )
                    p1.append("√"); p2.append("√")
                    n += 1

                no.append(n)
                action.append(task.name)
                loc_col.append(f"({self.__format_number(px)},{self.__format_number(py)})")
                t_col.append(float(self.__format_number(task.time)))
                p_col.append(float(self.__format_number(task.power)))
                p1.append("√"); p2.append("√")
                n += 1
                prev_loc = pt_name

            self.__plandf_days.append(pd.DataFrame({
                "No": no,
                "action": action,
                "location": loc_col,
                "time": t_col,
                "power": p_col,
                "1": p1,
                "2": p2,
            }))

    def __assemble_output(self, sol: _Solution):
        """
        Combine plandf_days with package DFs exactly like ea.py's add_package,
        producing self.__plandf and self.__voidindex.
        """
        self.__build_plandf(sol)

        pakdf = [
            [
                [self.__gen_packagedf(f"D{d+1}ss"),
                 self.__gen_packagedf(f"D{d+1}se")],
                [self.__gen_packagedf(f"D{d+1}es"),
                 self.__gen_packagedf(f"D{d+1}ee")],
            ]
            for d in range(3)
        ]

        plan = [None, None, None]
        df_break = [None, None, None]

        # "Begin of Day1" header
        df0 = pd.DataFrame({
            "No": [np.nan], "action": ["Begin of Day1"],
            "location": [np.nan], "time": [np.nan], "power": [np.nan],
            "1": [np.nan], "2": [np.nan],
        })

        # Detect whether tag tasks (12s/23s) are in each day
        isindf12 = [False] * 3
        isindf23 = [False] * 3
        tag12s_names = self.__task[
            self.__task["tag"] == "12s"
        ]["name"].values if "tag" in self.__task.columns else []
        tag23s_names = self.__task[
            self.__task["tag"] == "23s"
        ]["name"].values if "tag" in self.__task.columns else []

        for i in range(3):
            if self.__plandf_days[i].shape[0] > 0:
                isindf12[i] = self.__plandf_days[i]["action"].isin(
                    tag12s_names
                ).any()
                isindf23[i] = self.__plandf_days[i]["action"].isin(
                    tag23s_names
                ).any()

        for i in range(3):
            pday = self.__plandf_days[i]
            has_tag = isindf12[i] or isindf23[i]
            gap = self.__12gap if isindf12[i] else (
                self.__23gap if isindf23[i] else 0
            )

            if has_tag and pday.shape[0] >= 2:
                # tag day: ss | first_task | se | middle_tasks | es | last_task | ee
                # With wait if total < gap
                first = pday.iloc[:1, :]
                last = pday.iloc[-1:, :]
                middle = pday.iloc[1:-1, :] if pday.shape[0] > 2 else pd.DataFrame()

                temp_parts = [first, pakdf[i][0][1]]
                if middle.shape[0] > 0:
                    temp_parts.append(middle)
                temp_parts.extend([pakdf[i][1][0], last])
                temp = pd.concat(temp_parts)
                t_sum = sum(temp["time"])

                parts = [pakdf[i][0][0], first, pakdf[i][0][1]]
                if middle.shape[0] > 0:
                    parts.append(middle)
                parts.append(pakdf[i][1][0])

                if t_sum <= gap:
                    wait_time = float(self.__format_number(gap - t_sum))
                    origin = "探测起点1"
                    ox = self.__pointdf.loc[origin, "X"]
                    oy = self.__pointdf.loc[origin, "Y"]
                    adddf = pd.DataFrame({
                        "No": [np.nan],
                        "action": [f"Wait for {self.__format_number(gap - t_sum)}s"],
                        "location": [f"({self.__format_number(ox)},{self.__format_number(oy)})"],
                        "time": [wait_time],
                        "power": [0],
                        "1": ["√"], "2": ["√"],
                    })
                    parts.append(adddf)

                parts.extend([last, pakdf[i][1][1]])
                plan[i] = pd.concat(parts)
            else:
                # normal day: concat [ss, se] + planned + [es, ee]
                pak_start = pd.concat(pakdf[i][0])
                pak_end = pd.concat(pakdf[i][1])
                plan[i] = pd.concat([pak_start, pday, pak_end])

            t_str = "sum: {}".format(
                self.__format_number(sum(plan[i]["time"]))
            )
            p_str = "sum: {}".format(
                self.__format_number(sum(plan[i]["power"]))
            )
            df_break[i] = pd.DataFrame({
                "No": [np.nan], "action": ["xxx"],
                "location": [np.nan], "time": [t_str], "power": [p_str],
                "1": [np.nan], "2": [np.nan],
            })

        df_break[0]["action"] = "Break between Day1 and Day2"
        df_break[1]["action"] = "Break between Day2 and Day3"
        df_break[2]["action"] = "End of Day3"

        self.__plandf = pd.concat([
            df0, plan[0], df_break[0],
            plan[1], df_break[1],
            plan[2], df_break[2],
        ])
        self.__plandf.reset_index(drop=True, inplace=True)

        # Compute voidindex for yellow highlighting (same as ea.py)
        idx0 = self.__plandf[
            self.__plandf["action"] == "Begin of Day1"
        ].index[0] + 2
        idx1 = self.__plandf[
            self.__plandf["action"] == "Break between Day1 and Day2"
        ].index[0] + 2
        idx2 = self.__plandf[
            self.__plandf["action"] == "Break between Day2 and Day3"
        ].index[0] + 2
        idx3 = self.__plandf[
            self.__plandf["action"] == "End of Day3"
        ].index[0] + 2
        self.__voidindex = [int(idx0), int(idx1), int(idx2), int(idx3)]

        # Renumber No column (same logic as ea.py)
        no = list(range(self.__plandf.shape[0]))
        for ii in range(len(no)):
            if ii in [0, idx1 - 2, idx2 - 2, idx3 - 2]:
                no[ii] = np.nan
            elif ii > idx1 - 2 and ii < idx2 - 2:
                no[ii] = ii - 1
            elif ii > idx3 - 2:
                no[ii] = ii - 2
        self.__plandf["No"] = no

    # ================================================================
    #  write_excel — identical to ea.py
    # ================================================================

    def write_excel(self, path=None):
        if path is None:
            path = self.__outputpath
        with pd.ExcelWriter(path) as writer:
            self.__plandf.to_excel(writer, index=False)
            worksheet = writer.sheets[list(writer.sheets.keys())[0]]
            for column_cells in worksheet.columns:
                length = max(
                    max(len(str(cell.value)) for cell in column_cells), 5
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = (length + 2)
            for row in self.__voidindex:
                for cell in worksheet[row]:
                    cell.fill = PatternFill(
                        start_color="FFFF00",
                        end_color="FFFF00",
                        fill_type="solid",
                    )

    def schedule_frame(self):
        return self.__plandf.copy()

    def __write_schedule(self, sol, path):
        """Autosave helper: build output then write."""
        self.__assemble_output(sol)
        self.write_excel(path)

    # ================================================================
    #  Print status — same interface as ea.py
    # ================================================================

    def __calc_efficiencies(self, sol: _Solution) -> Tuple[float, float]:
        total_time_budget = sum(self.__TTime) + sum(self.__PKGTime)
        total_power_budget = sum(self.__TPower) + sum(self.__PKGPower)
        time_efficiency = (
            (sol.total_time + sum(self.__PKGTime)) / total_time_budget * 100
            if total_time_budget > 0 else 0.0
        )
        power_efficiency = (
            (sol.total_power + sum(self.__PKGPower)) / total_power_budget * 100
            if total_power_budget > 0 else 0.0
        )
        return time_efficiency, power_efficiency

    def print_status(self):
        best = self.__best_sol
        if best.penalty < 1e-6:
            print(f"SA feasible incumbent objective value: {self.__objvalue}")
            time_efficiency, power_efficiency = self.__calc_efficiencies(best)
            print(
                "Feasible objective value "
                f"{self.__format_number(self.__objvalue)} found with "
                f"{self.__format_number(time_efficiency)}% time-efficiency and "
                f"{self.__format_number(power_efficiency)}% power-efficiency"
            )
        else:
            raise MIPError(f"SA did not find a feasible solution; best penalty={best.penalty}")

    # ================================================================
    #  run() — identical call pattern to ea.py
    # ================================================================

    def run(self):
        self.test_IO()
        self.read_info()
        self.read_task()
        self.read_package()
        self.read_point()
        self.gen_void_point()
        self.check_remote()
        self.drop_O()
        self.__build_task_list()
        self.__run_sa()
        self.print_status()
        self.__assemble_output(self.__best_sol)
        if self.__write_output:
            self.write_excel()


# ============================================================================
#  solve() wrapper — identical signature to ea.py's solve()
# ============================================================================

def solve(case, mode="normal"):
    """Run the SA algorithm on UnifiedCase input."""
    if mode != "normal":
        raise NotImplementedError("sa only supports algorithm.mode=normal")

    from .schedule import (
        SchedulePlan,
        build_legacy_bundle,
        legacy_schedule_to_rows,
    )

    obj_map = {
        CONST.MAX_REVENUE: CONST.MAX_REVENUE,
        CONST.MIN_TIME: CONST.MIN_TIME,
        CONST.MIN_POWER: CONST.MIN_POWER,
    }
    objective = obj_map.get(case.config.algorithm.obj, CONST.MAX_REVENUE)

    bundle = build_legacy_bundle(case)
    kwargs = {
        "obj": objective,
        "decimal": case.config.algorithm.decimal,
        "autoSave": False,
        "writeOutput": False,
        "dataFrames": {
            "info": bundle.info,
            "task": bundle.task,
            "package": bundle.package[["name", "time", "power", "tag"]],
            "point": bundle.point[["name", "X", "Y", "备注"]],
            "distance": bundle.distance,
            "time": bundle.time,
            "power": bundle.power,
        },
    }
    if case.config.algorithm.time_limit is not None:
        kwargs["timeLimit"] = case.config.algorithm.time_limit
    if case.config.algorithm.random_seed is not None:
        kwargs["seed"] = case.config.algorithm.random_seed
    optimizer = task_optimize(**kwargs)
    optimizer.run()
    schedule_df = optimizer.schedule_frame()
    rows = legacy_schedule_to_rows(case, schedule_df)
    objective_value = getattr(
        optimizer, "_task_optimize__objvalue", None
    )
    return SchedulePlan(
        steps=[],
        rows=rows,
        objective_value=(
            None if objective_value is None
            else float(objective_value)
        ),
    )


# ============================================================================
#  __main__ — identical to ea.py
# ============================================================================

if __name__ == "__main__":
    opt = task_optimize(CONST.MAX_REVENUE, timeLimit=60 * 60 * 4)
    opt.run()
