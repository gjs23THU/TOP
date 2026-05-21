# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization (PSO) solver for task-path co-optimization.

Unified entry point ``solve(case, mode)`` consumes in-memory DataFrames built
from the normalized CSV case. The ``task_optimize`` class still accepts legacy
workbook path parameters for compatibility, but DataFrame inputs bypass Excel
I/O and return the generated schedule as a DataFrame.

Encoding
--------
A particle is a continuous random-key vector with four dimensions per real
task: selection, day, location, and within-day order. The solver updates
continuous positions and velocities with the canonical PSO equation, then
decodes the random keys into a complete 3-day route. A repair/validation
layer enforces the same structural constraints as ea.py: required tasks,
remote exactly-one, continuous exact-cardinality, day windows, tag boundary
constraints, time/power budgets, and per-step return safety.

This avoids any dependency on Gurobi.
"""

import os
import copy
import threading
import numpy as np
import pandas as pd
from itertools import product as iterproduct
from openpyxl.styles import PatternFill

from .models import MIPError, SubtourError


# ---------------------------------------------------------------------------
# Constants (same as ea.py)
# ---------------------------------------------------------------------------
class CONST(object):
    MAX_REVENUE = "maxRevenue"
    MIN_TIME = "minTime"
    MIN_POWER = "minPower"


class MaxDistanceError(Exception):
    def __init__(self, message, df, path):
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer)
        self.message = message

    def __str__(self):
        return self.message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_number(number, decimal=5):
    number = float(number)
    fmt = "{:." + str(decimal) + "f}"
    s = fmt.format(number)
    while s.endswith("0"):
        s = s[:-1]
    if s.endswith("."):
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Main class – public API identical to ea.py's task_optimize
# ---------------------------------------------------------------------------
class task_optimize(object):
    """PSO-based scheduler with the same interface as the Gurobi version."""

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
        # PSO hyper-parameters (extra, ignored by ea.py callers)
        n_particles=80,
        max_iter=2000,
        w_start=0.9,
        w_end=0.4,
        c1=2.0,
        c2=2.0,
        seed=None,
    ):
        self.__objective = obj
        self.__Obj = [CONST.MAX_REVENUE, CONST.MIN_TIME, CONST.MIN_POWER]
        self.__timeLimit = timeLimit
        self.__autosavestate = autoSave
        self.__decimal = decimal
        self.__lppath, self.__solpath = lpPath, solPath
        self.__infoPath, self.__taskPath, self.__pakpath = (
            infoPath, taskPath, packPath,
        )
        self.__pointpath = pointPath
        self.__distancepath = distancePath
        self.__timepath = timePath
        self.__powerpath = powerPath
        self.__outputpath = outputPath
        self.__dataframes = dataFrames or {}
        self.__write_output = writeOutput
        self.__solcount = 1

        # PSO parameters
        self.__n_particles = n_particles
        self.__max_iter = max_iter
        self.__w_start = w_start
        self.__w_end = w_end
        self.__c1 = c1
        self.__c2 = c2
        self.__rng = np.random.default_rng(seed)

        self.__objvalue = None
        return None

    # -----------------------------------------------------------------------
    # I/O (identical to ea.py)
    # -----------------------------------------------------------------------
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
        return None

    def read_info(self):
        info = self.__dataframes.get("info")
        info = info.copy() if info is not None else pd.read_excel(self.__infoPath)
        self.__MDistance = info["max-distance"][0]
        self.__TTime = list(map(float, info["total-time/day"][0].split(";")))
        self.__TPower = list(map(float, info["total-power/day"][0].split(";")))
        self.__Mincontinuous = info["min-continuous"][0]
        self.__12gap = info["12-gap"][0]
        self.__23gap = info["23-gap"][0]
        return None

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
        return None

    def read_package(self):
        package = self.__dataframes.get("package")
        self.__package = package.copy() if package is not None else pd.read_excel(self.__pakpath)
        tags_order = [
            "D1ss", "D1se", "D1es", "D1ee",
            "D2ss", "D2se", "D2es", "D2ee",
            "D3ss", "D3se", "D3es", "D3ee",
        ]
        time_vals = [
            sum(self.__package[self.__package["tag"] == tag]["time"])
            for tag in tags_order
        ]
        power_vals = [
            sum(self.__package[self.__package["tag"] == tag]["power"])
            for tag in tags_order
        ]
        self.__PKGTime = [sum(time_vals[i * 4:(i + 1) * 4]) for i in range(3)]
        self.__PKGPower = [sum(power_vals[i * 4:(i + 1) * 4]) for i in range(3)]
        self.__TTime = [self.__TTime[i] - self.__PKGTime[i] for i in range(3)]
        self.__TPower = [self.__TPower[i] - self.__PKGPower[i] for i in range(3)]
        return None

    def read_point(self):
        point = self.__dataframes.get("point")
        self.__pointdf = point.copy() if point is not None else pd.read_excel(self.__pointpath)
        self.__pointdf.set_index(self.__pointdf.columns[0], inplace=True)
        self.__pointdf.index.rename(None, inplace=True)
        return None

    # -----------------------------------------------------------------------
    # Virtual-node construction (mirrors ea.py exactly)
    # -----------------------------------------------------------------------
    def __add_void_matrix(self, matrix):
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
        for i, j in iterproduct(
            ["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
            ["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
        ):
            if i != j:
                matrix.loc[i, j] = 0
        return matrix

    def gen_void_point(self):
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
            index=["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
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
        location = self.__task["location"].values
        pool = []
        for i in location:
            if pd.isna(i):
                pool.append(np.nan)
            else:
                pool.append(list(i.split(",")))
        self.__task["location"] = pd.Series(pool)
        voidtaskdf = pd.DataFrame(
            {
                "No": [self.__task.shape[0] + k for k in range(4)],
                "name": ["void1", "void2", "void3", "void4"],
                "revenue": [0, 0, 0, 0],
                "location": [["探测起点1"], ["探测起点2"], ["探测起点3"], ["探测起点4"]],
                "day": [np.nan] * 4,
                "time": [0, 0, 0, 0],
                "power": [0, 0, 0, 0],
                "required": [False] * 4,
                "continuous": [False] * 4,
                "remote": [False] * 4,
                "exceptO": [False] * 4,
                "tag": [np.nan] * 4,
            },
            index=[self.__task.shape[0] + k for k in range(4)],
        )
        self.__task = pd.concat([self.__task, voidtaskdf])
        return None

    # -----------------------------------------------------------------------
    # Coordinate helpers (same as ea.py)
    # -----------------------------------------------------------------------
    def __cartesian_to_polar(self, x, y, O):
        r = np.sqrt((x - O[0]) ** 2 + (y - O[1]) ** 2)
        theta = np.arctan2(y - O[1], x - O[0])
        return r, theta

    def __polar_to_cartesian(self, r, theta, O):
        x = r * np.cos(theta) + O[0]
        y = r * np.sin(theta) + O[1]
        return x, y

    def __cal_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def __from_distance(self, distance):
        return (distance, 0)

    def check_remote(self):
        remoteindex = self.__task[self.__task["remote"] == True].index.to_list()
        points = []
        for index in remoteindex:
            pts = []
            for point in self.__task.loc[index, "location"]:
                if self.__dmatrix.loc["探测起点1", point] >= self.__MDistance:
                    points.append(point)
                    pts.append(point)
            self.__task.at[index, "location"] = pts
        if len(points) == 0:
            start = 0
            path = "new-point.xlsx"
            sortedseries = self.__dmatrix.sort_values(
                by=["探测起点1"], ascending=False
            ).loc["探测起点1", :]
            while sortedseries.iloc[start] >= self.__MDistance:
                start += 1
            pts_names = (
                self.__dmatrix.sort_values(by=["探测起点1"], ascending=False)
                .iloc[start:start + 5, :]
                .index.values
            )
            r, _ = self.__cartesian_to_polar(
                self.__from_distance(self.__MDistance)[0],
                self.__from_distance(self.__MDistance)[1],
                (0, 0),
            )
            pts_polar = [
                self.__cartesian_to_polar(
                    self.__pointdf.loc[pt, "X"],
                    self.__pointdf.loc[pt, "Y"],
                    (self.__pointdf.loc["探测起点1", "X"],
                     self.__pointdf.loc["探测起点1", "Y"]),
                )
                for pt in pts_names
            ]
            pts_cart = [
                self.__polar_to_cartesian(
                    r, pt[1],
                    (self.__pointdf.loc["探测起点1", "X"],
                     self.__pointdf.loc["探测起点1", "Y"]),
                )
                for pt in pts_polar
            ]
            newpt = pd.DataFrame(
                pts_cart, columns=["X", "Y"],
                index=pd.Index(
                    ["新最远探测点1", "新最远探测点2", "新最远探测点3", "新最远探测点4", "新最远探测点5"],
                    name="name",
                ),
            )
            raise MaxDistanceError(
                f"No point meets the max distance requirement. "
                f"Recommended points have been generated in {path}",
                newpt, path,
            )
        return None

    def divide_task(self):
        self.__reqtaskindex = self.__task[
            self.__task["required"] == True
        ].index.to_list()
        self.__opttaskindex = self.__task[
            self.__task["required"] == False
        ].index.to_list()
        self.__daytaskindex = [
            self.__task[self.__task["day"] == 1].index.to_list(),
            self.__task[self.__task["day"] == 2].index.to_list(),
            self.__task[self.__task["day"] == 3].index.to_list(),
            self.__task[self.__task["day"] == "1,2"].index.to_list(),
            self.__task[self.__task["day"] == "2,3"].index.to_list(),
        ]
        self.__remtaskindex = self.__task[
            self.__task["remote"] == True
        ].index.to_list()
        self.__tagtaskindex = [
            self.__task[self.__task["tag"] == "12s"].index.to_list(),
            self.__task[self.__task["tag"] == "12e"].index.to_list(),
            self.__task[self.__task["tag"] == "23s"].index.to_list(),
            self.__task[self.__task["tag"] == "23e"].index.to_list(),
        ]
        self.__contaskindex = self.__task[
            self.__task["continuous"] == True
        ].index.to_list()
        self.__noOtaskindex = self.__task[
            self.__task["exceptO"] == True
        ].index.to_list()
        return None

    def drop_O(self):
        for i in self.__noOtaskindex:
            for vn in ["探测起点1", "探测起点2", "探测起点3", "探测起点4"]:
                if vn in self.__task.loc[i, "location"]:
                    self.__task.loc[i, "location"].remove(vn)
        return None

    def gen_point(self):
        self.__opoint = (
            (self.__pointdf.loc["探测起点1", "No"],
             self.__task[self.__task["name"] == "void1"].index[0]),
            (self.__pointdf.loc["探测起点2", "No"],
             self.__task[self.__task["name"] == "void2"].index[0]),
            (self.__pointdf.loc["探测起点3", "No"],
             self.__task[self.__task["name"] == "void3"].index[0]),
            (self.__pointdf.loc["探测起点4", "No"],
             self.__task[self.__task["name"] == "void4"].index[0]),
        )
        self.__point = []
        for t in self.__task.index:
            loc = self.__task.loc[t, "location"]
            if isinstance(loc, str):
                i = self.__pointdf.loc[loc, "No"]
                self.__point.append((i, t))
            elif isinstance(loc, list):
                for name in loc:
                    i = self.__pointdf.loc[name, "No"]
                    self.__point.append((i, t))
        return self.__point

    def gen_edges(self):
        pass  # Not needed for PSO – edges are implicit in the route

    # -----------------------------------------------------------------------
    # Pre-compute lookup tables for fast evaluation inside PSO
    # -----------------------------------------------------------------------
    def _build_lookup(self):
        """Build data structures for fast route evaluation."""
        pdf = self.__pointdf.copy()
        pdf.reset_index(inplace=True)
        pdf.set_index("No", inplace=True)
        self.__pdf_byno = pdf

        tdf = self.__task.copy()
        tdf.reset_index(inplace=True)
        tdf.set_index("No", inplace=True)
        self.__tdf_byno = tdf

        self.__ptname_from_no = {}
        for no in pdf.index:
            self.__ptname_from_no[no] = pdf.loc[no, "index"]

        all_names = list(self.__tmatrix.index)
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        n = len(all_names)
        self.__tt_np = np.zeros((n, n))
        self.__tp_np = np.zeros((n, n))
        for i_name in all_names:
            for j_name in all_names:
                ii, jj = name_to_idx[i_name], name_to_idx[j_name]
                self.__tt_np[ii, jj] = self.__tmatrix.loc[i_name, j_name]
                self.__tp_np[ii, jj] = self.__pmatrix.loc[i_name, j_name]
        self.__name_to_matidx = name_to_idx

        self.__depot_nos = [self.__opoint[d][0] for d in range(4)]
        self.__depot_tasks = [self.__opoint[d][1] for d in range(4)]

        # Per-task attributes (excluding void tasks)
        self.__task_indices = []
        self.__task_time = {}
        self.__task_power = {}
        self.__task_revenue = {}
        self.__task_locs = {}
        self.__task_required = {}
        self.__task_remote = {}
        self.__task_continuous = {}
        self.__task_day = {}
        self.__task_tag = {}

        void_names = {"void1", "void2", "void3", "void4"}
        for t in self.__task.index:
            name = self.__task.loc[t, "name"]
            if name in void_names:
                continue
            self.__task_indices.append(t)
            self.__task_time[t] = float(self.__task.loc[t, "time"])
            self.__task_power[t] = float(self.__task.loc[t, "power"])
            self.__task_revenue[t] = float(self.__task.loc[t, "revenue"])
            self.__task_required[t] = bool(self.__task.loc[t, "required"])
            self.__task_remote[t] = bool(self.__task.loc[t, "remote"])
            self.__task_continuous[t] = bool(self.__task.loc[t, "continuous"])
            self.__task_day[t] = self.__task.loc[t, "day"]
            tag = self.__task.loc[t, "tag"]
            self.__task_tag[t] = tag if isinstance(tag, str) else None
            locs = self.__task.loc[t, "location"]
            if isinstance(locs, str):
                locs = [locs]
            self.__task_locs[t] = [self.__pointdf.loc[ln, "No"] for ln in locs]

        self.__real_task_set = set(self.__task_indices)

    # -----------------------------------------------------------------------
    # Travel cost helpers
    # -----------------------------------------------------------------------
    def _travel_time(self, pt_no_a, pt_no_b):
        na = self.__ptname_from_no[pt_no_a]
        nb = self.__ptname_from_no[pt_no_b]
        return self.__tt_np[self.__name_to_matidx[na], self.__name_to_matidx[nb]]

    def _travel_power(self, pt_no_a, pt_no_b):
        na = self.__ptname_from_no[pt_no_a]
        nb = self.__ptname_from_no[pt_no_b]
        return self.__tp_np[self.__name_to_matidx[na], self.__name_to_matidx[nb]]

    # -----------------------------------------------------------------------
    # Solution representation
    # -----------------------------------------------------------------------
    # A solution (decoded particle) is a dict:
    #   "selected": set of task indices that are selected
    #   "days": [[task_idx, ...], [...], [...]]  – ordered visit per day
    #   "loc":  {task_idx: point_No}  – chosen location for each task

    def _initial_solution(self):
        """Generate one random feasible (or near-feasible) solution."""
        rng = self.__rng
        selected = set()
        loc_choice = {}

        # 1. Always include required tasks
        for t in self.__reqtaskindex:
            if t in self.__real_task_set:
                selected.add(t)
                locs = self.__task_locs[t]
                loc_choice[t] = locs[rng.integers(len(locs))]

        # 2. Continuous tasks: select exactly Mincontinuous
        cont_real = [t for t in self.__contaskindex if t in self.__real_task_set]
        if cont_real:
            n_cont = min(int(self.__Mincontinuous), len(cont_real))
            chosen_cont = list(rng.choice(cont_real, size=n_cont, replace=False))
            for t in chosen_cont:
                selected.add(t)
                if t not in loc_choice:
                    locs = self.__task_locs[t]
                    loc_choice[t] = locs[rng.integers(len(locs))]

        # 3. Remote tasks: select exactly one
        rem_real = [t for t in self.__remtaskindex if t in self.__real_task_set]
        if rem_real:
            rt = rem_real[rng.integers(len(rem_real))]
            selected.add(rt)
            if rt not in loc_choice:
                locs = self.__task_locs[rt]
                loc_choice[rt] = locs[rng.integers(len(locs))]

        # 4. Randomly include some optional tasks
        opt_real = [t for t in self.__opttaskindex
                    if t in self.__real_task_set and t not in selected]
        if opt_real:
            n_opt = rng.integers(0, len(opt_real) + 1)
            for t in rng.choice(opt_real, size=n_opt, replace=False):
                selected.add(t)
                if t not in loc_choice:
                    locs = self.__task_locs[t]
                    loc_choice[t] = locs[rng.integers(len(locs))]

        # 5. Assign tasks to days
        days = [[], [], []]
        for t in selected:
            vd = self._get_valid_days(t)
            days[vd[rng.integers(len(vd))]].append(t)

        # shuffle order within each day
        for d in range(3):
            rng.shuffle(days[d])

        sol = {"selected": selected, "days": days, "loc": loc_choice}
        sol = self._repair(sol)
        return sol

    # -----------------------------------------------------------------------
    # Repair operator – enforce all constraints
    # -----------------------------------------------------------------------
    def _repair(self, sol):
        """Repair a solution to satisfy constraints. Modifies in-place."""
        rng = self.__rng
        selected = sol["selected"]
        days = sol["days"]
        loc = sol["loc"]

        # --- Ensure required tasks present ---
        for t in self.__reqtaskindex:
            if t in self.__real_task_set and t not in selected:
                selected.add(t)
                locs = self.__task_locs[t]
                loc[t] = locs[rng.integers(len(locs))]

        # --- Remote constraint: exactly one ---
        rem_real = [t for t in self.__remtaskindex if t in self.__real_task_set]
        if rem_real:
            rem_sel = [t for t in rem_real if t in selected]
            if len(rem_sel) == 0:
                rt = rem_real[rng.integers(len(rem_real))]
                selected.add(rt)
                locs = self.__task_locs[rt]
                loc[rt] = locs[rng.integers(len(locs))]
            elif len(rem_sel) > 1:
                keep = rem_sel[rng.integers(len(rem_sel))]
                for rt in rem_sel:
                    if rt != keep and not self.__task_required.get(rt, False):
                        selected.discard(rt)

        # --- Continuous constraint: exactly Mincontinuous ---
        cont_real = [t for t in self.__contaskindex if t in self.__real_task_set]
        if cont_real:
            n_need = int(self.__Mincontinuous)
            cont_sel = [t for t in cont_real if t in selected]
            while len(cont_sel) < n_need:
                pool = [t for t in cont_real if t not in selected]
                if not pool:
                    break
                t = pool[rng.integers(len(pool))]
                selected.add(t)
                locs = self.__task_locs[t]
                loc[t] = locs[rng.integers(len(locs))]
                cont_sel.append(t)
            while len(cont_sel) > n_need:
                removable = [t for t in cont_sel
                             if not self.__task_required.get(t, False)]
                if not removable:
                    break
                t = removable[rng.integers(len(removable))]
                selected.discard(t)
                cont_sel.remove(t)

        # --- Ensure location choice is valid ---
        for t in list(selected):
            if t not in loc or loc[t] not in self.__task_locs[t]:
                locs = self.__task_locs[t]
                if locs:
                    loc[t] = locs[rng.integers(len(locs))]
                else:
                    selected.discard(t)

        # --- Ensure every selected task in exactly one day ---
        in_day = {}
        for d in range(3):
            new_list = []
            for t in days[d]:
                if t in selected and t not in in_day:
                    in_day[t] = d
                    new_list.append(t)
            days[d] = new_list
        for t in selected:
            if t not in in_day:
                vd = self._get_valid_days(t)
                d = vd[rng.integers(len(vd))]
                days[d].append(t)
                in_day[t] = d

        # --- Day validity ---
        for d in range(3):
            new_list = []
            for t in days[d]:
                valid_days = self._get_valid_days(t)
                if d in valid_days:
                    new_list.append(t)
                else:
                    nd = valid_days[rng.integers(len(valid_days))]
                    days[nd].append(t)
            days[d] = new_list

        # --- Tag constraints ---
        self._repair_tags(sol)

        # --- Time and power feasibility per day (greedy removal) ---
        self._repair_time_power(sol)

        sol["selected"] = selected
        sol["days"] = days
        sol["loc"] = loc
        return sol

    def _get_valid_days(self, t):
        """Return list of valid day indices (0,1,2) for task t."""
        day_val = self.__task_day.get(t)
        tag_val = self.__task_tag.get(t)
        if day_val == 1:
            return [0]
        elif day_val == 2:
            return [1]
        elif day_val == 3:
            return [2]
        elif day_val == "1,2":
            return [0, 1]
        elif day_val == "2,3":
            return [1, 2]
        elif tag_val in ("12s", "12e"):
            return [0, 1]
        elif tag_val in ("23s", "23e"):
            return [1, 2]
        return [0, 1, 2]

    def _repair_tags(self, sol):
        """Enforce tag constraints."""
        rng = self.__rng
        days = sol["days"]
        selected = sol["selected"]

        tag12s = [t for t in self.__tagtaskindex[0] if t in selected]
        tag12e = [t for t in self.__tagtaskindex[1] if t in selected]
        tag23s = [t for t in self.__tagtaskindex[2] if t in selected]
        tag23e = [t for t in self.__tagtaskindex[3] if t in selected]

        # 12s must be first in its day (right after depot start)
        for t in tag12s:
            for d in range(3):
                if t in days[d]:
                    days[d].remove(t)
                    days[d].insert(0, t)

        # 12e must be last in its day
        for t in tag12e:
            for d in range(3):
                if t in days[d]:
                    days[d].remove(t)
                    days[d].append(t)

        # 23s must be first in its day
        for t in tag23s:
            for d in range(3):
                if t in days[d]:
                    days[d].remove(t)
                    days[d].insert(0, t)

        # 23e must be last in its day
        for t in tag23e:
            for d in range(3):
                if t in days[d]:
                    days[d].remove(t)
                    days[d].append(t)

        # sameday: 12s and 12e on the same day
        if tag12s and tag12e:
            for ts in tag12s:
                for d in range(3):
                    if ts in days[d]:
                        for te in tag12e:
                            for d2 in range(3):
                                if te in days[d2] and d2 != d:
                                    days[d2].remove(te)
                                    days[d].append(te)

        # sameday: 23s and 23e on the same day
        if tag23s and tag23e:
            for ts in tag23s:
                for d in range(3):
                    if ts in days[d]:
                        for te in tag23e:
                            for d2 in range(3):
                                if te in days[d2] and d2 != d:
                                    days[d2].remove(te)
                                    days[d].append(te)

        # startnotsameday: 12s and 23s not on same day
        if tag12s and tag23s:
            for ts12 in tag12s:
                for ts23 in tag23s:
                    for d in range(3):
                        if ts12 in days[d] and ts23 in days[d]:
                            vd = [dd for dd in self._get_valid_days(ts23) if dd != d]
                            if vd:
                                days[d].remove(ts23)
                                days[vd[rng.integers(len(vd))]].append(ts23)

        # endnotsameday: 12e and 23e not on same day
        if tag12e and tag23e:
            for te12 in tag12e:
                for te23 in tag23e:
                    for d in range(3):
                        if te12 in days[d] and te23 in days[d]:
                            vd = [dd for dd in self._get_valid_days(te23) if dd != d]
                            if vd:
                                days[d].remove(te23)
                                days[vd[rng.integers(len(vd))]].append(te23)

    def _repair_time_power(self, sol):
        """Remove tasks greedily if a day violates time/power budgets,
        respecting ea.py's per-step safety constraints."""
        safe_depot = self.__opoint[0][0]

        for d in range(3):
            while True:
                route = sol["days"][d]
                if not route:
                    break
                cum_t, cum_p = 0.0, 0.0
                cur_pt = self.__opoint[d][0]
                end_pt = self.__opoint[d + 1][0]
                feasible = True
                violator = None
                for idx, t in enumerate(route):
                    pt = sol["loc"][t]
                    cum_t += self._travel_time(cur_pt, pt) + self.__task_time[t]
                    cum_p += self._travel_power(cur_pt, pt) + self.__task_power[t]
                    ret_t = self._travel_time(pt, safe_depot)
                    ret_p = self._travel_power(safe_depot, pt)
                    if (cum_t + ret_t > self.__TTime[d]
                            or cum_p + ret_p > self.__TPower[d]):
                        feasible = False
                        violator = idx
                        break
                    cur_pt = pt
                if feasible and route:
                    last_pt = sol["loc"][route[-1]]
                    final_t = cum_t + self._travel_time(last_pt, end_pt)
                    final_p = cum_p + self._travel_power(last_pt, end_pt)
                    if final_t > self.__TTime[d] or final_p > self.__TPower[d]:
                        feasible = False
                        violator = len(route) - 1
                if feasible:
                    break
                if violator is not None:
                    t_rem = route[violator]
                    if not self.__task_required.get(t_rem, False):
                        route.pop(violator)
                        sol["selected"].discard(t_rem)
                    else:
                        removed = False
                        for ri in range(len(route) - 1, -1, -1):
                            tt = route[ri]
                            if not self.__task_required.get(tt, False):
                                route.pop(ri)
                                sol["selected"].discard(tt)
                                removed = True
                                break
                        if not removed:
                            break

    # -----------------------------------------------------------------------
    # Objective evaluation
    # -----------------------------------------------------------------------
    def _evaluate(self, sol):
        """Return objective value (higher is better for all modes)."""
        if not self._is_feasible(sol):
            return -1e18

        if self.__objective == CONST.MAX_REVENUE:
            return sum(self.__task_revenue[t] for t in sol["selected"])

        _, total_t, total_p = self._calc_solution_metrics(sol)

        if self.__objective == CONST.MIN_TIME:
            return -total_t
        return -total_p

    def _calc_solution_metrics(self, sol):
        total_revenue = sum(self.__task_revenue[t] for t in sol["selected"])
        total_t, total_p = 0.0, 0.0
        for d in range(3):
            cur_pt = self.__opoint[d][0]
            end_pt = self.__opoint[d + 1][0]
            for t in sol["days"][d]:
                pt = sol["loc"][t]
                total_t += self._travel_time(cur_pt, pt) + self.__task_time[t]
                total_p += self._travel_power(cur_pt, pt) + self.__task_power[t]
                cur_pt = pt
            total_t += self._travel_time(cur_pt, end_pt)
            total_p += self._travel_power(cur_pt, end_pt)
        return total_revenue, total_t, total_p

    def _calc_efficiencies(self, sol):
        total_time_budget = sum(self.__TTime) + sum(self.__PKGTime)
        total_power_budget = sum(self.__TPower) + sum(self.__PKGPower)
        _, total_time, total_power = self._calc_solution_metrics(sol)
        time_efficiency = (
            (total_time + sum(self.__PKGTime)) / total_time_budget * 100
            if total_time_budget > 0 else 0.0
        )
        power_efficiency = (
            (total_power + sum(self.__PKGPower)) / total_power_budget * 100
            if total_power_budget > 0 else 0.0
        )
        return time_efficiency, power_efficiency

    def _tag_expected_location(self, tag, day_idx):
        """Return the virtual depot point No required by ea.py tag constraints."""
        if tag == "12s" and day_idx in (0, 1):
            return self.__opoint[day_idx][0]
        if tag == "12e" and day_idx in (0, 1):
            return self.__opoint[day_idx + 1][0]
        if tag == "23s" and day_idx in (1, 2):
            return self.__opoint[day_idx][0]
        if tag == "23e" and day_idx in (1, 2):
            return self.__opoint[day_idx + 1][0]
        return None

    def _task_day_in_solution(self, sol, task_idx):
        for d in range(3):
            if task_idx in sol["days"][d]:
                return d
        return None

    def _is_tag_feasible(self, sol):
        """Validate ea.py's boundary tag constraints, not merely day windows."""
        selected, days, loc = sol["selected"], sol["days"], sol["loc"]
        tag_groups = self.__tagtaskindex

        # In ea.py, each tag equation has RHS == 1, so a tagged task is forced.
        for group in tag_groups:
            for t in group:
                if t in self.__real_task_set and t not in selected:
                    return False

        day_of = {}
        for d in range(3):
            for pos, t in enumerate(days[d]):
                tag = self.__task_tag.get(t)
                if tag is None:
                    continue
                day_of[t] = d
                expected_pt = self._tag_expected_location(tag, d)
                if expected_pt is None or loc.get(t) != expected_pt:
                    return False
                if tag in ("12s", "23s") and pos != 0:
                    return False
                if tag in ("12e", "23e") and pos != len(days[d]) - 1:
                    return False

        # Multiple starts or multiple ends at the same day boundary would violate
        # the virtual depot degree equations in ea.py.
        for d in range(3):
            starts = [t for t in days[d] if self.__task_tag.get(t) in ("12s", "23s")]
            ends = [t for t in days[d] if self.__task_tag.get(t) in ("12e", "23e")]
            if len(starts) > 1 or len(ends) > 1:
                return False

        tag12s = [t for t in tag_groups[0] if t in selected]
        tag12e = [t for t in tag_groups[1] if t in selected]
        tag23s = [t for t in tag_groups[2] if t in selected]
        tag23e = [t for t in tag_groups[3] if t in selected]

        if bool(tag12s) != bool(tag12e):
            return False
        if bool(tag23s) != bool(tag23e):
            return False
        if tag12s and tag12e:
            d12s = {day_of.get(t) for t in tag12s}
            d12e = {day_of.get(t) for t in tag12e}
            if len(d12s) != 1 or len(d12e) != 1 or d12s != d12e:
                return False
        if tag23s and tag23e:
            d23s = {day_of.get(t) for t in tag23s}
            d23e = {day_of.get(t) for t in tag23e}
            if len(d23s) != 1 or len(d23e) != 1 or d23s != d23e:
                return False
        if tag12s and tag23s:
            if next(iter({day_of.get(t) for t in tag12s})) == 1 and next(iter({day_of.get(t) for t in tag23s})) == 1:
                return False
        if tag12e and tag23e:
            if next(iter({day_of.get(t) for t in tag12e})) == 1 and next(iter({day_of.get(t) for t in tag23e})) == 1:
                return False
        return True

    def _is_feasible(self, sol):
        """Strict feasibility check against the ea.py model semantics."""
        selected = sol["selected"]
        days = sol["days"]
        loc = sol["loc"]

        for t in selected:
            if t not in self.__real_task_set or t not in loc:
                return False
            if loc[t] not in self.__task_locs[t]:
                return False

        for t in self.__reqtaskindex:
            if t in self.__real_task_set and t not in selected:
                return False

        rem_real = [t for t in self.__remtaskindex if t in self.__real_task_set]
        if rem_real and sum(1 for t in rem_real if t in selected) != 1:
            return False

        cont_real = [t for t in self.__contaskindex if t in self.__real_task_set]
        if cont_real and sum(1 for t in cont_real if t in selected) != int(self.__Mincontinuous):
            return False

        all_in_day = []
        for d in range(3):
            seen_day = set()
            for t in days[d]:
                if t in seen_day:
                    return False
                seen_day.add(t)
                if t not in selected:
                    return False
                if d not in self._get_valid_days(t):
                    return False
                all_in_day.append(t)
        if set(all_in_day) != selected or len(all_in_day) != len(selected):
            return False

        if not self._is_tag_feasible(sol):
            return False

        # Time/power propagation and safety constraints. Time safety follows
        # current -> depot; power safety follows ea.py's depot -> current form.
        for d in range(3):
            cum_t, cum_p = 0.0, 0.0
            cur_pt = self.__opoint[d][0]
            end_pt = self.__opoint[d + 1][0]
            for t in days[d]:
                pt = loc[t]
                cum_t += self._travel_time(cur_pt, pt) + self.__task_time[t]
                cum_p += self._travel_power(cur_pt, pt) + self.__task_power[t]
                safe_t = self._travel_time(pt, self.__opoint[0][0])
                safe_p = self._travel_power(self.__opoint[0][0], pt)
                if cum_t + safe_t > self.__TTime[d] + 1e-6:
                    return False
                if cum_p + safe_p > self.__TPower[d] + 1e-6:
                    return False
                cur_pt = pt
            cum_t += self._travel_time(cur_pt, end_pt)
            cum_p += self._travel_power(cur_pt, end_pt)
            if cum_t > self.__TTime[d] + 1e-6 or cum_p > self.__TPower[d] + 1e-6:
                return False
        return True

    # -----------------------------------------------------------------------
    # PSO operators (discrete / combinatorial)
    # -----------------------------------------------------------------------
    def _crossover(self, sol_a, sol_b):
        """Create a child by mixing two solutions."""
        rng = self.__rng
        child_sel = set()
        child_loc = {}
        child_days = [[], [], []]

        all_tasks = sol_a["selected"] | sol_b["selected"]
        for t in all_tasks:
            in_a = t in sol_a["selected"]
            in_b = t in sol_b["selected"]
            if in_a and in_b:
                child_sel.add(t)
                child_loc[t] = sol_a["loc"][t] if rng.random() < 0.5 else sol_b["loc"][t]
            elif in_a and rng.random() < 0.5:
                child_sel.add(t)
                child_loc[t] = sol_a["loc"][t]
            elif in_b and rng.random() < 0.5:
                child_sel.add(t)
                child_loc[t] = sol_b["loc"][t]

        # Day assignment from parents
        for t in child_sel:
            placed = False
            for parent in (sol_a, sol_b):
                for d in range(3):
                    if t in parent["days"][d]:
                        child_days[d].append(t)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                vd = self._get_valid_days(t)
                child_days[vd[rng.integers(len(vd))]].append(t)

        return self._repair({"selected": child_sel, "days": child_days, "loc": child_loc})

    def _mutate(self, sol, intensity=0.3):
        """Mutate: toggle tasks, swap order, change locations, move tasks."""
        rng = self.__rng
        sol = copy.deepcopy(sol)

        # Toggle optional tasks
        opt_real = [t for t in self.__opttaskindex if t in self.__real_task_set]
        for t in opt_real:
            if rng.random() < intensity * 0.3:
                if t in sol["selected"]:
                    if not self.__task_required.get(t, False):
                        sol["selected"].discard(t)
                        for d in range(3):
                            if t in sol["days"][d]:
                                sol["days"][d].remove(t)
                else:
                    sol["selected"].add(t)
                    locs = self.__task_locs[t]
                    sol["loc"][t] = locs[rng.integers(len(locs))]
                    vd = self._get_valid_days(t)
                    sol["days"][vd[rng.integers(len(vd))]].append(t)

        # Swap within day
        for d in range(3):
            if len(sol["days"][d]) >= 2 and rng.random() < intensity:
                i1, i2 = rng.choice(len(sol["days"][d]), size=2, replace=False)
                sol["days"][d][i1], sol["days"][d][i2] = (
                    sol["days"][d][i2], sol["days"][d][i1])

        # Change location
        for t in list(sol["selected"]):
            if rng.random() < intensity * 0.2:
                locs = self.__task_locs[t]
                if len(locs) > 1:
                    sol["loc"][t] = locs[rng.integers(len(locs))]

        # Move between days
        all_sel = list(sol["selected"])
        if all_sel and rng.random() < intensity * 0.5:
            t = all_sel[rng.integers(len(all_sel))]
            vd = self._get_valid_days(t)
            if len(vd) > 1:
                cur_d = None
                for d in range(3):
                    if t in sol["days"][d]:
                        cur_d = d
                        break
                new_opts = [dd for dd in vd if dd != cur_d]
                if new_opts and cur_d is not None:
                    sol["days"][cur_d].remove(t)
                    sol["days"][new_opts[rng.integers(len(new_opts))]].append(t)

        return self._repair(sol)

    def _local_search(self, sol):
        """2-opt local search within each day."""
        improved = True
        best_obj = self._evaluate(sol)
        while improved:
            improved = False
            for d in range(3):
                route = sol["days"][d]
                n = len(route)
                if n < 3:
                    continue
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        old_route = sol["days"][d]
                        sol["days"][d] = new_route
                        new_obj = self._evaluate(sol)
                        if new_obj > best_obj:
                            best_obj = new_obj
                            improved = True
                        else:
                            sol["days"][d] = old_route
        return sol


    # -----------------------------------------------------------------------
    # Canonical continuous PSO with random-key decoding
    # -----------------------------------------------------------------------
    def _sigmoid(self, x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _init_pso_space(self):
        self.__pso_tasks = list(self.__task_indices)
        self.__pso_index = {t: i for i, t in enumerate(self.__pso_tasks)}
        self.__tag_forced = set(
            t for group in self.__tagtaskindex for t in group
            if t in self.__real_task_set
        )
        req_remote = [t for t in self.__remtaskindex
                      if t in self.__real_task_set and self.__task_required.get(t, False)]
        if len(req_remote) > 1:
            raise MIPError("Model is infeasible: more than one required remote task")
        req_cont = [t for t in self.__contaskindex
                    if t in self.__real_task_set and self.__task_required.get(t, False)]
        if len(req_cont) > int(self.__Mincontinuous):
            raise MIPError("Model is infeasible: required continuous tasks exceed min-continuous")
        for t in self.__tag_forced:
            if not self.__task_required.get(t, False):
                # ea.py still forces these tasks. Do not silently rely on data assumptions.
                self.__task_required[t] = True

    def _new_particle(self):
        n = len(self.__pso_tasks)
        pos = {
            "select": self.__rng.normal(0.0, 1.0, n),
            "day": self.__rng.normal(0.0, 1.0, n),
            "loc": self.__rng.normal(0.0, 1.0, n),
            "order": self.__rng.normal(0.0, 1.0, n),
        }
        vel = {k: self.__rng.normal(0.0, 0.2, n) for k in pos}
        return {"pos": pos, "vel": vel}

    def _copy_particle_position(self, pos):
        return {k: np.array(v, copy=True) for k, v in pos.items()}

    def _choose_by_key(self, candidates, key_array, reverse=True):
        if not candidates:
            return []
        return sorted(
            candidates,
            key=lambda t: key_array[self.__pso_index[t]],
            reverse=reverse,
        )

    def _decode_particle(self, pos):
        """Decode continuous random keys to a schedule, then repair it."""
        selected = set()
        loc = {}

        select_score = self._sigmoid(pos["select"])
        day_score = self._sigmoid(pos["day"])
        loc_score = self._sigmoid(pos["loc"])

        # Required tasks, including ea.py-forced tag tasks.
        for t in self.__pso_tasks:
            if self.__task_required.get(t, False) or t in self.__tag_forced:
                selected.add(t)

        # Optional selection random key.
        for t in self.__pso_tasks:
            if t in selected:
                continue
            if select_score[self.__pso_index[t]] >= 0.5:
                selected.add(t)

        # Remote exactly-one. Required remote wins; otherwise choose highest key.
        rem_real = [t for t in self.__remtaskindex if t in self.__real_task_set]
        if rem_real:
            required_rem = [t for t in rem_real if self.__task_required.get(t, False)]
            if len(required_rem) > 1:
                selected.update(required_rem)
            elif len(required_rem) == 1:
                keep = required_rem[0]
                selected.add(keep)
                for t in rem_real:
                    if t != keep:
                        selected.discard(t)
            else:
                keep = self._choose_by_key(rem_real, pos["select"], reverse=True)[0]
                selected.add(keep)
                for t in rem_real:
                    if t != keep:
                        selected.discard(t)

        # Continuous exact-cardinality. Required continuous tasks are fixed.
        cont_real = [t for t in self.__contaskindex if t in self.__real_task_set]
        if cont_real:
            n_need = int(self.__Mincontinuous)
            req_cont = [t for t in cont_real if self.__task_required.get(t, False)]
            chosen = list(req_cont)
            if len(chosen) < n_need:
                pool = [t for t in cont_real if t not in chosen]
                chosen.extend(self._choose_by_key(pool, pos["select"], reverse=True)[:n_need - len(chosen)])
            chosen_set = set(chosen)
            selected.update(chosen_set)
            for t in cont_real:
                if t not in chosen_set and not self.__task_required.get(t, False):
                    selected.discard(t)

        # Location random key.
        for t in list(selected):
            locs = self.__task_locs.get(t, [])
            if not locs:
                selected.discard(t)
                continue
            raw = loc_score[self.__pso_index[t]]
            idx = min(int(raw * len(locs)), len(locs) - 1)
            loc[t] = locs[idx]

        # Day random key and order random key.
        days = [[], [], []]
        for t in selected:
            valid = self._get_valid_days(t)
            raw = day_score[self.__pso_index[t]]
            d = valid[min(int(raw * len(valid)), len(valid) - 1)]
            days[d].append(t)
        for d in range(3):
            days[d].sort(key=lambda t: pos["order"][self.__pso_index[t]])

        sol = {"selected": selected, "days": days, "loc": loc}
        return self._repair_decoded_solution(sol, pos)

    def _set_task_day(self, sol, task_idx, day_idx):
        for d in range(3):
            if task_idx in sol["days"][d]:
                sol["days"][d].remove(task_idx)
        sol["days"][day_idx].append(task_idx)

    def _repair_decoded_solution(self, sol, pos=None):
        """Deterministic repair aligned with ea.py constraints."""
        selected, days, loc = sol["selected"], sol["days"], sol["loc"]

        # Required/tag-forced tasks.
        for t in self.__pso_tasks:
            if self.__task_required.get(t, False) or t in self.__tag_forced:
                selected.add(t)
                if t not in loc and self.__task_locs[t]:
                    loc[t] = self.__task_locs[t][0]

        # Ensure valid locations for all selected tasks.
        for t in list(selected):
            if t not in loc or loc[t] not in self.__task_locs[t]:
                if self.__task_locs[t]:
                    loc[t] = self.__task_locs[t][0]
                else:
                    selected.discard(t)

        # Normalize day membership.
        new_days = [[], [], []]
        already = set()
        for d in range(3):
            for t in days[d]:
                if t in selected and t not in already:
                    vd = self._get_valid_days(t)
                    nd = d if d in vd else vd[0]
                    new_days[nd].append(t)
                    already.add(t)
        for t in selected:
            if t not in already:
                vd = self._get_valid_days(t)
                new_days[vd[0]].append(t)
        sol["days"] = new_days
        days = sol["days"]

        # Re-apply remote and continuous cardinalities after day normalization.
        # These operations only remove optional tasks; required conflicts remain invalid.
        rem_real = [t for t in self.__remtaskindex if t in self.__real_task_set]
        if rem_real:
            required_rem = [t for t in rem_real if self.__task_required.get(t, False)]
            if len(required_rem) <= 1:
                if required_rem:
                    keep = required_rem[0]
                else:
                    pool = [t for t in rem_real if t in selected] or rem_real
                    key = pos["select"] if pos is not None else np.zeros(len(self.__pso_tasks))
                    keep = self._choose_by_key(pool, key, reverse=True)[0]
                selected.add(keep)
                for t in rem_real:
                    if t != keep and not self.__task_required.get(t, False):
                        selected.discard(t)
                        for d in range(3):
                            if t in days[d]:
                                days[d].remove(t)

        cont_real = [t for t in self.__contaskindex if t in self.__real_task_set]
        if cont_real:
            n_need = int(self.__Mincontinuous)
            req_cont = [t for t in cont_real if self.__task_required.get(t, False)]
            chosen = list(req_cont)
            key = pos["select"] if pos is not None else np.zeros(len(self.__pso_tasks))
            if len(chosen) < n_need:
                pool = [t for t in cont_real if t not in chosen]
                chosen.extend(self._choose_by_key(pool, key, reverse=True)[:n_need - len(chosen)])
            chosen_set = set(chosen)
            selected.update(chosen_set)
            for t in chosen_set:
                if all(t not in days[d] for d in range(3)):
                    days[self._get_valid_days(t)[0]].append(t)
            for t in cont_real:
                if t not in chosen_set and not self.__task_required.get(t, False):
                    selected.discard(t)
                    for d in range(3):
                        if t in days[d]:
                            days[d].remove(t)

        self._repair_boundary_tags(sol, pos)
        self._sort_days_by_particle_order(sol, pos)
        self._repair_time_power(sol)
        # Boundary tags may have been displaced by time/power repair; restore them.
        self._repair_boundary_tags(sol, pos)
        self._sort_days_by_particle_order(sol, pos)
        return sol

    def _sort_days_by_particle_order(self, sol, pos=None):
        for d in range(3):
            if pos is not None:
                sol["days"][d].sort(key=lambda t: pos["order"][self.__pso_index[t]])
            starts = [t for t in sol["days"][d] if self.__task_tag.get(t) in ("12s", "23s")]
            ends = [t for t in sol["days"][d] if self.__task_tag.get(t) in ("12e", "23e")]
            middle = [t for t in sol["days"][d] if t not in starts and t not in ends]
            sol["days"][d] = starts + middle + ends

    def _repair_boundary_tags(self, sol, pos=None):
        """Place tag tasks on the exact virtual boundary points required by ea.py."""
        selected, loc = sol["selected"], sol["loc"]
        tag12s = [t for t in self.__tagtaskindex[0] if t in self.__real_task_set]
        tag12e = [t for t in self.__tagtaskindex[1] if t in self.__real_task_set]
        tag23s = [t for t in self.__tagtaskindex[2] if t in self.__real_task_set]
        tag23e = [t for t in self.__tagtaskindex[3] if t in self.__real_task_set]

        for t in tag12s + tag12e + tag23s + tag23e:
            selected.add(t)

        def choose_pair_day(starts, ends, allowed):
            tasks = starts + ends
            if not tasks:
                return None
            if pos is None:
                return allowed[0]
            vals = [self._sigmoid(pos["day"][self.__pso_index[t]]) for t in tasks]
            raw = float(np.mean(vals))
            return allowed[min(int(raw * len(allowed)), len(allowed) - 1)]

        d12 = choose_pair_day(tag12s, tag12e, [0, 1])
        d23 = choose_pair_day(tag23s, tag23e, [1, 2])
        # ea.py's not-same-day constraints prohibit both transition packages on day 2.
        if d12 == 1 and d23 == 1:
            # Move the 23 pair to day 3 unless particle strongly prefers moving 12 to day 1.
            d23 = 2

        for t in tag12s + tag12e:
            if d12 is not None:
                self._set_task_day(sol, t, d12)
                tag = self.__task_tag[t]
                expected = self._tag_expected_location(tag, d12)
                if expected in self.__task_locs[t]:
                    loc[t] = expected
        for t in tag23s + tag23e:
            if d23 is not None:
                self._set_task_day(sol, t, d23)
                tag = self.__task_tag[t]
                expected = self._tag_expected_location(tag, d23)
                if expected in self.__task_locs[t]:
                    loc[t] = expected

    def _fitness_from_position(self, pos):
        sol = self._decode_particle(pos)
        fit = self._evaluate(sol)
        return fit, sol

    def _save_incumbent(self, sol):
        if not self.__autosavestate:
            return
        try:
            self.__best_sol = sol
            self._build_res_from_sol(sol)
            self.cal_route()
            self.add_package()
            p = self.__outputpath
            if p.endswith(".xlsx"):
                p = p[:-5] + f"-sol{self.__solcount}.xlsx"
            elif p.endswith(".xls"):
                p = p[:-4] + f"-sol{self.__solcount}.xls"
            p = os.path.join("autosave", p)
            self.write_excel(p)
            self.__solcount += 1
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # PSO main loop
    # -----------------------------------------------------------------------
    def run_opt(self):
        """Canonical PSO loop over continuous random-key particles."""
        import time as _time

        self._init_pso_space()
        n_pop = self.__n_particles
        max_iter = self.__max_iter
        rng = self.__rng
        start_time = _time.time()
        time_limit = self.__timeLimit if self.__timeLimit != np.inf else 1e18
        vmax = 4.0

        swarm = [self._new_particle() for _ in range(n_pop)]
        fitness, sols = [], []
        for particle in swarm:
            fit, sol = self._fitness_from_position(particle["pos"])
            fitness.append(fit)
            sols.append(sol)

        pbest_pos = [self._copy_particle_position(p["pos"]) for p in swarm]
        pbest_fit = fitness[:]
        pbest_sol = [copy.deepcopy(s) for s in sols]

        best_idx = int(np.argmax(fitness))
        gbest_pos = self._copy_particle_position(swarm[best_idx]["pos"])
        gbest_fit = fitness[best_idx]
        gbest_sol = copy.deepcopy(sols[best_idx])
        init_rev, _, _ = self._calc_solution_metrics(gbest_sol)
        print(
            f"[PSO] Initial fitness={gbest_fit:.4f}  "
            f"rev={init_rev:.2f}  feasible={self._is_feasible(gbest_sol)}"
        )

        stagnation = 0
        no_improve_limit = max(200, max_iter // 5)

        for it in range(max_iter):
            if _time.time() - start_time >= time_limit:
                print("[PSO] Time limit reached.")
                break

            w = self.__w_start - (self.__w_start - self.__w_end) * it / max(max_iter - 1, 1)
            for i, particle in enumerate(swarm):
                for key in particle["pos"]:
                    r1 = rng.random(len(self.__pso_tasks))
                    r2 = rng.random(len(self.__pso_tasks))
                    cognitive = self.__c1 * r1 * (pbest_pos[i][key] - particle["pos"][key])
                    social = self.__c2 * r2 * (gbest_pos[key] - particle["pos"][key])
                    particle["vel"][key] = w * particle["vel"][key] + cognitive + social
                    particle["vel"][key] = np.clip(particle["vel"][key], -vmax, vmax)
                    particle["pos"][key] = particle["pos"][key] + particle["vel"][key]

                # Low-probability turbulence prevents all particles from collapsing
                # to identical random keys after long stagnation.
                if rng.random() < 0.03:
                    k = rng.choice(list(particle["pos"].keys()))
                    nudge_idx = rng.integers(len(self.__pso_tasks), size=max(1, len(self.__pso_tasks) // 20))
                    particle["pos"][k][nudge_idx] += rng.normal(0.0, 1.0, len(nudge_idx))

                fit, sol = self._fitness_from_position(particle["pos"])
                fitness[i] = fit
                sols[i] = sol
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = self._copy_particle_position(particle["pos"])
                    pbest_sol[i] = copy.deepcopy(sol)

            iter_best_idx = int(np.argmax(fitness))
            if fitness[iter_best_idx] > gbest_fit:
                gbest_fit = fitness[iter_best_idx]
                gbest_pos = self._copy_particle_position(swarm[iter_best_idx]["pos"])
                gbest_sol = copy.deepcopy(sols[iter_best_idx])
                stagnation = 0
                self._save_incumbent(gbest_sol)
            else:
                stagnation += 1

            # Periodic 2-opt is a route-improvement decoder post-processing step;
            # it does not replace the PSO velocity update.
            if it > 0 and it % 75 == 0 and self._is_feasible(gbest_sol):
                improved = self._local_search(copy.deepcopy(gbest_sol))
                improved_fit = self._evaluate(improved)
                if improved_fit > gbest_fit:
                    gbest_sol = improved
                    gbest_fit = improved_fit
                    self._save_incumbent(gbest_sol)

            if stagnation >= no_improve_limit:
                # Keep the elite fifth, reinitialize the rest as canonical particles.
                sorted_idx = np.argsort(fitness)[::-1]
                keep = set(sorted_idx[:max(1, n_pop // 5)])
                for i in range(n_pop):
                    if i not in keep:
                        swarm[i] = self._new_particle()
                        fitness[i], sols[i] = self._fitness_from_position(swarm[i]["pos"])
                        pbest_pos[i] = self._copy_particle_position(swarm[i]["pos"])
                        pbest_fit[i] = fitness[i]
                        pbest_sol[i] = copy.deepcopy(sols[i])
                stagnation = 0

        if self._is_feasible(gbest_sol):
            gbest_sol = self._local_search(gbest_sol)
            gbest_fit = self._evaluate(gbest_sol)

        self.__best_sol = gbest_sol
        self.__objvalue = gbest_fit
        best_rev, best_time, best_power = self._calc_solution_metrics(gbest_sol)
        print(
            f"[PSO] Best fitness={gbest_fit:.4f}  "
            f"rev={best_rev:.2f}  "
            f"time={best_time:.2f}  "
            f"power={best_power:.2f}  "
            f"feasible={self._is_feasible(gbest_sol)}"
        )

        if not self._is_feasible(gbest_sol):
            raise MIPError("No feasible solution found")
        return None

    # -----------------------------------------------------------------------
    # Convert PSO solution to ea.py result format
    # -----------------------------------------------------------------------
    def _build_res_from_sol(self, sol):
        loop = []
        plan = [[], [], []]
        cum_t_total = 0.0

        for d in range(3):
            depot_start = self.__opoint[d]
            loop.append((*depot_start, cum_t_total))
            plan[d].append((*depot_start, cum_t_total))

            cur_pt = depot_start[0]
            for t in sol["days"][d]:
                pt = sol["loc"][t]
                tt = self._travel_time(cur_pt, pt)
                cum_t_total += tt + self.__task_time[t]
                loop.append((pt, t, cum_t_total))
                plan[d].append((pt, t, cum_t_total))
                cur_pt = pt

            depot_end = self.__opoint[d + 1]
            tt_ret = self._travel_time(cur_pt, depot_end[0])
            cum_t_total += tt_ret
            loop.append((*depot_end, cum_t_total))
            plan[d].append((*depot_end, cum_t_total))

        w1 = plan[0][-1][2] if plan[0] else 0.0
        w2 = plan[1][-1][2] if plan[1] else 0.0
        self.__res = [loop, plan, w1, w2]
        return None

    # -----------------------------------------------------------------------
    # Route formatting / output (identical logic to ea.py)
    # -----------------------------------------------------------------------
    def __format_number(self, number):
        return _format_number(number, self.__decimal)

    def cal_route(self):
        pointdf = self.__pointdf.copy()
        pointdf.reset_index(inplace=True)
        pointdf.set_index("No", inplace=True)
        taskdf = self.__task.copy()
        taskdf.reset_index(inplace=True)
        taskdf.set_index("No", inplace=True)
        plan = [list(p) for p in self.__res[1]]
        n = 1
        self.__plandf = []
        curpt = self.__opoint[0][0]
        for pli in range(3):
            no, action, location, time, power, player1, player2 = (
                [], [], [], [], [], [], [],
            )
            if (*self.__opoint[0], 0) in plan[0]:
                plan[0].remove((*self.__opoint[0], 0))
            opoint_pts = set(self.__opoint[d][0] for d in range(4))
            for i, k, t in plan[pli]:
                if i == curpt or (i in opoint_pts and curpt in opoint_pts):
                    no.append(n)
                    action.append(taskdf.loc[k, "name"])
                    location.append(
                        f"({self.__format_number(pointdf.loc[i, 'X'])},"
                        f"{self.__format_number(pointdf.loc[i, 'Y'])})"
                    )
                    time.append(float(self.__format_number(taskdf.loc[k, "time"])))
                    power.append(float(self.__format_number(taskdf.loc[k, "power"])))
                    player1.append("√")
                    player2.append("√")
                    curpt = i
                    n += 1
                elif i != curpt:
                    pti = pointdf.loc[i, "index"]
                    ptcur = pointdf.loc[curpt, "index"]
                    if pti != ptcur:
                        no.append(n)
                        action.append(
                            f"Travel from ({self.__format_number(pointdf.loc[curpt, 'X'])},"
                            f"{self.__format_number(pointdf.loc[curpt, 'X'])}) to "
                            f"({self.__format_number(pointdf.loc[i, 'X'])},"
                            f"{self.__format_number(pointdf.loc[i, 'Y'])})"
                        )
                        ttime = self.__tmatrix.loc[
                            pointdf.loc[curpt, "index"]
                        ][pointdf.loc[i, "index"]]
                        time.append(float(self.__format_number(ttime)))
                        tpower = self.__pmatrix.loc[
                            pointdf.loc[curpt, "index"]
                        ][pointdf.loc[i, "index"]]
                        power.append(float(self.__format_number(tpower)))
                        location.append(
                            f"({self.__format_number(pointdf.loc[curpt, 'X'])},"
                            f"{self.__format_number(pointdf.loc[curpt, 'Y'])})→"
                            f"({self.__format_number(pointdf.loc[i, 'X'])},"
                            f"{self.__format_number(pointdf.loc[i, 'Y'])})"
                        )
                        player1.append("√")
                        player2.append("√")
                        n += 1
                    no.append(n)
                    action.append(taskdf.loc[k, "name"])
                    location.append(
                        f"({self.__format_number(pointdf.loc[i, 'X'])},"
                        f"{self.__format_number(pointdf.loc[i, 'Y'])})"
                    )
                    time.append(float(self.__format_number(taskdf.loc[k, "time"])))
                    power.append(float(self.__format_number(taskdf.loc[k, "power"])))
                    player1.append("√")
                    player2.append("√")
                    curpt = i
                    n += 1
            if no:
                no.pop()
                action.pop()
                location.pop()
                time.pop()
                power.pop()
                player1.pop()
                player2.pop()
            self.__plandf.append(
                pd.DataFrame({
                    "No": no, "action": action, "location": location,
                    "time": time, "power": power, "1": player1, "2": player2,
                })
            )
        return None

    def __gen_packagedf(self, package, tag):
        pak = self.__package[package["tag"] == tag]
        no = list(range(pak.shape[0]))
        location = (
            f"({self.__format_number(self.__pointdf.loc['探测起点1', 'X'])},"
            f"{self.__format_number(self.__pointdf.loc['探测起点1', 'X'])})"
        )
        loc = [location] * pak.shape[0]
        player = ["√"] * pak.shape[0]
        return pd.DataFrame({
            "No": no, "action": pak.loc[:, "name"].values,
            "location": loc, "time": pak.loc[:, "time"].values,
            "power": pak.loc[:, "power"].values, "1": player, "2": player,
        })

    def add_package(self):
        plan = [df.copy() for df in self.__plandf]
        isindf12 = [False, False, False]
        isindf23 = [False, False, False]
        df = [False, False, False]
        df0 = pd.DataFrame({
            "No": [np.nan], "action": ["Begin of Day1"],
            "location": [np.nan], "time": [np.nan], "power": [np.nan],
            "1": [np.nan], "2": [np.nan],
        })
        for i in range(3):
            isindf12[i] = (
                self.__plandf[i]
                .isin(self.__task[self.__task["tag"] == "12s"]["name"].values)
                .any().any()
            )
            isindf23[i] = (
                self.__plandf[i]
                .isin(self.__task[self.__task["tag"] == "23s"]["name"].values)
                .any().any()
            )
        pakdf = [
            [[self.__gen_packagedf(self.__package, f"D{i+1}ss"),
              self.__gen_packagedf(self.__package, f"D{i+1}se")],
             [self.__gen_packagedf(self.__package, f"D{i+1}es"),
              self.__gen_packagedf(self.__package, f"D{i+1}ee")]]
            for i in range(3)
        ]
        for i in range(3):
            if isindf12[i] or isindf23[i]:
                gap_val = self.__12gap if isindf12[i] else self.__23gap
                if self.__plandf[i].shape[0] == 2:
                    temp = pd.concat([
                        self.__plandf[i].iloc[:1, :], pakdf[i][0][1],
                        pakdf[i][1][0], self.__plandf[i].iloc[-1:, :],
                    ])
                elif self.__plandf[i].shape[0] > 2:
                    temp = pd.concat([
                        self.__plandf[i].iloc[:1, :], pakdf[i][0][1],
                        self.__plandf[i].iloc[1:-1, :], pakdf[i][1][0],
                        self.__plandf[i].iloc[-1:, :],
                    ])
                else:
                    temp = self.__plandf[i]
                time_val = sum(temp["time"])
                if time_val <= gap_val:
                    adddf = pd.DataFrame({
                        "No": [np.nan],
                        "action": [f"Wait for {self.__format_number(gap_val - time_val)}s"],
                        "location": [
                            f"({self.__format_number(self.__pointdf.loc['探测起点1', 'X'])},"
                            f"{self.__format_number(self.__pointdf.loc['探测起点1', 'X'])})"
                        ],
                        "time": [float(self.__format_number(gap_val - time_val))],
                        "power": [0], "1": ["√"], "2": ["√"],
                    })
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat([
                            pakdf[i][0][0], self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1], pakdf[i][1][0], adddf,
                            self.__plandf[i].iloc[-1:, :], pakdf[i][1][1],
                        ])
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat([
                            pakdf[i][0][0], self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1], self.__plandf[i].iloc[1:-1, :],
                            pakdf[i][1][0], adddf,
                            self.__plandf[i].iloc[-1:, :], pakdf[i][1][1],
                        ])
                else:
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat([
                            pakdf[i][0][0], self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1], pakdf[i][1][0],
                            self.__plandf[i].iloc[-1:, :], pakdf[i][1][1],
                        ])
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat([
                            pakdf[i][0][0], self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1], self.__plandf[i].iloc[1:-1, :],
                            pakdf[i][1][0], self.__plandf[i].iloc[-1:, :],
                            pakdf[i][1][1],
                        ])
                time_str = "sum: {}".format(self.__format_number(sum(plan[i]["time"])))
                power_str = "sum: {}".format(self.__format_number(sum(plan[i]["power"])))
                df[i] = pd.DataFrame({
                    "No": [np.nan], "action": ["xxx"], "location": [np.nan],
                    "time": [time_str], "power": [power_str],
                    "1": [np.nan], "2": [np.nan],
                })
            else:
                pakdf[i][0] = pd.concat(pakdf[i][0])
                pakdf[i][1] = pd.concat(pakdf[i][1])
                plan[i] = pd.concat([pakdf[i][0], self.__plandf[i], pakdf[i][1]])
                time_str = "sum: {}".format(self.__format_number(sum(plan[i]["time"])))
                power_str = "sum: {}".format(self.__format_number(sum(plan[i]["power"])))
                df[i] = pd.DataFrame({
                    "No": [np.nan], "action": ["xxx"], "location": [np.nan],
                    "time": [time_str], "power": [power_str],
                    "1": [np.nan], "2": [np.nan],
                })
        df[0]["action"] = "Break between Day1 and Day2"
        df[1]["action"] = "Break between Day2 and Day3"
        df[2]["action"] = "End of Day3"
        self.__plandf = pd.concat(
            [df0, plan[0], df[0], plan[1], df[1], plan[2], df[2]]
        )
        self.__plandf.reset_index(drop=True, inplace=True)
        index0 = self.__plandf[self.__plandf["action"] == "Begin of Day1"].index[0] + 2
        index1 = self.__plandf[self.__plandf["action"] == "Break between Day1 and Day2"].index[0] + 2
        index2 = self.__plandf[self.__plandf["action"] == "Break between Day2 and Day3"].index[0] + 2
        index3 = self.__plandf[self.__plandf["action"] == "End of Day3"].index[0] + 2
        self.__voidindex = [int(index0), int(index1), int(index2), int(index3)]
        no = list(range(self.__plandf.shape[0]))
        for i_no in no:
            if i_no in [0, index1 - 2, index2 - 2, index3 - 2]:
                no[i_no] = np.nan
            elif i_no > index1 - 2 and i_no < index2 - 2:
                no[i_no] = i_no - 1
            elif i_no > index3 - 2:
                no[i_no] = i_no - 2
        self.__plandf["No"] = no
        return None

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
                        start_color="FFFF00", end_color="FFFF00",
                        fill_type="solid",
                    )
        return None

    def schedule_frame(self):
        return self.__plandf.copy()

    def print_status(self):
        best = self.__best_sol
        if best is None or not self._is_feasible(best):
            raise MIPError("PSO did not find a feasible solution")

        print(f"PSO feasible incumbent objective value: {self.__objvalue}")
        time_efficiency, power_efficiency = self._calc_efficiencies(best)
        print(
            "Feasible objective value "
            f"{self.__format_number(self.__objvalue)} found with "
            f"{self.__format_number(time_efficiency)}% time-efficiency and "
            f"{self.__format_number(power_efficiency)}% power-efficiency"
        )
        return None

    # -----------------------------------------------------------------------
    # Main entry point (same signature as ea.py)
    # -----------------------------------------------------------------------
    def run(self):
        self.test_IO()
        self.read_info()
        self.read_task()
        self.read_package()
        self.read_point()
        self.gen_void_point()
        self.check_remote()
        self.divide_task()
        self.drop_O()
        self.gen_point()
        self.gen_edges()
        self._build_lookup()
        self.run_opt()
        self.print_status()
        self._build_res_from_sol(self.__best_sol)
        self.cal_route()
        self.add_package()
        if self.__write_output:
            self.write_excel()
        return None


# ---------------------------------------------------------------------------
# Module-level solve() – same interface as ea.py
# ---------------------------------------------------------------------------
def solve(case, mode="normal"):
    """Run the PSO algorithm on UnifiedCase input."""
    if mode != "normal":
        raise NotImplementedError("pso only supports algorithm.mode=normal")

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
    objective_value = getattr(optimizer, "_task_optimize__objvalue", None)
    return SchedulePlan(
        steps=[],
        rows=rows,
        objective_value=None if objective_value is None else float(objective_value),
    )


if __name__ == "__main__":
    opt = task_optimize(CONST.MAX_REVENUE, timeLimit=60 * 60 * 4)
    opt.run()
