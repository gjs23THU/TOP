# -*- coding: utf-8 -*-
"""
COPT backend migrated from the 2024 exact-algorithm delivery.

Original model author: chen wentian
"""
import coptpy as cp
import gc
import os
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from openpyxl.styles import PatternFill

from .models import MIPError, SubtourError, exact_big_m


_EAC_SOLVE_LOCK = threading.Lock()


class _LinearTupleDict:
    """Keep empty COPT tupledict sums as symbolic linear expressions.

    COPT returns the Python number ``0`` when a tupledict wildcard matches no
    variables. Comparing that value with a bound creates a Python boolean
    before the expression reaches ``addConstr(s)``. Wrapping every sum in a
    ``LinExpr`` preserves the intended constant model constraint.
    """

    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def items(self):
        return self._values.items()

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def sum(self, *patterns):
        return cp.LinExpr(self._values.sum(*patterns))

    def __getattr__(self, name):
        return getattr(self._values, name)


class CONST(object):
    MAX_REVENUE = "maxRevenue"
    MIN_TIME = "minTime"
    MIN_POWER = "minPower"


class task_optimize(object):
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
        iisPath=None,
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
        parallelWorkers=None,
    ):
        self.__objective = obj
        self.__Obj = [CONST.MAX_REVENUE, CONST.MIN_TIME, CONST.MIN_POWER]
        self.__env = None
        self.__m = None
        self.__time_limit = timeLimit
        try:
            self.__env = cp.Envr()
            self.__m = self.__env.createModel("schedule-optimization")
            if timeLimit != np.inf:
                self.__m.setParam(cp.COPT.Param.TimeLimit, timeLimit)
            if parallelWorkers is not None:
                threads_param = getattr(cp.COPT.Param, "Threads", None)
                if threads_param is not None:
                    self.__m.setParam(threads_param, max(1, int(parallelWorkers)))
        except Exception as exc:
            close_env = getattr(self.__env, "close", None)
            if callable(close_env):
                close_env()
            self.__env = None
            raise RuntimeError(
                "COPT environment initialization failed; verify coptpy==8.0.7 "
                "and a compatible COPT 8.x license in COPT_LICENSE_DIR"
            ) from exc
        # COPT integration intentionally disables the Gurobi-specific callback
        # autosave path. Unified outputs are written after a validated solve.
        self.__autosavestate = False
        self.__decimal = decimal
        self.__lppath, self.__solpath = lpPath, solPath
        self.__iispath = iisPath
        self.__infoPath, self.__taskPath, self.__pakpath = (
            infoPath,
            taskPath,
            packPath,
        )
        (
            self.__pointpath,
            self.__distancepath,
            self.__timepath,
            self.__powerpath,
        ) = (pointPath, distancePath, timePath, powerPath)
        self.__outputpath = outputPath
        self.__dataframes = dataFrames or {}
        self.__write_output = writeOutput
        self.__solcount = 1
        self.__time_upper_bound = None
        self.__power_upper_bound = None
        self.__parallel_workers = None if parallelWorkers is None else max(1, int(parallelWorkers))
        self.__step_timings = {}
        self.__run_elapsed = None
        self.__progress_events = []
        return None

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
        if not os.path.exists(self.__infoPath):
            FileNotFound.append(self.__infoPath)
        if not os.path.exists(self.__taskPath):
            FileNotFound.append(self.__taskPath)
        if not os.path.exists(self.__pakpath):
            FileNotFound.append(self.__pakpath)
        if not os.path.exists(self.__pointpath):
            FileNotFound.append(self.__pointpath)
        if not os.path.exists(self.__distancepath):
            FileNotFound.append(self.__distancepath)
        if not os.path.exists(self.__timepath):
            FileNotFound.append(self.__timepath)
        if not os.path.exists(self.__powerpath):
            FileNotFound.append(self.__powerpath)
        if FileNotFound != []:
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
        self.__RawTTime = list(self.__TTime)
        self.__RawTPower = list(self.__TPower)
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
        time = [
            sum(self.__package[self.__package["tag"] == tag]["time"])
            for tag in [
                "D1ss",
                "D1se",
                "D1es",
                "D1ee",
                "D2ss",
                "D2se",
                "D2es",
                "D2ee",
                "D3ss",
                "D3se",
                "D3es",
                "D3ee",
            ]
        ]
        power = [
            sum(self.__package[self.__package["tag"] == tag]["power"])
            for tag in [
                "D1ss",
                "D1se",
                "D1es",
                "D1ee",
                "D2ss",
                "D2se",
                "D2es",
                "D2ee",
                "D3ss",
                "D3se",
                "D3es",
                "D3ee",
            ]
        ]
        time = [
            time[0] + time[1] + time[2] + time[3],
            time[4] + time[5] + time[6] + time[7],
            time[8] + time[9] + time[10] + time[11],
        ]
        power = [
            power[0] + power[1] + power[2] + power[3],
            power[4] + power[5] + power[6] + power[7],
            power[8] + power[9] + power[10] + power[11],
        ]
        self.__TTime = [self.__TTime[i] - time[i] for i in range(3)]
        self.__TPower = [self.__TPower[i] - power[i] for i in range(3)]
        return None

    def read_point(self):
        point = self.__dataframes.get("point")
        self.__pointdf = point.copy() if point is not None else pd.read_excel(self.__pointpath)
        self.__pointdf.set_index(self.__pointdf.columns[0], inplace=True)
        self.__pointdf.index.rename(None, inplace=True)
        return None

    def __add_void_matrix(self, matrix):
        voidrow = [matrix.loc["探测起点", :].values.tolist() for i in range(4)]
        voidrowdf = pd.DataFrame(
            voidrow,
            index=["探测起点1", "探测起点2", "探测起点3", "探测起点4"],
            columns=matrix.columns,
        )
        matrix = pd.concat([voidrowdf, matrix])
        matrix.drop(["探测起点"], inplace=True)
        voidcol = np.array(
            [matrix.loc[:, "探测起点"].values.tolist() for i in range(4)]
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
        voidpoint = pd.DataFrame(
            {
                "X": [
                    self.__pointdf.loc["探测起点", "X"],
                    self.__pointdf.loc["探测起点", "X"],
                    self.__pointdf.loc["探测起点", "X"],
                    self.__pointdf.loc["探测起点", "X"],
                ],
                "Y": [
                    self.__pointdf.loc["探测起点", "Y"],
                    self.__pointdf.loc["探测起点", "Y"],
                    self.__pointdf.loc["探测起点", "Y"],
                    self.__pointdf.loc["探测起点", "Y"],
                ],
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
        self.__task["location"] = self.__task["location"].fillna(
            ",".join(self.__pointdf.index.values)
        )
        location = self.__task["location"].values
        pool = []
        for i in location:
            if pd.isna(i):
                pool.append(np.nan)
            else:
                split = list(i.split(","))
                pool.append(split)
        self.__task["location"] = pd.Series(pool)
        voidtaskdf = pd.DataFrame(
            {
                "No": [
                    self.__task.shape[0],
                    self.__task.shape[0] + 1,
                    self.__task.shape[0] + 2,
                    self.__task.shape[0] + 3,
                ],
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
            index=[
                self.__task.shape[0],
                self.__task.shape[0] + 1,
                self.__task.shape[0] + 2,
                self.__task.shape[0] + 3,
            ],
        )
        self.__task = pd.concat([self.__task, voidtaskdf])
        return None

    def __cartesian_to_polar(self, x, y, O):
        r = np.sqrt((x - O[0]) ** 2 + (y - O[1]) ** 2)
        theta = np.arctan2(y - O[1], x - O[0])
        return r, theta

    def __polar_to_cartesian(self, r, theta, O):
        x = r * np.cos(theta) + O[0]
        y = r * np.sin(theta) + O[1]
        return x, y

    def __cal_distance(self, point1, point2):
        dis = (
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        ) ** 0.5
        return dis

    def __from_distance(self, distance):
        pt = (distance, 0)
        return pt

    def check_remote(self):
        remoteindex = self.__task[
            self.__task["remote"] == True
        ].index.to_list()
        # Remote feasibility is expressed by add_remote_constrs().  In
        # particular, an empty remote task set adds no constraint, while an
        # existing remote task set with no qualified location adds 0 >= 1 and
        # is reported through the normal infeasible/IIS path.  Do not recreate
        # the legacy new-point.xlsx side effect in the unified backend.
        if not remoteindex:
            return None
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
            self.__task.loc[i, "location"].remove("探测起点1")
            self.__task.loc[i, "location"].remove("探测起点2")
            self.__task.loc[i, "location"].remove("探测起点3")
            self.__task.loc[i, "location"].remove("探测起点4")
        return None

    def gen_point(self):
        self.__opoint = (
            (
                self.__pointdf.loc["探测起点1", "No"],
                self.__task[self.__task["name"] == "void1"].index[0],
            ),
            (
                self.__pointdf.loc["探测起点2", "No"],
                self.__task[self.__task["name"] == "void2"].index[0],
            ),
            (
                self.__pointdf.loc["探测起点3", "No"],
                self.__task[self.__task["name"] == "void3"].index[0],
            ),
            (
                self.__pointdf.loc["探测起点4", "No"],
                self.__task[self.__task["name"] == "void4"].index[0],
            ),
        )
        self.__point = cp.tuplelist()
        for t in self.__task.index:
            if type(self.__task.loc[t, "location"]) == str:
                i = self.__pointdf.loc[self.__task.loc[t, "location"], "No"]
                self.__point.append((i, t))
            else:
                for name in self.__task.loc[t, "location"]:
                    i = self.__pointdf.loc[name, "No"]
                    self.__point.append((i, t))
        return self.__point

    def __task_stage_range(self, task_index):
        if task_index == self.__opoint[0][1]:
            return 0, 0
        if task_index == self.__opoint[1][1]:
            return 1, 1
        if task_index == self.__opoint[2][1]:
            return 2, 2
        if task_index == self.__opoint[3][1]:
            return 3, 3

        tag = self.__task.loc[task_index, "tag"]
        if isinstance(tag, str):
            tag = tag.strip()
            if tag in {"12s", "12e"}:
                return 0, 1
            if tag in {"23s", "23e"}:
                return 1, 2

        day = self.__task.loc[task_index, "day"]
        if pd.isna(day):
            return 0, 2
        text = str(day).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text == "1":
            return 0, 0
        if text == "2":
            return 1, 1
        if text == "3":
            return 2, 2
        if text == "1,2":
            return 0, 1
        if text == "2,3":
            return 1, 2
        return 0, 2

    def __edge_is_allowed(self, source, target, stage_ranges):
        if source == target:
            return False
        _, source_task = source
        _, target_task = target
        if source_task == target_task:
            return False
        if target == self.__opoint[0] or source == self.__opoint[-1]:
            return False

        source_is_void = source in self.__opoint
        target_is_void = target in self.__opoint
        if source_is_void and target_is_void:
            return self.__opoint.index(target) == self.__opoint.index(source) + 1

        source_min, source_max = stage_ranges[source_task]
        target_min, target_max = stage_ranges[target_task]
        if source_is_void:
            return target_max >= source_min
        if target_is_void:
            return source_min <= target_min - 1 <= source_max
        return source_min <= target_max

    def gen_edges(self):
        stage_ranges = {
            task_index: self.__task_stage_range(task_index)
            for task_index in self.__task.index
        }
        self.__edges = cp.tuplelist()
        for i in self.__point:
            for j in self.__point:
                if self.__edge_is_allowed(i, j, stage_ranges):
                    self.__edges.append((*i, *j))
        return None

    def __resource_upper_bound(self, budgets):
        values = [float(value) for value in budgets if np.isfinite(float(value))]
        if not values:
            return 0.0
        return max(0.0, sum(values))

    def __time_bound(self):
        if self.__time_upper_bound is None:
            self.__time_upper_bound = self.__resource_upper_bound(self.__TTime)
        return self.__time_upper_bound

    def __power_bound(self):
        if self.__power_upper_bound is None:
            self.__power_upper_bound = self.__resource_upper_bound(self.__TPower)
        return self.__power_upper_bound

    def add_variables(self):
        edges = self.__edges
        point = self.__point
        self.__x = _LinearTupleDict(
            self.__m.addVars(edges, vtype=cp.COPT.BINARY, nameprefix="x")
        )
        self.__W = _LinearTupleDict(self.__m.addVars(
            point,
            vtype=cp.COPT.CONTINUOUS,
            nameprefix="W",
            ub=self.__time_bound(),
        ))
        self.__Q = _LinearTupleDict(self.__m.addVars(
            point,
            vtype=cp.COPT.CONTINUOUS,
            nameprefix="Q",
            ub=self.__power_bound(),
        ))
        self.__Ws1 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Wsafe1"
        ))
        self.__Ws2 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Wsafe2"
        ))
        self.__Ws3 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Wsafe3"
        ))
        self.__Qs1 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Qsafe1"
        ))
        self.__Qs2 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Qsafe2"
        ))
        self.__Qs3 = _LinearTupleDict(self.__m.addVars(
            point, vtype=cp.COPT.BINARY, nameprefix="Qsafe3"
        ))
        return None

    def set_objective(self):
        taskdf = self.__task
        expr = []
        if self.__objective == self.__Obj[0]:
            for t in taskdf.index.values:
                expr.append(
                    self.__x.sum("*", "*", "*", t) * taskdf["revenue"][t]
                )
        if self.__objective == self.__Obj[1]:
            expr.append(-self.__W[*self.__opoint[-1]])
        if self.__objective == self.__Obj[2]:
            expr.append(-self.__Q[*self.__opoint[-1]])
        expr = cp.quicksum(expr)
        self.__m.setObjective(expr, cp.COPT.MAXIMIZE)
        return expr

    def add_indegree_constrs(self):
        temp = cp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[0])
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) <= 1 for i, k in temp),
            nameprefix="indegree0",
        )
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 0 for i, k in temp),
            nameprefix="indegree1",
        )

    def add_outdegree_constrs(self):
        temp = cp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[-1])
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") <= 1 for i, k in temp),
            nameprefix="outdegree0",
        )
        temp = [self.__opoint[-1]]
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 0 for i, k in temp),
            nameprefix="outdegree1",
        )
        return None

    def add_equaldegree_constrs(self):
        temp = cp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[0])
        temp.remove(self.__opoint[-1])
        self.__m.addConstrs(
            (
                self.__x.sum(i, k, "*", "*") == self.__x.sum("*", "*", i, k)
                for i, k in temp
            ),
            nameprefix="equaldegree",
        )
        return None

    def add_oodegree_constrs(self):
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 1 for i, k in temp),
            nameprefix="oodegree0",
        )
        temp = [self.__opoint[-1]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 1 for i, k in temp),
            nameprefix="oodegree1",
        )
        temp = [self.__opoint[1], self.__opoint[2]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 1 for i, k in temp),
            nameprefix="oodegree2",
        )
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 1 for i, k in temp),
            nameprefix="oodegree3",
        )
        temp = self.__opoint
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[-1], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[-1], j, k2) in self.__x
            ),
            nameprefix="oodegree4",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[2], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[2], j, k2) in self.__x
            ),
            nameprefix="oodegree5",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[1], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[1], j, k2) in self.__x
            ),
            nameprefix="oodegree6",
        )
        return None

    def add_rtask_constrs(self):
        self.__m.addConstrs(
            (self.__x.sum("*", "*", "*", k) == 1 for k in self.__reqtaskindex),
            nameprefix="rtask",
        )
        return None

    def add_otask_constrs(self):
        self.__m.addConstrs(
            (self.__x.sum("*", "*", "*", k) <= 1 for k in self.__opttaskindex),
            nameprefix="otask",
        )
        return None

    def add_remote_constrs(self):
        if not self.__remtaskindex:
            return None
        point_name = self.__pointdf.reset_index().set_index("No")["index"].to_dict()
        remote_points = [
            (i, k)
            for i, k in self.__point
            if k in self.__remtaskindex
            and self.__dmatrix.loc["探测起点1", point_name[i]] >= self.__MDistance
        ]
        self.__m.addConstr(
            cp.quicksum(
                (self.__x.sum("*", "*", i, k) for i, k in remote_points)
            )
            >= 1,
            name="remote",
        )
        return None

    def add_time_constrs(self):
        point = cp.tuplelist(self.__point.copy())
        time_bound = self.__time_bound()
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__W.sum(i, k) == 0 for i, k in temp), nameprefix="time0"
        )
        pointdfbackup = self.__pointdf.copy()
        pointdfbackup.reset_index(inplace=True)
        pointdfbackup.set_index("No", inplace=True)
        taskbackup = self.__task.copy()
        taskbackup.reset_index(inplace=True)
        taskbackup.set_index("No", inplace=True)
        self.__m.addConstrs(
            (
                self.__W[i, a]
                + self.__tmatrix.loc[pointdfbackup.loc[i, "index"]][
                    pointdfbackup.loc[j, "index"]
                ]
                + taskbackup.loc[b, "time"]
                - self.__W[j, b]
                <= exact_big_m(
                    time_bound,
                    self.__tmatrix.loc[pointdfbackup.loc[i, "index"]][
                        pointdfbackup.loc[j, "index"]
                    ]
                    + taskbackup.loc[b, "time"],
                )
                * (1 - self.__x[i, a, j, b])
                for i, a, j, b in self.__edges
            ),
            nameprefix="time1",
        )
        temp = list(point.select(self.__opoint[1][0], "*"))
        temp.remove(point.select(self.__opoint[1][0], self.__opoint[1][1])[0])
        self.__m.addConstrs(
            (
                self.__W[k[0], k[1]] <= self.__TTime[0]
                for k in list(point.select(self.__opoint[1][0], "*"))
            ),
            nameprefix="time2",
        )
        temp = list(point.select(self.__opoint[2][0], "*"))
        temp.remove(point.select(self.__opoint[2][0], self.__opoint[2][1])[0])
        self.__m.addConstr(
            self.__W[*self.__opoint[2]] - self.__W[*self.__opoint[1]]
            <= self.__TTime[1],
            name="time3",
        )
        temp = list(point.select(self.__opoint[-1][0], "*"))
        temp.remove(
            point.select(self.__opoint[-1][0], self.__opoint[-1][1])[0]
        )
        self.__m.addConstrs(
            (
                self.__W[*self.__opoint[-1]] >= self.__W.sum(k[0], k[1])
                for k in temp
            ),
            nameprefix="time4",
        )
        self.__m.addConstr(
            (
                self.__W[*self.__opoint[-1]] - self.__W[*self.__opoint[2]]
                <= self.__TTime[2]
            ),
            name="time5",
        )
        temp = [
            [self.__opoint[0], self.__opoint[1]],
            [self.__opoint[1], self.__opoint[2]],
            [self.__opoint[2], self.__opoint[-1]],
        ]
        self.__m.addConstrs(
            (
                self.__W.sum(i, k1) <= self.__W.sum(j, k2)
                for (i, k1), (j, k2) in temp
            ),
            nameprefix="time6",
        )
        return None

    def add_day_constrs(self):
        point = cp.tuplelist(self.__point.copy())
        if self.__daytaskindex[0] != []:
            for k in self.__daytaskindex[0]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    nameprefix=f"day1{k}",
                )
        if self.__daytaskindex[1] != []:
            for k in self.__daytaskindex[1]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    nameprefix=f"day21{k}",
                )
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    nameprefix=f"day22{k}",
                )
        if self.__daytaskindex[2] != []:
            for k in self.__daytaskindex[2]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    nameprefix=f"day3{k}",
                )
        if self.__daytaskindex[3] != []:
            for k in self.__daytaskindex[3]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    nameprefix=f"day12{k}",
                )
        if self.__daytaskindex[4] != []:
            for k in self.__daytaskindex[4]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    nameprefix=f"day23{k}",
                )
        if self.__tagtaskindex[0] != []:
            for k in self.__tagtaskindex[0]:
                self.__m.addConstr(
                    self.__x.sum(*self.__opoint[0], self.__opoint[0][0], k)
                    + self.__x.sum(*self.__opoint[1], self.__opoint[1][0], k)
                    == 1,
                    name=f"day12s{k}",
                )
            for k in self.__tagtaskindex[1]:
                self.__m.addConstr(
                    self.__x.sum(self.__opoint[1][0], k, *self.__opoint[1])
                    + self.__x.sum(self.__opoint[2][0], k, *self.__opoint[2])
                    == 1,
                    name=f"day12e{k}",
                )
            for k in self.__tagtaskindex[2]:
                self.__m.addConstr(
                    self.__x.sum(*self.__opoint[1], self.__opoint[1][0], k)
                    + self.__x.sum(*self.__opoint[2], self.__opoint[2][0], k)
                    == 1,
                    name=f"day23s{k}",
                )
            for k in self.__tagtaskindex[3]:
                self.__m.addConstr(
                    self.__x.sum(self.__opoint[2][0], k, *self.__opoint[2])
                    + self.__x.sum(self.__opoint[-1][0], k, *self.__opoint[-1])
                    == 1,
                    name=f"day23e{k}",
                )
            self.__m.addConstr(
                self.__x.sum(
                    *self.__opoint[1],
                    self.__opoint[1][0],
                    self.__tagtaskindex[0][0],
                )
                + self.__x.sum(
                    *self.__opoint[1],
                    self.__opoint[1][0],
                    self.__tagtaskindex[2][0],
                )
                <= 1,
                name=f"startnotsameday{k}",
            )
            self.__m.addConstr(
                self.__x.sum(
                    self.__opoint[2][0],
                    self.__tagtaskindex[1][0],
                    *self.__opoint[2],
                )
                + self.__x.sum(
                    self.__opoint[2][0],
                    self.__tagtaskindex[3][0],
                    *self.__opoint[2],
                )
                <= 1,
                name=f"endnotsameday{k}",
            )
            self.__m.addConstr(
                self.__x.sum(
                    *self.__opoint[0],
                    self.__opoint[0][0],
                    self.__tagtaskindex[0],
                )
                == self.__x.sum(
                    self.__opoint[1][0],
                    self.__tagtaskindex[1],
                    *self.__opoint[1],
                ),
                name="sameday0",
            )
            self.__m.addConstr(
                self.__x.sum(
                    *self.__opoint[1],
                    self.__opoint[1][0],
                    self.__tagtaskindex[0],
                )
                == self.__x.sum(
                    self.__opoint[2][0],
                    self.__tagtaskindex[1],
                    *self.__opoint[2],
                ),
                name="sameday1",
            )
            self.__m.addConstr(
                self.__x.sum(
                    *self.__opoint[1],
                    self.__opoint[1][0],
                    self.__tagtaskindex[2],
                )
                == self.__x.sum(
                    self.__opoint[2][0],
                    self.__tagtaskindex[3],
                    *self.__opoint[2],
                ),
                name="sameday2",
            )
            self.__m.addConstr(
                self.__x.sum(
                    *self.__opoint[2],
                    self.__opoint[2][0],
                    self.__tagtaskindex[2],
                )
                == self.__x.sum(
                    self.__opoint[-1][0],
                    self.__tagtaskindex[3],
                    *self.__opoint[-1],
                ),
                name="sameday3",
            )
        return None

    def add_power_constrs(self):
        point = cp.tuplelist(self.__point.copy())
        power_bound = self.__power_bound()
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__Q.sum(i, k) == 0 for i, k in temp), nameprefix="power0"
        )
        pointdfbackup = self.__pointdf.copy()
        pointdfbackup.reset_index(inplace=True)
        pointdfbackup.set_index("No", inplace=True)
        taskbackup = self.__task.copy()
        taskbackup.reset_index(inplace=True)
        taskbackup.set_index("No", inplace=True)
        self.__m.addConstrs(
            (
                self.__Q[i, a]
                + self.__pmatrix.loc[pointdfbackup.loc[i, "index"]][
                    pointdfbackup.loc[j, "index"]
                ]
                + taskbackup.loc[b, "power"]
                - self.__Q[j, b]
                <= exact_big_m(
                    power_bound,
                    self.__pmatrix.loc[pointdfbackup.loc[i, "index"]][
                        pointdfbackup.loc[j, "index"]
                    ]
                    + taskbackup.loc[b, "power"],
                )
                * (1 - self.__x[i, a, j, b])
                for i, a, j, b in self.__edges
            ),
            nameprefix="power1",
        )
        temp = list(point.select(self.__opoint[1][0], "*"))
        temp.remove(point.select(self.__opoint[1][0], self.__opoint[1][1])[0])
        self.__m.addConstrs(
            (
                self.__Q[k[0], k[1]] <= self.__TPower[0]
                for k in list(point.select(self.__opoint[1][0], "*"))
            ),
            nameprefix="power2",
        )
        temp = list(point.select(self.__opoint[2][0], "*"))
        temp.remove(point.select(self.__opoint[2][0], self.__opoint[2][1])[0])
        self.__m.addConstr(
            self.__Q[*self.__opoint[2]] - self.__Q[*self.__opoint[1]]
            <= self.__TPower[1],
            name="power3",
        )
        temp = list(point.select(self.__opoint[-1][0], "*"))
        temp.remove(
            point.select(self.__opoint[-1][0], self.__opoint[-1][1])[0]
        )
        self.__m.addConstrs(
            (
                self.__Q[*self.__opoint[-1]] >= self.__Q.sum(k[0], k[1])
                for k in temp
            ),
            nameprefix="power4",
        )
        self.__m.addConstr(
            (
                self.__Q[*self.__opoint[-1]] - self.__Q[*self.__opoint[2]]
                <= self.__TPower[2]
            ),
            name="power5",
        )
        temp = [
            [self.__opoint[0], self.__opoint[1]],
            [self.__opoint[1], self.__opoint[2]],
            [self.__opoint[2], self.__opoint[-1]],
        ]
        self.__m.addConstrs(
            (
                self.__Q.sum(i, k1) <= self.__Q.sum(j, k2)
                for (i, k1), (j, k2) in temp
            ),
            nameprefix="power6",
        )
        return None

    def add_safe_constrs(self):
        point = cp.tuplelist(self.__point.copy())
        eps = 0.0001
        time_bound = self.__time_bound()
        power_bound = self.__power_bound()
        task_time = self.__task.set_index("No")["time"].to_dict()
        task_power = self.__task.set_index("No")["power"].to_dict()
        point_name = self.__pointdf.reset_index().set_index("No")["index"].to_dict()
        safe_time_to_base = {
            i: self.__tmatrix.loc[name, "探测起点1"] for i, name in point_name.items()
        }
        safe_power_from_base = {
            i: self.__pmatrix.loc["探测起点1", name] for i, name in point_name.items()
        }
        for i, k in point:
            time_m = exact_big_m(time_bound, safe_time_to_base[i], task_time[k], eps)
            power_m = exact_big_m(power_bound, safe_power_from_base[i], task_power[k], eps)
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                >= self.__W[i, k] + eps - time_m * (1 - self.__Ws1[i, k]),
                name=f"safetimeday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                <= self.__W[i, k] + time_m * self.__Ws1[i, k],
                name=f"safetimeday1_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Ws1[i, k] == 1)
                >> (
                    self.__W[i, k]
                    - task_time[k]
                    + safe_time_to_base[i]
                    <= self.__TTime[0]
                ),
                name=f"safetimeday1_indicator_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                >= self.__W[*self.__opoint[2]]
                + eps
                - time_m * (1 - self.__Ws3[i, k]),
                name=f"safetimeday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                <= self.__W[*self.__opoint[2]] + time_m * self.__Ws3[i, k],
                name=f"safetimeday3_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Ws3[i, k] == 1)
                >> (
                    self.__W[i, k]
                    - self.__W[*self.__opoint[2]]
                    - task_time[k]
                    + safe_time_to_base[i]
                    <= self.__TTime[2]
                ),
                name=f"safetimeday3_indicator_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                >= self.__Ws1[i, k]
                + self.__Ws3[i, k]
                + eps
                - time_m * (1 - self.__Ws2[i, k]),
                name=f"safetimeday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Ws1[i, k] + self.__Ws3[i, k] + time_m * self.__Ws2[i, k],
                name=f"safetimeday2_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Ws2[i, k] == 1)
                >> (
                    self.__W[i, k]
                    - self.__W[*self.__opoint[1]]
                    - task_time[k]
                    + safe_time_to_base[i]
                    <= self.__TTime[1]
                ),
                name=f"safetimeday2_indicator_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[*self.__opoint[1]]
                >= self.__Q[i, k] + eps - power_m * (1 - self.__Qs1[i, k]),
                name=f"safepowerday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[*self.__opoint[1]]
                <= self.__Q[i, k] + power_m * self.__Qs1[i, k],
                name=f"safepowerday1_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Qs1[i, k] == 1)
                >> (
                    self.__Q[i, k]
                    - task_power[k]
                    + safe_power_from_base[i]
                    <= self.__TPower[0]
                ),
                name=f"safepowerday1_indicator_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                >= self.__Q[*self.__opoint[2]]
                + eps
                - power_m * (1 - self.__Qs3[i, k]),
                name=f"safepowerday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                <= self.__Q[*self.__opoint[2]] + power_m * self.__Qs3[i, k],
                name=f"safepowerday3_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Qs3[i, k] == 1)
                >> (
                    self.__Q[i, k]
                    - self.__Q[*self.__opoint[2]]
                    - task_power[k]
                    + safe_power_from_base[i]
                    <= self.__TPower[2]
                ),
                name=f"safepowerday3_indicator_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                >= self.__Qs1[i, k]
                + self.__Qs3[i, k]
                + eps
                - power_m * (1 - self.__Qs2[i, k]),
                name=f"safepowerday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Qs1[i, k] + self.__Qs3[i, k] + power_m * self.__Qs2[i, k],
                name=f"safepowerday2_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                (self.__Qs2[i, k] == 1)
                >> (
                    self.__Q[i, k]
                    - self.__Q[*self.__opoint[1]]
                    - task_power[k]
                    + safe_power_from_base[i]
                    <= self.__TPower[1]
                ),
                name=f"safepowerday2_indicator_constr0[{i},{k}]",
            )
        return None

    def add_continuous_constr(self):
        if self.__Mincontinuous > 0:
            expr = [self.__x.sum("*", "*", "*", k) for k in self.__contaskindex]
            self.__m.addConstr(
                cp.quicksum(expr) == self.__Mincontinuous, name="continuous"
            )
        return None

    def add_noii_constrs(self):
        point = cp.tuplelist(self.__point.copy())
        self.__m.addConstrs(
            (
                self.__x[i, k, i, k] == 0
                for i, k in point
                if (i, k, i, k) in self.__x
            ),
            nameprefix="noii",
        )
        return None

    def run_opt(self):
        if self.__lppath is not None:
            self.__m.write(self.__lppath)
        self.__m.solve()
        incumbent = self.__model_metric("ObjVal")
        best_bound = self.__model_metric("BestBnd", "BestBound")
        absolute_gap = (
            abs(incumbent - best_bound)
            if incumbent is not None and best_bound is not None
            else None
        )
        progress_events = getattr(self, "_task_optimize__progress_events", None)
        if progress_events is None:
            progress_events = []
            self.__progress_events = progress_events
        progress_events.append(
            {
                "event": "final",
                "elapsed_seconds": self.__model_metric("SolvingTime", "Runtime"),
                "incumbent_objective": incumbent,
                "best_bound": best_bound,
                "absolute_gap": absolute_gap,
                "relative_gap": self.__model_metric("BestGap", "MipGap", "MIPGap"),
                "solution_source": "solver" if incumbent is not None else None,
            }
        )
        if self.__m.Status == cp.COPT.INFEASIBLE:
            self.__m.computeIIS()
            iis_path = Path(self.__iispath or "infeasible.ilp")
            iis_path.parent.mkdir(parents=True, exist_ok=True)
            self.__m.writeIIS(str(iis_path))
            raise MIPError(f"Model is infeasible. Refer to {iis_path}")
        if self.__m.HasSol:
            self.__objvalue = float(self.__m.ObjVal)
        return None

    def print_status(self):
        if self.__m.Status == cp.COPT.OPTIMAL:
            print(f"Optimal objective value: {self.__m.ObjVal}")
        elif self.__m.HasSol:
            print(f"Feasible objective value: {self.__m.ObjVal}")
        else:
            raise MIPError("No feasible solution found")
        return None

    def proc_res(self, x=None, W=None):
        xbackup, Wbackup = 1, 1
        if x == None:
            x = self.__x
            xbackup = None
        if W == None:
            W = self.__W
            Wbackup = None
        loop, plan = [], [[], [], []]
        res = cp.tuplelist()
        for i, k1, j, k2 in self.__edges:
            if xbackup == None:
                if x[i, k1, j, k2].x <= 1.1 and x[i, k1, j, k2].x >= 0.9:
                    res.append((i, k1, j, k2))
            else:
                if x[i, k1, j, k2] <= 1.1 and x[i, k1, j, k2] >= 0.9:
                    res.append((i, k1, j, k2))
        i, k, t = *self.__opoint[0], 0
        loop.append((i, k, t))
        subtour = False
        for p in range(len(res)):
            cord = res.select(i, k, "*", "*")
            try:
                res.remove((i, k, cord[0][2], cord[0][3]))
                i, k = cord[0][2], cord[0][3]
            except IndexError:
                subtour = True
            if Wbackup == None:
                loop.append((i, k, W[i, k].x))
            else:
                loop.append((i, k, W[i, k]))
        day = 0
        for i, k, t in loop:
            plan[day].append((i, k, t))
            if (i, k) in self.__opoint[1:-1]:
                day += 1
        if xbackup == None:
            self.__res = [
                loop,
                plan,
                W[*self.__opoint[1]].x,
                W[*self.__opoint[2]].x,
            ]
        else:
            self.__res = [
                loop,
                plan,
                W[*self.__opoint[1]],
                W[*self.__opoint[2]],
            ]
        if res != [] or subtour:
            raise SubtourError("Subtour found")
        return None

    def __format_number(self, number):
        number = float(number)
        format_str = "{:." + str(self.__decimal) + "f}"
        formatted_number = format_str.format(number)
        while formatted_number.endswith("0"):
            formatted_number = formatted_number[:-1]
        if formatted_number.endswith("."):
            formatted_number = formatted_number[:-1]
        return formatted_number

    def cal_route(self):
        pointdf = self.__pointdf.copy()
        pointdf.reset_index(inplace=True)
        pointdf.set_index("No", inplace=True)
        taskdf = self.__task.copy()
        taskdf.reset_index(inplace=True)
        taskdf.set_index("No", inplace=True)
        plan = self.__res[1].copy()
        n = 1
        self.__plandf = []
        curpt = self.__opoint[0][0]
        for pli in range(3):
            no, action, location, time, power, player1, player2 = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            if (*self.__opoint[0], 0) in plan[0]:
                plan[0].remove((*self.__opoint[0], 0))
            for i, k, t in plan[pli]:
                if i == curpt or (
                    (i in (self.__opoint[i][0] for i in range(4)))
                    and (curpt in (self.__opoint[i][0] for i in range(4)))
                ):
                    no.append(n)
                    pti = pointdf.loc[i, "index"]
                    ptcur = pointdf.loc[curpt, "index"]
                    action.append(taskdf.loc[k, "name"])
                    location.append(
                        f"({self.__format_number(pointdf.loc[i,'X'])},{self.__format_number(pointdf.loc[i,'Y'])})"
                    )
                    time.append(
                        float(self.__format_number(taskdf.loc[k, "time"]))
                    )
                    power.append(
                        float(self.__format_number(taskdf.loc[k, "power"]))
                    )
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
                            f"Travel from ({self.__format_number(pointdf.loc[curpt,'X'])},{self.__format_number(pointdf.loc[curpt,'Y'])}) to ({self.__format_number(pointdf.loc[i,'X'])},{self.__format_number(pointdf.loc[i,'Y'])})"
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
                            f"({self.__format_number(pointdf.loc[curpt,'X'])},{self.__format_number(pointdf.loc[curpt,'Y'])})→({self.__format_number(pointdf.loc[i,'X'])},{self.__format_number(pointdf.loc[i,'Y'])})"
                        )
                        player1.append("√")
                        player2.append("√")
                        n += 1
                    no.append(n)
                    action.append(taskdf.loc[k, "name"])
                    location.append(
                        f"({self.__format_number(pointdf.loc[i,'X'])},{self.__format_number(pointdf.loc[i,'Y'])})"
                    )
                    time.append(
                        float(self.__format_number(taskdf.loc[k, "time"]))
                    )
                    power.append(
                        float(self.__format_number(taskdf.loc[k, "power"]))
                    )
                    player1.append("√")
                    player2.append("√")
                    curpt = i
                    n += 1
            no.pop()
            action.pop()
            location.pop()
            time.pop()
            power.pop()
            player1.pop()
            player2.pop()
            self.__plandf.append(
                pd.DataFrame(
                    {
                        "No": no,
                        "action": action,
                        "location": location,
                        "time": time,
                        "power": power,
                        "1": player1,
                        "2": player2,
                    }
                )
            )
        return None

    def __gen_packagedf(self, package, tag):
        pak = self.__package[package["tag"] == tag]
        no = [i for i in range(pak.shape[0])]
        location = f"({self.__format_number(self.__pointdf.loc['探测起点1','X'])},{self.__format_number(self.__pointdf.loc['探测起点1','Y'])})"
        loc = [location for i in range(pak.shape[0])]
        player = ["√" for i in range(pak.shape[0])]
        df = pd.DataFrame(
            {
                "No": no,
                "action": pak.loc[:, "name"].values,
                "location": loc,
                "time": pak.loc[:, "time"].values,
                "power": pak.loc[:, "power"].values,
                "1": player,
                "2": player,
            }
        )
        return df

    def add_package(self):
        plan = self.__plandf.copy()
        isindf12 = [False, False, False]
        isindf23 = [False, False, False]
        df = [False, False, False]
        df0 = pd.DataFrame(
            {
                "No": [np.nan],
                "action": ["Begin of Day1"],
                "location": [np.nan],
                "time": [np.nan],
                "power": [np.nan],
                "1": [np.nan],
                "2": [np.nan],
            }
        )
        for i in range(3):
            isindf12[i] = (
                self.__plandf[i]
                .isin(self.__task[self.__task["tag"] == "12s"]["name"].values)
                .any()
                .any()
            )
            isindf23[i] = (
                self.__plandf[i]
                .isin(self.__task[self.__task["tag"] == "23s"]["name"].values)
                .any()
                .any()
            )
        pakdf = [
            [
                [
                    self.__gen_packagedf(self.__package, "D1ss"),
                    self.__gen_packagedf(self.__package, "D1se"),
                ],
                [
                    self.__gen_packagedf(self.__package, "D1es"),
                    self.__gen_packagedf(self.__package, "D1ee"),
                ],
            ],
            [
                [
                    self.__gen_packagedf(self.__package, "D2ss"),
                    self.__gen_packagedf(self.__package, "D2se"),
                ],
                [
                    self.__gen_packagedf(self.__package, "D2es"),
                    self.__gen_packagedf(self.__package, "D2ee"),
                ],
            ],
            [
                [
                    self.__gen_packagedf(self.__package, "D3ss"),
                    self.__gen_packagedf(self.__package, "D3se"),
                ],
                [
                    self.__gen_packagedf(self.__package, "D3es"),
                    self.__gen_packagedf(self.__package, "D3ee"),
                ],
            ],
        ]
        for i in range(3):
            if isindf12[i]:
                if self.__plandf[i].shape[0] == 2:
                    temp = pd.concat(
                        [
                            self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1],
                            pakdf[i][1][0],
                            self.__plandf[i].iloc[-1:, :],
                        ]
                    )
                elif self.__plandf[i].shape[0] > 2:
                    temp = pd.concat(
                        [
                            self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1],
                            self.__plandf[i].iloc[1:-1, :],
                            pakdf[i][1][0],
                            self.__plandf[i].iloc[-1:, :],
                        ]
                    )
                time = sum(temp["time"])
                if time <= self.__12gap:
                    adddf = pd.DataFrame(
                        {
                            "No": [np.nan],
                            "action": [
                                f"Wait for {self.__format_number(self.__12gap-time)}s"
                            ],
                            "location": [
                                f"({self.__format_number(self.__pointdf.loc['探测起点1','X'])},{self.__format_number(self.__pointdf.loc['探测起点1','Y'])})"
                            ],
                            "time": [
                                float(
                                    self.__format_number(self.__12gap - time)
                                )
                            ],
                            "power": [0],
                            "1": ["√"],
                            "2": ["√"],
                        }
                    )
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                pakdf[i][1][0],
                                adddf,
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                self.__plandf[i].iloc[1:-1, :],
                                pakdf[i][1][0],
                                adddf,
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                else:
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                pakdf[i][1][0],
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                self.__plandf[i].iloc[1:-1, :],
                                pakdf[i][1][0],
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                time = "sum: {}".format(
                    self.__format_number(sum(plan[i]["time"]))
                )
                power = "sum: {}".format(
                    self.__format_number(sum(plan[i]["power"]))
                )
                df[i] = pd.DataFrame(
                    {
                        "No": [np.nan],
                        "action": ["xxx"],
                        "location": [np.nan],
                        "time": [time],
                        "power": [power],
                        "1": [np.nan],
                        "2": [np.nan],
                    }
                )
            elif isindf23[i]:
                if self.__plandf[i].shape[0] == 2:
                    temp = pd.concat(
                        [
                            self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1],
                            pakdf[i][1][0],
                            self.__plandf[i].iloc[-1:, :],
                        ]
                    )
                elif self.__plandf[i].shape[0] > 2:
                    temp = pd.concat(
                        [
                            self.__plandf[i].iloc[:1, :],
                            pakdf[i][0][1],
                            self.__plandf[i].iloc[1:-1, :],
                            pakdf[i][1][0],
                            self.__plandf[i].iloc[-1:, :],
                        ]
                    )
                time = sum(temp["time"])
                if time <= self.__23gap:
                    adddf = pd.DataFrame(
                        {
                            "No": [np.nan],
                            "action": [
                                f"Wait for {self.__format_number(self.__23gap-time)}s"
                            ],
                            "location": [
                                f"({self.__format_number(self.__pointdf.loc['探测起点1','X'])},{self.__format_number(self.__pointdf.loc['探测起点1','Y'])})"
                            ],
                            "time": [
                                float(
                                    self.__format_number(self.__23gap - time)
                                )
                            ],
                            "power": [0],
                            "1": ["√"],
                            "2": ["√"],
                        }
                    )
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                pakdf[i][1][0],
                                adddf,
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                self.__plandf[i].iloc[1:-1, :],
                                pakdf[i][1][0],
                                adddf,
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                else:
                    if self.__plandf[i].shape[0] == 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                pakdf[i][1][0],
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                    elif self.__plandf[i].shape[0] > 2:
                        plan[i] = pd.concat(
                            [
                                pakdf[i][0][0],
                                self.__plandf[i].iloc[:1, :],
                                pakdf[i][0][1],
                                self.__plandf[i].iloc[1:-1, :],
                                pakdf[i][1][0],
                                self.__plandf[i].iloc[-1:, :],
                                pakdf[i][1][1],
                            ]
                        )
                time = "sum: {}".format(
                    self.__format_number(sum(plan[i]["time"]))
                )
                power = "sum: {}".format(
                    self.__format_number(sum(plan[i]["power"]))
                )
                df[i] = pd.DataFrame(
                    {
                        "No": [np.nan],
                        "action": ["xxx"],
                        "location": [np.nan],
                        "time": [time],
                        "power": [power],
                        "1": [np.nan],
                        "2": [np.nan],
                    }
                )
            else:
                pakdf[i][0] = pd.concat(pakdf[i][0])
                pakdf[i][1] = pd.concat(pakdf[i][1])
                plan[i] = pd.concat(
                    [pakdf[i][0], self.__plandf[i], pakdf[i][1]]
                )
                time = "sum: {}".format(
                    self.__format_number(sum(plan[i]["time"]))
                )
                power = "sum: {}".format(
                    self.__format_number(sum(plan[i]["power"]))
                )
                df[i] = pd.DataFrame(
                    {
                        "No": [np.nan],
                        "action": ["xxx"],
                        "location": [np.nan],
                        "time": [time],
                        "power": [power],
                        "1": [np.nan],
                        "2": [np.nan],
                    }
                )
        df[0]["action"] = "Break between Day1 and Day2"
        df[1]["action"] = "Break between Day2 and Day3"
        df[2]["action"] = "End of Day3"
        self.__plandf = pd.concat(
            [df0, plan[0], df[0], plan[1], df[1], plan[2], df[2]]
        )
        self.__plandf.reset_index(drop=True, inplace=True)
        index0 = (
            self.__plandf[self.__plandf["action"] == "Begin of Day1"].index[0]
            + 2
        )
        index1 = (
            self.__plandf[
                self.__plandf["action"] == "Break between Day1 and Day2"
            ].index[0]
            + 2
        )
        index2 = (
            self.__plandf[
                self.__plandf["action"] == "Break between Day2 and Day3"
            ].index[0]
            + 2
        )
        index3 = (
            self.__plandf[self.__plandf["action"] == "End of Day3"].index[0]
            + 2
        )
        self.__voidindex = [int(index0), int(index1), int(index2), int(index3)]
        no = list(range(self.__plandf.shape[0]))
        for i in no:
            if i in [0, index1 - 2, index2 - 2, index3 - 2]:
                no[i] = np.nan
            elif i > index1 - 2 and i < index2 - 2:
                no[i] = i - 1
            elif i > index3 - 2:
                no[i] = i - 2
        self.__plandf["No"] = no
        return None

    def write_excel(self, path=None):
        if path == None:
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
        return None

    def schedule_frame(self):
        return self.__plandf.copy()

    @staticmethod
    def __finite_metric(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def __model_metric(self, *names):
        if self.__m is None:
            return None
        for name in names:
            try:
                value = getattr(self.__m, name)
            except Exception:
                continue
            metric = self.__finite_metric(value)
            if metric is not None:
                return metric
        return None

    def metrics(self):
        incumbent = self.__model_metric("ObjVal")
        best_bound = self.__model_metric("BestBnd", "BestBound")
        absolute_gap = (
            abs(incumbent - best_bound)
            if incumbent is not None and best_bound is not None
            else None
        )
        model_steps = {
            "gen_edges",
            "add_variables",
            "set_objective",
            "add_indegree_constrs",
            "add_outdegree_constrs",
            "add_equaldegree_constrs",
            "add_oodegree_constrs",
            "add_rtask_constrs",
            "add_otask_constrs",
            "add_remote_constrs",
            "add_time_constrs",
            "add_day_constrs",
            "add_power_constrs",
            "add_safe_constrs",
            "add_continuous_constr",
            "add_noii_constrs",
        }
        postprocess_steps = {
            "print_status",
            "proc_res",
            "cal_route",
            "add_package",
            "validate_schedule",
            "write_excel",
        }
        first_feasible = next(
            (
                event["elapsed_seconds"]
                for event in self.__progress_events
                if event.get("event") == "incumbent"
            ),
            None,
        )
        raw_status = getattr(self.__m, "Status", None)
        status_map = {
            getattr(cp.COPT, "OPTIMAL", object()): "optimal",
            getattr(cp.COPT, "INFEASIBLE", object()): "infeasible",
            getattr(cp.COPT, "UNBOUNDED", object()): "unbounded",
            getattr(cp.COPT, "INF_OR_UNB", object()): "infeasible_or_unbounded",
            getattr(cp.COPT, "TIMEOUT", object()): "time_limit",
            getattr(cp.COPT, "NODELIMIT", object()): "node_limit",
            getattr(cp.COPT, "INTERRUPTED", object()): "interrupted",
            getattr(cp.COPT, "NUMERICAL", object()): "numerical_error",
        }
        solver_version = ".".join(
            str(getattr(cp.COPT, name, "unknown"))
            for name in ("VERSION_MAJOR", "VERSION_MINOR", "VERSION_TECHNICAL")
        )
        return {
            "solver_name": "copt",
            "solver_version": solver_version,
            "termination_status": status_map.get(raw_status, str(raw_status)),
            "model_build_seconds": sum(
                value for key, value in self.__step_timings.items() if key in model_steps
            ),
            "solve_seconds": self.__model_metric("SolvingTime", "Runtime"),
            "postprocess_seconds": sum(
                value for key, value in self.__step_timings.items() if key in postprocess_steps
            ),
            "algorithm_total_seconds": self.__run_elapsed,
            "first_feasible_seconds": first_feasible,
            "first_feasible_observed": first_feasible is not None,
            "incumbent_objective": incumbent,
            "best_bound": best_bound,
            "absolute_gap": absolute_gap,
            "relative_gap": self.__model_metric("BestGap", "MipGap", "MIPGap"),
            "node_count": self.__model_metric("NodeCnt", "NodeCount"),
            "model_state_count": len(self.__point),
            "model_edge_count": len(self.__edges),
            "model_variable_count": sum(
                len(values)
                for values in (
                    self.__x,
                    self.__W,
                    self.__Q,
                    self.__Ws1,
                    self.__Ws2,
                    self.__Ws3,
                    self.__Qs1,
                    self.__Qs2,
                    self.__Qs3,
                )
            ),
            "solver_reported_variable_count": self.__model_metric("Cols", "NumVars"),
            "model_constraint_count": self.__model_metric("Rows", "NumConstrs"),
            "time_limit_seconds": None
            if self.__time_limit == np.inf
            else self.__finite_metric(self.__time_limit),
            "threads": self.__parallel_workers,
            "warm_start_enabled": False,
            "warm_start_accepted": False,
            "warm_start_source": None,
            "fallback_used": False,
            "native_solution_found": incumbent is not None,
            "business_validation_passed": getattr(
                self, "_task_optimize__business_validation_passed", None
            ),
            "step_timings": dict(self.__step_timings),
        }

    def progress_events(self):
        return [dict(event) for event in self.__progress_events]

    def close(self):
        model = getattr(self, "_task_optimize__m", None)
        env = getattr(self, "_task_optimize__env", None)
        try:
            dispose_model = getattr(model, "dispose", None)
            if callable(dispose_model):
                dispose_model()
        finally:
            # Keep a fallback for coptpy builds without explicit dispose().
            # Dropping the final model reference before closing the environment
            # also prevents a model from retaining a license token.
            self.__m = None
            model = None
            gc.collect()
            dispose_env = getattr(env, "dispose", None)
            if callable(dispose_env):
                dispose_env()
            else:
                close_env = getattr(env, "close", None)
                if callable(close_env):
                    close_env()
            self.__env = None
        return None

    def __validate_business_schedule(self, schedule_df, label="schedule"):
        if not hasattr(self, "_task_optimize__RawTTime"):
            return True
        task_by_name = {
            str(row["name"]).strip(): row
            for _, row in self.__task.iterrows()
            if not str(row["name"]).startswith("void")
        }
        package_names = {str(name).strip() for name in self.__package["name"].values}
        day = 0
        totals = [[0.0, 0.0] for _ in range(3)]
        task_entries = [[] for _ in range(3)]
        remote_count = 0
        for _, row in schedule_df.iterrows():
            action = row.get("action")
            if pd.isna(action):
                continue
            action = str(action).strip()
            if action == "Begin of Day1":
                day = 0
                continue
            if action == "Break between Day1 and Day2":
                day = 1
                continue
            if action == "Break between Day2 and Day3":
                day = 2
                continue
            if action == "End of Day3":
                continue
            if day not in {0, 1, 2}:
                continue
            row_time = pd.to_numeric(row.get("time"), errors="coerce")
            row_power = pd.to_numeric(row.get("power"), errors="coerce")
            if pd.notna(row_time):
                totals[day][0] += float(row_time)
            if pd.notna(row_power):
                totals[day][1] += float(row_power)
            if (
                action not in package_names
                and not action.startswith("Travel from")
                and not action.startswith("Wait for")
                and action not in {"xxx"}
                and action in task_by_name
            ):
                tag = task_by_name[action]["tag"]
                tag = None if pd.isna(tag) else str(tag).strip()
                task_entries[day].append((action, tag))
                if bool(task_by_name[action]["remote"]):
                    remote_count += 1

        violations = []
        for idx, (used_time, used_power) in enumerate(totals):
            if used_time > self.__RawTTime[idx] + 1e-6:
                violations.append(
                    f"day{idx + 1}_time={used_time:.6f}>{self.__RawTTime[idx]:.6f}"
                )
            if used_power > self.__RawTPower[idx] + 1e-6:
                violations.append(
                    f"day{idx + 1}_power={used_power:.6f}>{self.__RawTPower[idx]:.6f}"
                )
        if self.__remtaskindex and remote_count < 1:
            violations.append("remote_count=0<1")

        tag_days = {"12s": [], "12e": [], "23s": [], "23e": []}
        for idx, entries in enumerate(task_entries):
            for pos, (_, tag) in enumerate(entries):
                if tag not in tag_days:
                    continue
                tag_days[tag].append(idx)
                if tag in {"12s", "12e"} and idx not in {0, 1}:
                    violations.append(f"{tag}_invalid_day={idx + 1}")
                if tag in {"23s", "23e"} and idx not in {1, 2}:
                    violations.append(f"{tag}_invalid_day={idx + 1}")
                if tag in {"12s", "23s"} and pos != 0:
                    violations.append(f"{tag}_not_day_start=day{idx + 1}")
                if tag in {"12e", "23e"} and pos != len(entries) - 1:
                    violations.append(f"{tag}_not_day_end=day{idx + 1}")

        for start_tag, end_tag in [("12s", "12e"), ("23s", "23e")]:
            if tag_days[start_tag] and tag_days[end_tag] and tag_days[start_tag] != tag_days[end_tag]:
                violations.append(
                    f"{start_tag}_{end_tag}_not_same_day={tag_days[start_tag]}!={tag_days[end_tag]}"
                )
        if set(tag_days["12s"]) & set(tag_days["23s"]):
            violations.append("12s_23s_same_day")
        if set(tag_days["12e"]) & set(tag_days["23e"]):
            violations.append("12e_23e_same_day")

        day_time = ",".join(f"{value[0]:.5f}" for value in totals)
        day_power = ",".join(f"{value[1]:.5f}" for value in totals)
        if violations:
            print(
                f"[eac] VALIDATE {label} failed day_time={day_time} day_power={day_power} "
                f"violations={';'.join(violations)}",
                flush=True,
            )
            return False
        print(
            f"[eac] VALIDATE {label} ok day_time={day_time} day_power={day_power}",
            flush=True,
        )
        return True

    def validate_schedule(self):
        self.__business_validation_passed = self.__validate_business_schedule(
            self.__plandf, "output"
        )
        if not self.__business_validation_passed:
            raise MIPError("Output schedule violates business validation rules")
        return None


    def __run_logged_step(self, index, total, step, func):
        print(f"[eac] START {index:02d}/{total:02d} {step}", flush=True)
        started = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.__step_timings[step] = elapsed
            print(
                f"[eac] FAIL  {index:02d}/{total:02d} {step} elapsed={elapsed:.3f}s "
                f"error={exc.__class__.__name__}: {exc}",
                flush=True,
            )
            raise
        elapsed = time.perf_counter() - started
        self.__step_timings[step] = elapsed
        print(f"[eac] END   {index:02d}/{total:02d} {step} elapsed={elapsed:.3f}s", flush=True)
        return result

    def run(self):
        print(
            f"[eac] RUN start objective={self.__objective} timeLimit={self.__time_limit}",
            flush=True,
        )
        total_started = time.perf_counter()
        steps = [
            ("test_IO", self.test_IO),
            ("read_info", self.read_info),
            ("read_task", self.read_task),
            ("read_package", self.read_package),
            ("read_point", self.read_point),
            ("gen_void_point", self.gen_void_point),
            ("check_remote", self.check_remote),
            ("divide_task", self.divide_task),
            ("drop_O", self.drop_O),
            ("gen_point", self.gen_point),
            ("gen_edges", self.gen_edges),
            ("add_variables", self.add_variables),
            ("set_objective", self.set_objective),
            ("add_indegree_constrs", self.add_indegree_constrs),
            ("add_outdegree_constrs", self.add_outdegree_constrs),
            ("add_equaldegree_constrs", self.add_equaldegree_constrs),
            ("add_oodegree_constrs", self.add_oodegree_constrs),
            ("add_rtask_constrs", self.add_rtask_constrs),
            ("add_otask_constrs", self.add_otask_constrs),
            ("add_remote_constrs", self.add_remote_constrs),
            ("add_time_constrs", self.add_time_constrs),
            ("add_day_constrs", self.add_day_constrs),
            ("add_power_constrs", self.add_power_constrs),
            ("add_safe_constrs", self.add_safe_constrs),
            ("add_continuous_constr", self.add_continuous_constr),
            ("add_noii_constrs", self.add_noii_constrs),
            ("run_opt", self.run_opt),
            ("print_status", self.print_status),
            ("proc_res", self.proc_res),
            ("cal_route", self.cal_route),
            ("add_package", self.add_package),
            ("validate_schedule", self.validate_schedule),
        ]
        if self.__write_output:
            steps.append(("write_excel", self.write_excel))
        total = len(steps)
        for index, (step, func) in enumerate(steps, start=1):
            self.__run_logged_step(index, total, step, func)
        self.__run_elapsed = time.perf_counter() - total_started
        print(f"[eac] RUN end elapsed={self.__run_elapsed:.3f}s", flush=True)
        return None


def solve(case, mode="normal"):
    """Run the COPT exact algorithm on UnifiedCase input."""
    if mode != "normal":
        raise NotImplementedError("eac only supports algorithm.mode=normal")

    from .schedule import (
        SchedulePlan,
        build_legacy_bundle,
        legacy_schedule_to_rows,
    )

    objective = case.config.algorithm.obj
    if objective not in {CONST.MAX_REVENUE, CONST.MIN_TIME, CONST.MIN_POWER}:
        objective = CONST.MAX_REVENUE

    bundle = build_legacy_bundle(case)
    kwargs = {
        "obj": objective,
        "decimal": case.config.algorithm.decimal,
        "autoSave": False,
        "writeOutput": False,
        "iisPath": str(case.output_dir / "eac_infeasible.ilp"),
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
    algorithm_raw = case.config.raw.get("algorithm", {})
    parallel_workers = (
        algorithm_raw.get("parallelWorkers")
        or algorithm_raw.get("workers")
        or algorithm_raw.get("threads")
    )
    if parallel_workers is not None:
        kwargs["parallelWorkers"] = parallel_workers

    with _EAC_SOLVE_LOCK:
        optimizer = task_optimize(**kwargs)
        try:
            optimizer.run()
            schedule_df = optimizer.schedule_frame()
            rows = legacy_schedule_to_rows(case, schedule_df)
            objective_value = getattr(optimizer, "_task_optimize__objvalue", None)
            metrics = optimizer.metrics()
            progress = optimizer.progress_events()
        finally:
            optimizer.close()

    return SchedulePlan(
        steps=[],
        rows=rows,
        objective_value=None if objective_value is None else float(objective_value),
        metrics=metrics,
        progress=progress,
    )


if __name__ == "__main__":
    opt = task_optimize(CONST.MAX_REVENUE, timeLimit=60 * 60 * 4)
    opt.run()
