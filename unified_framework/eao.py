# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:53:23 2024

@author: chen wentian
"""
import os
import signal
import threading
import time
import numpy as np
import pandas as pd
from itertools import product
from openpyxl.styles import PatternFill
from pyscipopt import (
    Eventhdlr,
    Model,
    SCIP_EVENTTYPE,
    SCIP_PARAMEMPHASIS,
    SCIP_PARAMSETTING,
    quicksum,
)

from .models import MIPError, SubtourError


class _ScipStatus:
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    TIMELIMIT = "timelimit"
    USER_INTERRUPT = "userinterrupt"


class _ScipCallback:
    MIPSOL = "MIPSOL"


class _ScipGRB:
    BINARY = "B"
    CONTINUOUS = "C"
    MAXIMIZE = "maximize"
    OPTIMAL = _ScipStatus.OPTIMAL
    INFEASIBLE = _ScipStatus.INFEASIBLE
    TIME_LIMIT = _ScipStatus.TIMELIMIT
    Callback = _ScipCallback


class _TupleList(list):
    def select(self, *patterns):
        return _TupleList(
            item
            for item in self
            if len(item) == len(patterns)
            and all(pattern == "*" or value == pattern for value, pattern in zip(item, patterns))
        )


class _VarDict(dict):
    def __init__(self, arity):
        super().__init__()
        self._arity = arity
        self._indexes = [dict() for _ in range(arity)]

    def add(self, key, var):
        self[key] = var
        for pos, value in enumerate(key):
            self._indexes[pos].setdefault(value, set()).add(key)

    def sum(self, *patterns):
        if len(patterns) != self._arity:
            return quicksum([])

        candidates = None
        for pos, pattern in enumerate(patterns):
            if pattern == "*":
                continue
            if isinstance(pattern, (list, tuple, set, _TupleList)):
                keys = set()
                for value in pattern:
                    keys.update(self._indexes[pos].get(value, ()))
            else:
                keys = set(self._indexes[pos].get(pattern, ()))
            candidates = keys if candidates is None else candidates & keys
            if not candidates:
                return quicksum([])

        if candidates is None:
            return quicksum(self.values())
        return quicksum(self[key] for key in candidates)


def _match_pattern(value, pattern):
    if pattern == "*":
        return True
    if isinstance(pattern, (list, tuple, set, _TupleList)):
        return value in pattern
    return value == pattern


class _ScipParams:
    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __setattr__(self, name, value):
        model = object.__getattribute__(self, "_model")
        if name == "TimeLimit":
            model.setParam("limits/time", float(value))
        elif name == "SolutionLimit":
            model.setParam("limits/solutions", int(value))
        elif name in {"MIPFocus", "Heuristics"}:
            # Gurobi-specific tuning parameters have no direct SCIP equivalent here.
            return
        else:
            object.__setattr__(self, name, value)


class _ScipModel:
    def __init__(self, name):
        self._model = Model(name)
        self._model.setParam("display/verblevel", 0)
        self._model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
        self._model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        self.Params = _ScipParams(self._model)
        self.Status = None
        self.SolCount = 0
        self.objVal = None
        self._event_handlers = []

    def addVars(self, keys, vtype="C", name=""):
        key_list = [tuple(key) for key in keys]
        arity = len(key_list[0]) if key_list else 0
        variables = _VarDict(arity)
        for key_tuple in key_list:
            suffix = ",".join(str(part) for part in key_tuple)
            variables.add(
                key_tuple,
                self._model.addVar(
                    vtype=vtype,
                    name=f"{name}[{suffix}]" if name else "",
                ),
            )
        return variables

    def addConstr(self, cons, name=""):
        if isinstance(cons, bool):
            if cons:
                return None
            raise MIPError(f"Infeasible constant constraint: {name}")
        return self._model.addCons(cons, name=name)

    def addConstrs(self, constrs, name=""):
        added = []
        for idx, cons in enumerate(constrs):
            if isinstance(cons, bool):
                if cons:
                    continue
                raise MIPError(f"Infeasible constant constraint: {name}[{idx}]")
            added.append(self._model.addCons(cons, name=f"{name}[{idx}]" if name else ""))
        return added

    def addConsIndicator(self, cons, binvar=None, activeone=True, name=""):
        return self._model.addConsIndicator(cons, binvar=binvar, activeone=activeone, name=name)

    def setObjective(self, expr, sense):
        return self._model.setObjective(expr, sense=sense)

    def update(self):
        return None

    def printStats(self):
        return None

    def optimize(self, callback=None):
        try:
            self._model.optimize()
        finally:
            self.Status = self._model.getStatus()
            self.SolCount = self._model.getNSols()
            if self.SolCount >= 1:
                self.objVal = self._model.getObjVal()

    def interruptSolve(self):
        return self._model.interruptSolve()

    def computeIIS(self):
        return None

    def write(self, path):
        if path.endswith(".sol") and self.SolCount >= 1:
            self._model.writeBestSol(path)
        else:
            self._model.writeProblem(path)

    def getVal(self, var):
        return self._model.getVal(var)

    def getBestSol(self):
        return self._model.getBestSol()

    def getSolVal(self, sol, var):
        return self._model.getSolVal(sol, var)

    def getSolObjVal(self, sol):
        return self._model.getSolObjVal(sol)

    def getSolvingTime(self):
        return self._model.getSolvingTime()

    def createPartialSol(self):
        return self._model.createPartialSol()

    def createSol(self):
        return self._model.createSol()

    def setSolVal(self, sol, var, value):
        return self._model.setSolVal(sol, var, value)

    def trySol(self, sol, printreason=False):
        return self._model.addSol(sol, free=True)

    def checkSol(self, sol):
        return self._model.checkSol(
            sol,
            printreason=True,
            completely=True,
            checkbounds=True,
            checkintegrality=True,
            checklprows=True,
            original=False,
        )

    def includeBestSolHandler(self, owner):
        handler = _BestSolEventHandler(owner)
        self._event_handlers.append(handler)
        self._model.includeEventhdlr(
            handler,
            "eaoBestSolHandler",
            "Autosaves every new SCIP incumbent solution for eao",
        )


class _ScipCompat:
    GRB = _ScipGRB
    Model = _ScipModel
    quicksum = staticmethod(quicksum)
    tuplelist = _TupleList


gp = _ScipCompat()


class _BestSolEventHandler(Eventhdlr):
    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
            self._owner._save_incumbent_solution()


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
    ):
        self.__bigM = 10**5
        self.__objective = obj
        self.__Obj = [CONST.MAX_REVENUE, CONST.MIN_TIME, CONST.MIN_POWER]
        self.__time_limit = None if timeLimit == np.inf else timeLimit
        self.__m = gp.Model("schedule-optimization")
        if timeLimit != np.inf:
            self.__m.Params.TimeLimit = timeLimit
        if solNum != np.inf:
            self.__m.Params.SolutionLimit = solNum
        if MIPFocus != None:
            self.__m.Params.MIPFocus = MIPFocus
        if heuristics != None:
            self.__m.Params.Heuristics = heuristics
        self.__autosavestate = autoSave
        self.__decimal = decimal
        self.__lppath, self.__solpath = lpPath, solPath
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
        self.__fallback_schedule_df = None
        self.__fallback_objvalue = None
        self.__using_fallback = False
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
        self.__RawTTime = self.__TTime.copy()
        self.__RawTPower = self.__TPower.copy()
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
        self.__task["location"] = self.__task["location"].replace(
            np.nan, ",".join(self.__pointdf.index.values)
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
            while sortedseries[start] >= self.__MDistance:
                start += 1
            pts = (
                self.__dmatrix.sort_values(by=["探测起点1"], ascending=False)
                .iloc[start : start + 5, :]
                .index.values
            )
            r, _ = self.__cartesian_to_polar(
                self.__from_distance(self.__MDistance)[0],
                self.__from_distance(self.__MDistance)[1],
                (0, 0),
            )
            pts = [
                self.__cartesian_to_polar(
                    self.__pointdf.loc[pt, "X"],
                    self.__pointdf.loc[pt, "Y"],
                    (
                        self.__pointdf.loc["探测起点1", "X"],
                        self.__pointdf.loc["探测起点1", "Y"],
                    ),
                )
                for pt in pts
            ]
            pts = [
                self.__polar_to_cartesian(
                    r,
                    pt[1],
                    (
                        self.__pointdf.loc["探测起点1", "X"],
                        self.__pointdf.loc["探测起点1", "Y"],
                    ),
                )
                for pt in pts
            ]
            newpt = pd.DataFrame(
                pts,
                columns=["X", "Y"],
                index=pd.Index(
                    ["新最远探测点1", "新最远探测点2", "新最远探测点3", "新最远探测点4", "新最远探测点5"],
                    name="name",
                ),
            )
            raise MaxDistanceError(
                f"No point meets the max distance requirement. Recommended points have been generated in {path}",
                newpt,
                path,
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
        self.__point = gp.tuplelist()
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
        source_point, source_task = source
        target_point, target_task = target
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
        self.__edges = gp.tuplelist()
        for source in self.__point:
            for target in self.__point:
                if self.__edge_is_allowed(source, target, stage_ranges):
                    self.__edges.append((*source, *target))
        return None

    def add_variables(self):
        edges = self.__edges
        point = self.__point
        self.__x = self.__m.addVars(edges, vtype=gp.GRB.BINARY, name="x")
        self.__W = self.__m.addVars(point, vtype=gp.GRB.CONTINUOUS, name="W")
        self.__Q = self.__m.addVars(point, vtype=gp.GRB.CONTINUOUS, name="Q")
        self.__Ws1 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Wsafe1"
        )
        self.__Ws2 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Wsafe2"
        )
        self.__Ws3 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Wsafe3"
        )
        self.__Qs1 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Qsafe1"
        )
        self.__Qs2 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Qsafe2"
        )
        self.__Qs3 = self.__m.addVars(
            point, vtype=gp.GRB.BINARY, name="Qsafe3"
        )
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
        expr = gp.quicksum(expr)
        self.__m.setObjective(expr, gp.GRB.MAXIMIZE)
        return expr

    def add_indegree_constrs(self):
        temp = gp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[0])
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) <= 1 for i, k in temp),
            name="indegree0",
        )
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 0 for i, k in temp),
            name="indegree1",
        )

    def add_outdegree_constrs(self):
        temp = gp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[-1])
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") <= 1 for i, k in temp),
            name="outdegree0",
        )
        temp = [self.__opoint[-1]]
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 0 for i, k in temp),
            name="outdegree1",
        )
        return None

    def add_equaldegree_constrs(self):
        temp = gp.tuplelist(self.__point.copy())
        temp.remove(self.__opoint[0])
        temp.remove(self.__opoint[-1])
        self.__m.addConstrs(
            (
                self.__x.sum(i, k, "*", "*") == self.__x.sum("*", "*", i, k)
                for i, k in temp
            ),
            name="equaldegree",
        )
        return None

    def add_oodegree_constrs(self):
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 1 for i, k in temp),
            name="oodegree0",
        )
        temp = [self.__opoint[-1]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 1 for i, k in temp),
            name="oodegree1",
        )
        temp = [self.__opoint[1], self.__opoint[2]]
        self.__m.addConstrs(
            (self.__x.sum("*", "*", i, k) == 1 for i, k in temp),
            name="oodegree2",
        )
        self.__m.addConstrs(
            (self.__x.sum(i, k, "*", "*") == 1 for i, k in temp),
            name="oodegree3",
        )
        temp = self.__opoint
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[-1], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[-1], j, k2) in self.__x
            ),
            name="oodegree4",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[2], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[2], j, k2) in self.__x
            ),
            name="oodegree5",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (
                self.__x[*self.__opoint[1], j, k2] == 0
                for j, k2 in temp
                if (*self.__opoint[1], j, k2) in self.__x
            ),
            name="oodegree6",
        )
        return None

    def add_rtask_constrs(self):
        self.__m.addConstrs(
            (self.__x.sum("*", "*", "*", k) == 1 for k in self.__reqtaskindex),
            name="rtask",
        )
        return None

    def add_otask_constrs(self):
        self.__m.addConstrs(
            (self.__x.sum("*", "*", "*", k) <= 1 for k in self.__opttaskindex),
            name="otask",
        )
        return None

    def add_remote_constrs(self):
        self.__m.addConstr(
            gp.quicksum(
                (self.__x.sum("*", "*", "*", k) for k in self.__remtaskindex)
            )
            >= 1,
            name="remote",
        )
        return None

    def add_time_constrs(self):
        point = gp.tuplelist(self.__point.copy())
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__W.sum(i, k) == 0 for i, k in temp), name="time0"
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
                <= self.__bigM * (1 - self.__x[i, a, j, b])
                for i, a, j, b in self.__edges
            ),
            name="time1",
        )
        self.__m.addConstr(
            self.__W[*self.__opoint[1]] <= self.__TTime[0],
            name="time2",
        )
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
            name="time4",
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
            name="time6",
        )
        return None

    def add_day_constrs(self):
        point = gp.tuplelist(self.__point.copy())
        if self.__daytaskindex[0] != []:
            for k in self.__daytaskindex[0]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    name=f"day1{k}",
                )
        if self.__daytaskindex[1] != []:
            for k in self.__daytaskindex[1]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    name=f"day21{k}",
                )
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    name=f"day22{k}",
                )
        if self.__daytaskindex[2] != []:
            for k in self.__daytaskindex[2]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    name=f"day3{k}",
                )
        if self.__daytaskindex[3] != []:
            for k in self.__daytaskindex[3]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) <= self.__W[*self.__opoint[2]]
                        for p in pt
                    ),
                    name=f"day12{k}",
                )
        if self.__daytaskindex[4] != []:
            for k in self.__daytaskindex[4]:
                pt = list(point.select("*", k))
                self.__m.addConstrs(
                    (
                        self.__W.sum(p[0], p[1]) >= self.__W[*self.__opoint[1]]
                        for p in pt
                    ),
                    name=f"day23{k}",
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
        point = gp.tuplelist(self.__point.copy())
        temp = [self.__opoint[0]]
        self.__m.addConstrs(
            (self.__Q.sum(i, k) == 0 for i, k in temp), name="power0"
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
                <= self.__bigM * (1 - self.__x[i, a, j, b])
                for i, a, j, b in self.__edges
            ),
            name="power1",
        )
        self.__m.addConstr(
            self.__Q[*self.__opoint[1]] <= self.__TPower[0],
            name="power2",
        )
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
            name="power4",
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
            name="power6",
        )
        return None

    def add_safe_constrs(self):
        point = gp.tuplelist(self.__point.copy())
        eps = 0.0001
        M = self.__bigM + eps
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
            if (i, k) in self.__opoint:
                continue
            selected = self.__x.sum("*", "*", i, k)
            inactive = M * (1 - selected)
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                >= self.__W[i, k] + eps - M * (1 - self.__Ws1[i, k]) - inactive,
                name=f"safetimeday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                <= self.__W[i, k] + M * self.__Ws1[i, k] + inactive,
                name=f"safetimeday1_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k] - task_time[k] + safe_time_to_base[i]
                <= self.__TTime[0] + M * (1 - self.__Ws1[i, k]) + inactive,
                name=f"safetimeday1_constr[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                >= self.__W[*self.__opoint[2]]
                + eps
                - M * (1 - self.__Ws3[i, k])
                - inactive,
                name=f"safetimeday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                <= self.__W[*self.__opoint[2]] + M * self.__Ws3[i, k] + inactive,
                name=f"safetimeday3_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                - self.__W[*self.__opoint[2]]
                - task_time[k]
                + safe_time_to_base[i]
                <= self.__TTime[2] + M * (1 - self.__Ws3[i, k]) + inactive,
                name=f"safetimeday3_constr[{i},{k}]",
            )
            self.__m.addConstr(
                1
                >= self.__Ws1[i, k]
                + self.__Ws3[i, k]
                + eps
                - M * (1 - self.__Ws2[i, k])
                - inactive,
                name=f"safetimeday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Ws1[i, k]
                + self.__Ws3[i, k]
                + M * self.__Ws2[i, k]
                + inactive,
                name=f"safetimeday2_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                - self.__W[*self.__opoint[1]]
                - task_time[k]
                + safe_time_to_base[i]
                <= self.__TTime[1] + M * (1 - self.__Ws2[i, k]) + inactive,
                name=f"safetimeday2_constr[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[*self.__opoint[1]]
                >= self.__Q[i, k] + eps - M * (1 - self.__Qs1[i, k]) - inactive,
                name=f"safepowerday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[*self.__opoint[1]]
                <= self.__Q[i, k] + M * self.__Qs1[i, k] + inactive,
                name=f"safepowerday1_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k] - task_power[k] + safe_power_from_base[i]
                <= self.__TPower[0] + M * (1 - self.__Qs1[i, k]) + inactive,
                name=f"safepowerday1_constr[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                >= self.__Q[*self.__opoint[2]]
                + eps
                - M * (1 - self.__Qs3[i, k])
                - inactive,
                name=f"safepowerday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                <= self.__Q[*self.__opoint[2]] + M * self.__Qs3[i, k] + inactive,
                name=f"safepowerday3_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                - self.__Q[*self.__opoint[2]]
                - task_power[k]
                + safe_power_from_base[i]
                <= self.__TPower[2] + M * (1 - self.__Qs3[i, k]) + inactive,
                name=f"safepowerday3_constr[{i},{k}]",
            )
            self.__m.addConstr(
                1
                >= self.__Qs1[i, k]
                + self.__Qs3[i, k]
                + eps
                - M * (1 - self.__Qs2[i, k])
                - inactive,
                name=f"safepowerday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Qs1[i, k]
                + self.__Qs3[i, k]
                + M * self.__Qs2[i, k]
                + inactive,
                name=f"safepowerday2_bigM_constr1[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                - self.__Q[*self.__opoint[1]]
                - task_power[k]
                + safe_power_from_base[i]
                <= self.__TPower[1] + M * (1 - self.__Qs2[i, k]) + inactive,
                name=f"safepowerday2_constr[{i},{k}]",
            )
        return None

    def add_continuous_constr(self):
        if self.__Mincontinuous > 0:
            expr = [self.__x.sum("*", "*", "*", k) for k in self.__contaskindex]
            self.__m.addConstr(
                gp.quicksum(expr) == self.__Mincontinuous, name="continuous"
            )
        return None

    def add_noii_constrs(self):
        point = gp.tuplelist(self.__point.copy())
        self.__m.addConstrs(
            (
                self.__x[i, k, i, k] == 0
                for i, k in point
                if (i, k, i, k) in self.__x
            ),
            name="noii",
        )
        return None

    def __callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            if self.__outputpath[-5:] == ".xlsx":
                path = (
                    self.__outputpath[:-5]
                    + "-sol"
                    + str(self.__solcount)
                    + ".xlsx"
                )
                path = os.path.join("autosave", path)
            elif self.__outputpath[-4:] == ".xls":
                path = (
                    self.__outputpath[:-4]
                    + "-sol"
                    + str(self.__solcount)
                    + ".xls"
                )
                path = os.path.join("autosave", path)
            Permitted = True
            if os.access(path, os.F_OK):
                try:
                    pd.read_excel(path)
                except PermissionError:
                    Permitted = False
            if Permitted:
                x = model.cbGetSolution(self.__x)
                W = model.cbGetSolution(self.__W)
                autosavethread = threading.Thread(
                    target=self.__autosave, args=(x, W, path)
                )
                autosavethread.start()
                autosavethread.join()
            self.__solcount += 1
        return None

    def __autosave(self, x, W, path):
        self.proc_res(x, W)
        self.cal_route()
        self.add_package()
        self.write_excel(path)
        return None

    def __next_autosave_path(self):
        output_path = os.path.abspath(self.__outputpath)
        output_dir = os.path.dirname(output_path) or os.getcwd()
        stem = os.path.splitext(os.path.basename(output_path))[0] or "schedule"
        autosave_dir = os.path.join(output_dir, "autosave")
        os.makedirs(autosave_dir, exist_ok=True)
        return os.path.join(autosave_dir, f"{stem}-sol{self.__solcount}.xlsx")

    def _save_incumbent_solution(self):
        try:
            sol = self.__m.getBestSol()
            if sol is None:
                return None
            obj_value = self.__m.getSolObjVal(sol)
            x = {key: self.__m.getSolVal(sol, var) for key, var in self.__x.items()}
            W = {key: self.__m.getSolVal(sol, var) for key, var in self.__W.items()}
            path = self.__next_autosave_path()
            self.__autosave(x, W, path)
            csv_path = os.path.splitext(path)[0] + ".csv"
            self.__plandf.to_csv(csv_path, index=False)
            self.__objvalue = obj_value
            self.__print_solution_summary(
                f"incumbent sol={self.__solcount}",
                self.__plandf,
                obj_value=obj_value,
                path=path,
            )
            print(
                f"[eao] INCUMBENT sol={self.__solcount} time={self.__m.getSolvingTime():.3f}s "
                f"obj={obj_value} xlsx={path} csv={csv_path}",
                flush=True,
            )
            self.__solcount += 1
        except Exception as exc:
            print(
                f"[eao] INCUMBENT autosave failed error={exc.__class__.__name__}: {exc}",
                flush=True,
            )
        return None

    def __route_values_from_schedule(self, schedule_df):
        task_by_name = {
            str(row["name"]).strip(): idx
            for idx, row in self.__task.iterrows()
            if not str(row["name"]).startswith("void")
        }
        package_names = {str(name).strip() for name in self.__package["name"].values}
        point_no_by_name = self.__pointdf["No"].to_dict()
        point_name_by_no = self.__pointdf.reset_index().set_index("No")["index"].to_dict()
        task_time = self.__task.set_index("No")["time"].to_dict()
        task_power = self.__task.set_index("No")["power"].to_dict()
        continuous_selected = 0
        depot_by_day = {
            1: self.__opoint[0][0],
            2: self.__opoint[1][0],
            3: self.__opoint[2][0],
        }
        tag_depot_by_day = {
            ("12s", 1): self.__opoint[0][0],
            ("12s", 2): self.__opoint[1][0],
            ("12e", 1): self.__opoint[1][0],
            ("12e", 2): self.__opoint[2][0],
            ("23s", 2): self.__opoint[1][0],
            ("23s", 3): self.__opoint[2][0],
            ("23e", 2): self.__opoint[2][0],
            ("23e", 3): self.__opoint[3][0],
        }
        day = 1
        route_by_day = {1: [self.__opoint[0]], 2: [self.__opoint[1]], 3: [self.__opoint[2]]}
        for _, row in schedule_df.iterrows():
            action = row.get("action")
            if pd.isna(action):
                continue
            action = str(action).strip()
            if action == "Begin of Day1":
                day = 1
                continue
            if action == "Break between Day1 and Day2":
                day = 2
                continue
            if action == "Break between Day2 and Day3":
                day = 3
                continue
            if action in {"End of Day3", "xxx"}:
                continue
            if action.startswith("Travel from") or action.startswith("Wait for"):
                continue
            if action in package_names or action not in task_by_name:
                continue

            task_index = task_by_name[action]
            task_row = self.__task.loc[task_index]
            if bool(task_row["continuous"]) and not bool(task_row["required"]):
                if continuous_selected >= self.__Mincontinuous:
                    continue
                continuous_selected += 1
            location = row.get("location")
            if pd.isna(location):
                continue
            location = str(location).strip()
            if location == "探测起点":
                tag = None if pd.isna(task_row["tag"]) else str(task_row["tag"]).strip()
                point_no = tag_depot_by_day.get((tag, day), depot_by_day[day])
            else:
                point_no = point_no_by_name.get(location)
            if point_no is None or (point_no, task_index) not in self.__point:
                continue
            route_by_day[day].append((point_no, task_index))

        route_by_day[1].append(self.__opoint[1])
        route_by_day[2].append(self.__opoint[2])
        route_by_day[3].append(self.__opoint[3])

        selected_edges = set()
        selected_nodes = set()
        w_values = {point: 0.0 for point in self.__point}
        q_values = {point: 0.0 for point in self.__point}
        for route in route_by_day.values():
            current_w = 0.0 if route[0] == self.__opoint[0] else w_values[route[0]]
            current_q = 0.0 if route[0] == self.__opoint[0] else q_values[route[0]]
            selected_nodes.add(route[0])
            for source, target in zip(route, route[1:]):
                edge = (*source, *target)
                if edge not in self.__x:
                    raise ValueError(f"Warm-start edge is not in SCIP graph: {edge}")
                source_name = point_name_by_no[source[0]]
                target_name = point_name_by_no[target[0]]
                current_w += float(self.__tmatrix.loc[source_name, target_name]) + float(task_time[target[1]])
                current_q += float(self.__pmatrix.loc[source_name, target_name]) + float(task_power[target[1]])
                selected_edges.add(edge)
                selected_nodes.add(target)
                w_values[target] = current_w
                q_values[target] = current_q

        w_day1_end = w_values[self.__opoint[1]]
        w_day2_end = w_values[self.__opoint[2]]
        q_day1_end = q_values[self.__opoint[1]]
        q_day2_end = q_values[self.__opoint[2]]
        for point in self.__point:
            if point in selected_nodes:
                continue
            stage_min, stage_max = self.__task_stage_range(point[1])
            if stage_min >= 2:
                w_values[point] = w_day2_end
                q_values[point] = q_day2_end
            elif stage_min >= 1:
                w_values[point] = w_day1_end
                q_values[point] = q_day1_end
            else:
                w_values[point] = 0.0
                q_values[point] = 0.0
        return selected_edges, w_values, q_values, selected_nodes

    def __safe_binary_values(self, values):
        eps = 0.0001
        first_break = values[self.__opoint[1]]
        second_break = values[self.__opoint[2]]
        safe1, safe2, safe3 = {}, {}, {}
        for point, value in values.items():
            if value <= first_break:
                safe1[point], safe2[point], safe3[point] = 1.0, 0.0, 0.0
            elif value >= second_break + eps:
                safe1[point], safe2[point], safe3[point] = 0.0, 0.0, 1.0
            else:
                safe1[point], safe2[point], safe3[point] = 0.0, 1.0, 0.0
        return safe1, safe2, safe3

    def __schedule_objective_value(self, schedule_df):
        revenue_by_name = {
            str(row["name"]).strip(): float(row["revenue"])
            for _, row in self.__task.iterrows()
            if not str(row["name"]).startswith("void")
        }
        seen = set()
        value = 0.0
        for _, row in schedule_df.iterrows():
            action = row.get("action")
            if pd.isna(action):
                continue
            action = str(action).strip()
            if action in revenue_by_name and action not in seen:
                value += revenue_by_name[action]
                seen.add(action)
        if self.__objective == self.__Obj[0]:
            return value
        return None

    def __schedule_summary(self, schedule_df):
        task_by_name = {
            str(row["name"]).strip(): row
            for _, row in self.__task.iterrows()
            if not str(row["name"]).startswith("void")
        }
        package_names = {str(name).strip() for name in self.__package["name"].values}
        day = 0
        totals = [[0.0, 0.0] for _ in range(3)]
        task_count = 0
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
                task_count += 1
                if bool(task_by_name[action]["remote"]):
                    remote_count += 1

        day_time = ",".join(f"{value[0]:.5f}" for value in totals)
        day_power = ",".join(f"{value[1]:.5f}" for value in totals)
        return f"tasks={task_count} remote={remote_count} day_time={day_time} day_power={day_power}"

    def __print_solution_summary(self, label, schedule_df, obj_value=None, path=None):
        if obj_value is None:
            obj_value = self.__schedule_objective_value(schedule_df)
        obj_text = "unknown" if obj_value is None else f"{obj_value}"
        path_text = "" if path is None else f" schedule={path}"
        print(
            f"[eao] SOLUTION {label} obj={obj_text} "
            f"{self.__schedule_summary(schedule_df)}{path_text}",
            flush=True,
        )

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
                f"[eao] VALIDATE {label} failed day_time={day_time} day_power={day_power} "
                f"violations={';'.join(violations)}",
                flush=True,
            )
            return False
        print(
            f"[eao] VALIDATE {label} ok day_time={day_time} day_power={day_power}",
            flush=True,
        )
        return True

    def validate_schedule(self):
        if not self.__validate_business_schedule(self.__plandf, "output"):
            raise MIPError("Output schedule violates business validation rules")
        return None

    def __make_start_solution(self, selected_edges, w_values, q_values):
        w_safe1, w_safe2, w_safe3 = self.__safe_binary_values(w_values)
        q_safe1, q_safe2, q_safe3 = self.__safe_binary_values(q_values)

        sol = self.__m.createSol()
        for key, var in self.__x.items():
            self.__m.setSolVal(sol, var, 1.0 if key in selected_edges else 0.0)
        for key, var in self.__W.items():
            self.__m.setSolVal(sol, var, w_values.get(key, 0.0))
        for key, var in self.__Q.items():
            self.__m.setSolVal(sol, var, q_values.get(key, 0.0))
        for key, var in self.__Ws1.items():
            self.__m.setSolVal(sol, var, w_safe1[key])
        for key, var in self.__Ws2.items():
            self.__m.setSolVal(sol, var, w_safe2[key])
        for key, var in self.__Ws3.items():
            self.__m.setSolVal(sol, var, w_safe3[key])
        for key, var in self.__Qs1.items():
            self.__m.setSolVal(sol, var, q_safe1[key])
        for key, var in self.__Qs2.items():
            self.__m.setSolVal(sol, var, q_safe2[key])
        for key, var in self.__Qs3.items():
            self.__m.setSolVal(sol, var, q_safe3[key])
        return sol

    def __try_warmstart_schedule(self, schedule_df, label, printreason, path=None):
        selected_edges, w_values, q_values, selected_nodes = self.__route_values_from_schedule(
            schedule_df
        )
        sol = self.__make_start_solution(selected_edges, w_values, q_values)
        accepted = self.__m.trySol(sol, printreason=printreason)
        print(
            f"[eao] WARMSTART {label} accepted={accepted} "
            f"selected_edges={len(selected_edges)} selected_nodes={len(selected_nodes)}",
            flush=True,
        )
        if accepted:
            self.__print_solution_summary(
                f"warmstart {label}",
                schedule_df,
                obj_value=self.__schedule_objective_value(schedule_df),
                path=path,
            )
        return accepted

    def build_heuristic_start(self):
        if not self.__dataframes or len(self.__edges) < 1000:
            print("[eao] WARMSTART skipped", flush=True)
            return None
        from . import ha

        print("[eao] WARMSTART running ha for initial feasible route", flush=True)
        optimizer = ha.heuristic(
            situation="normal",
            decimal=self.__decimal,
            writeOutput=False,
            dataFrames={
                "info": self.__dataframes["info"],
                "task": self.__dataframes["task"],
                "package": self.__dataframes["package"],
                "point": self.__dataframes["point"],
                "distance": self.__dataframes["distance"],
                "time": self.__dataframes["time"],
                "power": self.__dataframes["power"],
            },
        )
        optimizer.run()
        ha_schedule = optimizer.schedule_frame()
        self.__fallback_schedule_df = ha_schedule.copy()
        self.__fallback_objvalue = self.__schedule_objective_value(ha_schedule)
        autosave_dir = os.path.join(os.path.dirname(os.path.abspath(self.__outputpath)), "autosave")
        os.makedirs(autosave_dir, exist_ok=True)
        ha_path = os.path.join(autosave_dir, "eao_ha_warmstart.csv")
        ha_schedule.to_csv(ha_path, index=False)
        print(f"[eao] WARMSTART ha_schedule={ha_path}", flush=True)

        if not self.__validate_business_schedule(ha_schedule, "ha_warmstart"):
            print("[eao] WARMSTART skipped because HA schedule failed business validation", flush=True)
            return None
        self.__try_warmstart_schedule(ha_schedule, "ha", printreason=False, path=ha_path)
        return None

    def run_opt(self):
        if self.__lppath != None:
            self.__m.write(self.__lppath)
        else:
            self.__m.update()
        self.__m.printStats()
        interrupted = {"value": False}

        def stop_scip(signum, frame):
            interrupted["value"] = True
            print(
                "[eao] INTERRUPT Ctrl+C received; stopping SCIP and keeping current incumbent...",
                flush=True,
            )
            try:
                self.__m.interruptSolve()
            except Exception as exc:
                print(
                    f"[eao] INTERRUPT failed to request SCIP stop: {exc.__class__.__name__}: {exc}",
                    flush=True,
                )

        old_sigint_handler = None
        install_sigint_handler = threading.current_thread() is threading.main_thread()
        if install_sigint_handler:
            old_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, stop_scip)
        try:
            if self.__autosavestate:
                self.__m.includeBestSolHandler(self)
            try:
                self.__m.optimize()
            except KeyboardInterrupt:
                interrupted["value"] = True
                try:
                    self.__m.interruptSolve()
                except Exception:
                    pass
        finally:
            if install_sigint_handler:
                signal.signal(signal.SIGINT, old_sigint_handler)
        if interrupted["value"] or self.__m.Status == _ScipStatus.USER_INTERRUPT:
            obj_text = "none" if self.__m.objVal is None else self.__m.objVal
            print(
                f"[eao] INTERRUPT stopped status={self.__m.Status} "
                f"sol_count={self.__m.SolCount} obj={obj_text}",
                flush=True,
            )
        if self.__m.SolCount >= 1:
            self.__objvalue = self.__m.objVal
        if self.__m.Status == gp.GRB.INFEASIBLE:
            self.__m.computeIIS()
            self.__m.write("infeasible.ilp")
            raise MIPError("Model is infeasible. Refer to infeasible.ilp")
        else:
            if self.__solpath != None:
                self.__m.write(self.__solpath)
        return None

    def print_status(self):
        if self.__m.Status == gp.GRB.OPTIMAL:
            print(f"Optimal objective value: {self.__m.objVal}")
        elif self.__m.SolCount >= 1:
            print(f"Feasible objective value: {self.__m.objVal}")
        elif self.__fallback_schedule_df is not None:
            self.__using_fallback = True
            self.__plandf = self.__fallback_schedule_df.copy()
            self.__objvalue = self.__fallback_objvalue
            print(
                "[eao] FALLBACK using HA schedule because SCIP found no incumbent",
                flush=True,
            )
        elif self.__m.Status == gp.GRB.TIME_LIMIT:
            raise MIPError("No feasible solution found within timeLimit")
        elif self.__m.SolCount < 1:
            raise MIPError("No feasible solution found")
        return None

    def __value(self, var):
        return self.__m.getVal(var)

    def proc_res(self, x=None, W=None):
        if self.__using_fallback:
            return None
        xbackup, Wbackup = 1, 1
        if x == None:
            x = self.__x
            xbackup = None
        if W == None:
            W = self.__W
            Wbackup = None
        loop, plan = [], [[], [], []]
        res = gp.tuplelist()
        for i, k1, j, k2 in self.__edges:
            if xbackup == None:
                value = self.__value(x[i, k1, j, k2])
                if value <= 1.1 and value >= 0.9:
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
                loop.append((i, k, self.__value(W[i, k])))
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
                self.__value(W[*self.__opoint[1]]),
                self.__value(W[*self.__opoint[2]]),
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
        if self.__using_fallback:
            return None
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
        if self.__using_fallback:
            return None
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

    def __log_step_detail(self, step):
        if step == "gen_point":
            return f"points={len(self.__point)}"
        if step == "gen_edges":
            return f"edges={len(self.__edges)}"
        if step == "add_variables":
            return (
                f"x={len(self.__x)} W={len(self.__W)} Q={len(self.__Q)} "
                f"safe_binary={len(self.__Ws1) + len(self.__Ws2) + len(self.__Ws3) + len(self.__Qs1) + len(self.__Qs2) + len(self.__Qs3)}"
            )
        if step == "run_opt":
            return f"status={self.__m.Status} sol_count={self.__m.SolCount} obj={self.__m.objVal}"
        return ""

    def __run_logged_step(self, index, total, step, func):
        print(f"[eao] START {index:02d}/{total:02d} {step}", flush=True)
        started = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[eao] FAIL  {index:02d}/{total:02d} {step} elapsed={elapsed:.3f}s "
                f"error={exc.__class__.__name__}: {exc}",
                flush=True,
            )
            raise
        elapsed = time.perf_counter() - started
        detail = self.__log_step_detail(step)
        suffix = f" {detail}" if detail else ""
        print(f"[eao] END   {index:02d}/{total:02d} {step} elapsed={elapsed:.3f}s{suffix}", flush=True)
        return result

    def run(self):
        print(
            f"[eao] RUN start objective={self.__objective} timeLimit={self.__time_limit}",
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
            ("build_heuristic_start", self.build_heuristic_start),
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
        print(f"[eao] RUN end elapsed={time.perf_counter() - total_started:.3f}s", flush=True)
        return None


def solve(case, mode="normal"):
    """Run the SCIP exact algorithm on UnifiedCase input."""
    if mode != "normal":
        raise NotImplementedError("eao only supports algorithm.mode=normal")

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
        "autoSave": True,
        "writeOutput": False,
        "outputPath": str(case.output_dir / "eao_schedule.xlsx"),
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
    optimizer = task_optimize(**kwargs)
    optimizer.run()
    schedule_df = optimizer.schedule_frame()
    rows = legacy_schedule_to_rows(case, schedule_df)
    objective_value = getattr(optimizer, "_task_optimize__objvalue", None)
    if objective_value is None:
        objective_value = getattr(getattr(optimizer, "_task_optimize__m", None), "objVal", None)
    return SchedulePlan(
        steps=[],
        rows=rows,
        objective_value=None if objective_value is None else float(objective_value),
    )


if __name__ == "__main__":
    opt = task_optimize(CONST.MAX_REVENUE, timeLimit=60 * 60 * 4)
    opt.run()
