# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:53:23 2024

@author: chen wentian
"""
import gurobipy as gp
import os
import threading
import numpy as np
import pandas as pd
from itertools import product
from openpyxl.styles import PatternFill

from .models import MIPError, SubtourError


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
        self.__task["location"].replace(
            "探测起点", "探测起点1,探测起点2,探测起点3,探测起点4", inplace=True
        )
        self.__task["location"].replace(
            np.nan, ",".join(self.__pointdf.index.values), inplace=True
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

    def gen_edges(self):
        self.__edges = gp.tuplelist()
        for i in self.__point:
            for j in self.__point:
                if i == j:
                    continue
                self.__edges.append((*i, *j))
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
            (self.__x[*self.__opoint[-1], j, k2] == 0 for j, k2 in temp),
            name="oodegree4",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (self.__x[*self.__opoint[2], j, k2] == 0 for j, k2 in temp),
            name="oodegree5",
        )
        temp = temp[:-1]
        self.__m.addConstrs(
            (self.__x[*self.__opoint[1], j, k2] == 0 for j, k2 in temp),
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
            == 1,
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
        temp = list(point.select(self.__opoint[1][0], "*"))
        temp.remove(point.select(self.__opoint[1][0], self.__opoint[1][1])[0])
        self.__m.addConstrs(
            (
                self.__W[k[0], k[1]] <= self.__TTime[0]
                for k in list(point.select(self.__opoint[1][0], "*"))
            ),
            name="time2",
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
        temp = list(point.select(self.__opoint[1][0], "*"))
        temp.remove(point.select(self.__opoint[1][0], self.__opoint[1][1])[0])
        self.__m.addConstrs(
            (
                self.__Q[k[0], k[1]] <= self.__TPower[0]
                for k in list(point.select(self.__opoint[1][0], "*"))
            ),
            name="power2",
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
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                >= self.__W[i, k] + eps - M * (1 - self.__Ws1[i, k]),
                name=f"safetimeday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[*self.__opoint[1]]
                <= self.__W[i, k] + M * self.__Ws1[i, k],
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
                - M * (1 - self.__Ws3[i, k]),
                name=f"safetimeday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__W[i, k]
                <= self.__W[*self.__opoint[2]] + M * self.__Ws3[i, k],
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
                - M * (1 - self.__Ws2[i, k]),
                name=f"safetimeday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Ws1[i, k] + self.__Ws3[i, k] + M * self.__Ws2[i, k],
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
                >= self.__Q[i, k] + eps - M * (1 - self.__Qs1[i, k]),
                name=f"safepowerday1_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[*self.__opoint[1]]
                <= self.__Q[i, k] + M * self.__Qs1[i, k],
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
                - M * (1 - self.__Qs3[i, k]),
                name=f"safepowerday3_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                self.__Q[i, k]
                <= self.__Q[*self.__opoint[2]] + M * self.__Qs3[i, k],
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
                - M * (1 - self.__Qs2[i, k]),
                name=f"safepowerday2_bigM_constr0[{i},{k}]",
            )
            self.__m.addConstr(
                1
                <= self.__Qs1[i, k] + self.__Qs3[i, k] + M * self.__Qs2[i, k],
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

    def run_opt(self):
        if self.__lppath != None:
            self.__m.write(self.__lppath)
        else:
            self.__m.update()
        self.__m.printStats()
        if self.__autosavestate:
            self.__m.optimize(self.__callback)
        else:
            self.__m.optimize()
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
        elif self.__m.SolCount < 1:
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
        res = gp.tuplelist()
        for i, k1, j, k2 in self.__edges:
            if xbackup == None:
                if x[i, k1, j, k2].X <= 1.1 and x[i, k1, j, k2].X >= 0.9:
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
                loop.append((i, k, W[i, k].X))
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
                W[*self.__opoint[1]].X,
                W[*self.__opoint[2]].X,
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
        self.add_variables()
        self.set_objective()
        self.add_indegree_constrs()
        self.add_outdegree_constrs()
        self.add_equaldegree_constrs()
        self.add_oodegree_constrs()
        self.add_rtask_constrs()
        self.add_otask_constrs()
        self.add_remote_constrs()
        self.add_time_constrs()
        self.add_day_constrs()
        self.add_power_constrs()
        self.add_safe_constrs()
        self.add_continuous_constr()
        self.add_noii_constrs()
        self.run_opt()
        self.print_status()
        self.proc_res()
        self.cal_route()
        self.add_package()
        if self.__write_output:
            self.write_excel()
        return None


def solve(case, mode="normal"):
    """Run the migrated exact algorithm on UnifiedCase input."""
    if mode != "normal":
        raise NotImplementedError("ea only supports algorithm.mode=normal")

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
