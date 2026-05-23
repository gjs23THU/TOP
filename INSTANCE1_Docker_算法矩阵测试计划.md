# Instance1 Docker 算法矩阵测试计划

## 1. 测试目标

在 Docker 化的 TOP 算法服务中，对 `instance1` 输入执行算法矩阵验证，覆盖现阶段可运行的非精确算法和全部目标函数，确认：

- zip 上传接口可用；
- 各算法能被正确路由和执行；
- 三类目标函数配置能被正确读取；
- 成功求解时能下载 `schedule.csv`；
- 失败时能保留响应 JSON 和错误信息，便于定位。

## 2. 测试范围

本轮执行算法：

- `ha`
- `ga`
- `sa`
- `pso`

本轮跳过：

- `ea`：精确求解，现阶段 ban 掉，避免耗时过长或依赖 Gurobi 授权。
- `ai`：当前框架中是保留入口，尚未实现，不纳入能力验证。

目标函数：

- `maxRevenue`
- `minTime`
- `minPower`

总测试组合：

```text
4 algorithms * 3 objectives = 12 runs
```

## 3. 前置条件

1. Docker 镜像已导入或已构建：

```bash
docker images top-algorithm-service
```

2. 服务已启动：

```bash
docker run -d \
  --name top-algorithm-service \
  -p 8000:8000 \
  -v top_algorithm_runs:/data/runs \
  top-algorithm-service:latest
```

3. 健康检查通过：

```bash
curl http://127.0.0.1:8000/health
```

4. 当前目录存在 `instance1.zip`。

## 4. 执行方法

运行脚本：

```bash
./run_instance1_docker_matrix.sh
```

默认服务地址：

```text
http://127.0.0.1:8000
```

如端口不同：

```bash
BASE_URL=http://127.0.0.1:8080 ./run_instance1_docker_matrix.sh
```

默认输出目录：

```text
output/instance1_docker_matrix/
```

如需更改：

```bash
OUTPUT_DIR=output/my_test ./run_instance1_docker_matrix.sh
```

默认对 `ga/sa/pso` 设置 `timeLimit=120` 秒，避免批量测试卡住。可调整：

```bash
TIME_LIMIT_SECONDS=300 ./run_instance1_docker_matrix.sh
```

## 5. 输出文件命名

成功时下载：

```text
output/instance1_docker_matrix/csv/<algorithm>_<objective>.csv
```

示例：

```text
ha_maxRevenue.csv
ha_minTime.csv
ha_minPower.csv
ga_maxRevenue.csv
...
pso_minPower.csv
```

每次调用原始响应：

```text
output/instance1_docker_matrix/responses/<algorithm>_<objective>.json
```

每次临时 case zip：

```text
output/instance1_docker_matrix/cases/<algorithm>_<objective>.zip
```

汇总表：

```text
output/instance1_docker_matrix/summary.csv
```

字段：

```text
algorithm,objective,http_code,status,objective_value,run_id,schedule_file,error_type,error_message
```

## 6. 通过标准

最低通过标准：

- 12 个组合均能完成 HTTP 调用；
- 成功组合能下载 CSV；
- 失败组合有明确 `error.type` 和 `error.message`；
- `summary.csv` 能完整记录所有组合结果。

推荐通过标准：

- `ha` 三个目标函数均成功；
- `ga/sa/pso` 至少能在给定 timeLimit 内返回明确结果；
- Docker 日志中能看到每个 `run_id` 的阶段性日志。

查看实时日志：

```bash
docker logs -f top-algorithm-service
```

## 7. 结果判读

- `status=success` 且存在 `schedule_file`：该组合求解成功，CSV 已下载。
- `status=infeasible`：算法执行成功，但在该配置/时限下未找到可行解。
- `status=solver_error`：算法执行或运行环境异常，需要看 `error_message` 和 Docker 日志。
- HTTP 非 200：服务接口或上传请求异常，优先检查服务是否启动、zip 是否有效、端口是否正确。
