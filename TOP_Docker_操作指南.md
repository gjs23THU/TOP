# TOP 算法服务 Docker 操作指南

## 1. 交付文件

- `top-algorithm-service.tar`

当前镜像为 Windows 11 Docker Desktop 构建导出的版本，导入和启动时请使用 `top-algorithm-service.tar` 和 `top-algorithm-service:latest`。

## 2. 导入镜像

在拿到 tar 文件的机器上执行：

```bash
docker load -i top-algorithm-service.tar
```

确认镜像已导入：

```bash
docker images top-algorithm-service
```

## 3. 启动服务

```bash
docker run -d \
  --name top-algorithm-service \
  -p 8000:8000 \
  -v top_algorithm_runs:/data/runs \
  top-algorithm-service:latest
```

说明：

- 容器内服务端口是 `8000`。
- `-p 8000:8000` 表示宿主机也用 `8000` 端口访问。
- `top_algorithm_runs` 用来持久化每次求解产生的输入和输出。

## 4. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

正常返回示例：

```json
{"status":"ok","solver":"unified_framework","run_root":"/data/runs"}
```

## 5. 提交数据方式一：上传 7 个文件

每个 case 需要上传：

- `config.json`
- `task.csv`
- `package.csv`
- `point.csv`
- `distance.csv`
- `time.csv`
- `power.csv`

示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/solve" \
  -F "config=@config.json" \
  -F "task=@task.csv" \
  -F "package=@package.csv" \
  -F "point=@point.csv" \
  -F "distance=@distance.csv" \
  -F "time=@time.csv" \
  -F "power=@power.csv"
```

若不希望响应里直接返回完整 schedule，可加：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/solve?include_schedule=false" \
  -F "config=@config.json" \
  -F "task=@task.csv" \
  -F "package=@package.csv" \
  -F "point=@point.csv" \
  -F "distance=@distance.csv" \
  -F "time=@time.csv" \
  -F "power=@power.csv"
```

## 6. 提交数据方式二：上传 zip

zip 内可以把 7 个文件放在根目录，或放在同一个 case 文件夹内。

```bash
zip case.zip config.json task.csv package.csv point.csv distance.csv time.csv power.csv
```

Windows 环境可用系统右键压缩或 7-Zip，把这 7 个文件压成一个 zip。

上传：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/solve-zip" \
  -F "file=@case.zip"
```

## 7. 查看和下载结果

求解接口会返回 `run_id`，例如：

```json
{
  "run_id": "6b67b2e6ff4f4facbd69bbeb33a085e0",
  "status": "success",
  "result_url": "/api/v1/runs/6b67b2e6ff4f4facbd69bbeb33a085e0/result",
  "schedule_url": "/api/v1/runs/6b67b2e6ff4f4facbd69bbeb33a085e0/schedule.csv"
}
```

查看结果：

```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>/result
```

查看 schedule JSON：

```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>/schedule
```

下载 schedule CSV：

```bash
curl -o schedule.csv http://127.0.0.1:8000/api/v1/runs/<run_id>/schedule.csv
```

## 8. 停止服务

```bash
docker stop top-algorithm-service
docker rm top-algorithm-service
```

## 9. 注意事项

- `ha`、`ga`、`sa`、`pso` 可直接试用。
- `ea` 是 Gurobi 精确算法，部署环境需要可用的 Gurobi 授权。
- `eao` 是 SCIP/PySCIPOpt 精确算法，部署环境需要安装 `pyscipopt`。
- 默认单个上传文件大小上限为 200MB。
- 若宿主机 8000 端口被占用，可改成 `-p 8080:8000`，访问地址相应改为 `http://127.0.0.1:8080`。
