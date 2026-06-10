# Docker 复现说明

日期：2026-06-10

本文档记录最终提交前 Docker 复现的推荐命令、验证边界和当前状态。

## 1. 推荐命令

在项目根目录 `我们的代码/` 下执行：

```bash
docker build -t xa202608:submission .
docker run --rm xa202608:submission
```

默认容器入口为：

```bash
bash scripts/run_docker_smoke.sh
```

该入口执行：

```bash
python scripts/check_project.py
python scripts/check_demo_inputs.py
python scripts/check_html_demo.py
python scripts/check_pre_demo_readiness.py
python -m unittest discover -v
```

## 2. 验证边界

Docker smoke 不重新训练最终 50 epoch 模型，也不重新下载公开数据集。它用于验证：

- Python/PyTorch 基础环境可用；
- 项目代码可以 import；
- 合成 debug dataloader 和模型前向可用；
- `competition_artifacts` 中 demo 输入和动态 HTML demo 可读；
- TC 边界、报告口径和 pre-demo readiness 检查可运行；
- 单元测试通过。

完整实验复跑仍建议按最终报告中的配置和数据准备说明执行。

## 3. 当前状态

本机已检测到 Docker CLI：

```text
Docker version 29.3.1
```

但当前 Docker Desktop/Linux daemon 未启动，`docker info` 报错：

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

因此本轮已完成 Docker 文件和 smoke 入口加固，但尚未完成实际 `docker build` / `docker run` 验证。启动 Docker daemon 后，应优先执行第 1 节命令，并把日志保存到：

```text
competition_artifacts/05_report_assets/docker_smoke_log.txt
```

