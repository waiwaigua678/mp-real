# mp-real 中文文档

`mp-real` 是连接外部 WebSocket 策略服务器的真机部署、控制、录制和评测客户端；不包含
OpenPI 的训练和模型服务栈。英文文档见 [English documentation](../en/README.md)。

## 安装与入口

```bash
uv sync --extra piper --extra realsense --extra web --extra dev
uv pip install -e /path/to/pyAgxArm

# 或：RM2
uv sync --extra rm2 --extra realsense --extra web --extra dev
```

录制和离线数据工具需在同一次 `uv sync` 中加入 `--extra recording`、`--extra data`
或 `--extra evaluation`。连续分开执行 `uv sync --extra ...` 可能移除之前选择的 extra。
ROS 相机使用系统 ROS；RM2 启动前请依据
[`configs/rm2.env.example`](../../configs/rm2.env.example) 配置厂商 SDK。

```bash
uv run mp-piper-infer --help
uv run mp-rm2-infer --help
uv run mp-piper-web --help
uv run mp-camera-preview --help
uv run mp-data-inspect --help
uv run mp-data-validate --help
uv run mp-data-view --help
uv run mp-open-loop-eval --help
uv run mp-baseline --help
uv run mp-move-to-recorded-state --help
uv run mp-robot-replay --help
```

`mp-real-web` 是 `mp-piper-web` 的机器人无关别名。Web 有三种资源模式：
`DEPLOYMENT` 创建机器人、相机和策略客户端；`CAMERA_PREVIEW` 只创建相机；
`OFFLINE_REPLAY` 不创建这些资源。策略预热产生的动作会丢弃，控制环开始前会准备一份
新的动作块。

## 安全与状态

`--dry-run` 和 `--infer-only` 仍可能读取已连接机器人的反馈，并非离线模拟器。任何可能
运动的命令都必须经过操作员批准的安全流程。移动到录制状态和轨迹回放仍属实验功能，默认
安全校验会阻止物理执行。

RM2 仅 CLI 部署默认配置具有操作员记录的真机验证范围；这不代表 RM2 Web、录制状态移动
或回放已验证。Piper 尚未声明真机移动/回放验证。详见
[能力矩阵](robot_capability_matrix.md) 和 [已知限制](known_limitations.md)。硬件安全运行手册
维护在仓库外。

## 操作文档

- [评测工作流](evaluation_workflow.md)
- [Baseline 工作流](baseline_workflow.md)
- [LeRobot v2.1 数据模式](lerobot_v21_schema.md)
- [离线数据查看](offline_data_view.md)
- [开环评测](open_loop_evaluation.md)
- [开环评测验证](validation_stage_11.md)
- [能力矩阵](robot_capability_matrix.md)
- [已知限制](known_limitations.md)
