# mp-real

面向 Motrix 真机推理与 Piper Web 控制面板的轻量部署仓库。本仓库有意不包含
OpenPI 训练或模型服务栈，而是连接独立部署的 WebSocket 策略服务器。

源码包位于 `src/mp_real`。机器人专用入口位于 `mp_real.robots`；旧的
`mp_ex`、`mp_web` 和 `openpi_client` 模块名不属于这个部署包。

## 安装

```bash
cd mp-real
uv sync
```

一次 `uv sync` 中应选择全部需要的 extra。连续分别执行
`uv sync --extra ...` 会每次选择一个新的精确环境，后一次可能移除前一次选择的
extra。

标准 Piper 部署应先在一次同步中选择所有 extra，再将部署的 `pyAgxArm` 检出目录
作为可编辑包安装：

```bash
# Piper + RealSense + 更快的 Web JPEG 编码 + 本地 lint 工具
uv sync --extra piper --extra realsense --extra web --extra dev
uv pip install -e /home/server/prj/pyAgxArm
uv pip install   --python /home/server/prj/mp-real/.venv/bin/python   --editable /home/server/prj/pyAgxArm
```

若控制器也需要 RM2 和 V4L2 支持，请在同一个 `uv sync` 调用中包含它们：

```bash
# 控制器同时需要 RM2 和 V4L2 相机时加入这些 extra。
# RM2 厂商 SDK 单独配置；参见 configs/rm2.env.example。
uv sync --extra piper --extra rm2 --extra realsense --extra v4l2 --extra web --extra dev
uv pip install -e /home/server/prj/pyAgxArm
```

当 `pyAgxArm` 作为相邻的 `../pyAgxArm` 目录检出时，仍可使用
`./scripts/bootstrap-piper.sh --extra ...`。非 Piper 部署也采用同样的重复 flag
模式：

```bash
uv sync --extra rm2 --extra realsense --extra web --extra dev
```

若设备需要全部可选 Python 依赖，Piper 使用
`./scripts/bootstrap-piper.sh --all-extras`，不含 Piper 时使用 `uv sync --all-extras`。

可选引导脚本要求如下相邻目录布局：

```text
parent/
  mp-real/
  pyAgxArm/
```

若 `pyAgxArm` 部署在其他位置，请如上所示用绝对路径执行 `uv pip install -e`。
ROS 相机后端使用系统的 ROS 安装（`rospy`、`sensor_msgs`），因此刻意未列为 PyPI
依赖。RM2 请在启动推理前 source 一份 `configs/rm2.env.example`，以设置厂商 SDK
路径。

## 添加机器人

在 `mp_real.robots.<name>` 实现 `Robot` 边界：发布 `ActionSpec`、读取归一化策略
状态、执行归一化动作、复位和关闭。用 `register_robot` 注册工厂。共享 runtime
负责策略请求、动作块调度、RTC 融合、仅推理持久化及带时间戳的相机/状态观测。

Piper SDK 可从本地检出目录安装，或在运行时指向其位置：

```bash
uv pip install -e /path/to/pyAgxArm
export PYAGXARM_ROOT=/path/to/pyAgxArm
```

可编辑安装后 `PYAGXARM_ROOT` 可选；当 SDK 位于本仓库外时很有用。

## Piper CLI

```bash
uv run mp-piper-infer --help
```

最小化、不会发出运动指令的策略与相机接线检查：

```bash
uv run mp-piper-infer \
  --server-url ws://127.0.0.1:8000 \
  --cam-head-backend black \
  --cam-left-wrist-backend black \
  --cam-right-wrist-backend black \
  --dry-run \
  --no-reset-on-start \
  --no-use-rtc \
  --max-steps 1
```

该命令仍会读取已连接机器人的状态。`--infer-only` 也会读取关节反馈，因此它不是
离线模拟器模式。

## 机器人 Web

常规 Web 进程保持既有硬件相机默认值：

```bash
uv run mp-piper-web --host 0.0.0.0 --port 8765
```

`mp-real-web` 是相同入口点的机器人无关别名；现有 `mp-piper-web` 部署脚本继续
受支持。Web 服务器支持 Piper 与 RM2。连接前选择机器人，或者用命令行设定初始
运行时。为保护控制请求，可设定 `--access-key` 或 `MOTRIX_WEB_ACCESS_KEY`；浏览器
仅在本次会话保存输入的密钥，并以 `X-Motrix-Key` 发送。

```bash
MOTRIX_WEB_ACCESS_KEY=change-me \
uv run mp-piper-web --host 0.0.0.0 --port 8765 --robot rm2
```

在没有 RealSense/V4L2 依赖的电脑上检查部署或策略连接时，使用黑帧并禁止启动时
复位/使能：

```bash
uv run mp-piper-web \
  --host 0.0.0.0 \
  --port 8765 \
  --camera-profile black \
  --no-enable-on-start \
  --no-reset-on-start \
  --dry-run
```

访问 `http://<robot-computer-ip>:8765`。也可在点击连接前在 Settings 页面设置连接
参数。

### Web 运行时模式

Settings 页面提供明确的运行时模式；请在连接前选择，变更模式需要先断开连接。

- `DEPLOYMENT` 创建机器人、已配置相机和策略客户端。同步、RTC 和仅推理是此模式
  内的策略执行选项。
- `CAMERA_PREVIEW` 只创建已配置相机。它绝不打开 CAN、创建机器人或策略客户端、
  复位/使能机械臂，也不读取机器人状态。某一路相机读失败时相机页仍可用，错误会在
  对应流上显示。
- `OFFLINE_REPLAY` 有意不创建机器人、相机或策略资源。在实现录制会话回放之前，
  它显示 Stage 7 回放占位内容。

## 仅相机预览

使用独立命令打开已配置相机，而不创建 `Robot`、连接 CAN 或创建策略客户端：

```bash
uv run mp-camera-preview --robot piper --no-web --camera-backend cam_head=black
uv run mp-camera-preview --robot rm2 --no-web --camera-backend left_color=black
```

真实相机使用重复的 `--camera-backend ROLE=BACKEND` 与
`--camera-selector ROLE=SELECTOR` 选项。省略 `--no-web` 可启用相同的仅相机 Web
预览生命周期；`--duration` 与 `--save-preview DIR` 可选。

## 已录制的 LeRobot v2.1 数据

`save_data=true` 的评测会话默认在 `recordings/` 下写入自包含的 LeRobot v2.1
数据集。录制 worker 在控制线程之外写入 Parquet、MP4 和遥测数据，将实际执行的
动作作为标准 `action` 记录，并在最终标签后原子完成会话。

无需创建机器人、相机或策略客户端即可检查或验证本地数据集：

```bash
uv run mp-data-inspect recordings/<dataset>
uv run mp-data-validate recordings/<dataset>
```

`mp-data-inspect` 也接受没有可选 `meta/mp_real/` 或 `telemetry/` 扩展的普通
LeRobot v2.1 数据集。模式、时间戳、Parquet、元数据或视频对齐出错时，
`mp-data-validate` 返回非零状态。

### 安全的录制状态姿态规划

检查录制的 `observation.state` 时不会创建 `Robot`、相机或 `PolicyClient`。默认是
dry-run；`--execute` 还要求新近重新验证的计划 hash，并且只应在 Stage 9 硬件闸门
获批后使用。

```bash
uv run mp-move-to-recorded-state \
  --robot piper --dataset recordings/<dataset> --episode-index 0 --sample-index 0
```

对于明确批准的其他状态模式，向 CLI 传入同一份版本化 JSON 映射
`--config mapping.json`，或向 Web 服务器传入 `--pose-mapping-config mapping.json`。
映射必须完整并记录每次单位换算；拒绝位置式或隐式单位换算。

Web 服务可通过重复的 `--recorded-data-root PATH` 配置允许列表录制根目录。录制状态
面板只提交数据集/episode/sample 引用；服务器重新读取 `observation.state`、进行模式
预检，然后在低速移动前要求确认计划 hash。任何真实运动测试前请阅读
[`hardware_validation_stage_9.md`](hardware_validation_stage_9.md)。

### 离线数据查看器

专用只读 Episode Viewer 用于同步查看 LeRobot v2.1 视频、状态/动作曲线、运行时
事件、指标及可拖拽的 sample 时间线。它有意与真机轨迹回放分离：进程不导入机器人
SDK，绝不创建 `Robot` 或 `Camera`。浏览不创建 `PolicyClient`；只有明确提交的
Stage-11 开环作业才可能在其后台 worker 中创建一个。

```bash
uv run mp-data-view \
  --storage-root /home/pc4/.cache/huggingface/lerobot/local/piper_1armblowv01 \
  --dataset piper_1armblowv01 \
  --episode 0
```

打开 `http://127.0.0.1:8766`。存储根目录也可以包含多个数据集目录；UI 只公开由目录
生成的数据集 ID，绝不接受前端任意文件路径。缺少 mp-real 遥测的标准 LeRobot 数据集
仍可查看，相关字段显示为未记录而非推断值。

### Teacher-forced 开环策略评测

`mp-open-loop-eval` 针对真实 LeRobot v2.1 观测评估完整策略动作块。它不创建
`Robot`、不导入机器人 SDK，也绝不发送动作。预热动作块会被丢弃；每个 sample 的
新预测写入隔离的结果目录。

```bash
uv run mp-open-loop-eval \
  --dataset recordings/<dataset> \
  --episode 0 \
  --policy-url ws://127.0.0.1:8000 \
  --policy-label pi05-checkpoint-a \
  --target-source action \
  --alignment sample_index \
  --output open_loop_results/pi05-checkpoint-a
```

绝对控制步对齐要求数据集拥有有效的 `mp_real.chunk_cursor` 遥测，并且操作者明确
声明 `frame_index` 是控制步：

```bash
uv run mp-open-loop-eval ... \
  --alignment absolute_control_step \
  --allow-frame-index-as-control-step
```

一个 episode 有多个任务时必须提供 `--prompt-override`。不同 ActionSpec 或目标来源
的结果不得合并。比较真实策略 checkpoint 前请阅读
[`validation_stage_11.md`](validation_stage_11.md)。

### 可复现 Baseline 与 A/B 比较

`mp-baseline` 默认将小型、版本化实验定义存储在 `recordings/baselines/`。Baseline
记录 Git commit、策略身份、ActionSpec、相机/机器人/运行时/RTC/安全设置、评测协议，
以及指向真机和开环结果的紧凑链接；它绝不保存 API key 或 episode/video 载荷。

```bash
uv run mp-baseline list
uv run mp-baseline show <id>
uv run mp-baseline diff <id-a> <id-b>
uv run mp-baseline compare <id-a> <id-b>
```

机器人 Web 的 **Baseline** 页面是权威 UI 存储路径；创建和克隆操作会排入有界后台
写入器。从 Baseline 启动只创建手动评测会话，不预热策略、不开始 episode，也不发送
动作。若运行时有变更，系统会以分类配置 diff 拒绝，直到操作者明确创建派生 Baseline。
参见 [`baseline_workflow.md`](baseline_workflow.md)、Piper/RM2
[`robot_capability_matrix.md`](robot_capability_matrix.md) 与
[`piper_rm2_generality_audit.md`](piper_rm2_generality_audit.md)。

### 安全的机器人轨迹回放

`mp-robot-replay` 与离线查看不同：它不使用策略，默认生成完全离线的计划/报告。
它只接受显式声明的标准动作用于命令回放；状态轨迹跟随是单独且明显标记的模式。

```bash
uv run mp-robot-replay \
  --robot piper --dataset recordings/<dataset> --episode-index 0 \
  --mode command --timing recorded --speed-scale 0.1
```

`--execute` 还要求精确确认已审查的 `--confirm-plan-hash`。它不创建 `PolicyClient`
或相机，先以低速移动到录制起始状态，再等待确认后发送轨迹目标。物理测试前请审查
[Stage 10 硬件闸门](hardware_validation_stage_10.md)。

### 策略预热与首个动作

部署为连接、元数据、预热和稳态推理提供独立超时。默认先发送一个真实观测作为预热
请求，预热超时为 60 秒；验证并丢弃该动作块后，再在控制开始前预取一个新的实时动作块。
预热期间不执行动作，RTC 使用预取块启动而非空缓冲区。`检查服务` 只确认 WebSocket
连接和元数据，不能说明模型已经预热。

### 规范运行时时间与内存事件

旧有浮点 `timestamp_monotonic` 字段为兼容调用方而保留。新的采集和运行时记录还携带
来自 `time.monotonic_ns()` 的规范 `timestamp_monotonic_ns`；排序决策必须使用整数纳秒
字段。墙钟 ISO 时间戳仅用于事件显示和未来文件名。

每个 `CameraFrame`/`CameraSample` 都有可信 `frame_id`。Black 和 V4L2 仅在取得新帧时
递增；RealSense 可用时使用 SDK 帧号；ROS 只在图像回调中递增。重复读取 ROS 缓存保留
先前 id。源原生序列和采集延迟字段为可选元数据，不能替代本地帧 id。

`ObservationSnapshot.max_camera_skew_ns` 是纳入的相机时间戳中最新与最早之差。
`observation_age_ns` 在 snapshot 完成时相对于最旧相机/状态源测量。共享 runtime 将既有
hook 转换为有界内存事件；动作保留不同的原始块、选择动作、稳定化目标和实际执行载荷。
事件 ndarray 载荷会在派发前复制。评测启用 `save_data` 时，专用 Stage 6 录制器异步
消费这些复制事件并写入文件。

## 在其他位置运行的组件

- OpenPI WebSocket 策略服务器及其模型/checkpoint 位于推理服务器。
- 机器人 SDK 与 Linux 硬件访问位于机器人控制器。
- `mp-real` 仅包含客户端、相机采集、机器人调用和 Web 控制层。
