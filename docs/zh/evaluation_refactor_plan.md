# 评测平台重构计划

## 目标与非目标

这是从 `current_architecture.md` 所记录架构出发的增量迁移。它在既有 `Robot`、
`ActionSpec`、`InferenceAdapter` 和共享 runtime 边界上实现评测和录制，同时保护
Piper CLI 与 Web 行为。

本计划不提出框架迁移、数据库、替代策略客户端、第二条推理循环、固定的共享 14 维动作
schema，也不对 `web/server.py` 进行一次性重写。

## 兼容性与硬件护栏

- 每个阶段保留现有 CLI 入口和浏览器端点。
- 厂商 SDK 调用保留在 `robots/piper` 和 `robots/rm2`；共享 runtime 代码不引入厂商
  SDK。
- 共享边界使用 `ActionSpec` 维度和相机角色。
- 开发测试绝不使能、复位、移动、回放或以其他方式控制硬件。测试只使用 fake、mock
  policy 和 black camera。
- 延迟结果按适用情况携带 `session_id`、`generation_id`、`request_id`、`chunk_id`；
  runtime 事件以 `time.monotonic_ns()` 排序。
- 录制后台 worker 必须有界、可显式停止和 join，报告失败，并临时/原子化完成会话。

## Stage 0 — 基线与保护线（本变更）

文件：`docs/current_architecture.md`、本计划和 `tests/test_runtime.py`。

- 记录两条 Piper 调用链、RM2 Web 实际共享 runtime 路径、重复的 Piper Web 循环、
  生命周期、仅推理数据和完整 API 契约。
- 为共享 runtime、观测、registry 与 Web config/stop 行为添加不依赖硬件的 unittest
  表征覆盖。
- 不添加评测产品行为、不移动既有大文件，也不改变生产控制行为。

退出标准：已记录的行为和表征测试通过仓库 unittest 与 Ruff 检查。

## Stage 1 — 生命周期身份与共享 runtime 接缝

主要文件：`runtime/` 下的小型聚焦新增、`web/server.py` 的针对性修改与测试。不要整体
移动 `server.py`。

1. 定义归 runtime 所有的会话和请求身份，使用单调时间戳；在线程队列工作中传递它们，
   并在每次交接时拒绝过期结果。
2. 将 Piper Web 的循环逐个重构为调用既有 `run_policy_loop()`，而不是私有
   infer-only/sync/RTC 循环。实现一个读取既有预览帧源、并通过 `PiperRobot` 委派机器人
   工作的 Piper Web `InferenceAdapter`。它不是新的 Robot、Camera 或 Policy 抽象。
3. 用 `on_step` 和范围受限的 profiling hook 保留 Web 指标。在与通用 runtime 行为对齐
   前，先表征有意的 UI 差异。
4. 只在 worker 已有显式 stop 信号、有界等待、join 结果和传播异常时替换 daemon 生命周期
   掩盖；本阶段不改变硬件运动语义。

退出标准：Piper CLI、Piper Web 与 RM2 Web 在每个模式均选择同一共享推理循环，同时正常
API 响应形状和生命周期行为保持兼容。

## Stage 2 — Web 资源模式与策略启动

主要文件：聚焦的共享策略启动 helper、窄范围 Web 生命周期修改、现有静态页面和仅 fake
测试。本阶段不添加录制、评测会话或离线数据播放器。

### 2A — 资源模式与仅相机预览

Web runtime 有三种资源模式：

- `DEPLOYMENT` 创建机器人、相机和策略客户端，是唯一可以启动 `RuntimeController`
  的模式。同步、RTC 和仅推理仍是该模式下的执行选择。
- `CAMERA_PREVIEW` 只创建已配置相机。它不得创建 `Robot` 或 `PolicyClient`、连接 CAN、
  复位/使能机械臂，或读取机器人状态。相机错误按流保留，单个相机失败不应停止其他预览。
- `OFFLINE_REPLAY` 不创建这些资源。它只提供 API/UI 隔离和 Stage 7 占位内容；录制会话
  回放不属于本阶段。

预览 worker 为非 daemon，具备显式 stop event、有界 join 与确定性的相机关闭。fake 测试
覆盖工厂隔离、重复 start/stop、单相机失败、离线隔离和普通 deployment 资源创建。

### 2B — 预热与首块就绪

策略生命周期区分 `DISCONNECTED`、`CONNECTING`、`CONNECTED`、`WARMING_UP`、
`PREFETCHING_FIRST_CHUNK`、`READY`、`RUNNING`、`WARMUP_FAILED` 和 `ERROR`。它为
连接、元数据、预热和稳态推理使用独立可配置超时。

共享启动协调器获取真实观测，执行一次或多次预热推理，验证并丢弃每一个预热动作，然后
取得新的第一个实时动作块。控制器仅在此后可进入 `RUNNING`。共享同步与 RTC 循环接收
该初始动作块；RTC 在启动 producer 前为其 buffer 预置该块。预热失败保留带类型的根因，
而不是仅暴露泛化 RTC producer 错误。stop/disconnect 取消启动 worker，并在资源拆除前
关闭其策略连接。

状态暴露连接、元数据、冷推理、预热、第一个实时动作和稳态推理延迟字段。策略 ping
只表示元数据/连接状态，不代表模型已预热。

退出标准：fake 的首帧慢策略证明未准备 RTC 的失败、成功预热、丢弃预热动作、执行新的
第一个动作块、带类型的预热错误和干净的 stop/disconnect。

## Stage 3 — 规范时间与内存 runtime 事件

主要文件：`runtime/models.py`、`runtime/observation.py`、`runtime/events.py`，聚焦的
相机/runtime 修改和仅 fake 测试。本阶段不写录制文件。

1. 保留旧浮点 `timestamp_monotonic`，同时向状态和相机 sample 添加规范整数
   `timestamp_monotonic_ns`。
2. 只在后端收到新源帧时分配可信的逐相机 `frame_id`；缓存的 ROS 读取保留该 id。
3. 捕获观测身份、采集开始/结束、相机 frame id、时间戳、相机 skew 和源 age。相机 skew
   是最新减最早相机时间戳；age 在采集完成时相对于最旧状态或相机源测量。
4. 经由有界、非 daemon 的内存 dispatcher 将兼容推理 hook 转成复制后的类型化事件。
   event sink 不能阻塞机器人循环；复合 sink 的失败子项记录为 sink-failure 遥测，且不阻止
   其他 sink 收到事件。
5. 原始动作块、选中的动作、稳定化目标和实际执行动作分别放在不同事件 payload key 中。
   预热事件与正常控制循环事件分阶段保存。

退出标准：fake 测试证明时间戳/帧排序、ROS 缓存身份、skew 计算、预热就绪排序、同步/RTC
动作排序、generation gate、sink 隔离和浮点字段兼容性。

## Stage 4 — 录制 worker 与版本化会话格式

主要文件：聚焦的录制模块和窄范围 runtime 集成。不得在 HTTP handler 或机器人循环中执行
磁盘写入。

1. 定义版本化 manifest 和 telemetry schema，其中包含 `ActionSpec`、策略/服务器元数据、
   丢帧/丢遥测计数器和失败状态。
2. 创建有界生产者队列和非 daemon 写入 worker，带显式 stop、join 结果和异常通道。
3. 写入 `.inprogress` 会话目录，随后在可行时原子完成；若完成失败，保留不完整/失败元数据。
4. 复用 Stage 3 的事件与时间契约，在控制路径外编码视频。

退出标准：过载 fake 录制会报告丢失和 writer 失败；没有控制循环调用栈写数据到磁盘。

## Stage 5 — 评测编排与 Web 暴露

主要文件：聚焦的 evaluation/session 模块和窄范围 handler 新增。

1. 将评测命令放入 runtime 所有的命令队列；handler 只验证/入队并返回 ID/status。
2. 每次回放前强制状态、初始条件和录制 schema 验证。未明确批准时，保持可运动测试禁用。
3. 为会话身份、队列位置、错误和最终 artifact 元数据添加向后兼容的 status/API 字段。
4. 增量扩展现有 UI，不改变其框架。

退出标准：fake 端到端运行覆盖 start、stop、过期工作拒绝、录制完成、错误与重复生命周期
请求。

## Stage 6 — 硬件验证与发布

单元测试成功不等于真机验证。每一条可能运动的命令均需明确批准，并验证：

- 使用限定相机配置的 Piper/RM2 连接与预览；
- 复位、速度、单位和关节顺序的保留；
- 同步与 RTC 执行期间的停止延迟和关机；
- 策略延迟、重连和过期结果处理；
- 在帧和磁盘压力下的有界录制。

每条硬件命令均记录预期运动、停止流程和结果。在证明行为对等前，保留回退至旧行为的路径。

## 按文件的迁移顺序

1. 将 `robots/*/infer.py` 保留为拥有 SDK 的实现；只修改小范围 adapter/lifecycle 调用点。
2. 添加评测特性前，先通过 `runtime/inference.py` 合并循环。
3. 录制 telemetry 前先添加 telemetry model。
4. 暴露录制控制前先添加录制 worker。
5. worker 和生命周期契约经过测试后，再添加评测 API/UI。

每个阶段均运行：

~~~
uv run python -m unittest discover -s tests -v
uv run ruff check .
~~~

随后审查 diff 中的无关变更，并说明仍需真机验证的内容。

## Stage 12 — 可复现 Baseline 与 A/B 报告

Stage 12 添加基于文件系统的 Baseline store、分类配置 diff、指向评测/开环输出的紧凑链接，
以及始终保留分子、分母和无效处理的 A/B 报告。它复用 `EvaluationService`、`ActionSpec`、
共享 runtime 和既有资源租约；不添加数据库、策略客户端、机器人 adapter 或独立控制循环。

Baseline 写入排入有界的非 daemon writer。通过 Web 创建或运行 Baseline 都不会启动
episode；终态的 Baseline 评测会异步关联。兼容性不匹配要求明确创建派生 Baseline。本阶段
还记录其余 Web profile 的 Piper 假设，并要求在任何硬件验证前完成 fake Piper/RM2 回归
覆盖。
