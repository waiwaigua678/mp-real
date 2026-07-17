# 当前架构（Stage 0 基线）

本文仅用于描述：它记录为评测平台重构所检查的架构，不规定新行为。

## 范围与稳定边界

- `Robot` 是硬件边界。Piper 与 RM2 SDK 调用保留在各自的
  `robots/<name>/infer.py` 模块中。
- `ActionSpec` 是面向策略的形状契约；共享 runtime 不假设 Piper 的 14 元素动作
  布局。
- `InferenceAdapter` 向 `runtime/inference.py` 提供观测、解码、稳定化、执行和
  仅推理元数据。
- 当前有两个同步 WebSocket 策略客户端：用于 CLI 的 `WebsocketClientPolicy` 和用于
  Web 的 `web.server.PolicyClient`。
- `capture_observation` 在 `ObservationSnapshot` 中保留相机和机器人状态时间戳；
  `to_policy_observation()` 不会将它们放入策略 wire payload。

## A. `mp-piper-infer` CLI 调用链

~~~mermaid
flowchart TD
  A[mp-piper-infer entry point] --> B[piper.infer.cli / tyro Args]
  B --> C[piper.infer.main]
  C --> D[InferenceLoopConfig.from_args + validate]
  C --> E[create_robot Args]
  E --> F[create_piper_arm left/right; Piper SDK connect/configure/optional enable]
  F --> G[PiperRobot]
  C --> H{reset_only?}
  H -- yes --> I[robot.reset] --> Z[finally close robot]
  H -- no --> J{infer_only?}
  J -- no --> K[robot.reset]
  J -- yes --> L[skip reset action]
  K --> M[WebsocketClientPolicy]
  L --> M
  M --> N[make_cameras]
  N --> O[PiperInferenceAdapter: PiperRobot + cameras + Args]
  O --> P{loop mode}
  P -- infer_only --> Q[shared run_infer_only]
  P -- use_rtc --> R[shared run_rtc_loop]
  P -- otherwise --> S[shared run_sync_loop]
  Q --> T[finally close_cameras + robot.close]
  R --> T
  S --> T
~~~

`main()` 验证通用 loop 配置及 `command_rate_hz`，随后负责构造和清理。它在连接策略
前创建两条 Piper 机械臂，在策略连接后创建相机。第二条臂部分创建失败时关闭第一条臂；
`finally` 会关闭全部已创建相机与 `PiperRobot`。此路径中的 CLI WebSocket 客户端没有
显式 `close` 调用。

`PiperInferenceAdapter.observe()` 调用 `prepare_observation()`，再调用
`capture_observation()` 读取相机和机器人状态。通用循环随后进行下列之一：

- 同步：本地 deque 为空时拉取动作块，稳定化一个选中的原始动作，执行一次 transition，
  然后按 fps 休眠；
- RTC：启动 producer，在 `RealTimeChunkingBuffer` 中融合可用的重叠动作块，稳定化并
  执行；等待时可重复最后一个已执行动作；
- 仅推理：获取并打印新的动作块，不稳定化也不执行。

## B. Web 启动到 Piper 推理的调用链

~~~mermaid
flowchart TD
  A[mp-piper-web] --> B[web.server.main]
  B --> C[PiperWebServer + PiperWebRuntime]
  C --> D[HTTP POST /api/start]
  D --> E[PiperWebRuntime.start]
  E --> F{already connected?}
  F -- no --> G[_connect]
  G --> H[create_robot piper]
  H --> I[direct PiperRobot/PiperArm access]
  I --> J[optional infer_piper.maybe_reset_arms]
  J --> K[web.PolicyClient]
  K --> L[infer_piper.make_cameras]
  L --> M[start daemon camera thread]
  F -- yes --> N[apply direct Piper arm settings]
  M --> N
  N --> O[start daemon run thread]
  O --> P[_run_loop]
  P --> Q{infer_only / RTC / sync}
  Q --> R[_run_infer_only_loop]
  Q --> S[_run_rtc_control_loop + producer]
  Q --> T[_run_sync_control_loop]
  R --> U[_prepare_observation + _infer_action_chunk]
  S --> U
  T --> U
  U --> V[latest preview frames + direct infer_piper.read_state]
  V --> W[web.PolicyClient.infer]
  W --> X[direct infer_piper decode/stabilize/execute]
~~~

请求处理器调用 `PiperWebRuntime.start()`；实际推理在 `_run_thread` 中执行。
`start()` 可以先同步执行 `_connect()`，因此机械臂/相机创建、复位和策略连接可能在
HTTP 请求线程发生。`disconnect()` 和 `reset_arms()` 也会从请求线程进行生命周期或
机器人工作。这是基线事实，并非推荐做法。

Piper Web 的相机 worker 读取并 JPEG 编码图像。`_prepare_observation()` 复制最新
图像，并直接调用 `infer_piper.read_state()`。`FrameSnapshot` 仅保留 Web 收取时间
（`updated_at`），不保留 CLI 路径使用的源相机时间戳。

## 共享 runtime 与重复的 Piper Web 控制逻辑

`runtime/inference.py` 负责 `_fetch_chunk`、仅推理采集、同步规划、RTC producer、
`RealTimeChunkingBuffer` 使用、hold-last 行为和 producer 错误传播。两个 CLI adapter
都调用它。

| 通用 runtime 行为 | Piper Web 的重复实现 |
| --- | --- |
| `run_infer_only` | `_run_infer_only_loop` |
| `run_sync_loop` | `_run_sync_control_loop` |
| `_rtc_producer` 和 `run_rtc_loop` | `_rtc_action_producer_loop` 和 `_run_rtc_control_loop` |
| `_fetch_chunk` | `_prepare_observation` 加 `_infer_action_chunk` |
| `raise_rtc_producer_error` | `_raise_producer_error` |

Web 副本增加 UI 指标并使用预览帧，但也产生差异：Web 的仅推理不使用共享 `.npz` 写入器、
将 `max_steps` 作为动作块计数，且不能使用 adapter 的仅推理元数据。除 runtime stop
event 外，其 RTC producer 另有一个 `producer_stop`。

## Piper Web 对 `infer_piper` 的直接依赖

`web/server.py` 在模块导入时导入 `mp_real.robots.piper.infer`，并在整个
`PiperWebRuntime` 中使用它：

- 默认值、相机掩码和配置序列化使用 `infer_piper.Args` 与 Piper 专有字段；
- `_connect()` 创建 registry 名称 `"piper"`、断言 `PiperRobot`、取出 left/right，
  并直接调用复位、相机和机械臂关闭 helper；
- 观测、动作解码、机械臂速度更新、三个循环和复位都直接调用 Piper 模块函数；
- 机器人选择会实例化 `PiperWebRuntime` 或 `Rm2WebRuntime`。

因此 Web 层了解 Piper 机械臂句柄、相机接线、解码、稳定化和 transition 执行。这是
增量迁移在不改变浏览器 API 的情况下需要缩小的依赖面。

## Web 中的 RM2 选择

启动时 `--robot rm2` 创建 `Rm2WebRuntime`。运行中，`POST /api/robot` 调用
`PiperWebServer.select_robot()`；仅当前 runtime 既未连接也未运行时允许切换，随后
以新的 `Rm2WebRuntime` 替换 runtime 对象。

`Rm2WebRuntime.connect()` 在 HTTP 请求线程中运行：

1. 创建 Web `PolicyClient`；
2. 调用 `infer_rm2.make_cameras()`；
3. 调用 `create_robot("rm2", args)` 并要求其为 `Rm2Robot`；
4. 调用 `robot.reset()`（其检查 `reset_on_start`）；
5. 保留资源并启动预览相机 worker。

`start()` 创建 daemon 运行线程。`_run_loop()` 创建 `_Rm2WebAdapter`，并使用 Web
stop event 与 step callback 调用共享 `run_policy_loop()`。adapter 提供最新预览图像、
相机参数和 `Rm2Robot.read_state()`，并将解码/稳定化/执行委派给 RM2 代码。因此 RM2
Web 已共享推理循环，尽管其 Web 生命周期、预览 worker 和策略客户端仍为 Web 专用。

## 资源所有权、线程和故障行为

| 路径 | 创建内容 | 停止/关闭 | 线程与错误行为 |
| --- | --- | --- | --- |
| Piper CLI | `main()` 中的 `PiperRobot`、策略客户端、相机 | `finally`：`close_cameras`、`robot.close` | 同步模式没有线程；通用 RTC 启动 daemon producer，在 `finally` 设置 event、join 2 秒，然后重新抛出已入队 producer 错误。 |
| Piper Web | 机械臂/robot、Web client、相机；connect 时相机 worker、start 时运行 worker | `stop()` 发出运行信号；`disconnect()` join 运行线程 5 秒及相机线程 `camera_timeout + 1`，再关闭资源 | 所有 worker（包括 RTC producer）均为 daemon。循环错误会改变 Web error state；相机错误按帧保存；RTC producer 错误入队后由 consumer 重新抛出。 |
| RM2 Web | Web client、相机、`Rm2Robot`；connect 时相机 worker、start 时运行 worker | `disconnect()` 发出运行信号，join 运行线程 5 秒及相机线程 3 秒，再关闭资源 | 运行/相机与通用 RTC producer 都是 daemon。通用 producer 错误到达 `run_policy_loop`，由 `_run_loop` 捕获。 |

空闲时两个 Web `stop()` 都幂等：Piper 返回当前状态，RM2 设置已经设置的 event。两者
都有有界 join，但不会在更广泛生命周期清理前验证超时 worker 是否真的退出。通用 RTC
错误队列无界；`RealTimeChunkingBuffer` 将动作块保存在 dict 中，读取时才清除过期块。

共享 `run_policy_loop()` 将 stop event 传给同步和 RTC 模式。其仅推理分支调用的
`run_infer_only()` 没有该 event，所以 RM2 Web 仅推理仅受配置动作块数以及阻塞的
相机/策略调用限制；Piper Web 有自己的仅推理停止检查。

## 仅推理持久化格式

配置后，共享仅推理模式将压缩 NumPy `.npz` 写到 `infer_only_output`：

| Key | 值 |
| --- | --- |
| `actions` | `float32`，`[infer_only_chunks, T, action_dim]`；机器人验证/replan slicing 后、稳定化或执行前解码的原始策略动作块。 |
| `states` | `float32`，`[infer_only_chunks, state_dim]`；策略观测中的状态。 |
| `prompt` | 来自 `config.prompt` 的标量 NumPy array/string。 |
| extra keys | 来自 `infer_only_metadata` 的 object array；RM2 写入 `camera_params`，Piper 不写。 |

它会创建父目录但直接写入：没有 `.inprogress` 目录或原子重命名。它不保存图像、源时间戳、
原始服务器响应、选中动作、稳定化目标或实际执行动作。Piper Web 的仅推理只获取动作块并
更新指标，没有输出路径设置。

## Web API 清单

`PiperWebHandler` 提供以下路由。服务器配置 access key 后，每个 `POST` 都要求
`X-Motrix-Key`。JSON 错误为带对应 HTTP 状态的 `{"ok": false, "error": "..."}`。

| 方法 | 路径 | 请求 | 响应 | Runtime 方法 / 行为 |
| --- | --- | --- | --- | --- |
| GET | `/` | 无 | `index.html` | 静态文件 |
| GET | `/static/<path>` | 无 | 静态文件 | 仅限 static root 的静态文件 |
| GET | `/api/status` | 无 | status object | `runtime.status()` |
| GET | `/api/config` | 无 | config object | `runtime.get_config()` |
| GET | `/snapshot/<camera>.jpg` | 命名相机 | JPEG | `runtime.wait_frame(camera, -1, timeout=0.1)` |
| GET | `/stream/<camera>.mjpg` | 命名相机 | multipart MJPEG | 重复 `runtime.wait_frame(camera, sequence)` |
| POST | `/api/config` | JSON partial config | `{"ok": true, "config": ...}` | `runtime.update_config(payload)` |
| POST | `/api/connect` | 可接受空 JSON | `{"ok": true, "status": ...}` | `runtime.connect()` |
| POST | `/api/start` | 可接受空 JSON | `{"ok": true, "status": ...}` | `runtime.start()` |
| POST | `/api/stop` | 可接受空 JSON | `{"ok": true, "status": ...}` | `runtime.stop(wait=False)` |
| POST | `/api/disconnect` | 可接受空 JSON | `{"ok": true, "status": ...}` | `runtime.disconnect()` |
| POST | `/api/reset` | 可接受空 JSON | `{"ok": true, "status": ...}` | `runtime.reset_arms()` |
| POST | `/api/ping_policy` | 可接受空 JSON | `{ok, connected, metadata, latency_ms?}` 或错误 | `runtime.ping_policy()` |
| POST | `/api/robot` | `{"robot": "piper" or "rm2"}` | `{"ok": true, "config": ...}` | `server.select_robot()` |

status 包含 phase/connectivity/capability flags、task/counters、策略 metadata/error、
metrics、frame metadata 和 logs。Piper 还包含 `stop_requested`、仅推理与详细 metrics；
RM2 当前返回空的 metrics 和 metadata。未知路径返回 JSON 404；其他 HTTP 方法未显式
实现。

## Stage-0 表征保护线

`tests/test_runtime.py` 不依赖硬件、网络或策略服务器，锁定这里记录的行为：同步动作块
消费、RTC 重叠融合与 generation 拒绝、hold-last 执行、时间戳保留、registry 行为、
Piper Web 配置往返 coercion 与幂等 stop。它们是行为保护测试，不是产品特性。
