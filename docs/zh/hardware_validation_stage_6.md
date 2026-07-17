# Stage 6 硬件验证：LeRobot v2.1 录制

本文是运行手册，不是硬件运动授权。每个可能运动的级别都需要在当前任务中单独获得
明确批准。全程保持物理急停可用，并在生成的数据集通过验证前保留原部署版本。

## 每个级别之前

- 创建新的数据集根目录，绝不向已完成录制追加。
- 在运行日志记录机器人、相机角色、策略端点、操作员、任务和目标级别。
- 确认本地可运行 `mp-data-inspect <dataset>` 和 `mp-data-validate <dataset>`。
- 若策略报告安全拒绝、录制器报错、相机年龄超过阈值、写入队列持续增长，或操作员
  发现意外运动，立即停止。

## L1：只使用假资源

用假机器人、相机和策略运行 unittest。预期：两 episode 的 v2.1 数据集可验证、
视频与 Parquet 行对齐、可读取外部标准 v2.1 数据集。

```bash
uv run python -m unittest tests.test_lerobot_v21 tests.test_evaluation -v
uv run mp-data-validate <generated-dataset>
```

此级别不发出硬件命令。

## L2：实时相机与机器人状态，不发命令

需要明确批准，并使用刻意配置的无命令录制路径；Stage 6 不会自动把 infer-only
执行变成录制会话。录制静止观测 60 秒后正常停止。

预期：每个相机每个 Parquet 行恰有一个 MP4 帧；`camera_frame_reused` 与
`camera_age_ns` 可解释重复帧；CAN 或厂商命令日志没有动作命令。收集控制器日志、
写入器高水位、丢弃计数、`meta/mp_real/events` 和校验器输出。通过停止部署、关闭
相机、只删除新建且未完成的数据集目录来回退。

## L3：低速空载运动

在明确说明策略、预期工作区、速度限制和急停程序后才可批准。使用已有保守限制，
不可为了录制提高速度。空载录制 30 秒。

预期：标准 `action` 等于机器人边界报告的已执行动作；原始和稳定化动作仍保留在
mp-real 遥测；写入队列有界，控制周期遥测符合现有安全预期。运行后立即校验。

若发现不匹配，用正常 Web 停止或急停，禁用部署，保留 `.inprogress` 和日志诊断，
并回到原运行时版本。

## L4：生产 episode

每个机器人及每个运动会话均需明确批准。至少录制三个 Piper 和三个 RM2 episode，
每种机器人均含一次正常完成和一次人工中止。

```bash
uv run mp-data-inspect <dataset>
uv run mp-data-validate <dataset>
```

归档校验输出、会话/标签元数据、事件、遥测和恢复元数据。失败或中止运行必须清晰
标记为 `INCOMPLETE` 或 `INVALID`，不得改名为有效训练数据。
