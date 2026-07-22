# Piper / RM2 能力矩阵

`supported` 表示已有软件路径并有 fake/mock 覆盖，不等于完成硬件验证。RM2 CLI 部署默认
配置有单独记录的操作员验证范围，但不延伸到 RM2 Web、录制状态移动或轨迹回放；Piper
尚未声明真机移动/回放验证。

| 能力 | Piper | RM2 | 状态说明 |
| --- | --- | --- | --- |
| CLI 部署 | supported | supported | 共享 sync/RTC/infer-only 循环；RM2 默认配置有有限的已记录真机验证范围 |
| Web 部署 | supported | supported | 共享 `RobotWebRuntime` profile 路径 |
| CAMERA_PREVIEW | supported | supported | 不创建 Robot 或 PolicyClient |
| Web 策略预热 | supported | supported | 预热动作会丢弃 |
| CLI 启动预热 | unsupported | unsupported | CLI 仍走既有直接循环路径 |
| sync / RTC | supported | supported | 由 ActionSpec 驱动 |
| EvaluationSession | supported | supported | 假机器人覆盖 |
| LeRobot v2.1 录制 | supported | supported | 有界写入器覆盖 |
| 离线数据查看 | supported | supported | 无硬件资源 |
| 移动到录制状态 | experimental | experimental | 默认安全校验会阻止物理执行 |
| 从录制状态部署 | experimental | experimental | 需要已验证姿态交接 |
| 轨迹回放 | experimental | experimental | 等待安全验证，物理执行被阻止 |
| 开环评测 | supported | supported | 机器人中立，绝不执行动作 |
| Baseline 与 A/B 报告 | supported | supported | 通用 ActionSpec 和运行时快照 |

共享 `runtime/`、`data/`、`evaluation/`、姿态校验、回放规划和开环指标使用
`Robot`、`ActionSpec`、状态字段和相机角色。
