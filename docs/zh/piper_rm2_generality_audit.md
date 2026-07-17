# Piper / RM2 通用性审计

## 范围与方法

此 Stage-12 审计在非测试共享 Python 模块中搜索固定动作/状态维度 `14`、`6 + 6`
布局、固定相机角色、固定夹爪索引、Piper 机器人名称判断、CAN 字段及直接
Piper/SDK 导入；并沿 `ResourceLeaseManager` 和 Web 运行时的部署、评测、姿态、
回放路径追踪资源所有权。

## 结果

| 区域 | 结果 | 证据/处置 |
| --- | --- | --- |
| `runtime/`、`data/`、`evaluation/`、`common/` | 通过 | 没有固定 14 维或 Piper SDK 假设；Baseline 使用 `ActionSpec`、状态名和相机角色。 |
| Baseline 配置采集 | 通过 | 共享 Baseline 使用通用的 `camera_config`、`robot_config`、`safety_config`；Piper/RM2 在各自 `RobotWebProfile` 中构建它们。 |
| 动作/状态模式 | 通过 | `ActionSpec` 可往返模式版本、名称、动作模式、机械臂数、夹爪索引和能力，且不破坏位置参数构造。 |
| 相机/动作解释 | 通过 | 共享回放、开环和录制代码遵循 `ActionSpec` 字段和相机角色；Piper/RM2 解码保留在机器人模块/profile。 |
| 机器人包外的直接厂商 SDK 导入 | SDK 导入通过 | 共享 runtime/data/evaluation 不直接导入 `pyagx`/RM SDK。Web/profile 对机器人 `infer` 模块的导入是既有构造接线，而非 SDK 调用。 |
| 共享 Web 生命周期 | 部分通过 | `RobotWebRuntime` 以 profile 支持两机器人，但 `web/server.py` 仍保留 Piper CLI 字段、占位相机帮助函数和兼容的旧 RM2 Web 实现；这是已记录迁移债务，未复制到新 Baseline 代码。 |
| 资源所有权 | 单进程通过 | 假测试覆盖部署/评测/姿态/回放控制冲突、重复控制/回放租约、陈旧租约保护以及离线数据与相机预览共存。浏览 Baseline 不创建资源租约。 |

## 必须遵守的边界

- 厂商 SDK 调用必须留在 `robots/piper/` 与 `robots/rm2/`。
- 共享代码只能使用 `Robot`、`ActionSpec`、状态模式、相机角色、profile 能力和显式映射。
- 新机器人必须提供自己的 `RobotWebProfile` Baseline 分类回调，不得向
  `evaluation/baseline/` 添加厂商字段名。
- 进程内资源管理器不协调多个进程。

## 剩余迁移项

为保持现有部署行为，本阶段没有改动旧的 Piper 形状 Web CLI/配置代码。未来可在单独
审查的重构中将这些 flag 与相机占位帮助函数完全移到 profile 接口之后；必须保持当前
命令名称，并在硬件测试前用两个假机器人验证。
