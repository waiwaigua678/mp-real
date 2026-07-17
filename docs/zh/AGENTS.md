# mp-real 仓库说明

## 项目范围

mp-real 是一个轻量级真机部署、控制、评测、录制、回放和可视化项目。OpenPI 训练与
模型服务栈在仓库外；mp-real 连接单独部署的 WebSocket 策略服务器。

当前实现支持：

- Piper
- RM2
- CLI 部署
- Web 部署与控制
- 相机预览
- 同步动作块执行
- RTC 动作块执行
- 仅推理执行

修改行为前阅读 `README.md` 和相关源码。

## 既有架构

不要为既有抽象创建平行替代品。现有边界包括：

- `Robot` Protocol
- `ActionSpec`
- Robot registry
- `PiperRobot`
- `Rm2Robot`
- `InferenceAdapter`
- `PolicyClient`
- 共享同步与 RTC 推理 runtime
- RealSense、V4L2、ROS 与 black 相机后端

机器人专用 SDK 调用必须保留在：

```text
src/mp_real/robots/piper/
src/mp_real/robots/rm2/
```

共享 runtime 代码不得导入厂商机器人 SDK。优先扩展既有抽象，不要引入
`RobotAdapter`、`CameraAdapter`、另一个 `PolicyClient` 或另一条独立推理循环。

## 重要文件

架构变更前阅读：

- `README.md`
- `pyproject.toml`
- `src/mp_real/robots/base.py`
- `src/mp_real/robots/registry.py`
- `src/mp_real/runtime/models.py`
- `src/mp_real/runtime/inference.py`
- `src/mp_real/runtime/observation.py`
- `src/mp_real/common/camera.py`
- `src/mp_real/robots/piper/infer.py`
- `src/mp_real/robots/rm2/infer.py`
- `src/mp_real/web/server.py`
- `static/index.html`
- `static/app.js`
- `tests/test_runtime.py`

评测平台路线图还要阅读：

- `docs/plans/evaluation-platform.md`

## 架构约束

- 复用 `runtime/inference.py` 中的共享 runtime。
- 不要继续在 Web 层复制控制循环。
- Web 请求处理器不得直接执行机器人动作。
- Web 请求处理器不得阻塞推理或磁盘写入。
- 不要一次大改重写 `server.py`。
- 不要迁移既有前端框架。
- 除非明确批准，不要添加数据库、pandas、PyArrow、React、Vue 或 FastAPI。
- 保留现有 Piper/RM2 CLI 与 Web 行为。
- 新数据结构必须使用 `ActionSpec`，不能假设固定动作维度。
- 共享模块中不得硬编码 Piper 的 14 维动作布局。

## Runtime 与并发规则

每个后台 worker 必须具备：

- 显式停止机制
- 适用时的有界队列
- 明确定义的 join 行为
- 异常传播
- 确定性清理

不要用 daemon 线程掩盖生命周期问题。每个延迟结果必须携带足够身份信息来拒绝过期
结果：`session_id`、`generation_id`、`request_id`，以及适用时的 `chunk_id`。

运行时事件排序使用 `time.monotonic_ns()`；墙钟时间只用于显示与文件名。只要可用，
必须区分：

- 原始策略动作块
- 选中的原始动作
- 稳定化后的目标动作
- 机器人边界返回的实际执行动作

不得悄悄将其合并成一个动作字段。

## 录制要求

录制与视频编码不得阻塞机器人控制循环。使用：

- 有界生产者队列
- 后台写入 worker
- 版本化录制 schema
- 临时 `.inprogress` 目录
- 可行时的原子完成

始终记录丢帧或丢失遥测。绝不静默丢弃录制失败。

## 硬件安全

除非用户在当前任务中明确授权真机运动测试，不要运行可能使真机运动的命令。

任何可能运动的测试前：

1. 说明将运行的命令。
2. 说明其为何可能引起运动。
3. 说明预期运动。
4. 说明停止与回滚步骤。
5. 等待明确用户批准。

默认自动测试必须使用 fake robot、mock policy、black camera 和 dry-run 行为。
dry-run 不一定是离线仿真；不能仅因启用 dry-run 就假定没有硬件。

绝不：绕过关节或动作安全检查；在重构中提高机器人速度；在测试中自动使能或复位
硬件；没有 schema 与初始状态验证就回放轨迹；猜测动作单位、关节顺序或夹爪语义。

## 开发流程

非平凡工作按以下顺序进行：

1. 检查当前实现。
2. 总结当前调用路径。
3. 提出按文件划分的实现计划。
4. 标识兼容性与硬件风险。
5. 实现最小、连贯的变更。
6. 添加或更新自动测试。
7. 执行验证命令。
8. 审查 diff 中的无关变更。
9. 报告仍需真机验证的部分。

不要进行大范围无关格式化；没有迁移需求不要重命名公共 API；不要仅为简化抽象而删除
稳定行为。除非明确要求，不要提交或推送 Git 变更。

## 验证命令

报告完成前运行相关检查：

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

依赖变更时还要验证：

```bash
uv sync
```

确认这些入口仍可导入：

```bash
uv run mp-piper-infer --help
uv run mp-rm2-infer --help
uv run mp-piper-web --help
```

若命令因硬件或外部策略服务器不可用而无法运行，应明确报告，不得宣称已通过。

## 完成报告

每项任务结束时报告：

1. 修改的文件。
2. 重要设计决策。
3. 已执行测试及精确结果。
4. 未执行测试。
5. 仍需验证的硬件行为。
6. 已知风险或限制。
7. 建议的下一步。

分阶段路线图工作只实现请求的阶段；不要自动开始下一阶段。

## Runtime 模式与策略启动

Web runtime 必须区分资源模式：

- `deployment`：机器人、相机和策略服务器
- `camera_preview`：仅相机
- `offline_replay`：仅录制文件

相机预览与离线回放不得创建 `Robot` 或 `PolicyClient`。策略冷启动属于启动生命周期
问题，不是 RTC 故障。分别使用可配置超时处理：WebSocket 连接与元数据、策略预热、
稳态推理。

策略预热期间不得执行机器人动作。丢弃预热动作；控制循环进入运行阶段前必须准备好
全新的第一个动作块。保留推理 worker 失败的原始异常类型和消息；不得只用泛化 RTC
错误替代策略超时。
