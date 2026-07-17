# 可复现 Baseline 工作流

## 范围

Baseline 是实验不可变且带版本的定义，不是 LeRobot 数据的副本，也绝不包含
策略凭证。后端文件系统存储是唯一权威来源，不使用浏览器存储。

每个 Baseline 记录 Git commit、策略 URL/标签/元数据、可选 checkpoint hash、
ActionSpec、状态模式、相机角色与配置、机器人/运行时/安全设置、RTC 与预热
设置、评测协议、起始位姿协议，以及可选的源数据集/episode/sample 引用。无法
获得 checkpoint hash 时保存 `null`，绝不伪造。

## 创建、派生与运行

1. 连接目标 `DEPLOYMENT` 运行时并保存其配置。
2. 在 Web 的 **Baseline** 页面填写策略标签、任务、协议和操作员元数据来创建
   Baseline；有界后台写入器会原子持久化它。也可用 **从当前评测创建**，或
   `mp-baseline create --from-evaluation-snapshot snapshot.json` 捕获当前
   `EvaluationSession`。
3. 若修改 checkpoint、RTC、相机、FPS、安全限值或起始状态，克隆原 Baseline 并
   写明派生原因。正式 A/B 对比应尽量一次只改变一个主要变量。
4. **从 Baseline 创建评测**会比较实时配置与 Baseline；任何影响执行的差异都会
   按类别显示并被拒绝。
5. 该操作仅创建 `EvaluationSession`；操作员仍需人工预热、确认复位并开始每一轮。
6. Baseline 关联的评测达到终态后由后台写入器异步附加。开环结果使用
   `attach-open-loop` 或 Web 关联控件附加。

## 结果语义

成功率始终显示百分比、分子、分母和样本量。`SUCCESS`、`FAILURE`、`TIMEOUT`
和 `SAFETY_ABORT` 是有效试次；`INVALID`、`SYSTEM_ERROR` 和 `OPERATOR_ABORT`
仍会展示，但不进入成功率分母。`1/1` 会显示为 `100% (1/1; valid n=1)`，仅为
冒烟结果，不是稳定结论。

机器人、数据集格式、ActionSpec、状态模式或相机角色不同的结果不会汇总。开环
对比还要求目标来源与对齐模式相同。

## CLI

`mp-baseline create` 接受元数据 JSON 和已脱敏的运行时配置 JSON；对于实时部署，
Web 页面更方便。`run --web-url` 调用 Web Baseline 运行端点，仅创建人工评测
会话，绝不会自行启动机器人动作。
