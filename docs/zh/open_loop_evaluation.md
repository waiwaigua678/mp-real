# 开环评测

`mp-open-loop-eval` 是离线 teacher-forced 评测。它可以连接策略服务器，但绝不导入
机器人 SDK、创建 Robot 或发送动作。关联其 `summary.json` 和 `config.json` 时，会
保留数据集、episode 选择、目标来源、对齐、ActionSpec、状态模式和相机角色契约。
契约不同的开环结果保持分离，绝不在 Baseline A/B 报告中汇总。

开环误差不等于真机成功率。它只能与人工标注的真机 Baseline 结果并列解读，不能
替代后者。
