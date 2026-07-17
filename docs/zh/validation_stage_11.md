# Stage 11 验证：teacher-forced 开环评测

## 安全边界

`mp-open-loop-eval` 和 `mp-data-view` 的可选任务均为离线策略评测路径。它们不创建
`Robot`、相机、机器人注册项或机器人专属 SDK 对象，也绝不调用动作执行方法。只有
操作员启动命令/任务后，才可能连接提供的策略 WebSocket。

## 必需的真实数据验证

在可用策略服务器下运行每条命令；这不是机器人运动测试。

1. 先验证目标数据集：

   ```bash
   uv run mp-data-validate recordings/<piper-dataset>
   uv run mp-data-validate recordings/<rm2-dataset>
   ```

2. 对两个策略标签分别运行一个 Piper 和一个 RM2 episode，并让每个策略使用独立输出
   目录：

   ```bash
   uv run mp-open-loop-eval --dataset recordings/<piper-dataset> --episode 0 \
     --policy-url ws://<policy-host> --policy-label checkpoint-a \
     --target-source action --alignment sample_index \
     --output open_loop_results/piper-checkpoint-a

   uv run mp-open-loop-eval --dataset recordings/<rm2-dataset> --episode 0 \
     --policy-url ws://<policy-host> --policy-label checkpoint-b \
     --target-source action --alignment timestamp --max-timestamp-error 0.05 \
     --output open_loop_results/rm2-checkpoint-b
   ```

3. 检查 `config.json`、`summary.json`、每个 `reports/episode_*.json` 和相应的
   `predictions/*.npz`。确认 `teacher_forced=true`、每个结果根只有一个目标来源、
   ActionSpec/状态模式符合预期、预热动作被丢弃、有效掩码尾部处理正确、源数据集未变。

4. 只有在 `frame_index` 被明确批准为录制控制步，且每个被评估源样本都有非负
   `mp_real.chunk_cursor` 遥测时，才使用绝对控制步对齐。传入
   `--allow-frame-index-as-control-step`；否则使用样本索引或时间戳对齐。

## 解读限制

开环误差不是真机成功率。teacher forcing 始终使用录制状态/图像/prompt，因此无法暴露
全部闭环漂移或误差累积。多模态策略可以选择不同于录制 expert 的动作但仍有价值。除
逐点 MAE 外还应检查夹爪事件时间和 chunk 重叠一致性，且绝不可合并 ActionSpec 或
目标来源不同的结果。

## 本实现已运行的自动检查

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run mp-open-loop-eval --help
```

编写该实现时未连接真实策略服务器或录制的 Piper/RM2 部署数据集。Stage 11 离线评测
器本身不需要真机验证。
