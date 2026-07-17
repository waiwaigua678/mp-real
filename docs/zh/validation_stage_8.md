# Stage 8 验证：离线数据查看器

日期：2026-07-16

查看器使用提供的标准 LeRobot v2.1 数据集验证：

```text
/home/pc4/.cache/huggingface/lerobot/local/piper_1armblowv01
```

观察到的元数据：

- 机器人：Piper
- Episodes：165
- FPS：30
- 相机：`cam_head`、`cam_left_wrist`、`cam_right_wrist`
- Episode 0：344 个样本
- 为没有可选 mp-real 遥测的标准 LeRobot 数据。

本地 loopback HTTP 冒烟验证了目录枚举、episode 样本查找、`cam_head` JPEG 解码和
Episode Viewer 页面服务；未使用 Robot、相机工厂、策略客户端、CAN 接口或 SDK。

`tests/test_data_view.py` 的自动覆盖包括：目录访问、样本索引/时间戳/进度查找、可拖动
查看器游标计算、播放/暂停速率、同步视频/状态元数据、缺失视频和遥测、不完整数据集、
路径遍历拒绝、保峰下采样、动态动作维度、指标、运行时事件显示和所选样本状态。

该数据集未覆盖：mp-real 原始/稳定化/已执行动作和相机年龄遥测展示、RM2 物理相机
录制，以及真正很长的单个真实 episode（读取器使用 512 行 Parquet 批次且不缓存解码
后的 episode 表）。本阶段不需要也未进行真机验证。
