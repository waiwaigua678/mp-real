# LeRobot v2.1 模式与 Baseline

录制的 mp-real 数据集会保留带字段名、单位和语义的版本化 ActionSpec。Baseline 将
这一契约保留为紧凑 JSON 快照，不重复保存 episode Parquet 或视频。`ActionSpec`
新增模式版本和动作模式，同时保持旧构造方式兼容；动作/状态名称、机械臂数量和
夹爪索引由 `VectorField` 推导，并可通过录制模式往返序列化。

只有声明契约完全一致的结果才能用于 A/B 汇总。状态布局或单位不同仍必须使用显式
映射。
