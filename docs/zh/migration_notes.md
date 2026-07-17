# 迁移说明

Stage 12 新增 `mp-baseline` 与 Web Baseline 页面，未改变既有 Piper/RM2 CLI 名称、
共享推理循环、策略客户端或录制模式位置。由于新元数据字段均追加且有默认值，已有
`ActionSpec(...)` 调用仍然有效。

为兼容性保留了废弃的 RM2 专用 Web 实现；正常的
`mp-real-web --robot rm2` 使用共享 profile 运行时。剩余的 Piper 形状 Web 接线及
未来应迁移到 profile 接口的内容见 `piper_rm2_generality_audit.md`。
