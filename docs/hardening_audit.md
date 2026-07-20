# Hardening H0/H1 Audit

本文件记录 H0 静态审查复核结果和 H1 数据语义修复状态。H0 只建立复现测试和审计证据，H1 修复 LeRobot observation/action/control-step 对齐。

## 当前调用链

部署路径复核以当前工作区为准：

- Web/CLI runtime 通过 `RuntimeController` 进入 `src/mp_real/runtime/inference.py` 的共享推理循环。
- 同步 runtime 在 `run_sync_loop()` 中调用 `fetch_action_chunk()`；该函数先 `adapter.observe()`，再 `PolicyClient.infer()`，再解码 action chunk。
- `run_sync_loop()` 和 `run_rtc_loop()` 将一个 policy chunk 展开为多个 robot-bound action。H1 后，每个实际发送的 action 都会先采集一条新的 control-step observation。
- `RuntimeEventHooks` 将 observation、chunk、action selected/stabilized/executed 转成 `RuntimeEvent`。
- H1 新增 `ControlStepRecorded` 作为标准 LeRobot row 的权威事件。`ActionExecuted` 兼容路径仍保留，但同一 control step 已有 `ControlStepRecorded` 时不会重复写 row。
- `LeRobotV21EpisodeRecorder` 在后台线程消费事件；H1-aligned 数据直接使用 `ControlStepRecorded.state/images/executed_action` 写入 LeRobot v2.1 frame，policy observation 和 raw chunk 只作为 telemetry。
- replay 路径为 `ReplayPlanner` 离线生成 `ReplayPlan`，然后 `RobotReplayController.prepare()` move-to-start，`confirm_and_start(plan_hash)` 启动 replay。
- move-to-recorded-state 路径为 `MoveToStateValidator`、机器人 `validate_pose_target()`、`MoveToRecordedStatePlan.build()`、`PoseMoveController.start()`。

## 审查问题

| # | 审查问题 | 当前代码位置 | 确认 | 严重级别 | 可复现测试 | 影响范围 | 建议修复阶段 | 阻止真机运动 | 阻止数据用于训练 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LeRobot recorder 是否给同一个 action chunk 的多个动作重复配同一 observation/state/images | `src/mp_real/runtime/inference.py::run_sync_loop`; `src/mp_real/data/lerobot_v21.py::LeRobotV21EpisodeRecorder._run` | H1 已修复 | Critical | `test_h1_policy_chunk_actions_should_not_share_one_recorded_observation`; `test_h0_lerobot_semantic_audit_fixture_reports_risk_signals` | policy runtime 录制、后续训练、open-loop 评估 | H1 | 否，主要是数据语义 | 修复前阻止，H1-aligned 新数据不阻止 |
| 2 | 控制循环是否每个实际动作前采集新机器人状态和相机快照 | `fetch_action_chunk()` 只在取 chunk 时 observe；`run_sync_loop()` 从 plan deque 连续执行 | H1 已修复 | Critical | `test_h1_control_loop_should_capture_before_each_executed_action`; `test_h1_rtc_records_control_step_observations_for_prefetched_chunk` | 同步/RTC 数据语义、状态-动作对齐 | H1 | 条件性，不应声称逐动作闭环观测 | 修复前阻止，H1-aligned 新数据不阻止 |
| 3 | ObservationCaptured、ActionSelected、ActionExecuted 是否能通过同一个 control step 唯一关联 | H1 通过 `ControlStepRecorded` 聚合 control-step observation、selected、stabilized、executed action | H1 已修复 | High | `test_h1_observation_and_actions_should_join_by_control_step`; `test_sync_and_rtc_keep_action_event_order` | 审计、debug、数据清洗、训练过滤 | H1 | 否 | 修复前阻止，H1-aligned 新数据不阻止 |
| 4 | Piper/RM2 `validate_pose_target()` 是否总是产生 critical issues | `src/mp_real/robots/piper/infer.py::PiperRobot.validate_pose_target`; `src/mp_real/robots/rm2/infer.py::Rm2Robot.validate_pose_target` | 已确认 | Critical | `test_h0_piper_and_rm2_pose_validation_currently_block_move_to_state` | move-to-state、replay move-to-start | H2/H5 | 是 | 否 |
| 5 | `PoseValidationReport.require_valid()` 是否让真实 Piper/RM2 永远无法 move-to-state | `src/mp_real/pose/models.py::PoseValidationReport.require_valid`; `src/mp_real/web/server.py::_run_pose_connect`; `src/mp_real/replay/controller.py::_move_to_start` | 已确认 | Critical | `test_h0_piper_and_rm2_pose_validation_currently_block_move_to_state` | Web/CLI move-to-state、robot replay | H2/H5 | 是 | 否 |
| 6 | Replay/Pose plan 中 NumPy 数组是否仍可原地修改 | `src/mp_real/replay/models.py::ReplayStep`; `src/mp_real/pose/models.py::PoseWaypoint`; frozen dataclass 未冻结 ndarray buffer | 已确认 | Critical | `test_h3_replay_plan_hash_should_reject_step_array_mutation` expected failure; `test_h3_pose_plan_hash_should_reject_waypoint_array_mutation` expected failure | replay/move-to-state 安全确认 | H3 | 是 | 否 |
| 7 | 执行前是否只比较旧 `plan_hash`，没有重新计算 payload hash | `RobotReplayController.confirm_and_start`; `src/mp_real/web/server.py::replay_start/pose_execute/pose_prepare_deployment/pose_start_deployment`; `PoseMoveController.start` | 已确认 | Critical | 同 #6 两项 expected failure | replay/move-to-state 审核完整性 | H3 | 是 | 否 |
| 8 | ReplayController 是否发送命令后立即读状态，并将 sent 当 acknowledged | `src/mp_real/replay/controller.py::_run_replay` 设置 `sent_sample_index == acknowledged_sample_index` | 已确认 | High | `test_h3_replay_controller_should_not_acknowledge_sent_with_nonzero_tracking_error` expected failure | robot replay 状态语义、安全审计 | H3/H5 | 是，robot replay No-Go | 否 |
| 9 | ReplayPlanner 的 max_step、velocity、acceleration 是否同时作用于夹爪维度 | `src/mp_real/replay/planning.py::_kinematics` 对完整 target vector 计算 | 已确认 | High | `test_h2_replay_planner_should_not_apply_arm_limits_to_gripper_only_changes` expected failure | replay 规划、夹爪语义、误拒绝或误设限 | H2 | 是，robot replay No-Go | 条件性，影响 replay 不是训练本体 |
| 10 | Recorder 是否长期保留带图像的 observation cache | `src/mp_real/data/lerobot_v21.py::LeRobotV21EpisodeRecorder._run` 的 `observations` dict 只增不删 | 已确认 | High | `test_h4_recorder_should_release_matched_observation_images` expected failure | 长 episode 录制内存、稳定性 | H4 | 否 | 条件性，可能导致长录制失败或丢失 |
| 11 | telemetry 是否在 episode 结束前全部堆积在内存 | `src/mp_real/data/lerobot_v21.py::_EpisodeWriter` 的 telemetry list; `close()` 才写 `.npz` | 已确认 | High | `test_h4_telemetry_should_be_durable_before_episode_end` expected failure | 长 episode、崩溃恢复、审计完整性 | H4 | 否 | 条件性，影响数据完整性和可审计性 |
| 12 | 双臂 ReplayPlan 的 `arm_count` 是否被覆盖为 1 | `src/mp_real/replay/planning.py::_source_contract`; 只有 `arm_count <= 0` 时兜底为 1 并加入 error | 否定 | Low | `test_h0_dual_arm_replay_plan_preserves_declared_arm_count` | replay contract | 无需修复 | 否 | 否 |
| 13 | Recorder metadata 的 `action_mode` 是否写死 | `src/mp_real/data/lerobot_v21.py::_build_info`; `src/mp_real/data/models.py` legacy info | H1 已修复 LeRobot recorder | High | `test_h1_recorder_metadata_should_preserve_action_spec_action_mode` | replay metadata、训练/评估筛选 | H1 | 否 | 修复前阻止，H1-aligned 新数据不阻止 |
| 14 | README 引用的安全和能力文档是否真实存在 | `README.md` 链接 `docs/*.md`; 实际文件在 `docs/en/` 和 `docs/zh/` | 已确认路径不一致 | Medium | `test_h6_readme_local_doc_links_should_exist` expected failure | 文档可用性、操作员安全流程 | H6 | 条件性，阻止按 README 完成安全复核 | 否 |
| 15 | `av`、`pyarrow` 是否仍属于核心部署依赖 | `pyproject.toml [project].dependencies` | 已确认 | Medium | `test_h6_av_and_pyarrow_should_not_be_core_deployment_dependencies` expected failure | 轻量部署、安装面、非录制部署依赖 | H6 | 否 | 否 |

## 确认问题列表

已确认问题：1、2、3、4、5、6、7、8、9、10、11、13、14、15。

H1 已修复 1、2、3、13 的 LeRobot/runtime 数据语义问题，对应测试已解除 `expectedFailure` 并作为回归测试通过。H2/H3/H4/H6 项仍保留 `expectedFailure` 锚点。

## 否定问题列表

已否定问题：12。

证据：`ReplayPlanner._layout_metadata()` 会从 `ActionSpec` 的 joint semantics 和 `joint_dof_per_arm` 计算 arm count。`_source_contract()` 只有在计算结果 `<= 0` 时才兜底为 1，并同时加入 `arm_count` error。H0 测试 `test_h0_dual_arm_replay_plan_preserves_declared_arm_count` 用有效双臂 spec 生成 replay plan，确认 `result.plan.source.arm_count == 2`。

## H1-H6 修复映射

- H1: LeRobot 数据语义和 runtime event 对齐。修复 #1、#2、#3、#13，并把 H0 的语义审计 fixture 提升为常规数据审计入口。
- H2: 机器人 safety contract 和 gripper 语义。修复 #4、#5 的可验证能力表达，修复 #9 的夹爪/关节维度限制分离。
- H3: replay/pose plan 不可变性、payload hash 重新计算、ack 语义。修复 #6、#7、#8。
- H4: Recorder 生命周期和内存/耐久性。修复 #10、#11，要求 matched observation 可释放，telemetry 可分段/增量落盘。
- H5: 真机验证阶段。只在 H1-H4 修复后，按硬件清单验证 Piper/RM2 move-to-state 和 robot replay；H0 不执行真机。
- H6: 文档和依赖拆分。修复 #14、#15，README 链接指向真实文档，`av`/`pyarrow` 从核心部署依赖移到数据/录制相关 extra 或等效方案。

## Go / No-Go

- 当前数据能否用于训练：H1-aligned 新数据为 Go with validation，前提是 `control_step_aligned=true`、dataset validator 通过，并且语义审计没有 unexplained risk signal。legacy 或 unknown semantics 数据仍 No-Go by default。
- 当前 move-to-state 能否上真机：No-Go。Piper/RM2 当前 `validate_pose_target()` 明确返回 workspace、joint-limit、health validation unavailable；`require_valid()` 正确阻止运动。
- 当前 robot replay 能否上真机：No-Go。replay move-to-start 会被 pose validation 阻止；即使绕开，plan mutability/hash、ack 语义和 gripper limit 语义仍未 harden。

## 真机验证清单

H0 不需要真机，未执行任何可能移动、使能、复位或发送控制命令的命令。后续 H5 之前至少需要：

- Piper/RM2 SDK 能返回可验证的 joint limits、workspace 或等价 safety envelope、health/status 语义。
- move-to-state 的 stop procedure、expected motion、rollback procedure 已文档化并由用户明确授权。
- replay plan payload 在执行前重新 hash，任意 payload mutation 可被拒绝。
- replay ack 与 sent/observed state 语义分离，并可在日志中审计。
- gripper 维度不再被未声明的 arm joint velocity/acceleration 规则隐式处理。

## 已知限制

- H1 已解除 LeRobot 语义相关 expected failure；H2/H3/H4/H6 的 bug reproduction 仍是 `expectedFailure`。
- LeRobot 语义审计信号是风险检测，不把所有重复 frame/state 直接认定为错误；静止机器人、相机丢帧或合法 hold action 都可能产生重复。
- H0 没有扫描历史数据集；Go/No-Go 针对当前代码生成的数据语义和真机执行路径。
- Ruff 当前仍受现有工作区问题影响，H0 不主动清理非本阶段 lint。

## 建议 commit 拆分

- `test(hardening): reproduce data and replay safety issues`
- `docs(hardening): document confirmed audit findings`

## 建议阶段 tag

- `hardening-h0-audit`
