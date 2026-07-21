# Hardening H0-H6 Audit

本文件记录 H0 静态审查复核结果、H1 数据语义、H2 Recorder 内存 hardening、H3 plan integrity、H4 robot safety profile、H5 replay feedback 语义和 H6 metadata/CI/文档状态。
H0 只建立复现测试和审计证据，H1 修复 LeRobot observation/action/control-step 对齐，H2 修复 Recorder cache、图像和 telemetry 的无界内存增长，H3 修复 replay/pose plan 深不可变和执行前 canonical hash 复核，H4 修复真实机器人 safety profile/health 表达，H5 修复 replay sent/feedback/ack 语义与 joint/gripper 约束分离，H6 修复 metadata、optional dependency、README 文档和无硬件 CI。

## 当前调用链

部署路径复核以当前工作区为准：

- Web/CLI runtime 通过 `RuntimeController` 进入 `src/mp_real/runtime/inference.py` 的共享推理循环。
- 同步 runtime 在 `run_sync_loop()` 中调用 `fetch_action_chunk()`；该函数先 `adapter.observe()`，再 `PolicyClient.infer()`，再解码 action chunk。
- `run_sync_loop()` 和 `run_rtc_loop()` 将一个 policy chunk 展开为多个 robot-bound action。H1 后，每个实际发送的 action 都会先采集一条新的 control-step observation。
- `RuntimeEventHooks` 将 observation、chunk、action selected/stabilized/executed 转成 `RuntimeEvent`。
- H1 新增 `ControlStepRecorded` 作为标准 LeRobot row 的权威事件。`ActionExecuted` 兼容路径仍保留，但同一 control step 已有 `ControlStepRecorded` 时不会重复写 row。
- `LeRobotV21EpisodeRecorder` 在后台线程消费事件；H1-aligned 数据直接使用 `ControlStepRecorded.state/images/executed_action` 写入 LeRobot v2.1 frame，policy observation 和 raw chunk 只作为 telemetry。
- replay 路径为 `ReplayPlanner` 离线生成 deep-immutable `ReplayPlan`，Web 连接阶段绑定 resource lease/generation 后重新 hash，然后 `RobotReplayController.prepare()` move-to-start，`confirm_and_start(plan_hash)` 启动 replay。H5 后 replay worker 按计划发送命令、轮询机器人反馈，并分别推进 sent、feedback 和 acknowledged 游标。
- move-to-recorded-state 路径为 `MoveToStateValidator`、机器人 `validate_pose_target()`、`MoveToRecordedStatePlan.build()`、`PoseMoveController.start()`；H3 后计划内 ndarray 为只读，执行前重新计算 canonical payload hash。

## 审查问题

| # | 审查问题 | 当前代码位置 | 确认 | 严重级别 | 可复现测试 | 影响范围 | 建议修复阶段 | 阻止真机运动 | 阻止数据用于训练 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LeRobot recorder 是否给同一个 action chunk 的多个动作重复配同一 observation/state/images | `src/mp_real/runtime/inference.py::run_sync_loop`; `src/mp_real/data/lerobot_v21.py::LeRobotV21EpisodeRecorder._run` | H1 已修复 | Critical | `test_h1_policy_chunk_actions_should_not_share_one_recorded_observation`; `test_h0_lerobot_semantic_audit_fixture_reports_risk_signals` | policy runtime 录制、后续训练、open-loop 评估 | H1 | 否，主要是数据语义 | 修复前阻止，H1-aligned 新数据不阻止 |
| 2 | 控制循环是否每个实际动作前采集新机器人状态和相机快照 | `fetch_action_chunk()` 只在取 chunk 时 observe；`run_sync_loop()` 从 plan deque 连续执行 | H1 已修复 | Critical | `test_h1_control_loop_should_capture_before_each_executed_action`; `test_h1_rtc_records_control_step_observations_for_prefetched_chunk` | 同步/RTC 数据语义、状态-动作对齐 | H1 | 条件性，不应声称逐动作闭环观测 | 修复前阻止，H1-aligned 新数据不阻止 |
| 3 | ObservationCaptured、ActionSelected、ActionExecuted 是否能通过同一个 control step 唯一关联 | H1 通过 `ControlStepRecorded` 聚合 control-step observation、selected、stabilized、executed action | H1 已修复 | High | `test_h1_observation_and_actions_should_join_by_control_step`; `test_sync_and_rtc_keep_action_event_order` | 审计、debug、数据清洗、训练过滤 | H1 | 否 | 修复前阻止，H1-aligned 新数据不阻止 |
| 4 | Piper/RM2 `validate_pose_target()` 是否总是产生 critical issues | `src/mp_real/robots/piper/infer.py::PiperRobot.validate_pose_target`; `src/mp_real/robots/rm2/infer.py::Rm2Robot.validate_pose_target` | 已确认 | Critical | `test_h0_piper_and_rm2_pose_validation_currently_block_move_to_state` | move-to-state、replay move-to-start | H2/H5 | 是 | 否 |
| 5 | `PoseValidationReport.require_valid()` 是否让真实 Piper/RM2 永远无法 move-to-state | `src/mp_real/pose/models.py::PoseValidationReport.require_valid`; `src/mp_real/web/server.py::_run_pose_connect`; `src/mp_real/replay/controller.py::_move_to_start` | 已确认 | Critical | `test_h0_piper_and_rm2_pose_validation_currently_block_move_to_state` | Web/CLI move-to-state、robot replay | H2/H5 | 是 | 否 |
| 6 | Replay/Pose plan 中 NumPy 数组是否仍可原地修改 | H3 后为 `src/mp_real/common/plan_integrity.py::readonly_array`; `ReplayStep`; `RecordedPoseTarget`; `PoseWaypoint`; `MoveToRecordedStatePlan` | H3 已修复 | Critical | `test_h3_replay_plan_hash_should_reject_step_array_mutation`; `test_h3_pose_plan_hash_should_reject_waypoint_array_mutation`; `test_h3_replay_plan_arrays_are_readonly_and_inputs_are_copied`; `test_h3_pose_plan_arrays_are_readonly_and_inputs_are_copied` | replay/move-to-state 安全确认 | H3 | H3 后不因 plan mutability 阻止，仍受 H4/H5 阻止 | 否 |
| 7 | 执行前是否只比较旧 `plan_hash`，没有重新计算 payload hash | H3 后 `RobotReplayController.prepare/confirm_and_start/resume/_run_replay`; `PoseMoveController.start/_run`; Web replay/pose handoff 入口均调用 `require_integrity()` | H3 已修复 | Critical | `test_h3_replay_rehashes_before_arm_execute_resume_and_stale_identity`; `test_h3_pose_controller_rehashes_before_execute_and_rejects_expired_plan`; H0 两项 H3 regression tests | replay/move-to-state 审核完整性 | H3 | H3 后不因 hash 复核阻止，仍受 H4/H5 阻止 | 否 |
| 8 | ReplayController 是否发送命令后立即读状态，并将 sent 当 acknowledged | H5 后 `src/mp_real/replay/controller.py::_run_replay` 分离 command send、feedback poll 和 acknowledgement | H5 已修复 | High | `test_h5_replay_controller_should_not_acknowledge_sent_with_nonzero_tracking_error`; `test_h5_send_feedback_acknowledged_cursors_are_separate_and_recorded` | robot replay 状态语义、安全审计 | H5 | 软件语义已修复，真机仍需 H5 gates | 否 |
| 9 | ReplayPlanner 的 max_step、velocity、acceleration 是否同时作用于夹爪维度 | H5 后 `src/mp_real/replay/planning.py` 按 ActionSpec semantics 分离 joint/gripper kinematics | H5 已修复 | High | `test_h5_replay_planner_should_not_apply_arm_limits_to_gripper_only_changes`; `test_h5_gripper_constraints_are_independent_from_joint_limits` | replay 规划、夹爪语义、误拒绝或误设限 | H5 | 软件语义已修复，真机仍需 H5 gates | 条件性，影响 replay 不是训练本体 |
| 10 | Recorder 是否长期保留带图像的 observation cache | H2 后为 `LeRobotV21EpisodeRecorder._run` 内 bounded cache，匹配后 pop，episode end 清理 | H2 已修复 | High | `test_h2_recorder_should_release_matched_observation_images`; `test_h2_unmatched_observation_cache_is_bounded_and_measured` | 长 episode 录制内存、稳定性 | H2 | 否 | H2 后不阻止，仍需 validator/metrics 通过 |
| 11 | telemetry 是否在 episode 结束前全部堆积在内存 | H2 后为 `telemetry/chunk-*/episode_*/part_*.npz` + `index.json`，part 内 bounded padding | H2 已修复 | High | `test_h2_telemetry_should_be_durable_before_episode_end`; `test_h2_telemetry_parts_reader_and_sample_lookup`; `test_h2_ten_minute_fake_camera_soak_keeps_caches_bounded` | 长 episode、崩溃恢复、审计完整性 | H2 | 否 | H2 后不阻止，仍需 dropped counters 为 0 或可解释 |
| 12 | 双臂 ReplayPlan 的 `arm_count` 是否被覆盖为 1 | `src/mp_real/replay/planning.py::_source_contract`; H6 后 unknown layout 保持 invalid，不再兜底成 1 | H6 已修复保守表达 | Low | `test_h0_dual_arm_replay_plan_preserves_declared_arm_count`; H6 metadata tests | replay contract | H6 | 否 | 否 |
| 13 | Recorder metadata 的 `action_mode` 是否写死 | `src/mp_real/data/lerobot_v21.py::_build_info`; `schema.json`; `FakeRecordedEpisodeSource` | H6 已补全 round trip | High | `test_h1_recorder_metadata_should_preserve_action_spec_action_mode`; `test_h6_recording_metadata_round_trips_action_spec_and_action_source` | replay metadata、训练/评估筛选 | H1/H6 | 否 | 修复前阻止，H1/H6-aligned 新数据不阻止 |
| 14 | README 引用的安全和能力文档是否真实存在 | `README.md` 链接 `docs/*.md`; root hardening docs | H6 已修复 | Medium | `test_h6_readme_local_doc_links_should_exist`; `test_readme_local_doc_links_exist` | 文档可用性、操作员安全流程 | H6 | 条件性，阻止按 README 完成安全复核 | 否 |
| 15 | `av`、`pyarrow` 是否仍属于核心部署依赖 | `pyproject.toml [project].dependencies`; lazy imports in data/open-loop/Web entrypoints | H6 已修复 | Medium | `test_h6_av_and_pyarrow_should_not_be_core_deployment_dependencies`; `test_core_imports_do_not_require_av_or_pyarrow` | 轻量部署、安装面、非录制部署依赖 | H6 | 否 | 否 |

## 确认问题列表

已确认问题：1、2、3、4、5、6、7、8、9、10、11、13、14、15。

H1 已修复 1、2、3、13 的 LeRobot/runtime 数据语义问题，对应测试已解除 `expectedFailure` 并作为回归测试通过。H2 已修复 10、11 的 Recorder 内存/telemetry 问题。H3 已修复 6、7 的 plan immutability/hash integrity 问题。H4 已修复 4、5 的安全能力表达问题。H5 已修复 8、9 的 replay feedback acknowledgement 和 joint/gripper kinematic 语义问题。H6 已修复 README 文档断链、核心依赖拆分、metadata round trip、legacy/unknown schema warning 和无硬件 CI 锚点。

## 否定问题列表

已否定问题：12。

证据：`ReplayPlanner._layout_metadata()` 会从 `ActionSpec` 的 joint semantics 和 `joint_dof_per_arm` 计算 arm count。H0 测试 `test_h0_dual_arm_replay_plan_preserves_declared_arm_count` 用有效双臂 spec 生成 replay plan，确认 `result.plan.source.arm_count == 2`。H6 后，无法推导 arm count 的 legacy/unknown schema 会保留 invalid 状态并阻止 plan 生成，不再用 1 伪装。

## H1-H6 修复映射

- H1: LeRobot 数据语义和 runtime event 对齐。修复 #1、#2、#3、#13，并把 H0 的语义审计 fixture 提升为常规数据审计入口。
- H2: Recorder 生命周期和内存/耐久性。修复 #10、#11，要求 matched observation 可释放，telemetry 可分段/增量落盘，writer metrics 可观察。
- H3: replay/pose plan 不可变性、canonical payload hash、执行入口重新计算和 stale generation/lease 复核。修复 #6、#7。
- H4: 机器人 safety contract。修复 #4、#5 的 Piper/RM2 可验证能力表达。
- H5: replay 控制语义与真机验证阶段。修复 #8、#9，新增 sent/feedback/ack cursor、feedback acknowledgement 策略、stale/health abort 和 joint/gripper 分离约束。真机 replay 仍需按 `docs/hardware_validation_h5.md` 逐 gate 验证。
- H6: metadata、文档、依赖拆分和无硬件 CI。修复 #12 的 unknown layout 保守表达、#13 的 metadata round trip、#14 README 根文档、#15 core dependency 拆分；`av`/`pyarrow` 移到 `recording`/`data`/`evaluation` extra。

## Go / No-Go

- 当前数据能否用于训练：H1-aligned 新数据为 Go with validation，前提是 `control_step_aligned=true`、dataset validator 通过，并且语义审计没有 unexplained risk signal。legacy 或 unknown semantics 数据仍 No-Go by default。
- 当前 move-to-state 能否上真机：No-Go。H3 已解决 plan integrity，H4 已实现 profile/health 语义，但 Piper/RM2 真机 profile 和 health gates 仍未完成。
- 当前 robot replay 能否上真机：No-Go。H3/H4/H5 的软件语义已落地；真实 Piper/RM2 仍需完成 H4/H5 硬件 gate，尤其是 stop capability、feedback freshness、tracking threshold 和 gripper settle 验证。

## 真机验证清单

H0 不需要真机，未执行任何可能移动、使能、复位或发送控制命令的命令。后续 H5 之前至少需要：

- Piper/RM2 SDK 能返回可验证的 joint limits、workspace 或等价 safety envelope、health/status 语义。
- move-to-state 的 stop procedure、expected motion、rollback procedure 已文档化并由用户明确授权。
- replay/pose plan payload 在执行前重新 hash，任意 payload mutation 可被拒绝。H3 详见 `docs/plan_integrity.md`。
- replay ack 与 sent/observed state 语义分离，并可在日志中审计。H5 软件测试已覆盖，真机仍需 gate 验证。
- gripper 维度不再被未声明的 arm joint velocity/acceleration 规则隐式处理。H5 软件测试已覆盖，真机仍需 gate 验证。

## 已知限制

- H1 已解除 LeRobot 语义相关 expected failure；H2 已解除 Recorder cache/telemetry 相关 expected failure；H3 已解除 plan mutability/hash 相关 expected failure；H5 已解除 replay feedback/gripper 相关 expected failure；H6 已解除文档/依赖 bug reproduction 的 `expectedFailure`。
- LeRobot 语义审计信号是风险检测，不把所有重复 frame/state 直接认定为错误；静止机器人、相机丢帧或合法 hold action 都可能产生重复。
- H0 没有扫描历史数据集；Go/No-Go 针对当前代码生成的数据语义和真机执行路径。
- Ruff 基线由 H6 纳入无硬件 CI 清理范围。

## 建议 commit 拆分

- `fix(safety): make pose and replay plans deeply immutable`
- `fix(safety): revalidate canonical plan hash before execution`
- `test(safety): reject mutated and stale motion plans`
- `docs(safety): document plan integrity hardening`
- `fix(metadata): derive replay and recording metadata from ActionSpec`
- `refactor(deps): make recording dependencies optional`
- `docs: document hardening safety and dataset semantics`
- `ci: add core and recording validation jobs`

## 建议阶段 tag

- `hardening-h6-metadata-ci-docs`
