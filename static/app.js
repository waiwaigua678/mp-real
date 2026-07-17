const form = document.querySelector("#configForm");
const message = document.querySelector("#message");
const saveBtn = document.querySelector("#saveBtn");
const pingBtn = document.querySelector("#pingBtn");
const connectBtn = document.querySelector("#connectBtn");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const resetBtn = document.querySelector("#resetBtn");
const disconnectBtn = document.querySelector("#disconnectBtn");
const robotSelect = document.querySelector("#robotSelect");
const accessKeyInput = document.querySelector("#accessKey");
const evaluationForm = document.querySelector("#evaluationForm");
const createEvaluationBtn = document.querySelector("#createEvaluationBtn");
const abortEvaluationBtn = document.querySelector("#abortEvaluationBtn");
const completeEvaluationBtn = document.querySelector("#completeEvaluationBtn");
const warmupEvaluationBtn = document.querySelector("#warmupEvaluationBtn");
const resetReadyBtn = document.querySelector("#resetReadyBtn");
const startEpisodeBtn = document.querySelector("#startEpisodeBtn");
const stopEpisodeBtn = document.querySelector("#stopEpisodeBtn");
const labelSuccessBtn = document.querySelector("#labelSuccessBtn");
const labelFailureBtn = document.querySelector("#labelFailureBtn");
const labelInvalidBtn = document.querySelector("#labelInvalidBtn");
const failureReason = document.querySelector("#failureReason");
const operatorNote = document.querySelector("#operatorNote");
const baselineForm = document.querySelector("#baselineForm");
const baselineSelect = document.querySelector("#baselineSelect");

let currentStatus = null;
let firstLoad = true;
let dirty = false;
let accessKey = sessionStorage.getItem("motrixAccessKey") || "";
let cameraRoles = [];
let evaluation = null;
let evaluationRequestInFlight = false;
let poseStatus = null;
let replayStatus = null;
let baselines = [];
accessKeyInput.value = accessKey;

function setMessage(text, kind = "") {
  message.textContent = text || "";
  message.className = `message ${kind}`;
}

async function requestJson(url, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (accessKey) headers["X-Motrix-Key"] = accessKey;
  const response = await fetch(url, {
    headers,
    cache: "no-store",
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    const error = new Error(data.error || `${response.status} ${response.statusText}`);
    error.payload = data;
    throw error;
  }
  return data;
}

function numberOrEmpty(value) {
  return value === null || value === undefined ? "" : value;
}

function setForm(config) {
  if (config.robot) robotSelect.value = config.robot;
  ensureCameraPanels(config.camera_roles || []);
  for (const [key, value] of Object.entries(config)) {
    const field = form.elements[key];
    if (!field) continue;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else if (Array.isArray(value)) {
      field.value = value.join(" ");
    } else {
      field.value = numberOrEmpty(value);
    }
  }
  setRuntimeModeUi(config.runtime_mode || "deployment");
  evaluationForm.elements.robot_name.value = config.robot || robotSelect.value;
  if (!evaluation) evaluationForm.elements.prompt.value = config.prompt || "";
  dirty = false;
}

function collectEvaluationForm() {
  const data = { robot_name: robotSelect.value };
  for (const field of evaluationForm.elements) {
    if (!field.name || field.name === "robot_name") continue;
    if (field.type === "checkbox") {
      data[field.name] = field.checked;
    } else if (field.type === "number") {
      data[field.name] = Number(field.value);
    } else {
      data[field.name] = field.value.trim();
    }
  }
  data.save_video = data.save_data && data.save_video;
  return data;
}

function evaluationOperationAllowed(operation) {
  return !evaluationRequestInFlight && Boolean(evaluation?.legal_operations?.includes(operation));
}

function setEvaluationDisabled(disabled) {
  for (const field of evaluationForm.elements) {
    if (field.name !== "robot_name") field.disabled = disabled;
  }
}

function applyEvaluation(next) {
  evaluation = next;
  const state = next?.state || "无活动评测";
  const summary = next?.summary || {};
  const recording = next?.recording || {};
  const currentEpisode = next?.current_episode;
  const resultCounts = summary.result_counts || {};
  const startup = next?.policy_startup || {};

  document.querySelector("#evaluationState").textContent = state;
  document.querySelector("#evaluationPolicy").textContent = startup.warmed_up ? "READY" : startup.phase || "-";
  document.querySelector("#evaluationRound").textContent = next
    ? `${currentEpisode?.episode_index || summary.completed_episodes || 0}/${summary.planned_episodes || "-"}`
    : "-";
  document.querySelector("#evaluationElapsed").textContent = fmt(next?.current_episode_elapsed_s?.toFixed?.(1), " s");
  document.querySelector("#evaluationCompleted").textContent = summary.completed_episodes || 0;
  document.querySelector("#evaluationValid").textContent = summary.success_rate_denominator || 0;
  document.querySelector("#evaluationResults").textContent = `${resultCounts.SUCCESS || 0} / ${resultCounts.FAILURE || 0} / ${resultCounts.INVALID || 0}`;
  document.querySelector("#evaluationRate").textContent = `${summary.successes || 0}/${summary.success_rate_denominator || 0}`;
  document.querySelector("#evaluationStops").textContent = `${summary.timeout_count || 0} / ${summary.safety_abort_count || 0}`;
  document.querySelector("#evaluationRecorderQueue").textContent = recording.enabled
    ? `${recording.queue_depth || 0}/${recording.queue_capacity || "-"}`
    : "off";
  document.querySelector("#evaluationDrops").textContent = `${recording.dropped_event_count || 0} / ${recording.dropped_frame_count || 0}`;
  document.querySelector("#evaluationDataset").textContent = recording.dataset_root || "-";
  document.querySelector("#evaluationStopTrigger").textContent = currentEpisode?.stop_trigger || next?.episodes?.at?.(-1)?.stop_trigger || "-";
  document.querySelector("#evaluationError").textContent = next?.last_error || recording.failure || "-";
  document.querySelector("#evaluationLegalActions").textContent = (next?.legal_operations || []).join(" · ") || "-";

  const terminal = ["COMPLETED", "ABORTED", "ERROR"].includes(next?.state);
  const canCreate =
    (!next || terminal) && currentStatus?.runtime_mode === "deployment" && currentStatus?.connected && !currentStatus?.running;
  createEvaluationBtn.disabled = evaluationRequestInFlight || !canCreate;
  abortEvaluationBtn.disabled = !evaluationOperationAllowed("abort");
  completeEvaluationBtn.disabled = evaluationRequestInFlight || next?.state !== "COMPLETED";
  warmupEvaluationBtn.disabled = !evaluationOperationAllowed("warmup");
  resetReadyBtn.disabled = !evaluationOperationAllowed("reset-ready");
  startEpisodeBtn.disabled = !evaluationOperationAllowed("start-episode");
  stopEpisodeBtn.disabled = !evaluationOperationAllowed("stop-episode");
  labelSuccessBtn.disabled = !evaluationOperationAllowed("label");
  labelFailureBtn.disabled = !evaluationOperationAllowed("label");
  labelInvalidBtn.disabled = !evaluationOperationAllowed("label");
  failureReason.disabled = !evaluationOperationAllowed("label");
  operatorNote.disabled = !evaluationOperationAllowed("label");
  setEvaluationDisabled(Boolean(next) && !terminal);
}

async function evaluationRequest(url, payload = {}) {
  if (evaluationRequestInFlight) return;
  evaluationRequestInFlight = true;
  applyEvaluation(evaluation);
  try {
    const data = await requestJson(url, { method: "POST", body: JSON.stringify(payload) });
    applyEvaluation(data.evaluation);
    setMessage("评测状态已更新", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
  } finally {
    evaluationRequestInFlight = false;
    applyEvaluation(evaluation);
  }
}

function createEvaluation() {
  if (dirty) {
    setMessage("请先保存参数，再创建评测", "bad");
    return;
  }
  evaluationRequest("/api/evaluations", collectEvaluationForm());
}

function labelEvaluation(result) {
  const payload = { result, notes: operatorNote.value.trim() };
  if (result === "FAILURE") {
    if (!failureReason.value) {
      setMessage("标记失败时必须选择失败原因", "bad");
      return;
    }
    payload.failure_reason = failureReason.value;
  }
  evaluationRequest("/api/evaluations/current/label", payload);
}

function selectedBaselineId() {
  return baselineSelect.value || "";
}

function collectBaselineForm() {
  const value = {};
  for (const field of baselineForm.elements) {
    if (!field.name) continue;
    if (field.type === "number") value[field.name] = Number(field.value);
    else value[field.name] = field.value.trim();
  }
  return value;
}

function renderBaselines(payload) {
  baselines = payload.baselines || [];
  const previous = baselineSelect.value;
  for (const selector of [baselineSelect, document.querySelector("#baselineCompareA"), document.querySelector("#baselineCompareB")]) {
    selector.replaceChildren();
    for (const baseline of baselines) {
      const option = document.createElement("option");
      option.value = baseline.baseline_id;
      option.textContent = `${baseline.name} · ${baseline.policy_label} · ${baseline.baseline_id.slice(0, 8)}`;
      selector.appendChild(option);
    }
  }
  if (baselines.some((item) => item.baseline_id === previous)) baselineSelect.value = previous;
  if (baselines.length > 1) document.querySelector("#baselineCompareB").selectedIndex = 1;
  document.querySelector("#baselineCount").textContent = String(baselines.length);
  document.querySelector("#baselineWriterState").textContent = JSON.stringify(payload.writer || {});
  renderBaselineDetails();
}

function renderBaselineDetails() {
  const selected = baselines.find((item) => item.baseline_id === selectedBaselineId());
  document.querySelector("#baselineDetails").textContent = JSON.stringify(selected || { message: "尚未创建 Baseline" }, null, 2);
}

async function loadBaselines() {
  const data = await requestJson("/api/baselines");
  renderBaselines(data);
}

async function waitBaselineJob(jobId) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const data = await requestJson(`/api/baseline-jobs/${encodeURIComponent(jobId)}`);
    if (data.job.finished) {
      if (data.job.state !== "complete") throw new Error(data.job.error || "Baseline 后台任务失败");
      return data.job.result;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("等待 Baseline 后台任务超时");
}

async function createBaseline() {
  try {
    const data = await requestJson("/api/baselines", { method: "POST", body: JSON.stringify(collectBaselineForm()) });
    await waitBaselineJob(data.job.job_id);
    await loadBaselines();
    setMessage("Baseline 已保存", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function createBaselineFromEvaluation() {
  try {
    const data = await requestJson("/api/baselines/from-evaluation", {
      method: "POST",
      body: JSON.stringify({ name: baselineForm.elements.name.value.trim() || undefined }),
    });
    await waitBaselineJob(data.job.job_id);
    await loadBaselines();
    setMessage("已从当前评测创建 Baseline", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function cloneBaseline() {
  const baselineId = selectedBaselineId();
  const reason = document.querySelector("#baselineCloneReason").value.trim();
  const name = document.querySelector("#baselineCloneName").value.trim();
  if (!baselineId || !reason || !name) {
    setMessage("克隆需要选择 Baseline、名称和派生原因", "bad");
    return;
  }
  try {
    const data = await requestJson(`/api/baselines/${encodeURIComponent(baselineId)}/clone`, {
      method: "POST",
      body: JSON.stringify({ patch: { name }, derived_reason: reason }),
    });
    const created = await waitBaselineJob(data.job.job_id);
    await loadBaselines();
    baselineSelect.value = created.baseline_id;
    renderBaselineDetails();
    setMessage("Derived Baseline 已保存", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function runBaseline() {
  const baselineId = selectedBaselineId();
  if (!baselineId) return;
  try {
    const data = await requestJson(`/api/baselines/${encodeURIComponent(baselineId)}/run`, {
      method: "POST",
      body: "{}",
    });
    applyEvaluation(data.evaluation);
    showPage("evaluationPage");
    setMessage("已创建评测；仍需人工预热、复位和开始本轮", "ok");
  } catch (error) {
    const detail = error.payload?.diff || error.message;
    document.querySelector("#baselineComparison").textContent =
      typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
    setMessage(error.message, "bad");
  }
}

async function attachOpenLoop() {
  const baselineId = selectedBaselineId();
  const resultDir = document.querySelector("#baselineOpenLoopDir").value.trim();
  if (!baselineId || !resultDir) return;
  try {
    const data = await requestJson(`/api/baselines/${encodeURIComponent(baselineId)}/attach-open-loop`, {
      method: "POST",
      body: JSON.stringify({ result_dir: resultDir }),
    });
    await waitBaselineJob(data.job.job_id);
    await loadBaselines();
    setMessage("已关联开环结果", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function diffBaselines() {
  const a = document.querySelector("#baselineCompareA").value;
  const b = document.querySelector("#baselineCompareB").value;
  if (!a || !b) return;
  try {
    const data = await requestJson(`/api/baselines/${encodeURIComponent(a)}/diff`, {
      method: "POST",
      body: JSON.stringify({ other_baseline_id: b }),
    });
    document.querySelector("#baselineComparison").textContent = JSON.stringify(data.diff, null, 2);
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function compareBaselines() {
  const a = document.querySelector("#baselineCompareA").value;
  const b = document.querySelector("#baselineCompareB").value;
  if (!a || !b) return;
  try {
    const data = await requestJson("/api/baselines/compare", {
      method: "POST",
      body: JSON.stringify({ baseline_ids: [a, b] }),
    });
    document.querySelector("#baselineComparison").textContent = JSON.stringify(data.comparison, null, 2);
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

function ensureCameraPanels(roles) {
  const normalized = Array.from(roles || []);
  if (normalized.length === cameraRoles.length && normalized.every((role, index) => role === cameraRoles[index])) return;
  cameraRoles = normalized;
  const grid = document.querySelector("#cameraGrid");
  const template = document.querySelector("#cameraPanelTemplate");
  grid.replaceChildren();
  for (const role of cameraRoles) {
    const panel = template.content.firstElementChild.cloneNode(true);
    panel.dataset.cameraRole = role;
    panel.querySelector("h2").textContent = role;
    const meta = panel.querySelector(".panel-title span");
    meta.id = `meta-${role}`;
    const image = panel.querySelector("img");
    image.id = `stream-${role}`;
    image.alt = role;
    image.src = `/stream/${encodeURIComponent(role)}.mjpg?t=${Date.now()}`;
    grid.appendChild(panel);
  }
}

function runtimeMode(statusOrMode) {
  if (typeof statusOrMode === "string") return statusOrMode;
  return statusOrMode?.runtime_mode || form.elements.runtime_mode?.value || "deployment";
}

function setRuntimeModeUi(statusOrMode) {
  const rm2 = statusOrMode?.robot === "rm2" || robotSelect.value === "rm2";
  const mode = runtimeMode(statusOrMode);
  form.elements.runtime_mode.value = mode;
  form.elements.runtime_mode.disabled = false;
  const deployment = mode === "deployment";
  const preview = mode === "camera_preview";
  const offline = mode === "offline_replay";
  for (const node of document.querySelectorAll("[data-deployment-only='true']")) {
    node.hidden = !deployment;
  }
  for (const node of document.querySelectorAll("[data-piper-only='true']")) {
    node.hidden = rm2;
  }
  for (const node of document.querySelectorAll("[data-rm2-only='true']")) {
    node.hidden = !rm2;
  }
  document.querySelector("#offlineReplayNotice").hidden = !offline;
  connectBtn.textContent = preview ? "连接相机" : offline ? "进入回放" : "连接预览";
  startBtn.textContent = preview ? "开始预览" : offline ? "回放（阶段 7）" : "开始";
  stopBtn.textContent = preview ? "停止预览" : "停止";
}

async function selectRobot() {
  try {
    const data = await requestJson("/api/robot", { method: "POST", body: JSON.stringify({ robot: robotSelect.value }) });
    setForm(data.config);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message, "bad");
    await loadConfig();
  }
}

function collectForm() {
  const data = {};
  for (const field of form.elements) {
    if (!field.name) continue;
    if (field.type === "checkbox") {
      data[field.name] = field.checked;
    } else if (field.type === "number") {
      data[field.name] = field.value === "" ? null : Number(field.value);
    } else {
      data[field.name] = field.value;
    }
  }
  return data;
}

async function loadConfig() {
  const config = await requestJson("/api/config");
  setForm(config);
  document.querySelector("#serverLine").textContent = config.server_url;
  document.querySelector("#promptLine").textContent = config.prompt;
}

async function saveConfig() {
  const data = await requestJson("/api/config", {
    method: "POST",
    body: JSON.stringify(collectForm()),
  });
  setForm(data.config);
  document.querySelector("#serverLine").textContent = data.config.server_url;
  document.querySelector("#promptLine").textContent = data.config.prompt;
  setMessage("参数已保存", "ok");
}

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value}${suffix}`;
}

function setBadge(id, text, kind) {
  const node = document.querySelector(id);
  node.textContent = text;
  node.className = `badge ${kind || ""}`;
}

function setFieldLocks(status) {
  for (const field of form.elements) {
    if (!field.name) continue;
    const connectionField = field.dataset.connectionField === "true";
    field.disabled = !status.can_edit_config || (connectionField && !status.can_edit_connection_config);
  }
  saveBtn.disabled = !status.can_edit_config;
  const lockText = status.can_edit_config
    ? status.can_edit_connection_config
      ? "参数可编辑"
      : "运行参数可编辑，连接参数已锁定"
    : "参数已锁定";
  document.querySelector("#editState").textContent = dirty && status.can_edit_config ? `${lockText} · 未保存` : lockText;
  document.querySelector("#settingsLockLine").textContent = dirty && status.can_edit_config ? `${lockText} · 未保存` : lockText;
}

function applyStatus(status) {
  currentStatus = status;
  ensureCameraPanels(status.camera_roles || []);
  const metrics = status.metrics || {};

  setBadge("#phaseBadge", status.phase, status.phase === "running" ? "ok" : status.phase === "error" ? "bad" : "");
  const policyText = status.policy_state ? `policy ${status.policy_state.toLowerCase()}` : status.policy_connected ? "policy on" : "policy off";
  setBadge("#policyBadge", policyText, status.policy_connected ? "ok" : "warn");
  setBadge("#runBadge", status.running ? "running" : "stopped", status.running ? "ok" : "");

  document.querySelector("#serverLine").textContent = status.server_url || "";
  document.querySelector("#promptLine").textContent = status.prompt || "";
  document.querySelector("#stepLine").textContent = `${status.step}${status.max_steps ? ` / ${status.max_steps}` : ""}`;
  document.querySelector("#inferLatency").textContent = fmt(metrics.infer_latency_ms, " ms");
  document.querySelector("#inferHz").textContent = fmt(metrics.infer_hz, " Hz");
  document.querySelector("#loopMs").textContent = fmt(metrics.loop_ms, " ms");
  document.querySelector("#controlHz").textContent = fmt(metrics.control_hz, " Hz");
  document.querySelector("#queueLen").textContent = fmt(metrics.action_queue_len);
  document.querySelector("#uptime").textContent = fmt(metrics.uptime_s, " s");
  document.querySelector("#errorLine").textContent = status.last_error || "";
  document.querySelector("#metadata").textContent = JSON.stringify(status.server_metadata || {}, null, 2);
  document.querySelector("#policyTiming").textContent = JSON.stringify(
    {
      connect_latency_ms: metrics.connect_latency_ms,
      metadata_latency_ms: metrics.metadata_latency_ms,
      cold_inference_latency_ms: metrics.cold_inference_latency_ms,
      warmup_latency_ms: metrics.warmup_latency_ms,
      first_live_inference_latency_ms: metrics.first_live_inference_latency_ms,
      steady_inference_latency_ms: metrics.steady_inference_latency_ms,
    },
    null,
    2,
  );
  document.querySelector("#logBox").textContent = (status.logs || []).join("\n");

  for (const camera of cameraRoles) {
    const frame = (status.frames || {})[camera] || {};
    const meta = document.querySelector(`#meta-${camera}`);
    if (meta) {
      meta.textContent =
        `sequence ${frame.sequence || 0} · age ${fmt(frame.age_ms, " ms")}${frame.error ? ` · ${frame.error}` : ""}`;
    }
  }

  // Status polling must not overwrite a disconnected user's unsaved mode.
  // The backend still reports its persisted mode until Save succeeds.
  setRuntimeModeUi(dirty ? form.elements.runtime_mode.value : status);
  setFieldLocks(status);
  connectBtn.disabled = !status.can_connect;
  pingBtn.disabled = status.running || runtimeMode(status) !== "deployment";
  startBtn.disabled = !status.can_start;
  stopBtn.disabled = !status.can_stop;
  resetBtn.disabled = !status.can_reset;
  disconnectBtn.disabled = !status.can_disconnect;
  applyEvaluation(status.evaluation || null);
  applyPose(status.pose || null);
  applyReplay(status.replay || null);
}

function applyPose(pose) {
  poseStatus = pose;
  const phase = pose?.phase || "idle";
  document.querySelector("#posePhase").textContent = phase;
  document.querySelector("#poseDetails").textContent = JSON.stringify(
    {
      target: pose?.target || null,
      validation: pose?.offline_validation || null,
      plan: pose?.plan || null,
      progress: pose?.progress || null,
      error: pose?.error || null,
    },
    null,
    2,
  );
  document.querySelector("#poseSelectBtn").disabled = !currentStatus?.can_edit_connection_config;
  document.querySelector("#poseConnectBtn").disabled = phase !== "offline_preflighted";
  document.querySelector("#poseExecuteBtn").disabled = phase !== "awaiting_move_confirmation";
  document.querySelector("#poseStopBtn").disabled = phase !== "moving";
  document.querySelector("#posePrepareDeployBtn").disabled = !["reached", "reached_with_warning"].includes(phase);
  document.querySelector("#poseStartDeployBtn").disabled = phase !== "awaiting_deployment_confirmation";
}

async function poseRequest(url, payload = {}) {
  try {
    const data = await requestJson(url, { method: "POST", body: JSON.stringify(payload) });
    applyPose(data.pose);
    await refreshStatus();
    return data.pose;
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
    return null;
  }
}

function poseSelectionPayload() {
  return {
    dataset_id: document.querySelector("#poseDatasetId").value.trim(),
    episode_index: Number(document.querySelector("#poseEpisodeIndex").value),
    sample_index: Number(document.querySelector("#poseSampleIndex").value),
  };
}

async function selectPose() {
  await poseRequest("/api/pose/select", poseSelectionPayload());
}

async function executePose() {
  const planHash = poseStatus?.plan?.plan_hash;
  if (!planHash || !window.confirm(`确认以低速执行计划 ${planHash.slice(0, 12)}…？`)) return;
  await poseRequest("/api/pose/execute", { plan_hash: planHash });
}

async function preparePoseDeployment() {
  const planHash = poseStatus?.plan?.plan_hash;
  if (!planHash || !window.confirm("确认姿态已到达；将连接真实相机和 Policy，但不会 reset 或执行 warmup action。")) return;
  await poseRequest("/api/pose/prepare-deployment", { plan_hash: planHash });
}

async function startPoseDeployment() {
  const planHash = poseStatus?.plan?.plan_hash;
  if (!planHash || !window.confirm("确认使用当前真实相机帧和当前机器人状态开始实时推理？")) return;
  await poseRequest("/api/pose/start-deployment", { plan_hash: planHash });
}

function applyReplay(replay) {
  replayStatus = replay;
  const state = replay?.state || "idle";
  const locked = Boolean(replay?.view_cursor_locked);
  document.querySelector("#replayState").textContent = state;
  document.querySelector("#replayDetails").textContent = JSON.stringify(
    {
      safety_report: replay?.safety_report || null,
      plan: replay?.plan || null,
      robot_cursor: replay?.cursor || null,
      view_cursor_locked: locked,
      error: replay?.error || null,
    },
    null,
    2,
  );
  const planning = state === "planning";
  const editable = !locked && !planning && Boolean(currentStatus?.can_edit_connection_config);
  for (const id of [
    "#replayDatasetId",
    "#replayEpisodeIndex",
    "#replayStartSample",
    "#replayEndSample",
    "#replayMode",
    "#replayTiming",
    "#replayFps",
    "#replaySpeedScale",
  ]) {
    document.querySelector(id).disabled = !editable;
  }
  document.querySelector("#replayPlanBtn").disabled = !editable;
  document.querySelector("#replayConnectBtn").disabled = state !== "validated";
  document.querySelector("#replayStartBtn").disabled = state !== "armed";
  document.querySelector("#replayPauseBtn").disabled = state !== "running";
  document.querySelector("#replayResumeBtn").disabled = state !== "paused";
  document.querySelector("#replayStopBtn").disabled = !["connecting", "moving_to_start", "armed", "running", "paused", "stopping"].includes(state);
  document.querySelector("#replayEmergencyBtn").disabled = document.querySelector("#replayStopBtn").disabled;
}

function replayPlanPayload() {
  const optionalNumber = (id) => {
    const value = document.querySelector(id).value;
    return value === "" ? null : Number(value);
  };
  return {
    dataset_id: document.querySelector("#replayDatasetId").value.trim(),
    episode_index: Number(document.querySelector("#replayEpisodeIndex").value),
    start_sample: optionalNumber("#replayStartSample"),
    end_sample: optionalNumber("#replayEndSample"),
    mode: document.querySelector("#replayMode").value,
    timing_mode: document.querySelector("#replayTiming").value,
    fps: optionalNumber("#replayFps"),
    speed_scale: Number(document.querySelector("#replaySpeedScale").value),
  };
}

async function replayRequest(url, payload = {}) {
  try {
    const data = await requestJson(url, { method: "POST", body: JSON.stringify(payload) });
    applyReplay(data.replay);
    await refreshStatus();
    return data.replay;
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
    return null;
  }
}

async function createReplayPlan() {
  await replayRequest("/api/replay/plan", replayPlanPayload());
}

async function startReplay() {
  const planHash = replayStatus?.plan?.plan_hash;
  if (!planHash || !window.confirm(`确认以当前计划 ${planHash.slice(0, 12)}… 开始真机轨迹回放？`)) return;
  await replayRequest("/api/replay/start", { plan_hash: planHash });
}

async function refreshStatus() {
  try {
    const status = await requestJson("/api/status");
    applyStatus(status);
    if (firstLoad) {
      setMessage("");
      firstLoad = false;
    }
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function connectRuntime() {
  try {
    setMessage("正在连接...", "");
    const data = await requestJson("/api/connect", { method: "POST", body: "{}" });
    applyStatus(data.status);
    setMessage("已连接", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
  }
}

async function startRuntime() {
  try {
    if (dirty) {
      throw new Error("参数页有未保存修改");
    }
    setMessage(runtimeMode(currentStatus) === "deployment" ? "正在预热策略..." : "正在启动...", "");
    const data = await requestJson("/api/start", { method: "POST", body: "{}" });
    applyStatus(data.status);
    setMessage(runtimeMode(data.status) === "deployment" ? "策略启动中" : "已开始", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
  }
}

async function stopRuntime() {
  try {
    const data = await requestJson("/api/stop", { method: "POST", body: "{}" });
    applyStatus(data.status);
    setMessage("已请求停止", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function disconnectRuntime() {
  try {
    setMessage("正在断开...", "");
    const data = await requestJson("/api/disconnect", { method: "POST", body: "{}" });
    applyStatus(data.status);
    await loadConfig();
    setMessage("已断开", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
    await refreshStatus();
  }
}

async function resetRuntime() {
  try {
    const data = await requestJson("/api/reset", { method: "POST", body: "{}" });
    applyStatus(data.status);
    setMessage("已归位", "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

async function pingPolicy() {
  try {
    if (dirty) {
      await saveConfig();
    }
    setMessage("正在检查服务...", "");
    const data = await requestJson("/api/ping_policy", { method: "POST", body: "{}" });
    if (data.connected) {
      setMessage(`服务在线，延迟 ${fmt(data.latency_ms, " ms")}`, "ok");
      document.querySelector("#metadata").textContent = JSON.stringify(data.metadata || {}, null, 2);
    } else {
      setMessage(data.error || "服务未连接", "bad");
    }
  } catch (error) {
    setMessage(error.message, "bad");
  }
}

function initStreams() {
  for (const camera of cameraRoles) {
    const image = document.querySelector(`#stream-${camera}`);
    if (image) image.src = `/stream/${encodeURIComponent(camera)}.mjpg?t=${Date.now()}`;
  }
}

function showPage(pageId) {
  for (const page of document.querySelectorAll(".page")) {
    page.classList.toggle("active", page.id === pageId);
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.page === pageId);
  }
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showPage(tab.dataset.page));
}

form.addEventListener("input", () => {
  dirty = true;
  if (currentStatus) setFieldLocks(currentStatus);
});

accessKeyInput.addEventListener("input", () => {
  accessKey = accessKeyInput.value;
  if (accessKey) sessionStorage.setItem("motrixAccessKey", accessKey);
  else sessionStorage.removeItem("motrixAccessKey");
});
robotSelect.addEventListener("change", selectRobot);
form.elements.runtime_mode?.addEventListener("change", () => {
  dirty = true;
  setRuntimeModeUi(form.elements.runtime_mode.value);
  if (currentStatus) setFieldLocks(currentStatus);
});

saveBtn.addEventListener("click", () => saveConfig().catch((error) => setMessage(error.message, "bad")));
pingBtn.addEventListener("click", pingPolicy);
connectBtn.addEventListener("click", connectRuntime);
startBtn.addEventListener("click", startRuntime);
stopBtn.addEventListener("click", stopRuntime);
resetBtn.addEventListener("click", resetRuntime);
disconnectBtn.addEventListener("click", disconnectRuntime);
document.querySelector("#poseSelectBtn").addEventListener("click", selectPose);
document.querySelector("#poseConnectBtn").addEventListener("click", () => poseRequest("/api/pose/connect"));
document.querySelector("#poseExecuteBtn").addEventListener("click", executePose);
document.querySelector("#poseStopBtn").addEventListener("click", () => poseRequest("/api/pose/stop"));
document.querySelector("#posePrepareDeployBtn").addEventListener("click", preparePoseDeployment);
document.querySelector("#poseStartDeployBtn").addEventListener("click", startPoseDeployment);
document.querySelector("#replayPlanBtn").addEventListener("click", createReplayPlan);
document.querySelector("#replayConnectBtn").addEventListener("click", () => replayRequest("/api/replay/connect"));
document.querySelector("#replayStartBtn").addEventListener("click", startReplay);
document.querySelector("#replayPauseBtn").addEventListener("click", () => replayRequest("/api/replay/pause"));
document.querySelector("#replayResumeBtn").addEventListener("click", () => replayRequest("/api/replay/resume"));
document.querySelector("#replayStopBtn").addEventListener("click", () => replayRequest("/api/replay/stop"));
document.querySelector("#replayEmergencyBtn").addEventListener("click", () => replayRequest("/api/replay/emergency-stop"));
createEvaluationBtn.addEventListener("click", createEvaluation);
abortEvaluationBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/abort"));
completeEvaluationBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/complete"));
warmupEvaluationBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/warmup"));
resetReadyBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/reset-ready"));
startEpisodeBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/start-episode"));
stopEpisodeBtn.addEventListener("click", () => evaluationRequest("/api/evaluations/current/stop-episode"));
labelSuccessBtn.addEventListener("click", () => labelEvaluation("SUCCESS"));
labelFailureBtn.addEventListener("click", () => labelEvaluation("FAILURE"));
labelInvalidBtn.addEventListener("click", () => labelEvaluation("INVALID"));
document.querySelector("#refreshBaselinesBtn").addEventListener("click", () => loadBaselines().catch((error) => setMessage(error.message, "bad")));
document.querySelector("#createBaselineBtn").addEventListener("click", createBaseline);
document.querySelector("#createBaselineFromEvaluationBtn").addEventListener("click", createBaselineFromEvaluation);
baselineSelect.addEventListener("change", renderBaselineDetails);
document.querySelector("#cloneBaselineBtn").addEventListener("click", cloneBaseline);
document.querySelector("#runBaselineBtn").addEventListener("click", runBaseline);
document.querySelector("#attachOpenLoopBtn").addEventListener("click", attachOpenLoop);
document.querySelector("#diffBaselineBtn").addEventListener("click", diffBaselines);
document.querySelector("#compareBaselineBtn").addEventListener("click", compareBaselines);

loadConfig()
  .then(() => refreshStatus())
  .then(() => initStreams())
  .then(() => loadBaselines())
  .catch((error) => setMessage(error.message, "bad"));

setInterval(refreshStatus, 500);
