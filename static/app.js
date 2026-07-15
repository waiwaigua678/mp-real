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

let currentStatus = null;
let firstLoad = true;
let dirty = false;
let accessKey = sessionStorage.getItem("motrixAccessKey") || "";
let cameraRoles = [];
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
    throw new Error(data.error || `${response.status} ${response.statusText}`);
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
  dirty = false;
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

loadConfig()
  .then(() => refreshStatus())
  .then(() => initStreams())
  .catch((error) => setMessage(error.message, "bad"));

setInterval(refreshStatus, 500);
