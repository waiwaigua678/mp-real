const datasetSelect = document.querySelector("#datasetSelect");
const episodeSelect = document.querySelector("#episodeSelect");
const progress = document.querySelector("#progress");
const playbackRate = document.querySelector("#playbackRate");
const playPause = document.querySelector("#playPause");
const message = document.querySelector("#message");
const datasetImportMenu = document.querySelector("#datasetImportMenu");
const datasetPath = document.querySelector("#datasetPath");
const datasetImportStatus = document.querySelector("#datasetImportStatus");
const datasetRootList = document.querySelector("#datasetRootList");

let datasetId = null;
let episodeIndex = 0;
let metadata = null;
let currentIndex = 0;
let currentSample = null;
let playing = false;
let fractionalIndex = 0;
let lastTick = null;
let loadingSample = false;
let requestGeneration = 0;
let sampleAbortController = null;
let recordedActionSeries = [];
let openLoopActionSeries = [];
const cameraViews = new Map();
let openLoopJobId = null;
let openLoopPollTimer = null;
let replayStatus = null;
let replayPollTimer = null;

const COLORS = [
  "#23c5db", "#fa6d8a", "#a98cff", "#f4b526", "#3bd1a4", "#65a8fa", "#e86ab2",
  "#98dc2e", "#ff963c", "#2fcbb5", "#8587fb", "#db76ef", "#f5ce27", "#50da87",
];

function requestHeaders(extra = {}) {
  const accessKey = window.sessionStorage?.getItem("motrixAccessKey");
  return accessKey ? { ...extra, "X-Motrix-Key": accessKey } : extra;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: requestHeaders(options.headers || {}),
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function setMessage(text = "") {
  message.textContent = text;
}

function option(select, value, text) {
  const node = document.createElement("option");
  node.value = String(value);
  node.textContent = text;
  select.appendChild(node);
}

function clearDatasetView() {
  datasetId = null;
  metadata = null;
  currentIndex = 0;
  currentSample = null;
  fractionalIndex = 0;
  recordedActionSeries = [];
  openLoopActionSeries = [];
  playing = false;
  updatePlayButton();
  progress.max = "0";
  progress.value = "0";
  document.querySelector("#episodeMeta").textContent = "尚未选择录制数据集";
  document.querySelector("#currentLine").textContent = "0.00 s · frame 0";
  document.querySelector("#durationLine").textContent = "—";
  document.querySelector("#sampleDetails").replaceChildren();
  document.querySelector("#metrics").replaceChildren();
  document.querySelector("#eventList").replaceChildren();
  for (const view of cameraViews.values()) view.panel.remove();
  cameraViews.clear();
  renderStateChart([]);
  renderActionChart();
}

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${typeof value === "number" ? Number(value.toFixed?.(4) ?? value) : value}${suffix}`;
}

function vectorText(names, values) {
  if (!values) return "未记录";
  return values.map((value, index) => `${names?.[index] || `dim_${index}`}: ${fmt(value)}`).join(" · ");
}

function cameraOrder(role) {
  const value = role.toLowerCase();
  if (value.includes("head") || value.includes("d435_2")) return 0;
  if (value.includes("left") || value.includes("d435_0")) return 1;
  if (value.includes("right") || value.includes("d435_1")) return 2;
  return 10;
}

function cameraTitle(role) {
  const value = role.toLowerCase();
  if (value.includes("head") || value.includes("d435_2")) return "Head Color";
  if (value.includes("left") || value.includes("d435_0")) return "Left Color";
  if (value.includes("right") || value.includes("d435_1")) return "Right Color";
  return role.replaceAll("_", " ");
}

async function loadDatasets() {
  const payload = await requestJson("/api/data-view/datasets");
  datasetSelect.replaceChildren();
  for (const dataset of payload.datasets) {
    option(datasetSelect, dataset.dataset_id, `${dataset.name} · ${dataset.robot_name} · ${dataset.episode_count} episodes`);
  }
  if (!payload.datasets.length) {
    episodeSelect.replaceChildren();
    clearDatasetView();
    setMessage("尚未导入可读取的 LeRobot v2.1 数据集。可在“导入数据路径”中添加目录，或用 --recorded-data-root 启动服务。");
    return;
  }
  datasetId = datasetSelect.value;
  await loadEpisodes();
}

function renderDatasetRoots(dataView = {}) {
  const roots = Array.isArray(dataView.dataset_roots) ? dataView.dataset_roots : [];
  datasetRootList.replaceChildren();
  if (!roots.length) {
    const empty = document.createElement("span");
    empty.className = "data-view-root-meta";
    empty.textContent = "当前没有已导入的数据路径。";
    datasetRootList.appendChild(empty);
    return;
  }
  for (const root of roots) {
    const row = document.createElement("div");
    row.className = "data-view-root-row";
    const label = document.createElement("span");
    label.className = "data-view-root-label";
    label.textContent = root.label || root.root_id || "录制数据路径";
    label.title = label.textContent;
    const meta = document.createElement("span");
    meta.className = "data-view-root-meta";
    const origin = root.origin === "web" ? "本次服务" : "启动参数";
    const datasets = typeof root.dataset_count === "number" && Number.isFinite(root.dataset_count)
      ? `${root.dataset_count} 个数据集`
      : "已配置";
    meta.textContent = `${origin} · ${datasets}`;
    row.append(label, meta);
    if (root.origin === "web" && root.root_id) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "移除";
      remove.title = "只从本次 Web 服务的数据目录中移除，不会删除磁盘文件";
      remove.addEventListener("click", () => removeDatasetRoot(root).catch((error) => setMessage(error.message)));
      row.appendChild(remove);
    }
    datasetRootList.appendChild(row);
  }
}

function updateDatasetImportStatus(dataView = {}) {
  const roots = Array.isArray(dataView.dataset_roots) ? dataView.dataset_roots : [];
  const webCount = roots.filter((root) => root.origin === "web").length;
  datasetImportStatus.textContent = webCount
    ? `已导入 ${webCount} 个本次服务的数据路径；重启 Web 服务后需重新导入。`
    : "尚未导入额外数据路径。";
}

function applyDataViewStatus(dataView = {}) {
  renderDatasetRoots(dataView);
  updateDatasetImportStatus(dataView);
}

async function importDatasetPath() {
  const path = datasetPath.value.trim();
  if (!path) throw new Error("请输入运行 mp-real-web 的机器上可读取的数据集或存储目录");
  const button = document.querySelector("#importDatasetPath");
  button.disabled = true;
  try {
    const payload = await requestJson("/api/data-view/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    datasetPath.value = "";
    applyDataViewStatus(payload.data_view || {});
    const root = payload.root || {};
    datasetImportStatus.textContent = `已导入 ${root.label || "数据路径"} · ${root.dataset_count ?? 0} 个数据集。重启 Web 服务后需重新导入。`;
    await loadDatasets();
  } finally {
    button.disabled = false;
  }
}

async function removeDatasetRoot(root) {
  if (!window.confirm(`仅从本次 Web 服务移除“${root.label || root.root_id}”？不会删除磁盘文件。`)) return;
  const payload = await requestJson("/api/data-view/datasets/remove", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ root_id: root.root_id }),
  });
  applyDataViewStatus(payload.data_view || {});
  await loadDatasets();
}

async function loadEpisodes() {
  if (!datasetId) return;
  playing = false;
  updatePlayButton();
  const payload = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes`);
  episodeSelect.replaceChildren();
  for (const episode of payload.episodes) {
    const result = episode.labels?.result ? ` · ${episode.labels.result}` : "";
    option(episodeSelect, episode.episode_index, `Episode ${episode.episode_index} · ${episode.length} samples · ${episode.status}${result}`);
  }
  if (!payload.episodes.length) {
    setMessage("该数据集没有可读取的 episode。");
    return;
  }
  episodeIndex = Number(episodeSelect.value);
  await loadEpisode();
}

async function loadEpisode() {
  if (!datasetId) return;
  setMessage("");
  playing = false;
  updatePlayButton();
  openLoopActionSeries = [];
  metadata = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/metadata`);
  progress.max = Math.max(0, metadata.length - 1);
  currentIndex = 0;
  fractionalIndex = 0;
  document.querySelector("#episodeMeta").textContent = `${metadata.robot_name} · ${metadata.length} samples · ${fmt(metadata.duration_s, " s")} · ${metadata.status}${metadata.is_mp_real ? " · mp-real telemetry" : " · 标准 LeRobot"}`;
  document.querySelector("#durationLine").textContent = fmt(metadata.duration_s, "s");
  await Promise.all([loadSample(0), loadMetrics(), loadEvents(), loadCurves()]);
}

async function loadSample(index) {
  if (!metadata || !datasetId) return;
  const normalized = Math.max(0, Math.min(metadata.length - 1, Math.round(index)));
  sampleAbortController?.abort();
  const abortController = new AbortController();
  sampleAbortController = abortController;
  loadingSample = true;
  const generation = ++requestGeneration;
  try {
    const payload = await requestJson(
      `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/sample?sample_index=${normalized}`,
      { signal: abortController.signal },
    );
    if (generation !== requestGeneration) return;
    await renderCameras(payload.cameras || {}, normalized, generation, abortController.signal);
    if (generation !== requestGeneration) return;
    currentIndex = normalized;
    progress.value = String(normalized);
    currentSample = payload;
    renderSample(payload);
    renderStateChart();
    renderActionChart();
  } catch (error) {
    if (generation === requestGeneration && error.name !== "AbortError") setMessage(error.message);
  } finally {
    if (generation === requestGeneration) {
      loadingSample = false;
      sampleAbortController = null;
    }
  }
}

function renderSample(sample) {
  const cursor = sample.cursor;
  document.querySelector("#currentLine").textContent = `${fmt(cursor.timestamp, " s")} · frame ${sample.frame_index} · sample ${cursor.sample_index} / ${metadata.length - 1}`;
  const details = [
    ["任务", sample.task || "—"],
    ["sample / frame", `${cursor.sample_index} / ${sample.frame_index}（global ${sample.global_index}）`],
    ["录制时间", fmt(cursor.timestamp, " s")],
    ["robot state", vectorText(metadata.state_fields, sample.state)],
    ["录制 action", vectorText(metadata.action_fields, sample.action)],
    ["selected raw", vectorText(metadata.action_fields, sample.selected_raw_action)],
    ["stabilized", vectorText(metadata.action_fields, sample.stabilized_action)],
    ["executed", vectorText(metadata.action_fields, sample.executed_action)],
    ["inference / control", `${fmt(sample.inference_latency_ns === null ? null : sample.inference_latency_ns / 1e6, " ms")} / ${fmt(sample.control_cycle_ns === null ? null : sample.control_cycle_ns / 1e6, " ms")}`],
    ["camera skew", fmt(sample.camera_skew_ns === null ? null : sample.camera_skew_ns / 1e6, " ms")],
    ["安全标记", Array.isArray(sample.safety_flags) && sample.safety_flags.length ? sample.safety_flags.join(", ") : "未记录"],
  ];
  const list = document.querySelector("#sampleDetails");
  list.replaceChildren();
  for (const [key, value] of details) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    list.append(dt, dd);
  }
}

async function renderCameras(cameras, sampleAtRequest, generation, signal) {
  const grid = document.querySelector("#cameraGrid");
  const rendered = await Promise.all(Object.entries(cameras).map(async ([role, camera]) => {
    let bitmap = null;
    let missingText = null;
    let reused = camera.frame_reused;
    let renderedFrame = camera.frame_index;
    if (camera.missing) {
      missingText = "missing video frame";
    } else {
      const url = `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/frame?sample_index=${sampleAtRequest}&role=${encodeURIComponent(role)}&t=${Date.now()}`;
      try {
        const response = await fetch(url, { cache: "no-store", signal, headers: requestHeaders() });
        if (!response.ok) throw new Error("camera frame unavailable");
        bitmap = await createImageBitmap(await response.blob());
        reused = response.headers.get("X-Frame-Reused") === "true" || camera.frame_reused;
        renderedFrame = response.headers.get("X-Rendered-Frame-Index") || camera.frame_index;
      } catch (error) {
        bitmap?.close();
        if (error.name === "AbortError") throw error;
        missingText = "camera frame unavailable";
        reused = true;
        renderedFrame = "missing";
        bitmap = null;
      }
    }
    return { role, camera, bitmap, missingText, reused, renderedFrame };
  }));
  if (generation !== requestGeneration) {
    rendered.forEach((item) => item.bitmap?.close());
    return;
  }
  rendered.sort((left, right) => cameraOrder(left.role) - cameraOrder(right.role) || left.role.localeCompare(right.role));
  const activeRoles = new Set(rendered.map((item) => item.role));
  for (const [role, view] of cameraViews) {
    if (activeRoles.has(role)) continue;
    view.panel.remove();
    cameraViews.delete(role);
  }
  for (const item of rendered) {
    const view = cameraViews.get(item.role) || createCameraView(grid, item.role);
    grid.appendChild(view.panel);
    view.meta.textContent = cameraMetaText(item.camera, item.reused, item.renderedFrame);
    if (item.bitmap) {
      if (view.canvas.width !== item.bitmap.width || view.canvas.height !== item.bitmap.height) {
        view.canvas.width = item.bitmap.width;
        view.canvas.height = item.bitmap.height;
      }
      view.context.drawImage(item.bitmap, 0, 0);
      item.bitmap.close();
      view.canvas.hidden = false;
      view.missing.hidden = true;
      view.hasFrame = true;
    } else if (!view.hasFrame) {
      view.missing.textContent = item.missingText || "camera frame unavailable";
      view.canvas.hidden = true;
      view.missing.hidden = false;
    }
  }
}

function createCameraView(grid, role) {
  const panel = document.createElement("article");
  panel.className = "data-view-camera";
  const header = document.createElement("header");
  const heading = document.createElement("h2");
  heading.textContent = cameraTitle(role);
  const meta = document.createElement("div");
  meta.className = "camera-meta";
  meta.title = role;
  header.append(heading, meta);
  const media = document.createElement("div");
  media.className = "camera-media";
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", `${role} playback`);
  canvas.hidden = true;
  const missing = document.createElement("div");
  missing.className = "missing";
  missing.textContent = "loading camera frame";
  media.append(canvas, missing);
  panel.append(header, media);
  grid.appendChild(panel);
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error(`2D canvas is unavailable for camera ${role}`);
  const view = { panel, meta, canvas, context, missing, hasFrame: false };
  cameraViews.set(role, view);
  return view;
}

function cameraMetaText(camera, reused = camera.frame_reused, rendered = camera.frame_index) {
  return `frame ${camera.frame_index} · rendered ${rendered ?? "—"} · id ${camera.frame_id ?? "—"} · reused ${reused ? "yes" : "no"}`;
}

async function loadMetrics() {
  const payload = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/metrics`);
  const labels = {
    episode_duration_s: "episode duration", frame_count: "frame count", inference_latency_p50_ms: "inference P50",
    inference_latency_p95_ms: "inference P95", control_frequency_mean_hz: "control frequency",
    control_overrun_count: "control overruns", action_jump_mean: "action jump mean", action_jump_max: "action jump max",
    dropped_frame_count: "dropped frames", dropped_event_count: "dropped events",
  };
  const list = document.querySelector("#metrics");
  list.replaceChildren();
  for (const [key, label] of Object.entries(labels)) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = fmt(payload.metrics[key]);
    list.append(dt, dd);
  }
}

async function loadEvents() {
  const payload = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/events`);
  const list = document.querySelector("#eventList");
  list.replaceChildren();
  for (const event of payload.events) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${event.type} · sample ${event.sample_index}`;
    button.title = event.description;
    button.addEventListener("click", () => setIndex(event.sample_index));
    list.appendChild(button);
  }
}

async function loadCurves() {
  const endpoint = `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/curves?series=${encodeURIComponent("state,action")}&max_points=600`;
  const payload = await requestJson(endpoint);
  const all = payload.series || [];
  renderStateChart(all.filter((item) => String(item.id || "").startsWith("state.")));
  recordedActionSeries = all.filter((item) => String(item.id || "").startsWith("action."));
  renderActionChart();
}

function renderStateChart(series = null) {
  const source = series || [];
  if (series === null) {
    return;
  }
  renderChart("#stateChart", "#stateLegend", "#stateChartRange", source, { comparison: false });
}

function renderActionChart() {
  const series = openLoopActionSeries.length ? openLoopActionSeries : recordedActionSeries;
  const note = document.querySelector("#actionChartNote");
  note.textContent = openLoopActionSeries.length ? "实线：预测；虚线：录制目标" : "录制动作";
  renderChart("#actionChart", "#actionLegend", "#actionChartRange", series, { comparison: openLoopActionSeries.length > 0 });
}

function renderChart(chartSelector, legendSelector, rangeSelector, series, { comparison }) {
  const chart = document.querySelector(chartSelector);
  const legend = document.querySelector(legendSelector);
  const range = document.querySelector(rangeSelector);
  chart.replaceChildren();
  legend.replaceChildren();
  const points = series.flatMap((item) => item.points || []).filter((point) => Number.isFinite(Number(point[1])));
  if (!points.length || !metadata) {
    const text = svgNode("text", { x: 480, y: 180, "text-anchor": "middle", class: "curve-axis-label" });
    text.textContent = "此 episode 没有可绘制的数据";
    chart.appendChild(text);
    range.textContent = "—";
    return;
  }
  const values = points.map((point) => Number(point[1]));
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const width = 960;
  const height = 360;
  const pad = { left: 48, right: 12, top: 14, bottom: 32 };
  const x = (index) => pad.left + (Number(index) / Math.max(1, metadata.length - 1)) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((Number(value) - min) / (max - min)) * (height - pad.top - pad.bottom);
  for (let index = 0; index <= 4; index += 1) {
    const gridY = pad.top + index / 4 * (height - pad.top - pad.bottom);
    chart.appendChild(svgNode("line", { x1: pad.left, y1: gridY, x2: width - pad.right, y2: gridY, class: "curve-grid-line" }));
    const label = svgNode("text", { x: pad.left - 7, y: gridY + 4, "text-anchor": "end", class: "curve-axis-label" });
    label.textContent = fmt(max - index / 4 * (max - min));
    chart.appendChild(label);
  }
  for (let index = 0; index <= 4; index += 1) {
    const gridX = pad.left + index / 4 * (width - pad.left - pad.right);
    chart.appendChild(svgNode("line", { x1: gridX, y1: pad.top, x2: gridX, y2: height - pad.bottom, class: "curve-grid-line" }));
    const label = svgNode("text", { x: gridX, y: height - 10, "text-anchor": "middle", class: "curve-axis-label" });
    label.textContent = `+${fmt(index / 4 * (metadata.duration_s || 0), "s")}`;
    chart.appendChild(label);
  }
  const markerX = x(currentIndex);
  chart.appendChild(svgNode("line", { x1: markerX, y1: pad.top, x2: markerX, y2: height - pad.bottom, class: "curve-marker" }));
  series.forEach((item, index) => {
    const color = colorForSeries(item, index);
    const path = svgNode("path", {
      d: curvePath(item.points || [], x, y),
      stroke: color,
      class: `curve-line${comparison && curveKind(item) === "target" ? " curve-target" : ""}`,
    });
    chart.appendChild(path);
    const latest = lastFiniteValue(item.points || []);
    const line = document.createElement("span");
    line.style.setProperty("--curve-color", color);
    const dot = document.createElement("i");
    const label = document.createElement("b");
    label.textContent = item.label || item.id || `dim ${index}`;
    const value = document.createElement("em");
    value.textContent = fmt(latest);
    line.append(dot, label, value);
    legend.appendChild(line);
  });
  range.textContent = `${fmt(min)} … ${fmt(max)}`;
}

function svgNode(name, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function curvePath(points, x, y) {
  let started = false;
  let path = "";
  for (const point of points) {
    const value = Number(point[1]);
    if (!Number.isFinite(value)) { started = false; continue; }
    path += `${started ? "L" : "M"}${x(point[0]).toFixed(2)},${y(value).toFixed(2)}`;
    started = true;
  }
  return path;
}

function curveKind(item) {
  if (item.kind) return item.kind;
  return String(item.id || item.label || "").startsWith("target") ? "target" : "prediction";
}

function colorForSeries(item, fallbackIndex) {
  const normalized = String(item.id || item.label || fallbackIndex)
    .replace(/^prediction[._ ]*/, "")
    .replace(/^target[._ ]*/, "");
  let hash = 0;
  for (const char of normalized) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return COLORS[hash % COLORS.length] || COLORS[fallbackIndex % COLORS.length];
}

function lastFiniteValue(points) {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = Number(points[index][1]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function updatePlayButton() {
  playPause.textContent = playing ? "❚❚ 暂停" : "▶ 播放";
}

function setIndex(index) {
  if (!metadata) return;
  playing = false;
  updatePlayButton();
  fractionalIndex = Math.max(0, Math.min(metadata.length - 1, index));
  loadSample(Math.round(fractionalIndex));
}

function playbackTick(now) {
  if (playing && metadata && !loadingSample) {
    if (lastTick !== null) {
      const elapsed = Math.min(0.5, (now - lastTick) / 1000);
      fractionalIndex += elapsed * metadata.fps * Number(playbackRate.value);
      const next = Math.floor(fractionalIndex);
      if (next >= metadata.length - 1) {
        fractionalIndex = metadata.length - 1;
        playing = false;
        updatePlayButton();
        loadSample(metadata.length - 1);
      } else if (next !== currentIndex) {
        loadSample(next);
      }
    }
    lastTick = now;
  } else {
    lastTick = now;
  }
  requestAnimationFrame(playbackTick);
}

async function selectCurrentSample() {
  if (!currentSample || !datasetId) return;
  const payload = await requestJson("/api/data-view/selection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      episode_index: episodeIndex,
      sample_index: currentIndex,
      playing,
      playback_rate: Number(playbackRate.value),
    }),
  });
  document.querySelector("#selectionState").textContent = `已标记 dataset ${payload.selection.dataset_id} · episode ${payload.selection.episode_index} · sample ${payload.selection.sample_index}`;
}

async function startOpenLoop() {
  if (!datasetId || !metadata) throw new Error("请先选择可读取的 dataset 和 episode");
  const payload = await requestJson("/api/data-view/open-loop-evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      episode_index: episodeIndex,
      policy_url: document.querySelector("#openLoopPolicyUrl").value,
      policy_label: document.querySelector("#openLoopPolicyLabel").value,
      policy_api_key: document.querySelector("#openLoopPolicyApiKey").value || null,
      target_source: document.querySelector("#openLoopTargetSource").value,
      alignment: document.querySelector("#openLoopAlignment").value,
      max_timestamp_error: Number(document.querySelector("#openLoopTimestampError").value),
      camera_roles: document.querySelector("#openLoopCameraRoles").value.split(",").map((item) => item.trim()).filter(Boolean),
      allow_frame_index_as_control_step: document.querySelector("#openLoopFrameControl").checked,
    }),
  });
  openLoopActionSeries = [];
  renderActionChart();
  openLoopJobId = payload.job.job_id;
  document.querySelector("#stopOpenLoop").disabled = false;
  renderOpenLoopStatus(payload.job);
  window.clearTimeout(openLoopPollTimer);
  pollOpenLoop();
}

async function pollOpenLoop() {
  if (!openLoopJobId) return;
  try {
    const payload = await requestJson(`/api/data-view/open-loop-evaluations/${encodeURIComponent(openLoopJobId)}`);
    renderOpenLoopStatus(payload.job);
    if (["complete", "partial_error", "cancelled", "error"].includes(payload.job.state)) {
      document.querySelector("#stopOpenLoop").disabled = true;
      if (payload.job.state === "complete" || payload.job.state === "partial_error") await loadOpenLoopReport();
      openLoopPollTimer = null;
      return;
    }
  } catch (error) {
    setMessage(error.message);
    return;
  }
  openLoopPollTimer = window.setTimeout(pollOpenLoop, 500);
}

function renderOpenLoopStatus(job) {
  const progressInfo = job.progress || {};
  const suffix = progressInfo.total_samples ? ` · ${progressInfo.completed_samples || 0}/${progressInfo.total_samples} samples` : "";
  document.querySelector("#openLoopStatus").textContent = `job ${job.job_id} · ${job.state}${suffix}${job.error_message ? ` · ${job.error_message}` : ""}`;
}

async function stopOpenLoop() {
  if (!openLoopJobId) return;
  const payload = await requestJson(`/api/data-view/open-loop-evaluations/${encodeURIComponent(openLoopJobId)}/stop`, { method: "POST" });
  renderOpenLoopStatus(payload.job);
}

async function loadOpenLoopReport() {
  if (!openLoopJobId) return;
  const payload = await requestJson(`/api/data-view/open-loop-evaluations/${encodeURIComponent(openLoopJobId)}/reports/${episodeIndex}?curves=1`);
  const report = payload.report;
  openLoopActionSeries = report.curves || [];
  renderActionChart();
  const result = document.querySelector("#openLoopResult");
  const metrics = report.metrics || {};
  result.textContent = `teacher_forced=${report.teacher_forced} · valid=${report.valid_prediction_count} · MAE=${fmt(metrics.overall_mae)} · RMSE=${fmt(metrics.overall_rmse)}。动作图已显示预测（实线）与录制目标（虚线）。`;
}

function replayPayload() {
  const optionalNumber = (id) => {
    const value = document.querySelector(id).value;
    return value === "" ? null : Number(value);
  };
  return {
    dataset_id: datasetId,
    episode_index: episodeIndex,
    start_sample: optionalNumber("#replayStartSample"),
    end_sample: optionalNumber("#replayEndSample"),
    mode: document.querySelector("#replayMode").value,
    timing_mode: document.querySelector("#replayTiming").value,
    fps: optionalNumber("#replayFps"),
    speed_scale: Number(document.querySelector("#replaySpeedScale").value),
  };
}

async function replayRequest(url, payload = {}) {
  const response = await requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  applyReplay(response.replay);
  return response.replay;
}

function applyReplay(replay) {
  replayStatus = replay;
  const state = replay?.state || "idle";
  const progressInfo = replay?.progress || {};
  const report = replay?.safety_report;
  const status = document.querySelector("#replayStatus");
  status.textContent = `${state}${replay?.error ? ` · ${replay.error}` : ""} · sent ${Math.round(Number(progressInfo.sent || 0) * 100)}% · feedback ${Math.round(Number(progressInfo.feedback || 0) * 100)}%${report?.valid === false ? " · 安全预检未通过" : ""}`;
  const locked = Boolean(replay?.view_cursor_locked);
  document.querySelector("#replayPlanBtn").disabled = locked || state === "planning" || !datasetId;
  document.querySelector("#replayConnectBtn").disabled = state !== "validated";
  document.querySelector("#replayStartBtn").disabled = state !== "armed";
  document.querySelector("#replayPauseBtn").disabled = state !== "running";
  document.querySelector("#replayResumeBtn").disabled = state !== "paused";
  const stoppable = ["connecting", "moving_to_start", "armed", "running", "paused", "stopping"].includes(state);
  document.querySelector("#replayStopBtn").disabled = !stoppable;
  document.querySelector("#replayEmergencyBtn").disabled = !stoppable;
  if (replay?.state && !["idle", "validated", "completed", "aborted", "error"].includes(state)) {
    scheduleReplayPoll();
  }
}

function scheduleReplayPoll() {
  if (replayPollTimer !== null) return;
  replayPollTimer = window.setTimeout(async () => {
    replayPollTimer = null;
    try {
      const payload = await requestJson("/api/replay/status");
      applyReplay(payload.replay);
    } catch (error) {
      setMessage(error.message);
    }
  }, 400);
}

async function createReplayPlan() {
  if (!datasetId) throw new Error("请先选择 dataset");
  await replayRequest("/api/replay/plan", replayPayload());
}

async function startReplay() {
  const planHash = replayStatus?.plan?.plan_hash;
  if (!planHash || !window.confirm(`确认以已验证计划 ${planHash.slice(0, 12)}… 执行真机轨迹回放？`)) return;
  await replayRequest("/api/replay/start", { plan_hash: planHash });
}

async function setupWebMode() {
  try {
    const status = await requestJson("/api/status");
    if (status.runtime_mode !== "data_view") return;
    datasetImportMenu.hidden = false;
    applyDataViewStatus(status.data_view || {});
    document.querySelector("#robotReplayMenu").hidden = false;
    applyReplay(status.replay || null);
    const config = await requestJson("/api/config");
    if (config.server_url) document.querySelector("#openLoopPolicyUrl").value = config.server_url;
  } catch {
    // The standalone mp-data-view process deliberately has no Web runtime or replay endpoint.
  }
}

datasetSelect.addEventListener("change", async () => { datasetId = datasetSelect.value; await loadEpisodes(); });
episodeSelect.addEventListener("change", async () => { episodeIndex = Number(episodeSelect.value); await loadEpisode(); });
progress.addEventListener("input", () => setIndex(Number(progress.value)));
document.querySelector("#jumpStart").addEventListener("click", () => setIndex(0));
document.querySelector("#previousFrame").addEventListener("click", () => setIndex(currentIndex - 1));
document.querySelector("#nextFrame").addEventListener("click", () => setIndex(currentIndex + 1));
document.querySelector("#jumpEnd").addEventListener("click", () => setIndex(metadata.length - 1));
playPause.addEventListener("click", () => { if (!metadata) return; playing = !playing; fractionalIndex = currentIndex; updatePlayButton(); });
document.querySelector("#selectSample").addEventListener("click", () => selectCurrentSample().catch((error) => setMessage(error.message)));
document.querySelector("#importDatasetPath").addEventListener("click", () => importDatasetPath().catch((error) => setMessage(error.message)));
document.querySelector("#startOpenLoop").addEventListener("click", () => startOpenLoop().catch((error) => setMessage(error.message)));
document.querySelector("#stopOpenLoop").addEventListener("click", () => stopOpenLoop().catch((error) => setMessage(error.message)));
document.querySelector("#replayPlanBtn").addEventListener("click", () => createReplayPlan().catch((error) => setMessage(error.message)));
document.querySelector("#replayConnectBtn").addEventListener("click", () => replayRequest("/api/replay/connect").catch((error) => setMessage(error.message)));
document.querySelector("#replayStartBtn").addEventListener("click", () => startReplay().catch((error) => setMessage(error.message)));
document.querySelector("#replayPauseBtn").addEventListener("click", () => replayRequest("/api/replay/pause").catch((error) => setMessage(error.message)));
document.querySelector("#replayResumeBtn").addEventListener("click", () => replayRequest("/api/replay/resume").catch((error) => setMessage(error.message)));
document.querySelector("#replayStopBtn").addEventListener("click", () => replayRequest("/api/replay/stop").catch((error) => setMessage(error.message)));
document.querySelector("#replayEmergencyBtn").addEventListener("click", () => replayRequest("/api/replay/emergency-stop").catch((error) => setMessage(error.message)));

applyReplay(null);
Promise.all([setupWebMode(), loadDatasets()]).catch((error) => setMessage(error.message));
requestAnimationFrame(playbackTick);
