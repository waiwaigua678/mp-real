const datasetSelect = document.querySelector("#datasetSelect");
const episodeSelect = document.querySelector("#episodeSelect");
const progress = document.querySelector("#progress");
const playbackRate = document.querySelector("#playbackRate");
const playPause = document.querySelector("#playPause");
const message = document.querySelector("#message");

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

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
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

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${typeof value === "number" ? Number(value.toFixed?.(4) ?? value) : value}${suffix}`;
}

function vectorText(names, values) {
  if (!values) return "未记录";
  return values.map((value, index) => `${names?.[index] || `dim_${index}`}: ${fmt(value)}`).join(" · ");
}

async function loadDatasets() {
  const payload = await requestJson("/api/data-view/datasets");
  datasetSelect.replaceChildren();
  for (const dataset of payload.datasets) {
    option(datasetSelect, dataset.dataset_id, `${dataset.name} · ${dataset.robot_name} · ${dataset.episode_count} episodes`);
  }
  if (!payload.datasets.length) {
    setMessage("所选 storage root 下没有可读取的 LeRobot v2.1 数据集。");
    return;
  }
  datasetId = datasetSelect.value;
  await loadEpisodes();
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
  metadata = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/metadata`);
  progress.max = Math.max(0, metadata.length - 1);
  currentIndex = 0;
  fractionalIndex = 0;
  document.querySelector("#episodeMeta").textContent = `${metadata.robot_name} · ${metadata.length} samples · ${fmt(metadata.duration_s, " s")} · ${metadata.status}${metadata.is_mp_real ? " · mp-real telemetry" : " · 标准 LeRobot 数据"}`;
  await Promise.all([loadSample(0), loadMetrics(), loadEvents(), loadCurves()]);
}

async function loadSample(index) {
  if (!metadata || !datasetId) return;
  const normalized = Math.max(0, Math.min(metadata.length - 1, Math.round(index)));
  currentIndex = normalized;
  progress.value = String(normalized);
  loadingSample = true;
  const generation = ++requestGeneration;
  try {
    const payload = await requestJson(
      `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/sample?sample_index=${normalized}`,
    );
    if (generation !== requestGeneration) return;
    currentSample = payload;
    renderSample(payload);
    renderCameras(payload.cameras || {});
  } catch (error) {
    if (generation === requestGeneration) setMessage(error.message);
  } finally {
    if (generation === requestGeneration) loadingSample = false;
  }
}

function renderSample(sample) {
  const cursor = sample.cursor;
  document.querySelector("#currentLine").textContent = `sample ${cursor.sample_index} / ${metadata.length - 1} · frame ${sample.frame_index} · t=${fmt(cursor.timestamp, " s")} · ${(cursor.progress_ratio * 100).toFixed(1)}%`;
  const details = [
    ["task prompt", sample.task || "—"],
    ["sample / frame", `${cursor.sample_index} / ${sample.frame_index} (global ${sample.global_index})`],
    ["episode timestamp", fmt(cursor.timestamp, " s")],
    ["monotonic timestamp", sample.timestamp_monotonic_ns ?? "未记录"],
    ["robot state", vectorText(metadata.state_fields, sample.state)],
    ["expert action", vectorText(metadata.action_fields, sample.action)],
    ["selected raw action", vectorText(metadata.action_fields, sample.selected_raw_action)],
    ["stabilized action", vectorText(metadata.action_fields, sample.stabilized_action)],
    ["executed action", vectorText(metadata.action_fields, sample.executed_action)],
    ["raw chunk action", vectorText(metadata.action_fields, sample.raw_action)],
    ["chunk id / cursor", `${sample.chunk_id ?? "—"} / ${sample.chunk_cursor ?? "—"}`],
    ["inference / control", `${fmt(sample.inference_latency_ns === null ? null : sample.inference_latency_ns / 1e6, " ms")} / ${fmt(sample.control_cycle_ns === null ? null : sample.control_cycle_ns / 1e6, " ms")}`],
    ["camera skew", fmt(sample.camera_skew_ns === null ? null : sample.camera_skew_ns / 1e6, " ms")],
    ["safety status", Array.isArray(sample.safety_flags) && sample.safety_flags.length ? sample.safety_flags.join(", ") : "未记录"],
    ["人工结果 / failure", `${sample.labels?.result || "—"} / ${sample.labels?.failure_reason || "—"}`],
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

function renderCameras(cameras) {
  const grid = document.querySelector("#cameraGrid");
  grid.replaceChildren();
  for (const [role, camera] of Object.entries(cameras)) {
    const panel = document.createElement("article");
    panel.className = "camera";
    const heading = document.createElement("h3");
    heading.textContent = role;
    const meta = document.createElement("div");
    meta.className = "camera-meta";
    const setMeta = (reused = camera.frame_reused, rendered = camera.frame_index) => {
      meta.textContent = `frame ${camera.frame_index} · rendered ${rendered ?? "—"} · frame_id ${camera.frame_id ?? "—"} · ts ${camera.camera_timestamp_ns ?? "—"} · reused ${reused ?? "—"} · age ${camera.camera_age_ns === null ? "—" : `${(camera.camera_age_ns / 1e6).toFixed(1)} ms`}`;
    };
    setMeta();
    panel.append(heading, meta);
    if (camera.missing) {
      const missing = document.createElement("div");
      missing.className = "missing";
      missing.textContent = "missing video frame";
      panel.appendChild(missing);
    } else {
      const image = document.createElement("img");
      image.alt = `${role} sample ${currentIndex}`;
      const sampleAtRequest = currentIndex;
      const url = `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/frame?sample_index=${sampleAtRequest}&role=${encodeURIComponent(role)}&t=${Date.now()}`;
      fetch(url, { cache: "no-store" })
        .then(async (response) => {
          if (!response.ok) throw new Error("camera frame unavailable");
          const imageUrl = URL.createObjectURL(await response.blob());
          if (sampleAtRequest !== currentIndex) {
            URL.revokeObjectURL(imageUrl);
            return;
          }
          image.onload = () => URL.revokeObjectURL(imageUrl);
          image.src = imageUrl;
          setMeta(
            response.headers.get("X-Frame-Reused") === "true" || camera.frame_reused,
            response.headers.get("X-Rendered-Frame-Index") || camera.frame_index,
          );
        })
        .catch(() => {
          if (sampleAtRequest === currentIndex) setMeta(true, "missing");
        });
      panel.appendChild(image);
    }
    grid.appendChild(panel);
  }
}

async function loadMetrics() {
  const payload = await requestJson(`/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/metrics`);
  const labels = {
    episode_duration_s: "episode duration",
    frame_count: "frame count",
    inference_latency_p50_ms: "inference latency P50",
    inference_latency_p95_ms: "inference latency P95",
    inference_latency_p99_ms: "inference latency P99",
    control_frequency_mean_hz: "control frequency mean",
    control_overrun_count: "control overruns",
    camera_skew_p50_ms: "camera skew P50",
    camera_skew_p95_ms: "camera skew P95",
    action_jump_mean: "action jump mean",
    action_jump_max: "action jump max",
    velocity_rms: "velocity RMS",
    acceleration_rms: "acceleration RMS",
    jerk_rms: "jerk RMS",
    raw_to_stabilized_mean: "raw → stabilized magnitude",
    stabilized_to_executed_mean: "stabilized → executed magnitude",
    safety_modification_count: "safety modifications",
    dropped_frame_count: "dropped frames",
    dropped_event_count: "dropped events",
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
    button.innerHTML = `<span class="event-type"></span>`;
    button.querySelector(".event-type").textContent = event.type;
    button.append(` · sample ${event.sample_index} · ${event.description}`);
    button.addEventListener("click", () => loadSample(event.sample_index));
    list.appendChild(button);
  }
}

function selectedCurveGroups() {
  return Array.from(document.querySelector("#curveSeries").selectedOptions).map((item) => item.value);
}

async function loadCurves() {
  const series = selectedCurveGroups();
  const payload = await requestJson(
    `/api/data-view/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}/curves?series=${encodeURIComponent(series.join(","))}&max_points=600`,
  );
  renderCurves(payload.series || []);
}

function renderCurves(series) {
  const chart = document.querySelector("#curveChart");
  const legend = document.querySelector("#curveLegend");
  chart.replaceChildren();
  legend.replaceChildren();
  const colors = ["#08777f", "#c45424", "#6a4c93", "#2f855a", "#b7791f", "#c53030", "#2b6cb0", "#805ad5", "#319795", "#9c4221"];
  const points = series.flatMap((item) => item.points || []);
  if (!points.length) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "450"); text.setAttribute("y", "140"); text.setAttribute("text-anchor", "middle"); text.textContent = "选中的数据在此 episode 中未记录";
    chart.appendChild(text);
    return;
  }
  const values = points.map((point) => Number(point[1])).filter(Number.isFinite);
  let min = Math.min(...values); let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const width = 900; const height = 280; const pad = 12;
  series.forEach((item, index) => {
    const color = colors[index % colors.length];
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const d = (item.points || []).map((point, pointIndex) => {
      const x = pad + (Number(point[0]) / Math.max(1, metadata.length - 1)) * (width - 2 * pad);
      const y = height - pad - ((Number(point[1]) - min) / (max - min)) * (height - 2 * pad);
      return `${pointIndex ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    path.setAttribute("d", d); path.setAttribute("stroke", color); path.setAttribute("class", "curve-line");
    chart.appendChild(path);
    const itemLegend = document.createElement("span");
    itemLegend.style.setProperty("--curve-color", color);
    itemLegend.textContent = item.label;
    legend.appendChild(itemLegend);
  });
}

function updatePlayButton() {
  playPause.textContent = playing ? "❚❚ 暂停" : "▶ 播放";
}

function setIndex(index) {
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

datasetSelect.addEventListener("change", async () => { datasetId = datasetSelect.value; await loadEpisodes(); });
episodeSelect.addEventListener("change", async () => { episodeIndex = Number(episodeSelect.value); await loadEpisode(); });
progress.addEventListener("input", () => setIndex(Number(progress.value)));
document.querySelector("#jumpStart").addEventListener("click", () => setIndex(0));
document.querySelector("#previousFrame").addEventListener("click", () => setIndex(currentIndex - 1));
document.querySelector("#nextFrame").addEventListener("click", () => setIndex(currentIndex + 1));
document.querySelector("#jumpEnd").addEventListener("click", () => setIndex(metadata.length - 1));
playPause.addEventListener("click", () => { playing = !playing; fractionalIndex = currentIndex; updatePlayButton(); });
document.querySelector("#curveSeries").addEventListener("change", () => loadCurves().catch((error) => setMessage(error.message)));
document.querySelector("#selectSample").addEventListener("click", async () => {
  if (!currentSample) return;
  const payload = await requestJson("/api/data-view/selection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id: datasetId, episode_index: episodeIndex, sample_index: currentIndex, playback_rate: Number(playbackRate.value) }) });
  document.querySelector("#selectionState").textContent = `已选择 dataset ${payload.selection.dataset_id} · episode ${payload.selection.episode_index} · sample ${payload.selection.sample_index}，供后续阶段使用。`;
});

loadDatasets().catch((error) => setMessage(error.message));
requestAnimationFrame(playbackTick);
