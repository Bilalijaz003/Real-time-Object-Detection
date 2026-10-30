// Real-time Object Detection – web interface.
// Captures frames from the camera, a video file or a photo, sends them to
// the Flask backend (/api/detect) and draws the returned boxes.

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), stage: $("stage"), video: $("video"), photo: $("photo"),
  overlay: $("overlay"), empty: $("empty"), emptyText: $("empty-text"),
  camStart: $("cam-start"), camStop: $("cam-stop"),
  videoFile: $("video-file"), videoToggle: $("video-toggle"),
  photoFile: $("photo-file"),
  conf: $("conf"), confOut: $("conf-out"), filter: $("filter"),
  classList: $("class-list"),
  fps: $("fps"), latency: $("latency"), total: $("total"), counts: $("counts"),
  summary: $("summary"), statsBody: $("stats-body"), scans: $("scans"),
  clearHistory: $("clear-history"),
};

const SEND_WIDTH = 640;          // frames are resized to this before upload
const EMPTY_TEXT = {
  camera: "Start the camera to detect objects live.",
  video: "Choose a video file to run detection on it.",
  photo: "Choose a photo to detect the objects in it.",
};

const state = {
  mode: "camera",
  stream: null,       // webcam MediaStream
  running: false,     // frame loop active
  scanId: null,       // history session id for camera/video
  frameTimes: [],
  videoUrl: null,
  photoUrl: null,
  last: null,         // last detection result
  gen: 0,             // increments per loop start, so an old loop can't continue
};

const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d");
const ctx = els.overlay.getContext("2d");

// ---------------------------------------------------------------- helpers
function colorFor(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${h} 80% 60%)`;
}

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = "status " + kind;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function detectionParams(form) {
  form.append("conf", els.conf.value);
  const f = els.filter.value.trim();
  if (f) form.append("classes", f);
}

// ---------------------------------------------------------------- drawing
function sizeOverlay(w, h) {
  if (els.overlay.width !== w || els.overlay.height !== h) {
    els.overlay.width = w;
    els.overlay.height = h;
  }
}

function draw(result, srcW, srcH) {
  // result boxes are in the sent image's coordinates (result.width x height);
  // the overlay uses the media's own resolution (srcW x srcH).
  sizeOverlay(srcW, srcH);
  ctx.clearRect(0, 0, srcW, srcH);
  if (!result) return;
  const sx = srcW / result.width, sy = srcH / result.height;
  const lw = Math.max(2, Math.round(srcW / 320));
  const fs = Math.max(13, Math.round(srcW / 48));
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px "Bricolage Grotesque", system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const d of result.detections) {
    const [x1, y1, x2, y2] = d.box;
    const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    const color = colorFor(d.class);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const label = `${d.class} ${d.confidence.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 10, th = fs + 8;
    const ly = y - th < 0 ? y : y - th;
    ctx.fillStyle = color;
    ctx.fillRect(x - lw / 2, ly, tw, th);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 4, ly + 4);
  }
}

function showReadout(result, fps) {
  els.fps.textContent = fps == null ? "–" : fps.toFixed(1);
  els.latency.textContent = result ? `${Math.round(result.inference_ms)} ms` : "–";
  els.total.textContent = result ? result.count : "–";
  els.counts.innerHTML = "";
  if (!result) return;
  for (const [name, n] of Object.entries(result.counts)) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = colorFor(name);
    label.append(sw, name);
    const num = document.createElement("strong");
    num.textContent = n;
    li.append(label, num);
    els.counts.append(li);
  }
}

function clearView() {
  state.last = null;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  showReadout(null, null);
}

// ---------------------------------------------------------------- sessions
async function startScan(source, name) {
  const { scan_id } = await api("/api/scans", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, name }),
  });
  state.scanId = scan_id;
}

async function endScan() {
  if (state.scanId == null) return;
  const id = state.scanId;
  state.scanId = null;
  try { await api(`/api/scans/${id}/end`, { method: "POST" }); } catch {}
  loadHistory();
}

window.addEventListener("pagehide", () => {
  if (state.scanId != null) navigator.sendBeacon(`/api/scans/${state.scanId}/end`);
});

// ---------------------------------------------------------------- frame loop
async function loop(gen) {
  const v = els.video;
  const active = () => state.running && state.gen === gen;
  while (active()) {
    if (v.paused || v.ended || v.readyState < 2 || !v.videoWidth) {
      await new Promise((r) => setTimeout(r, 100));
      continue;
    }
    const scale = Math.min(1, SEND_WIDTH / v.videoWidth);
    grab.width = Math.round(v.videoWidth * scale);
    grab.height = Math.round(v.videoHeight * scale);
    grabCtx.drawImage(v, 0, 0, grab.width, grab.height);
    const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.8));

    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    detectionParams(form);
    // Live frames belong to a session; never save them as separate photos.
    if (state.scanId != null) form.append("scan_id", state.scanId);
    else form.append("save", "0");
    if (!active()) break;

    const t0 = performance.now();
    try {
      const result = await api("/api/detect", { method: "POST", body: form });
      if (!active()) break;
      state.frameTimes.push(performance.now() - t0);
      if (state.frameTimes.length > 20) state.frameTimes.shift();
      const avg = state.frameTimes.reduce((a, b) => a + b, 0) / state.frameTimes.length;
      state.last = result;
      draw(result, v.videoWidth, v.videoHeight);
      showReadout(result, 1000 / avg);
      setStatus("Detecting", "ok");
    } catch (e) {
      if (!active()) break;   // session ended while this frame was in flight
      setStatus(e.message, "err");
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

function startLoop() {
  if (state.running) return;
  state.running = true;
  state.gen += 1;
  state.frameTimes = [];
  loop(state.gen);
}

function stopLoop() {
  state.running = false;
}

// ---------------------------------------------------------------- media display
function showMedia(kind) {
  els.video.hidden = kind !== "video";
  els.photo.hidden = kind !== "photo";
  els.empty.hidden = kind !== null;
  els.stage.classList.toggle("has-media", kind !== null);
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  els.camStart.disabled = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    els.camStart.disabled = false;
    const denied = e.name === "NotAllowedError";
    setStatus(denied
      ? "Camera access was blocked. Allow it in the browser's address bar, then try again."
      : "No camera found. Connect one, or use the Video file or Photo tab.", "err");
    return;
  }
  els.video.srcObject = state.stream;
  await els.video.play();
  showMedia("video");
  els.camStart.hidden = true;
  els.camStart.disabled = false;
  els.camStop.hidden = false;
  try { await startScan("camera"); } catch (e) { setStatus(e.message, "err"); }
  startLoop();
}

async function stopCamera() {
  stopLoop();
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  els.video.srcObject = null;
  els.camStop.hidden = true;
  els.camStart.hidden = false;
  showMedia(null);
  clearView();
  setStatus("Camera stopped. Session saved to history.", "ok");
  await endScan();
}

// ---------------------------------------------------------------- video file
async function openVideo(file) {
  await endScan();
  stopLoop();
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(file);
  els.video.srcObject = null;
  els.video.src = state.videoUrl;
  els.video.loop = false;
  showMedia("video");
  clearView();
  try {
    await els.video.play();
  } catch {
    setStatus("This video format can't play in the browser. Try an MP4 or WebM file.", "err");
    return;
  }
  els.videoToggle.hidden = false;
  els.videoToggle.textContent = "Pause";
  await startScan("video", file.name);
  startLoop();
}

els.video.addEventListener("ended", async () => {
  if (state.mode !== "video") return;
  stopLoop();
  els.videoToggle.textContent = "Play again";
  setStatus("Video finished. Session saved to history.", "ok");
  await endScan();
});

async function toggleVideo() {
  const v = els.video;
  if (v.ended) {
    v.currentTime = 0;
    await v.play();
    els.videoToggle.textContent = "Pause";
    await startScan("video", els.videoFile.files[0]?.name);
    startLoop();
  } else if (v.paused) {
    await v.play();
    els.videoToggle.textContent = "Pause";
  } else {
    v.pause();
    els.videoToggle.textContent = "Play";
  }
}

// ---------------------------------------------------------------- photo
async function detectPhoto(file) {
  if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  state.photoUrl = URL.createObjectURL(file);
  els.photo.src = state.photoUrl;
  await els.photo.decode().catch(() => {});
  showMedia("photo");
  clearView();
  setStatus("Detecting…");
  const form = new FormData();
  form.append("image", file, file.name);
  detectionParams(form);
  try {
    const result = await api("/api/detect", { method: "POST", body: form });
    state.last = result;
    draw(result, els.photo.naturalWidth, els.photo.naturalHeight);
    showReadout(result, null);
    setStatus(result.count
      ? `Found ${result.count} object${result.count === 1 ? "" : "s"}. Saved to history.`
      : "No objects found. Try lowering the confidence.", "ok");
    loadHistory();
  } catch (e) {
    setStatus(e.message, "err");
  }
}

// ---------------------------------------------------------------- tabs
async function switchMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "camera" && state.stream) await stopCamera();
  if (state.mode === "video") {
    stopLoop();
    els.video.pause();
    els.video.removeAttribute("src");
    els.video.load();
    els.videoToggle.hidden = true;
    els.videoFile.value = "";
    await endScan();
  }
  state.mode = mode;
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  document.querySelectorAll(".mode-actions").forEach((el) =>
    (el.hidden = el.dataset.for !== mode));
  els.emptyText.textContent = EMPTY_TEXT[mode];
  showMedia(null);
  clearView();
}

// ---------------------------------------------------------------- history
function timeAgo(iso) {
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

const SOURCE_LABEL = { image: "Photo", camera: "Camera session", video: "Video" };

async function loadHistory() {
  let stats, history;
  try {
    [stats, history] = await Promise.all([api("/api/stats"), api("/api/history?limit=12")]);
  } catch { return; }

  const t = stats.totals;
  els.summary.textContent = t.scans
    ? `${t.scans} scan${t.scans === 1 ? "" : "s"}: ${t.images} photo${t.images === 1 ? "" : "s"}, ` +
      `${t.camera_sessions} camera session${t.camera_sessions === 1 ? "" : "s"}, ` +
      `${t.video_sessions} video${t.video_sessions === 1 ? "" : "s"}. ` +
      `${t.frames.toLocaleString()} frames analysed.`
    : "No scans yet.";

  els.statsBody.innerHTML = "";
  if (!stats.classes.length) {
    els.statsBody.innerHTML = '<tr><td colspan="3" class="muted">Detected objects will appear here.</td></tr>';
  }
  for (const c of stats.classes) {
    const tr = document.createElement("tr");
    for (const v of [c.class, c.times_detected, c.total_objects]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    }
    els.statsBody.append(tr);
  }

  els.scans.innerHTML = "";
  for (const s of history.scans) {
    const li = document.createElement("li");
    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("span");
    title.textContent = SOURCE_LABEL[s.source] + (s.name ? `: ${s.name}` : "");
    const time = document.createElement("time");
    time.dateTime = s.started_at;
    time.textContent = timeAgo(s.started_at);
    meta.append(title, time);
    const objs = document.createElement("div");
    objs.className = "objs";
    objs.textContent = s.objects.length
      ? s.objects.map((o) => `${o.objects} ${o.class}`).join(", ")
      : "Nothing detected";
    if (s.source !== "image") objs.textContent += ` (${s.frames} frames)`;
    li.append(meta, objs);
    els.scans.append(li);
  }
}

els.clearHistory.addEventListener("click", async () => {
  if (!confirm("Delete all saved scans? This can't be undone.")) return;
  try {
    await api("/api/history", { method: "DELETE" });
    setStatus("History cleared.", "ok");
  } catch (e) { setStatus(e.message, "err"); }
  loadHistory();
});

// ---------------------------------------------------------------- wiring
els.camStart.addEventListener("click", startCamera);
els.camStop.addEventListener("click", stopCamera);
els.videoFile.addEventListener("change", () => els.videoFile.files[0] && openVideo(els.videoFile.files[0]));
els.videoToggle.addEventListener("click", toggleVideo);
els.photoFile.addEventListener("change", () => els.photoFile.files[0] && detectPhoto(els.photoFile.files[0]));
document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => switchMode(b.dataset.mode)));
els.conf.addEventListener("input", () => (els.confOut.textContent = Number(els.conf.value).toFixed(2)));

async function init() {
  try {
    const h = await api("/api/health");
    setStatus(`Backend ready, model ${h.model}`, "ok");
    const { classes } = await api("/api/classes");
    for (const c of classes) {
      const o = document.createElement("option");
      o.value = c;
      els.classList.append(o);
    }
  } catch {
    setStatus("Can't reach the backend. Start it with: python backend/app.py", "err");
  }
  loadHistory();
  setInterval(() => state.running && loadHistory(), 5000);
}
init();
