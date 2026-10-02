const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const toggle = document.getElementById("toggle");
const statusEl = document.getElementById("status");
const iconEl = document.getElementById("icon");
const labelEl = document.getElementById("label");
const barsEl = document.getElementById("bars");
const grab = document.createElement("canvas");

let stream = null, timer = null, busy = false, lastSave = 0;

// build one bar per emotion
LABELS.forEach((l) => {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `<span>${l[0].toUpperCase() + l.slice(1)}</span>
    <div class="track"><div class="fill" id="fill-${l}" style="width:0;background:var(--${l})"></div></div>
    <span class="bar-pct" id="pct-${l}">0%</span>`;
  barsEl.appendChild(row);
});

function resetUI() {
  iconEl.textContent = "🙂";
  labelEl.textContent = "-";
  LABELS.forEach((l) => setBar(l, 0));
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

function setBar(label, pct) {
  document.getElementById(`fill-${label}`).style.width = pct + "%";
  document.getElementById(`pct-${label}`).textContent = Math.round(pct) + "%";
}

function drawBox(box, frameSize) {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const k = overlay.width / frameSize[0];
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 3;
  ctx.strokeRect(box[0] * k, box[1] * k, box[2] * k, box[3] * k);
}

function render(d) {
  if (!d.face) {
    statusEl.textContent = "No face found. Face the camera in good light.";
    resetUI();
    return;
  }
  statusEl.textContent = "";
  iconEl.textContent = EMOJI[d.emotion] || "🙂";
  labelEl.textContent = d.emotion[0].toUpperCase() + d.emotion.slice(1);
  LABELS.forEach((l) => setBar(l, d.probs[l] || 0));
  drawBox(d.box, d.frame_size);
}

async function tick() {
  if (busy || !video.videoWidth) return;
  busy = true;
  const w = 320, h = Math.round((320 * video.videoHeight) / video.videoWidth);
  grab.width = w;
  grab.height = h;
  grab.getContext("2d").drawImage(video, 0, 0, w, h);
  const save = Date.now() - lastSave > 3000;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: grab.toDataURL("image/jpeg", 0.7), save }),
    });
    const d = await res.json();
    if (!res.ok) {
      statusEl.textContent = d.error || "Prediction failed.";
    } else {
      if (save && d.face) lastSave = Date.now();
      render(d);
    }
  } catch (e) {
    statusEl.textContent = "Lost connection to the server. Check that app.py is still running.";
  } finally {
    busy = false;
  }
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    statusEl.textContent = "Camera is blocked. Allow camera access for this site in your browser, then try again.";
    return;
  }
  video.srcObject = stream;
  await video.play();
  toggle.textContent = "Stop camera";
  toggle.classList.add("stop");
  statusEl.textContent = "Looking for a face...";
  timer = setInterval(tick, 400);
}

function stop() {
  clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  toggle.textContent = "Start camera";
  toggle.classList.remove("stop");
  statusEl.textContent = "";
  resetUI();
}

toggle.addEventListener("click", () => (stream ? stop() : start()));
