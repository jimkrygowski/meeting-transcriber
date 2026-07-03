const $ = (id) => document.getElementById(id);
const STAGE_LABELS = {
  uploading: "Uploading…",
  converting: "Converting audio…",
  transcribing: "Transcribing — a long meeting takes a few minutes…",
  diarizing: "Identifying speakers…",
  finishing: "Finalizing…",
};
let jobId = null;
let pollTimer = null;

// --- upload ---
const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

async function uploadFile(file) {
  showError(null);
  $("result").hidden = true;
  $("progress").hidden = false;
  $("stage-label").textContent = "Uploading…";
  const form = new FormData();
  form.append("file", file);
  form.append("context", $("context").value);
  const r = await fetch("/api/jobs", { method: "POST", body: form });
  if (!r.ok) {
    $("progress").hidden = true;
    showError((await r.json()).detail || "Upload failed");
    return;
  }
  jobId = (await r.json()).job_id;
  startPolling();
}

// --- polling ---
function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(refresh, 1500);
  refresh();
}

async function refresh() {
  const r = await fetch(jobId ? `/api/jobs/${jobId}` : "/api/jobs/latest");
  if (!r.ok) { clearInterval(pollTimer); return; }
  const body = await r.json();
  jobId = body.job.id;
  render(body);
  const busy = body.job.status === "processing" ||
    (body.transcript && body.transcript.summary_status === "running");
  if (!busy) clearInterval(pollTimer);
}

// --- rendering ---
function render({ job, transcript }) {
  if (job.status === "processing") {
    $("progress").hidden = false;
    $("stage-label").textContent = STAGE_LABELS[job.stage] || job.stage;
    return;
  }
  $("progress").hidden = true;
  if (job.status === "error") { showError(job.error); return; }

  $("result").hidden = false;
  $("meta").textContent =
    `${job.original_name} · ${fmtTime(job.duration)} · ` +
    `${Object.keys(transcript.speaker_map).length} speaker(s)`;
  $("warning").hidden = !job.warning;
  $("warning").textContent = job.warning || "";
  renderSpeakers(transcript);
  renderTranscript(transcript);
  renderSummary(transcript);
  $("export-md").href = `/api/jobs/${jobId}/export?fmt=md`;
  $("export-txt").href = `/api/jobs/${jobId}/export?fmt=txt`;
}

function renderSpeakers(t) {
  const el = $("speakers");
  el.innerHTML = "";
  for (const [sid, name] of Object.entries(t.speaker_map)) {
    const row = document.createElement("div");
    row.className = "speaker-row";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/jobs/${jobId}/snippets/${sid}`;
    audio.addEventListener("error", () => { audio.hidden = true; });
    const input = document.createElement("input");
    input.value = name;
    input.addEventListener("change", async () => {
      const r = await fetch(`/api/jobs/${jobId}/speakers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speaker_id: sid, name: input.value }),
      });
      if (r.ok) {
        t.speaker_map = (await r.json()).speaker_map;
        renderTranscript(t);
      }
    });
    row.append(audio, input);
    el.append(row);
  }
}

function renderTranscript(t) {
  const el = $("transcript");
  el.innerHTML = "";
  let block = null;
  for (const s of t.segments) {
    if (!block || block.speaker !== s.speaker) {
      block = { speaker: s.speaker, start: s.start, texts: [] };
      const div = document.createElement("div");
      div.className = "block";
      div.innerHTML = `<span class="who"></span> <span class="when"></span><p></p>`;
      div.querySelector(".who").textContent = t.speaker_map[s.speaker] || s.speaker;
      div.querySelector(".when").textContent = fmtTime(s.start);
      block.p = div.querySelector("p");
      el.append(div);
    }
    block.texts.push(s.text);
    block.p.textContent = block.texts.join(" ");
  }
}

function renderSummary(t) {
  const btn = $("summarize-btn");
  const status = $("summary-status");
  const out = $("summary");
  btn.disabled = t.summary_status === "running";
  status.hidden = true;
  if (t.summary_status === "running") {
    status.hidden = false;
    status.textContent = "Generating summary locally — this can take a minute…";
  } else if (t.summary_status === "error") {
    status.hidden = false;
    status.textContent = t.summary_error;
  }
  out.hidden = !t.summary;
  out.textContent = t.summary || "";
}

$("summarize-btn").addEventListener("click", async () => {
  await fetch(`/api/jobs/${jobId}/summarize`, { method: "POST" });
  startPolling();
});

// --- helpers ---
function fmtTime(sec) {
  if (sec == null) return "";
  const s = Math.floor(sec), h = Math.floor(s / 3600),
    m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function showError(msg) {
  $("error").hidden = !msg;
  $("error").textContent = msg || "";
}

// context persists across sessions so recurring names stay filled in
$("context").value = localStorage.getItem("transcriber-context") || "";
$("context").addEventListener("input", () =>
  localStorage.setItem("transcriber-context", $("context").value));

// restore the most recent job on load
refresh();
