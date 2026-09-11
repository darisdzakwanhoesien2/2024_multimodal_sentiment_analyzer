const $ = (id) => document.getElementById(id);

$("doTranscribe").addEventListener("change", (e) => {
  $("whisperOptions").classList.toggle("hidden", !e.target.checked);
});

$("verifyBtn").addEventListener("click", async () => {
  const token = $("hfToken").value.trim();
  const out = $("verifyResult");
  if (!token) {
    out.textContent = "Enter a token first.";
    out.className = "status-line status-error";
    return;
  }
  out.textContent = "Verifying…";
  out.className = "status-line";

  const form = new FormData();
  form.append("hf_token", token);
  try {
    const res = await fetch("/api/hf/verify", { method: "POST", body: form });
    const data = await res.json();
    if (!data.token_valid) {
      out.textContent = `Invalid token: ${data.error || "unknown error"}`;
      out.className = "status-line status-error";
      return;
    }
    const lines = data.models.map(
      (m) => `${m.accessible ? "✅" : "❌"} ${m.model} — ${m.detail}`
    );
    out.textContent = `Logged in as ${data.identity}. ` + lines.join(" | ");
    out.className = "status-line status-success";
  } catch (e) {
    out.textContent = `Request failed: ${e}`;
    out.className = "status-line status-error";
  }
});

let pollTimer = null;

$("runBtn").addEventListener("click", async () => {
  const fileInput = $("fileInput");
  const status = $("runStatus");

  if (!fileInput.files.length) {
    status.textContent = "Choose a file first.";
    status.className = "status-line status-error";
    return;
  }
  const hfToken = $("hfToken").value.trim();
  if (!hfToken) {
    status.textContent = "A Hugging Face token is required to load the diarization pipeline.";
    status.className = "status-line status-error";
    return;
  }

  $("resultsPanel").classList.add("hidden");
  $("runBtn").disabled = true;
  if (pollTimer) clearInterval(pollTimer);

  const form = new FormData();
  form.append("file", fileInput.files[0]);
  form.append("hf_token", hfToken);
  form.append("model_choice", $("modelChoice").value);
  form.append("num_speakers", $("numSpeakers").value);
  form.append("min_speakers", $("minSpeakers").value);
  form.append("max_speakers", $("maxSpeakers").value);
  form.append("do_transcribe", $("doTranscribe").checked);
  form.append("whisper_model_size", $("whisperModel").value);
  form.append("whisper_language", $("whisperLanguage").value.trim());

  status.textContent = "Uploading…";
  status.className = "status-line";

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const { job_id } = await res.json();
    pollJob(job_id);
  } catch (e) {
    status.textContent = `Failed to start job: ${e.message}`;
    status.className = "status-line status-error";
    $("runBtn").disabled = false;
  }
});

function pollJob(jobId) {
  const status = $("runStatus");
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();

    if (job.status === "error") {
      clearInterval(pollTimer);
      status.textContent = `❌ ${job.error}`;
      status.className = "status-line status-error";
      $("runBtn").disabled = false;
      return;
    }
    if (job.status === "done") {
      clearInterval(pollTimer);
      status.textContent = "✅ Done";
      status.className = "status-line status-success";
      $("runBtn").disabled = false;
      loadResult(jobId);
      return;
    }
    status.textContent = `⏳ ${job.progress}`;
    status.className = "status-line";
  }, 2000);
}

async function loadResult(jobId) {
  const res = await fetch(`/api/jobs/${jobId}/result`);
  const data = await res.json();

  $("resultsPanel").classList.remove("hidden");
  $("summaryLine").textContent =
    `${data.num_speakers} speaker(s), ${data.segments.length} segments` +
    (data.detected_language ? ` — language: ${data.detected_language} (${Math.round(data.language_probability * 100)}%)` : "");

  renderTimeline(data.segments);
  renderTable("segmentsTable", data.segments, (r) => [r.start, r.end, r.duration, r.speaker]);
  renderTable("statsTable", data.speaker_stats, (r) => [r.speaker, r.segments, r.total_time, r.avg_segment, r.percentage]);

  const transcriptSection = $("transcriptSection");
  if (data.transcript_text) {
    transcriptSection.classList.remove("hidden");
    $("transcriptBox").textContent = data.transcript_text;
    renderTable("alignedTable", data.speaker_aligned_transcript || [], (r) => [r.start, r.end, r.speaker, r.text]);
    $("dlTranscript").href = `/api/jobs/${jobId}/transcript.txt`;
    $("dlTranscript").classList.remove("hidden");
  } else {
    transcriptSection.classList.add("hidden");
    $("dlTranscript").classList.add("hidden");
  }

  $("dlRttm").href = `/api/jobs/${jobId}/rttm`;
  $("dlCsv").onclick = (e) => {
    e.preventDefault();
    downloadCsv(data.segments);
  };
}

function renderTimeline(segments) {
  const el = $("timeline");
  el.textContent = "";
  if (!segments.length) return;

  const speakers = [...new Set(segments.map((s) => s.speaker))];
  const colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"];
  const colorFor = (spk) => colors[speakers.indexOf(spk) % colors.length];
  const totalEnd = Math.max(...segments.map((s) => s.end));

  for (const spk of speakers) {
    const row = document.createElement("div");
    row.className = "timeline-row";

    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = spk;
    row.appendChild(label);

    const track = document.createElement("div");
    track.className = "timeline-track";
    for (const seg of segments.filter((s) => s.speaker === spk)) {
      const bar = document.createElement("div");
      bar.className = "timeline-seg";
      bar.style.left = `${(seg.start / totalEnd) * 100}%`;
      bar.style.width = `${Math.max((seg.duration / totalEnd) * 100, 0.3)}%`;
      bar.style.background = colorFor(spk);
      bar.title = `${seg.start}s - ${seg.end}s`;
      track.appendChild(bar);
    }
    row.appendChild(track);
    el.appendChild(row);
  }
}

function renderTable(tableId, rows, toCells) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.textContent = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of toCells(row)) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function downloadCsv(segments) {
  const header = "start,end,duration,speaker\n";
  const body = segments.map((s) => `${s.start},${s.end},${s.duration},${s.speaker}`).join("\n");
  const blob = new Blob([header + body], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "segments.csv";
  a.click();
  URL.revokeObjectURL(url);
}
