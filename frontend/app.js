const $ = (id) => document.getElementById(id);

function formatDuration(seconds) {
  if (seconds == null) return "";
  seconds = Math.round(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

function formatCacheAge(seconds) {
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

async function fetchChannel(forceRefresh) {
  const url = $("channelUrl").value.trim();
  const status = $("channelStatus");
  const results = $("channelResults");

  if (!url) {
    status.textContent = "Paste a channel URL first.";
    status.className = "status-line status-error";
    return;
  }

  results.classList.add("hidden");
  status.textContent = forceRefresh ? "Refreshing video list…" : "Fetching video list…";
  status.className = "status-line";
  $("fetchChannelBtn").disabled = true;
  $("refreshChannelBtn").disabled = true;

  try {
    const params = { url };
    if (forceRefresh) params.force_refresh = "true";
    const res = await fetch(`/api/youtube/channel?${new URLSearchParams(params)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    status.textContent = `✅ Found ${data.videos.length} video(s)`;
    status.className = "status-line status-success";
    $("channelTitle").textContent = data.channel_title;
    $("channelCacheInfo").textContent = data.cached
      ? ` (cached, fetched ${formatCacheAge(data.cache_age_seconds)})`
      : " (just fetched)";

    const list = $("channelVideoList");
    list.textContent = "";
    for (const video of data.videos) {
      const item = document.createElement("div");
      item.className = "channel-video-item";

      if (video.thumbnail) {
        const img = document.createElement("img");
        img.src = video.thumbnail;
        img.alt = "";
        item.appendChild(img);
      }

      const info = document.createElement("div");
      info.className = "cv-info";
      const title = document.createElement("div");
      title.className = "cv-title";
      title.textContent = video.title;
      title.title = video.title;
      const meta = document.createElement("div");
      meta.className = "cv-meta";
      meta.textContent = [
        formatDuration(video.duration),
        video.view_count != null ? `${video.view_count.toLocaleString()} views` : null,
      ].filter(Boolean).join(" · ");
      info.appendChild(title);
      info.appendChild(meta);
      item.appendChild(info);

      const selectBtn = document.createElement("button");
      selectBtn.type = "button";
      selectBtn.className = "secondary";
      selectBtn.textContent = "Select";
      selectBtn.onclick = () => {
        $("youtubeUrl").value = video.url;
        $("fileInput").value = "";
        $("youtubeUrl").scrollIntoView({ behavior: "smooth", block: "center" });
        $("youtubeUrl").focus();
      };
      item.appendChild(selectBtn);

      list.appendChild(item);
    }
    results.classList.remove("hidden");
  } catch (e) {
    status.textContent = `Failed to fetch channel: ${e.message}`;
    status.className = "status-line status-error";
  } finally {
    $("fetchChannelBtn").disabled = false;
    $("refreshChannelBtn").disabled = false;
  }
}

$("fetchChannelBtn").addEventListener("click", () => fetchChannel(false));
$("refreshChannelBtn").addEventListener("click", () => fetchChannel(true));

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
  const youtubeUrl = $("youtubeUrl").value.trim();
  const status = $("runStatus");

  const hasFile = fileInput.files.length > 0;
  if (hasFile && youtubeUrl) {
    status.textContent = "Choose a file OR a YouTube URL, not both.";
    status.className = "status-line status-error";
    return;
  }
  if (!hasFile && !youtubeUrl) {
    status.textContent = "Choose a file or paste a YouTube URL.";
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
  if (hasFile) {
    form.append("file", fileInput.files[0]);
  } else {
    form.append("youtube_url", youtubeUrl);
  }
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
    loadHistory();
  } catch (e) {
    status.textContent = `Failed to start job: ${e.message}`;
    status.className = "status-line status-error";
    $("runBtn").disabled = false;
  }
});

let downloadPollTimer = null;

$("downloadVideoBtn").addEventListener("click", async () => {
  const youtubeUrl = $("youtubeUrl").value.trim();
  const status = $("downloadStatus");

  if (!youtubeUrl) {
    status.textContent = "Paste a YouTube URL first.";
    status.className = "status-line status-error";
    return;
  }

  $("downloadResult").classList.add("hidden");
  $("downloadVideoBtn").disabled = true;
  if (downloadPollTimer) clearInterval(downloadPollTimer);

  const form = new FormData();
  form.append("youtube_url", youtubeUrl);

  status.textContent = "Starting download…";
  status.className = "status-line";

  try {
    const res = await fetch("/api/downloads", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const { job_id } = await res.json();
    pollDownload(job_id);
    loadHistory();
  } catch (e) {
    status.textContent = `Failed to start download: ${e.message}`;
    status.className = "status-line status-error";
    $("downloadVideoBtn").disabled = false;
  }
});

function pollDownload(jobId) {
  const status = $("downloadStatus");
  downloadPollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();

    if (job.status === "error") {
      clearInterval(downloadPollTimer);
      status.textContent = `❌ ${job.error}`;
      status.className = "status-line status-error";
      $("downloadVideoBtn").disabled = false;
      loadHistory();
      return;
    }
    if (job.status === "done") {
      clearInterval(downloadPollTimer);
      status.textContent = "✅ Done";
      status.className = "status-line status-success";
      $("downloadVideoBtn").disabled = false;
      $("downloadResult").classList.remove("hidden");
      $("downloadedVideoPlayer").src = `/api/jobs/${jobId}/video`;
      $("dlVideo").href = `/api/jobs/${jobId}/video`;
      loadHistory();
      return;
    }
    status.textContent = `⏳ ${job.progress}`;
    status.className = "status-line";
  }, 2000);
}

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
      loadHistory();
      return;
    }
    if (job.status === "done") {
      clearInterval(pollTimer);
      status.textContent = "✅ Done";
      status.className = "status-line status-success";
      $("runBtn").disabled = false;
      loadResult(jobId);
      loadHistory();
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

  if (data.has_video) {
    $("videoSection").classList.remove("hidden");
    $("resultVideoPlayer").src = `/api/jobs/${jobId}/video`;
    $("dlResultVideo").href = `/api/jobs/${jobId}/video`;
    $("dlResultVideo").classList.remove("hidden");
  } else {
    $("videoSection").classList.add("hidden");
    $("resultVideoPlayer").removeAttribute("src");
    $("dlResultVideo").classList.add("hidden");
  }

  if (data.has_audio) {
    $("audioSection").classList.remove("hidden");
    $("audioPlayer").src = `/api/jobs/${jobId}/audio`;
    $("dlAudio").href = `/api/jobs/${jobId}/audio`;
    $("dlAudio").classList.remove("hidden");
  } else {
    $("audioSection").classList.add("hidden");
    $("audioPlayer").removeAttribute("src");
    $("dlAudio").classList.add("hidden");
  }

  renderTimeline(data.segments, data.total_duration);
  renderTable("segmentsTable", data.segments, (r) => [r.start, r.end, r.duration, r.speaker]);
  renderTable("statsTable", data.speaker_stats, (r) => [r.speaker, r.segments, r.total_time, r.avg_segment, r.percentage]);

  const transcriptSection = $("transcriptSection");
  if (data.transcript_text) {
    transcriptSection.classList.remove("hidden");
    $("transcriptBox").textContent = data.transcript_text;
    renderTable("alignedTable", data.speaker_aligned_transcript || [], (r) => [r.start, r.end, r.speaker, r.text]);
    $("dlTranscript").href = `/api/jobs/${jobId}/transcript.txt`;
    $("dlTranscript").classList.remove("hidden");
    $("noTranscriptNote").classList.add("hidden");
  } else {
    transcriptSection.classList.add("hidden");
    $("dlTranscript").classList.add("hidden");
    $("noTranscriptNote").classList.remove("hidden");
  }

  $("dlRttm").href = `/api/jobs/${jobId}/rttm`;
  $("dlCsv").onclick = (e) => {
    e.preventDefault();
    downloadCsv(data.segments);
  };
}

function renderTimeline(segments, totalDuration) {
  const el = $("timeline");
  el.textContent = "";
  if (!segments.length) return;

  const speakers = [...new Set(segments.map((s) => s.speaker))];
  const colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"];
  const colorFor = (spk) => colors[speakers.indexOf(spk) % colors.length];
  // Prefer the media's real duration (from ffprobe server-side) so the
  // timeline's full width matches the actual video/audio length. Falling
  // back to the last segment's end time undershoots whenever there's
  // trailing silence after the last detected speech segment.
  const totalEnd = totalDuration || Math.max(...segments.map((s) => s.end));

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

const STATUS_ICON = { queued: "⏳", running: "⏳", done: "✅", error: "❌" };
const KIND_LABEL = { diarize: "🎙️ Diarize", download: "🎬 Download" };

async function loadHistory() {
  const tbody = document.querySelector("#historyTable tbody");
  const res = await fetch("/api/jobs");
  if (!res.ok) return;
  const jobs = await res.json();

  tbody.textContent = "";
  for (const job of jobs) {
    const tr = document.createElement("tr");

    const whenTd = document.createElement("td");
    whenTd.textContent = new Date(job.created_at * 1000).toLocaleString();
    tr.appendChild(whenTd);

    const kindTd = document.createElement("td");
    kindTd.textContent = KIND_LABEL[job.kind] || job.kind;
    tr.appendChild(kindTd);

    const sourceTd = document.createElement("td");
    sourceTd.textContent = job.file_name;
    sourceTd.title = job.file_name;
    tr.appendChild(sourceTd);

    const statusTd = document.createElement("td");
    statusTd.textContent = `${STATUS_ICON[job.status] || ""} ${job.status}`;
    tr.appendChild(statusTd);

    const linkTd = document.createElement("td");
    const link = document.createElement("a");
    // A finished download job links straight to the file; everything else
    // (including a still-running download, to show its progress) opens the
    // SPA view via ?job=.
    link.href = (job.kind === "download" && job.status === "done")
      ? `/api/jobs/${job.job_id}/video`
      : `?job=${job.job_id}`;
    link.target = "_blank";
    link.className = "dl";
    link.textContent = "Open ↗";
    linkTd.appendChild(link);
    tr.appendChild(linkTd);

    tbody.appendChild(tr);
  }
}

$("refreshHistoryBtn").addEventListener("click", loadHistory);

async function loadJobFromQueryParam() {
  const jobId = new URLSearchParams(location.search).get("job");
  if (!jobId) return;

  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) {
    $("runStatus").textContent = "That job wasn't found (server may have restarted since).";
    $("runStatus").className = "status-line status-error";
    return;
  }
  const job = await res.json();

  if (job.kind === "download") {
    if (job.status === "done") {
      $("downloadResult").classList.remove("hidden");
      $("downloadedVideoPlayer").src = `/api/jobs/${jobId}/video`;
      $("dlVideo").href = `/api/jobs/${jobId}/video`;
      $("downloadStatus").textContent = "✅ Done";
      $("downloadStatus").className = "status-line status-success";
    } else if (job.status === "error") {
      $("downloadStatus").textContent = `❌ ${job.error}`;
      $("downloadStatus").className = "status-line status-error";
    } else {
      pollDownload(jobId);
    }
    return;
  }

  if (job.status === "done") {
    loadResult(jobId);
  } else if (job.status === "error") {
    $("runStatus").textContent = `❌ ${job.error}`;
    $("runStatus").className = "status-line status-error";
  } else {
    pollJob(jobId);
  }
}

async function checkCookiesHealth() {
  const line = $("cookiesHealthLine");
  try {
    const res = await fetch("/api/youtube/health");
    const data = await res.json();
    line.classList.remove("hidden");
    if (data.ok) {
      line.textContent = "✅ YouTube cookies OK";
      line.className = "status-line status-success";
    } else {
      line.textContent = `⚠️ ${data.detail}`;
      line.className = "status-line status-error";
    }
  } catch (e) {
    // Non-critical — just skip showing the banner if the check itself fails.
  }
}

loadJobFromQueryParam();
loadHistory();
checkCookiesHealth();
