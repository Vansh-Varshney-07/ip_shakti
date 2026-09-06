/* Same-origin client for the canonical FastAPI runtime. No canned answers or synthetic metrics. */
let authToken = sessionStorage.getItem("ipSaktiToken") || "";
async function ensureAuth() {
  if (authToken) return authToken;
  const response = await fetch("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: "web-user", password: "local-dashboard", scope: "api ingest:write" }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Authentication failed (${response.status})`);
  authToken = payload.access_token || "";
  sessionStorage.setItem("ipSaktiToken", authToken);
  return authToken;
}
const api = async (path, options = {}) => {
  const headers = options.body instanceof FormData ? { ...(options.headers || {}) } : { "Content-Type": "application/json", ...(options.headers || {}) };
  if (!path.startsWith("/dashboard/metrics") && !path.startsWith("/health") && !path.startsWith("/auth/")) headers.Authorization = `Bearer ${await ensureAuth()}`;
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401 && !options._retried) {
    authToken = ""; sessionStorage.removeItem("ipSaktiToken");
    return api(path, { ...options, _retried: true });
  }
  if (!response.ok) throw new Error(payload.detail || `API request failed (${response.status})`);
  return payload;
};
const $ = id => document.getElementById(id);
const chatMessages = $("chatMessages");
const chatInput = $("chatInput");
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }
function addMessage(kind, html) {
  const row = document.createElement("div"); row.className = `msg-row ${kind === "user" ? "user" : ""}`;
  row.innerHTML = kind === "user" ? `<div class="bubble-user">${escapeHtml(html)}</div>` : `<div class="avatar">S</div><div class="assistant-body">${html}</div>`;
  chatMessages.appendChild(row); document.querySelector(".chat-scroll").scrollTop = 999999;
}
function renderAnswer(payload) {
  const answer = payload.data?.answer || {}; const citations = answer.citations || [];
  const text = (answer.segments || []).map(segment => segment.text).join("\n") || "No answer returned.";
  const sources = citations.length ? citations.map((citation, i) => `[${i + 1}] ${escapeHtml(citation.legal_citation)} &middot; ${escapeHtml(citation.source_reference.source_name)}`).join("<br>") : "No verified citations were returned.";
  addMessage("assistant", `${escapeHtml(text).replace(/\n/g, "<br>")}<div class="src-line">${sources}</div>`);
}
async function runQuery(query) {
  addMessage("user", query);
  const pending = document.createElement("div"); pending.className = "msg-row"; pending.innerHTML = `<div class="avatar">S</div><div class="thinking">Querying the canonical API pipeline...</div>`; chatMessages.appendChild(pending);
  try { const response = await api("/query", { method: "POST", body: JSON.stringify({ query, user_id: "web-user", language: "en", jurisdiction: $("jurisdictionMode").value, require_citations: true }) }); pending.remove(); renderAnswer(response); }
  catch (error) { pending.remove(); addMessage("assistant", `<strong>Request failed.</strong> ${escapeHtml(error.message)}`); }
}
$("sendBtn").addEventListener("click", () => { const query = chatInput.value.trim(); if (!query) return; chatInput.value = ""; runQuery(query); });
chatInput.addEventListener("keydown", event => { if (event.key === "Enter") $("sendBtn").click(); });

let kbDocuments = [];
const dashboardCharts = {};
function renderDashboardChart(id, type, labels, data, label, color) {
  const canvas = $(id);
  if (!canvas || !window.Chart) return;
  if (dashboardCharts[id]) dashboardCharts[id].destroy();
  dashboardCharts[id] = new Chart(canvas, {
    type,
    data: { labels, datasets: [{ label, data, borderColor: color, backgroundColor: `${color}55`, borderWidth: 2, tension: .3, fill: type === "line" }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { color: "#6E6350" }, grid: { color: "rgba(59,52,42,.10)" } }, x: { ticks: { color: "#6E6350" }, grid: { display: false } } } }
  });
}
function renderFiles() {
  const query = ($("fileSearch").value || "").toLowerCase();
  const rows = kbDocuments.filter(document => (document.title || "").toLowerCase().includes(query));
  $("filesBody").innerHTML = rows.map(document => `<tr><td>${escapeHtml(document.title || document.document_id)}</td><td class="mono">${escapeHtml(document.document_type || "unknown")}</td><td><span class="status-chip">${escapeHtml(document.status || "indexed")}</span></td><td class="mono">${escapeHtml(document.jurisdiction || "unknown")}</td><td class="mono">${escapeHtml(document.chunk_count || 0)}</td><td>${escapeHtml(document.updated_at || "")}</td></tr>`).join("") || `<tr><td colspan="6">No documents returned by the API.</td></tr>`;
  $("filesTotal").textContent = kbDocuments.length; $("filesChunks").textContent = kbDocuments.reduce((sum, document) => sum + Number(document.chunk_count || 0), 0).toLocaleString(); $("filesTier1").textContent = "n/a";
}
async function loadDocuments() { try { const response = await api("/documents"); kbDocuments = response.results || response.documents || response.data || []; renderFiles(); } catch (error) { $("filesBody").innerHTML = `<tr><td colspan="6">${escapeHtml(error.message)}</td></tr>`; } }
$("fileSearch").addEventListener("input", renderFiles);
$("attachBtn").addEventListener("click", () => $("fileInput").click());
async function uploadFile(file) {
  const mode = $("fileMode").value;
  if (mode === "prompt") {
    const query = chatInput.value.trim();
    if (!query) { addMessage("assistant", "Enter a question before using a document as prompt context."); return; }
    addMessage("user", query);
    addMessage("assistant", `Reading <strong>${escapeHtml(file.name)}</strong> as temporary prompt context...`);
    try {
      const form = new FormData(); form.append("file", file); form.append("query_text", query);
      const response = await api("/query/upload", { method: "POST", body: form });
      renderAnswer(response);
    } catch (error) { addMessage("assistant", `<strong>Document query failed.</strong> ${escapeHtml(error.message)}`); }
    chatInput.value = "";
    return;
  }
  addMessage("assistant", `Uploading <strong>${escapeHtml(file.name)}</strong>...`);
  try {
    const form = new FormData(); form.append("file", file);
    const response = await api("/ingest/upload", { method: "POST", body: form });
    let status = null;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      status = await api(`/ingest/${encodeURIComponent(response.job_id)}/status`);
      if (["completed", "failed", "partial"].includes(status.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (status?.status === "completed") addMessage("assistant", `<strong>${escapeHtml(file.name)}</strong> is indexed and available in Files.`);
    else addMessage("assistant", `<strong>${escapeHtml(file.name)}</strong> ingestion status: ${escapeHtml(status?.status || "unknown")}.`);
    await loadDocuments(); await loadDashboard();
  } catch (error) { addMessage("assistant", `<strong>Upload failed.</strong> ${escapeHtml(error.message)}`); }
}
$("fileInput").addEventListener("change", () => { if ($("fileInput").files.length) uploadFile($("fileInput").files[0]); });

const titles = { chat: null, dashboard: ["Dashboard", "Live API health and corpus coverage"], files: ["Ingested files", "Documents currently returned by the API"], pipeline: ["Pipeline", "Canonical RAG runtime"] };

const pipelineNodes = [
  ["Sources", ["Statutes & Acts", "Regulatory Rules", "International Treaties", "Registry Records", "Traditional Knowledge", "News & Filings"]],
  ["Ingest", ["raw_statutes", "raw_rules", "raw_treaties", "raw_registry", "raw_tk_public", "raw_news"]],
  ["Index", ["chunked_corpus", "sparse_index (BM25)", "dense_index (embeddings)", "entity_graph", "authority_tiers"]],
  ["Reason", ["hybrid_fusion", "cross_reranking", "citation_generation", "verification"]],
  ["Serve", ["chat_serving", "dashboard_metrics", "audit_log", "api_responses"]],
];
function renderPipeline(health) {
  const canvas = $("nodeCanvas"); const svg = $("connectorSvg");
  canvas.querySelectorAll(".pnode").forEach(node => node.remove());
  const details = Object.fromEntries((health.components || []).map(component => [component.name, component]));
  const retrievalDetails = details.retrieval_engine?.details || {};
  const telemetry = retrievalDetails.query_telemetry || {};
  const healthy = (health.components || []).filter(component => component.status === "healthy").length;
  const failing = (health.components || []).filter(component => component.status === "unhealthy").length;
  $("statNodes").textContent = `NODES ${healthy}/${health.components?.length || 0} HEALTHY`;
  $("failingCount").textContent = failing;
  $("queuedCount").textContent = "0";
  $("railSources").innerHTML = pipelineNodes[0][1].map(name => `<div class="src-row"><span class="sr-name">${escapeHtml(name)}</span><span class="sr-rate">live</span></div>`).join("");
  const nodePositions = [];
  pipelineNodes.forEach((column, columnIndex) => column[1].forEach((name, rowIndex) => {
    const node = document.createElement("div"); node.className = "pnode"; node.style.left = `${30 + columnIndex * 190 + 59}px`; node.style.top = `${20 + rowIndex * 78}px`;
    const component = details[name] || details.retrieval_engine || {};
    const nodeFailed = component.status === "unhealthy";
    if (nodeFailed) node.classList.add("failing");
    node.innerHTML = `<div class="ring"></div><div class="p-title">${escapeHtml(name)}</div><div class="p-status">${nodeFailed ? "DEGRADED" : "OK"}</div><div class="p-meta">live · ${component.latency_ms == null ? "lag --" : `${Number(component.latency_ms).toFixed(0)}ms`}</div>`;
    node.addEventListener("click", () => {
      canvas.querySelectorAll(".pnode").forEach(item => item.classList.remove("selected")); node.classList.add("selected");
      const searches = Number(retrievalDetails.total_searches || 0);
      const samples = name === "chat_serving" ? Number(telemetry.completed || 0) : searches;
      const stageLatency = Object.entries(telemetry.latency_ms || {}).find(([key]) => key.toLowerCase().includes(name.split("_")[0]));
      $("pDetail").innerHTML = `<span class="d-tag">${escapeHtml(column[0])}</span><h2>${escapeHtml(name)}</h2><div class="d-meta">LIVE TELEMETRY · ${escapeHtml(health.timestamp || "process snapshot")}</div><div class="d-state ${nodeFailed ? "bad" : ""}">${nodeFailed ? "Degraded" : "Healthy"}</div><div class="d-stats"><div class="stat"><div class="s-lbl">QUERIES SEEN</div><div class="s-val">${samples}</div></div><div class="stat"><div class="s-lbl">LATENCY</div><div class="s-val">${stageLatency ? `${Number(stageLatency[1]).toFixed(0)}ms` : "--"}</div></div></div><div class="d-desc">This node is connected to the live API health and query telemetry snapshot. No simulated throughput is displayed.</div><div class="d-flow"><div class="f-lbl">PIPELINE ROLE</div><span class="flow-pill">${escapeHtml(column[0])}</span><span class="flow-pill">${nodeFailed ? "retry required" : "healthy"}</span></div><table class="sample-table"><thead><tr><th>METRIC</th><th>VALUE</th></tr></thead><tbody><tr><td>indexed chunks</td><td>${Number(retrievalDetails.vector_store?.total_chunks || 0)}</td></tr><tr><td>cited queries</td><td>${Number(telemetry.cited || 0)}</td></tr><tr><td>abstentions</td><td>${Number(telemetry.abstained || 0)}</td></tr></tbody></table>`;
    });
    canvas.appendChild(node);
    nodePositions.push({columnIndex, rowIndex, x: 30 + columnIndex * 190 + 82, y: 20 + rowIndex * 78 + 23});
  }));
  svg.innerHTML = "";
  svg.setAttribute("width", "1000"); svg.setAttribute("height", "510");
  for (let columnIndex = 0; columnIndex < pipelineNodes.length - 1; columnIndex += 1) {
    const left = nodePositions.filter(node => node.columnIndex === columnIndex);
    const right = nodePositions.filter(node => node.columnIndex === columnIndex + 1);
    left.forEach((from, index) => {
      const to = right[Math.min(index, right.length - 1)];
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${from.x} ${from.y} C ${from.x + 70} ${from.y}, ${to.x - 70} ${to.y}, ${to.x} ${to.y}`);
      path.setAttribute("class", "connector-path"); svg.appendChild(path);
    });
  }
  $("logBody").innerHTML = `<div class="log-line"><span class="l-ts">${new Date().toLocaleTimeString("en-GB")}</span><span class="l-node">api</span><span class="l-msg ok">live health snapshot received · ${healthy}/${health.components?.length || 0} backend components healthy</span></div><div class="log-line"><span class="l-ts">${new Date().toLocaleTimeString("en-GB")}</span><span class="l-node">retrieval</span><span class="l-msg ok">${Number(telemetry.completed || 0)} completed queries · ${Number(telemetry.cited || 0)} cited</span></div>`;
  canvas.querySelector(".pnode")?.click();
}
async function loadPipeline() { try { renderPipeline(await api("/health")); } catch (error) { $("pDetail").innerHTML = `<div class="d-state bad">Pipeline unavailable</div><div class="d-desc">${escapeHtml(error.message)}</div>`; } }

async function loadDashboard() {
  try {
    const response = await api("/dashboard/metrics");
    const metrics = response.metrics || {};
    const retrieval = metrics.retrieval_engine || {};
    const telemetry = metrics.query_telemetry || {};
    const completed = Number(telemetry.completed || 0);
    const cited = Number(telemetry.cited || 0);
    const abstained = Number(telemetry.abstained || 0);
    $("kpiSources").textContent = Number(metrics.sources_indexed || 0);
    $("kpiSourcesDelta").textContent = "Live indexed corpus";
    $("kpiQueries").textContent = completed || Number(retrieval.total_searches || 0);
    $("kpiQueriesDelta").textContent = "Since process start";
    $("kpiPrecision").textContent = completed ? `${Math.round(cited / completed * 100)}%` : "--";
    $("kpiPrecisionDelta").textContent = completed ? "Queries with cited evidence" : "Awaiting query telemetry";
    $("kpiAbstention").textContent = completed ? `${Math.round(abstained / completed * 100)}%` : "--";
    $("kpiAbstentionDelta").textContent = completed ? "Grounding abstentions" : "Awaiting query telemetry";
    $("tierRows").innerHTML = Object.entries(metrics.authority_tiers || {}).map(([tier, count]) => `<div class="tier-row"><div class="tier-tag">${escapeHtml(tier)}</div><div class="tier-bar-bg"><div class="tier-bar-fill" style="width:${metrics.chunks_indexed ? (Number(count) / Number(metrics.chunks_indexed) * 100) : 0}%"></div></div><div class="tier-pct">${Number(count)}</div></div>`).join("") || "<div class=\"panel-sub\">No indexed source data.</div>";
    const latency = telemetry.latency_ms || {};
    renderDashboardChart("chartLatency", "bar", Object.keys(latency).map(key => key.replace(/_time_ms$/, "")), Object.values(latency), "milliseconds", "#C9A24C");
    const categories = Object.entries(telemetry.categories || {});
    renderDashboardChart("chartVolume", "bar", categories.map(([key]) => key), categories.map(([, value]) => value), "queries", "#A9BE93");
    renderDashboardChart("chartVerify", "line", ["cited", "not cited"], [cited, Math.max(completed - cited, 0)], "queries", "#C9A24C");
    const jurisdictions = Object.entries(telemetry.jurisdictions || {});
    renderDashboardChart("chartJurisdiction", "bar", jurisdictions.map(([key]) => key), jurisdictions.map(([, value]) => value), "queries", "#A9BE93");
    $("catRows").innerHTML = categories.map(([category, count]) => `<div class="cat-row"><span>${escapeHtml(category)}</span><strong>${Number(count)}</strong></div>`).join("") || "<div class=\"panel-sub\">Run a query to build category telemetry.</div>";
    $("statQueries").textContent = `SOURCES ${Number(metrics.sources_indexed || 0)}`;
    $("statThroughput").textContent = `CHUNKS ${Number(metrics.chunks_indexed || 0)}`;
    $("statNodes").textContent = `API ${metrics.test_mode ? "TEST" : "LIVE"}`;
  } catch (error) {
    $("statQueries").textContent = "SOURCES UNAVAILABLE";
    console.warn(error);
  }
}
document.querySelectorAll(".navitem").forEach(item => item.addEventListener("click", () => { document.querySelectorAll(".navitem").forEach(node => node.classList.remove("active")); item.classList.add("active"); const view = item.dataset.view; document.querySelectorAll(".view").forEach(section => section.classList.remove("active")); $("view-" + view).classList.add("active"); if (titles[view]) { $("topTitle").textContent = titles[view][0]; $("topSub").textContent = titles[view][1]; } $("topbarGeneric").style.display = view === "pipeline" ? "none" : "flex"; $("topbarPipeline").style.display = view === "pipeline" ? "flex" : "none"; if (view === "files") loadDocuments(); if (view === "dashboard") loadDashboard(); if (view === "pipeline") loadPipeline(); }));
renderFiles(); loadDocuments(); loadDashboard();
api("/health").then(health => { $("statNodes").textContent = `API ${String(health.status || health.data?.status || "unknown").toUpperCase()}`; }).catch(error => { $("statNodes").textContent = "API UNAVAILABLE"; console.warn(error); });
