/**
 * crypo-side UI — 扫描 + 结果 + 关系图（悬停详情、点击钻取）
 */

const API_BASE = "";

const $ = (sel) => document.querySelector(sel);

function domainFromUrl(url) {
  try {
    const u = new URL(url.startsWith("http") ? url : `https://${url}`);
    return u.hostname.replace(/^www\./, "");
  } catch {
    return url.replace(/^https?:\/\//, "").split("/")[0].replace(/^www\./, "");
  }
}

function verdictLevel(prob) {
  if (prob < 0.3) return { key: "safe", label: "低风险", color: "var(--vt-safe)" };
  if (prob < 0.7) return { key: "suspicious", label: "可疑", color: "var(--vt-warn)" };
  return { key: "malicious", label: "高风险", color: "var(--vt-danger)" };
}

function formatPct(p) {
  return `${Math.round(p * 100)}%`;
}

function parseGangHint(hint) {
  if (!hint) return { louvain: "—", fraudNeighbors: "—" };
  const out = { louvain: "—", fraudNeighbors: "—" };
  for (const part of hint.split(",")) {
    const [k, v] = part.split(":");
    if (k === "louvain") out.louvain = v || "—";
    if (k === "fraud_neighbors") out.fraudNeighbors = v || "—";
  }
  return out;
}

function modeLabel(mode) {
  return mode === "gnn" ? "GraphSAGE（图卷积）" : "启发式";
}

function modeShortLabel(mode) {
  return mode === "gnn" ? "图卷积" : "启发式";
}

function setVerdictRing(pct, color) {
  const ring = $("#verdict-ring");
  const fill = ring?.querySelector(".fill");
  const pctEl = $("#verdict-pct");
  if (!fill || !pctEl) return;
  const r = 42;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - pct / 100);
  fill.style.strokeDasharray = `${c}`;
  fill.style.strokeDashoffset = String(offset);
  fill.style.stroke = color;
  pctEl.textContent = `${pct}%`;
}

async function predictUrl(url) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, label: null }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || res.statusText || "请求失败");
  }
  return data;
}

/** @typedef {{ label: string, domain?: string, url?: string, centerId?: string }} DrillCrumb */

let network = null;
let lastResult = null;
/** @type {Map<string, object>} */
let graphNodeById = new Map();
let graphCenterId = null;
/** @type {DrillCrumb[]} */
let drillStack = [];
let currentGraphQuery = null;

function buildGraphQueryParams(query) {
  const params = new URLSearchParams();
  if (query.centerId) params.set("center_id", query.centerId);
  else if (query.url) params.set("url", query.url);
  else if (query.domain) params.set("domain", query.domain);
  return params;
}

async function fetchGraph(query) {
  const params = buildGraphQueryParams(query);
  const res = await fetch(`${API_BASE}/graph?${params}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || res.statusText || "无法加载关系图");
  }
  return data;
}

function nodeDisplayLabel(n) {
  const p = n.properties || {};
  if (p.domain) return p.domain;
  if (p.url) return domainFromUrl(p.url) || p.url.slice(0, 28);
  if (p.address) return p.address;
  if (p.email) return p.email;
  if (p.ns_name) return p.ns_name;
  if (p.tracking_id) return String(p.tracking_id).slice(0, 20);
  if (p.fingerprint) return String(p.fingerprint).slice(0, 16) + "…";
  return (n.labels && n.labels[0]) || "?";
}

function nodeColor(n) {
  const p = n.properties || {};
  const prob = p.gnn_fraud_prob;
  if (prob == null) return "#94a3b8";
  if (prob < 0.3) return "#16a34a";
  if (prob < 0.7) return "#d97706";
  return "#dc2626";
}

const PROP_LABELS = {
  url: "URL",
  domain: "域名",
  label: "数据集标签",
  gnn_fraud_prob: "欺诈概率",
  pagerank: "PageRank",
  louvain_community: "Louvain 社区",
  address: "IP 地址",
  email: "注册邮箱",
  ns_name: "NS 主机名",
  tracking_id: "Analytics ID",
  fingerprint: "证书指纹",
  registrar: "注册商",
  creation_date: "注册日期",
  title: "页面标题",
  status_code: "HTTP 状态",
  content_length: "内容长度",
};

function formatPropValue(key, val) {
  if (val == null || val === "") return null;
  if (key === "gnn_fraud_prob" && typeof val === "number") return formatPct(val);
  if (typeof val === "number" && (key === "pagerank" || key.includes("score"))) {
    return Number(val).toFixed(4);
  }
  if (typeof val === "boolean") return val ? "是" : "否";
  const s = String(val);
  return s.length > 200 ? s.slice(0, 200) + "…" : s;
}

function buildNodeTooltip(n) {
  const p = n.properties || {};
  const lines = [`【${(n.labels || []).join(" / ")}】`];
  const keys = Object.keys(p).sort();
  for (const k of keys) {
    if (["text_embedding", "feat_vector", "graph_embedding"].includes(k)) continue;
    const formatted = formatPropValue(k, p[k]);
    if (formatted == null) continue;
    const label = PROP_LABELS[k] || k;
    lines.push(`${label}: ${formatted}`);
  }
  if (lines.length === 1) lines.push("（无额外属性）");
  return lines.join("\n");
}

function showNodeDetailPanel(n) {
  const panel = $("#graph-node-detail");
  if (!panel || !n) return;
  const p = n.properties || {};
  const rows = [];
  rows.push(`<dt>类型</dt><dd>${(n.labels || []).join(", ")}</dd>`);
  const keys = Object.keys(p).sort();
  for (const k of keys) {
    if (["text_embedding", "feat_vector", "graph_embedding"].includes(k)) continue;
    const formatted = formatPropValue(k, p[k]);
    if (formatted == null) continue;
    rows.push(`<dt>${PROP_LABELS[k] || k}</dt><dd>${escapeHtml(formatted)}</dd>`);
  }
  panel.innerHTML = `<strong>${escapeHtml(nodeDisplayLabel(n))}</strong><dl>${rows.join("")}</dl>`;
  panel.classList.remove("hidden");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function hideNodeDetailPanel() {
  $("#graph-node-detail")?.classList.add("hidden");
}

function drillQueryFromNode(n) {
  const p = n.properties || {};
  const labels = n.labels || [];
  const label = nodeDisplayLabel(n);

  if (labels.includes("Website")) {
    if (p.url) return { label, url: p.url };
    if (p.domain) return { label, domain: p.domain };
    return { label, domain: label };
  }
  return { label, centerId: n.id };
}

function renderBreadcrumb() {
  const nav = $("#graph-breadcrumb");
  const backBtn = $("#graph-drill-back");
  if (!nav) return;

  if (drillStack.length <= 1) {
    nav.classList.add("hidden");
    backBtn?.classList.add("hidden");
    return;
  }

  nav.classList.remove("hidden");
  backBtn?.classList.remove("hidden");

  const parts = drillStack.map((crumb, i) => {
    const isLast = i === drillStack.length - 1;
    if (isLast) {
      return `<span class="current">${escapeHtml(crumb.label)}</span>`;
    }
    return `<button type="button" class="crumb" data-idx="${i}">${escapeHtml(crumb.label)}</button>`;
  });
  nav.innerHTML = parts.join('<span class="sep">›</span>');

  nav.querySelectorAll("button.crumb").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.idx);
      drillStack = drillStack.slice(0, idx + 1);
      const target = drillStack[drillStack.length - 1];
      loadGraph(target, { pushStack: false });
    });
  });
}

function renderGraph(graphData) {
  const container = document.getElementById("graph-viz");
  if (!container || typeof vis === "undefined") return;

  graphNodeById = new Map();
  graphCenterId = graphData.center_id || null;

  for (const n of graphData.nodes || []) {
    graphNodeById.set(n.id, n);
  }

  const nodes = new vis.DataSet(
    (graphData.nodes || []).map((n) => {
      const isCenter = n.id === graphCenterId;
      const bg = nodeColor(n);
      return {
        id: n.id,
        label: nodeDisplayLabel(n),
        group: (n.labels && n.labels[0]) || "Node",
        size: isCenter ? 22 : n.labels?.includes("Website") ? 18 : 14,
        borderWidth: isCenter ? 4 : 2,
        color: {
          background: bg,
          border: isCenter ? "#1e293b" : "#fff",
          highlight: { background: bg, border: "#394eff" },
        },
        font: { color: "#fff", size: isCenter ? 12 : 11 },
        title: buildNodeTooltip(n),
        _raw: n,
      };
    })
  );

  const edges = new vis.DataSet(
    (graphData.edges || []).map((e, i) => ({
      id: i,
      from: e.source,
      to: e.target,
      label: e.type || "",
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      font: { size: 9, align: "middle" },
      color: { color: "#cbd5e1", highlight: "#394eff" },
    }))
  );

  const options = {
    nodes: {
      shape: "dot",
      shadow: true,
    },
    edges: {
      smooth: { type: "continuous" },
      width: 1.5,
    },
    physics: {
      stabilization: { iterations: 150 },
      barnesHut: { gravitationalConstant: -4000, springLength: 120 },
    },
    interaction: { hover: true, tooltipDelay: 80, selectConnectedEdges: true },
    groups: {
      Website: { shape: "dot" },
      IP: { shape: "square" },
      NameServer: { shape: "diamond" },
      RegistrantEmail: { shape: "triangle" },
    },
  };

  if (network) {
    network.destroy();
    network = null;
  }

  if (!graphData.nodes?.length) {
    container.innerHTML =
      '<p style="padding:48px;text-align:center;color:#64748b">暂无图数据，请确认 Neo4j 中已有该站点或关联节点。</p>';
    hideNodeDetailPanel();
    return;
  }

  container.innerHTML = "";
  network = new vis.Network(container, { nodes, edges }, options);

  network.on("click", (params) => {
    if (!params.nodes.length) return;
    const nodeId = params.nodes[0];
    const raw = graphNodeById.get(nodeId);
    if (!raw) return;
    showNodeDetailPanel(raw);
    drillIntoNode(raw);
  });

  network.on("hoverNode", (params) => {
    const raw = graphNodeById.get(params.node);
    if (raw) showNodeDetailPanel(raw);
  });

  network.once("stabilizationIterationsDone", () => {
    if (graphCenterId) network.focus(graphCenterId, { scale: 1.1, animation: true });
    else network.fit({ animation: true });
  });
}

async function drillIntoNode(n) {
  const next = drillQueryFromNode(n);
  const top = drillStack[drillStack.length - 1];
  const same =
    top &&
    ((next.centerId && top.centerId === next.centerId) ||
      (next.url && top.url === next.url) ||
      (next.domain && top.domain === next.domain && !next.url && !next.centerId));
  if (same) return;

  await loadGraph(next, { pushStack: true });
}

/**
 * @param {DrillCrumb} query
 * @param {{ pushStack?: boolean }} opts
 */
async function loadGraph(query, opts = {}) {
  const { pushStack = false } = opts;
  const status = $("#graph-status");

  if (pushStack) {
    drillStack.push({ ...query });
  } else if (!drillStack.length) {
    drillStack = [{ ...query }];
  } else {
    drillStack[drillStack.length - 1] = { ...query };
  }

  currentGraphQuery = { ...query };
  renderBreadcrumb();
  syncGraphDomainInputs(query);

  try {
    status.textContent = "加载中…";
    const graph = await fetchGraph(query);
    const n = (graph.nodes || []).length;
    const e = (graph.edges || []).length;
    const centerLabel =
      graph.nodes?.find((x) => x.id === graph.center_id)?.properties?.domain ||
      drillStack[drillStack.length - 1]?.label ||
      "—";
    status.textContent = n
      ? `中心: ${centerLabel} · ${n} 节点 · ${e} 关系`
      : "暂无关联数据（请先完成扫描或导入图库）";
    renderGraph(graph);
  } catch (err) {
    status.textContent = err.message || "加载失败";
    if (pushStack) drillStack.pop();
    renderBreadcrumb();
  }
}

function syncGraphDomainInputs(query) {
  const val = query.domain || (query.url ? domainFromUrl(query.url) : "") || query.label || "";
  const empty = $("#graph-domain-input");
  const inline = $("#graph-domain-input-inline");
  if (empty && document.activeElement !== empty) empty.value = val;
  if (inline && document.activeElement !== inline) inline.value = val;
}

function resetDrillStack(query) {
  drillStack = [{ ...query }];
  currentGraphQuery = { ...query };
  renderBreadcrumb();
}

function showGraphSection(graphOnly = false) {
  $("#empty-state")?.classList.add("hidden");
  const section = $("#result-section");
  section?.classList.remove("hidden");
  section?.classList.toggle("graph-explore-only", graphOnly);
}

function resolveDomainInput(raw) {
  let s = (raw || "").trim();
  if (!s) return null;
  if (!/^https?:\/\//i.test(s)) s = `https://${s}`;
  try {
    const u = new URL(s);
    return { domain: u.hostname.replace(/^www\./, ""), label: u.hostname.replace(/^www\./, "") };
  } catch {
    const domain = s.replace(/^https?:\/\//, "").split("/")[0].replace(/^www\./, "");
    return { domain, label: domain };
  }
}

async function onLoadGraphOnly(fromInline = false) {
  const input = fromInline ? $("#graph-domain-input-inline") : $("#graph-domain-input");
  const query = resolveDomainInput(input?.value);
  if (!query) {
    showError("请输入域名或 URL");
    return;
  }
  hideError();
  showGraphSection(true);
  resetDrillStack(query);
  await loadGraph(query, { pushStack: false });
  document.querySelector('.tab-btn[data-tab="graph"]')?.click();
}

function showResult(data) {
  lastResult = data;
  const prob = data.fraud_probability;
  const verdict = verdictLevel(prob);
  const domain = domainFromUrl(data.url);
  const gang = parseGangHint(data.gang_hint);

  showGraphSection(false);

  $("#result-domain").textContent = domain;
  $("#result-full-url").textContent = data.url;

  const badge = $("#verdict-badge");
  badge.textContent = verdict.label;
  badge.className = `verdict-badge ${verdict.key}`;

  setVerdictRing(Math.round(prob * 100), verdict.color);

  $("#stat-prob").textContent = formatPct(prob);
  $("#stat-mode").textContent = modeShortLabel(data.mode);
  $("#stat-neighbors").textContent = gang.fraudNeighbors;
  $("#stat-community").textContent = gang.louvain;

  $("#detail-mode").textContent = modeLabel(data.mode);
  $("#detail-prob").textContent = `${prob.toFixed(4)} (${formatPct(prob)})`;
  $("#detail-gang").textContent = data.gang_hint || "无";
  $("#detail-url").textContent = data.url;

  $("#json-output").textContent = JSON.stringify(data, null, 2);

  resetDrillStack({ domain, url: data.url, label: domain });
  loadGraph({ domain, url: data.url, label: domain }, { pushStack: false });
}

function showError(msg) {
  const el = $("#alert-box");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden");
  el.className = "alert error";
}

function hideError() {
  $("#alert-box")?.classList.add("hidden");
}

async function onScan() {
  const input = $("#url-input");
  const btn = $("#scan-btn");
  const row = $(".search-row");
  let url = (input?.value || "").trim();
  if (!url) {
    showError("请输入 URL 或域名");
    return;
  }
  if (!/^https?:\/\//i.test(url)) url = `https://${url}`;

  hideError();
  btn.disabled = true;
  row?.classList.add("loading");

  try {
    const data = await predictUrl(url);
    showResult(data);
    if (input) input.value = data.url;
  } catch (err) {
    showError(err.message || "分析失败，请检查服务与 Neo4j 连接");
  } finally {
    btn.disabled = false;
    row?.classList.remove("loading");
  }
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.id === `panel-${tab}`);
      });
      if (tab === "graph" && network) network.fit({ animation: true });
    });
  });
}

function initGraphControls() {
  $("#graph-load-btn")?.addEventListener("click", () => onLoadGraphOnly(false));
  $("#graph-load-btn-inline")?.addEventListener("click", () => onLoadGraphOnly(true));
  $("#graph-domain-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onLoadGraphOnly(false);
  });
  $("#graph-domain-input-inline")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onLoadGraphOnly(true);
  });
  $("#graph-refresh")?.addEventListener("click", () => {
    if (currentGraphQuery) loadGraph(currentGraphQuery, { pushStack: false });
    else if (lastResult) {
      const domain = domainFromUrl(lastResult.url);
      resetDrillStack({ domain, url: lastResult.url, label: domain });
      loadGraph({ domain, url: lastResult.url, label: domain }, { pushStack: false });
    }
  });
  $("#graph-drill-back")?.addEventListener("click", () => {
    if (drillStack.length <= 1) return;
    drillStack.pop();
    loadGraph(drillStack[drillStack.length - 1], { pushStack: false });
  });
}

function init() {
  $("#scan-btn")?.addEventListener("click", onScan);
  $("#url-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onScan();
  });
  initTabs();
  initGraphControls();

  const params = new URLSearchParams(location.search);
  const qUrl = params.get("url");
  const qDomain = params.get("domain");
  if (qUrl) {
    const input = $("#url-input");
    if (input) {
      input.value = qUrl;
      onScan();
    }
  } else if (qDomain) {
    const empty = $("#graph-domain-input");
    if (empty) {
      empty.value = qDomain;
      onLoadGraphOnly(false);
    }
  }
}

document.addEventListener("DOMContentLoaded", init);
