/* ═══════════════════════════════════════════════
   ROAD-SENSE — Application Script
   ═══════════════════════════════════════════════ */

/* ─── Data ─── */
const docs = [
  { title: "Quick Setup Guide", description: "Install, validate, preprocess, and verify in a fast flow.", path: "./docs/QUICK_SETUP_GUIDE.md", category: "getting-started" },
  { title: "Project Details", description: "Full dataset context, workflow design, and milestone structure.", path: "./docs/PROJECT_DETAILS.md", category: "getting-started" },
  { title: "Preprocessing and Augmentation", description: "Details of data conversion and augmentation pipeline.", path: "./docs/PREPROCESSING_AND_AUGMENTATION_GUIDE.md", category: "dataset" },
  { title: "Dataset Exploration", description: "Dataset profile and supporting statistical analysis.", path: "./docs/DATASET_EXPLORATION_REPORT.md", category: "dataset" },
  { title: "Data Quality Report", description: "Validation checks for corruption, labels, and box constraints.", path: "./docs/data_quality_report.md", category: "dataset" },
  { title: "Dataset Download Instructions", description: "How to obtain and place KITTI and related datasets.", path: "./docs/DATASET_DOWNLOAD_INSTRUCTIONS.md", category: "dataset" },
  { title: "Dataset Upload Guidelines", description: "Repository data handling and large-file best practices.", path: "./docs/DATASET_UPLOAD_GUIDELINES.md", category: "dataset" },
  { title: "Training Report", description: "Training configuration, curves, and run-specific observations.", path: "./docs/TRAINING_REPORT_EXP34332.md", category: "training" },
  { title: "Multi-Dataset Training Strategy", description: "Plan for integrating additional datasets and training stages.", path: "./docs/MULTI_DATASET_TRAINING_STRATEGY.md", category: "training" },
  { title: "Model Comparison Report", description: "Model-level comparison details and performance perspective.", path: "./docs/models/MODEL_COMPARISON_REPORT.md", category: "models" },
  { title: "Project Proposal", description: "Overview, objectives, scope, and expected outcomes.", path: "./docs/project_planning/PROJECT_PROPOSAL.md", category: "project-planning" },
  { title: "Project Plan & Gantt Chart", description: "Timeline, milestones, Gantt chart, and resource allocation.", path: "./docs/project_planning/PROJECT_PLAN.md", category: "project-planning" },
  { title: "Task Assignment & Roles", description: "Defined responsibilities and task ownership for all members.", path: "./docs/project_planning/TASK_ASSIGNMENT.md", category: "project-planning" },
  { title: "Risk Assessment & Mitigation", description: "Identified risks, impact analysis, and mitigation strategies.", path: "./docs/project_planning/RISK_ASSESSMENT.md", category: "project-planning" },
  { title: "KPIs", description: "Key performance indicators for project success.", path: "./docs/project_planning/KPIS.md", category: "project-planning" },
  { title: "Feedback & Evaluation", description: "Advisor evaluation, peer assessment, and project strengths.", path: "./docs/literature_review/FEEDBACK_AND_EVALUATION.md", category: "project-planning" },
  { title: "Suggested Improvements", description: "Prioritized technical and process improvements.", path: "./docs/literature_review/SUGGESTED_IMPROVEMENTS.md", category: "project-planning" },
  { title: "Final Grading Criteria", description: "Grading breakdown, rubrics, scorecard, and timeline.", path: "./docs/literature_review/GRADING_CRITERIA.md", category: "project-planning" }
];

const reports = [
  { title: "Milestone 2 Executive Summary", description: "High-level training outcomes and readiness statement.", path: "./reports/MILESTONE_2_EXECUTIVE_SUMMARY_EXP34332.md", category: "milestones" },
  { title: "Milestone 2 Technical Report", description: "Technical details, metrics, and artifact inventory.", path: "./reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md", category: "milestones" },
  { title: "Reports Index", description: "One-stop map for all major reporting documents.", path: "./reports/REPORTS_INDEX.md", category: "status" },
  { title: "Project Status Report", description: "Progress across milestones and current priorities.", path: "./reports/PROJECT_STATUS_REPORT.md", category: "status" },
  { title: "Week 1 Deliverables", description: "Early milestone summary and key outputs.", path: "./reports/WEEK_1_DELIVERABLES_SUMMARY.md", category: "status" },
  { title: "Abdallah Dataset Analysis", description: "Research analysis for dataset options and rationale.", path: "./reports/research/Abdallah_dataset_analysis.md", category: "research" },
  { title: "Aya Dataset Analysis", description: "Research notes and comparative insights by Aya Ahmed.", path: "./reports/research/AyaAhmed_dataset_analysis.md", category: "research" },
  { title: "Dataset Analysis Template", description: "Template used for standardized analysis documentation.", path: "./reports/templates/dataset_analysis_template.md", category: "research" }
];

const visuals = [
  { title: "Class Distribution", src: "./experiments/visualization/dataset_analysis/class_distribution.png", category: "dataset" },
  { title: "Merged Class Distribution", src: "./experiments/visualization/dataset_analysis/class_distribution_after_merge.png", category: "dataset" },
  { title: "Bounding Box Area by Class", src: "./experiments/visualization/dataset_analysis/bbox_area_by_class.png", category: "dataset" },
  { title: "Bounding Box Width vs Height", src: "./experiments/visualization/dataset_analysis/bbox_width_vs_height.png", category: "dataset" },
  { title: "Occlusion Distribution", src: "./experiments/visualization/dataset_analysis/occlusion_distribution.png", category: "dataset" },
  { title: "Truncation Distribution", src: "./experiments/visualization/dataset_analysis/truncation_distribution.png", category: "dataset" },
  { title: "Results Curve", src: "./experiments/visualization/runsV/exp34332/results.png", category: "runs" },
  { title: "Confusion Matrix", src: "./experiments/visualization/runsV/exp34332/confusion_matrix.png", category: "runs" },
  { title: "Normalized Confusion Matrix", src: "./experiments/visualization/runsV/exp34332/confusion_matrix_normalized.png", category: "runs" },
  { title: "PR Curve", src: "./experiments/visualization/runsV/exp34332/BoxPR_curve.png", category: "runs" },
  { title: "F1 Curve", src: "./experiments/visualization/runsV/exp34332/BoxF1_curve.png", category: "runs" },
  { title: "Precision Curve", src: "./experiments/visualization/runsV/exp34332/BoxP_curve.png", category: "runs" },
  { title: "Recall Curve", src: "./experiments/visualization/runsV/exp34332/BoxR_curve.png", category: "runs" },
  { title: "Validation Prediction Sample", src: "./experiments/visualization/runsV/exp34332/val_batch0_pred.jpg", category: "runs" },
  { title: "Validation Prediction Sample 2", src: "./experiments/visualization/runsV/exp34332/val_batch1_pred.jpg", category: "runs" },
  { title: "Validation Prediction Sample 3", src: "./experiments/visualization/runsV/exp34332/val_batch2_pred.jpg", category: "runs" },
  { title: "Project Gantt Chart", src: "./docs/project_planning/Gantt.png", category: "planning" }
];

const metricRows = [
  { name: "mAP50-95", best: 0.76786, final: 0.76517 },
  { name: "mAP50",    best: 0.94159, final: 0.93534 },
  { name: "Precision", best: 0.92421, final: 0.90017 },
  { name: "Recall",   best: 0.91550, final: 0.90167 }
];

/* ─── Helpers ─── */
function escapeHtml(v) {
  return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-link">$1</a>');
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g,"\n").split("\n");
  let html = "", inCode = false, inUl = false, inOl = false, inTable = false;

  const closeLists = () => { if (inUl) { html+="</ul>"; inUl=false; } if (inOl) { html+="</ol>"; inOl=false; } };
  const closeTable = () => { if (inTable) { html+="</tbody></table></div>"; inTable=false; } };

  lines.forEach((rawLine, idx) => {
    const trimmed = rawLine.trim();
    if (trimmed.startsWith("```")) {
      closeLists(); closeTable();
      if (!inCode) { inCode = true; html += "<pre><code>"; }
      else { inCode = false; html += "</code></pre>"; }
      return;
    }
    if (inCode) { html += `${escapeHtml(rawLine)}\n`; return; }
    if (!trimmed) { closeLists(); closeTable(); html += "<br />"; return; }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)/);
    if (heading) {
      closeLists(); closeTable();
      const lvl = heading[1].length;
      const id = heading[2].toLowerCase().replace(/[^a-z0-9\s-]/g,"").trim().replace(/\s+/g,"-");
      html += `<h${lvl} id="${id}">${inlineMarkdown(escapeHtml(heading[2]))}</h${lvl}>`;
      return;
    }
    if (/^>\s+/.test(trimmed)) { closeLists(); closeTable(); html += `<blockquote>${inlineMarkdown(escapeHtml(trimmed.replace(/^>\s+/,"")))}</blockquote>`; return; }
    if (/^\|(.+)\|$/.test(trimmed)) {
      closeLists();
      if (!inTable) { html += "<div class='table-responsive'><table><tbody>"; inTable = true; }
      const nextLine = lines[idx+1] ? lines[idx+1].trim() : "";
      if (/^\|?\s*[-:]+/.test(nextLine)) {
        const headers = trimmed.split("|").map(c=>c.trim()).filter(Boolean).map(c=>`<th>${inlineMarkdown(escapeHtml(c))}</th>`).join("");
        html += `<thead><tr>${headers}</tr></thead><tbody>`;
        return;
      }
      const cols = trimmed.split("|").map(c=>c.trim()).filter(Boolean).map(c=>`<td>${inlineMarkdown(escapeHtml(c))}</td>`).join("");
      html += `<tr>${cols}</tr>`;
      return;
    }
    if (/^[-*]\s+/.test(trimmed)) { closeTable(); if (!inUl) { closeLists(); html+="<ul>"; inUl=true; } html+=`<li>${inlineMarkdown(escapeHtml(trimmed.replace(/^[-*]\s+/,"")))}</li>`; return; }
    if (/^\d+\.\s+/.test(trimmed)) { closeTable(); if (!inOl) { closeLists(); html+="<ol>"; inOl=true; } html+=`<li>${inlineMarkdown(escapeHtml(trimmed.replace(/^\d+\.\s+/,"")))}</li>`; return; }
    closeLists(); closeTable();
    html += `<p>${inlineMarkdown(escapeHtml(trimmed))}</p>`;
  });
  closeLists(); closeTable();
  if (inCode) html += "</code></pre>";
  return html;
}

/* ─── Document Loading ─── */
async function loadDocumentIntoContainer(docItem, container) {
  const statusEl = container.querySelector(".status-message") || (() => {
    const s = document.createElement("div");
    s.className = "status-message";
    container.insertBefore(s, container.firstChild);
    return s;
  })();

  // Fade out effect
  container.style.opacity = '0.5';
  container.style.transition = 'opacity 0.2s ease';

  try {
    statusEl.textContent = "Loading document...";
    statusEl.style.display = "block";
    const res = await fetch(docItem.path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const md = await res.text();

    setTimeout(() => {
      container.innerHTML = markdownToHtml(md);
      container.style.opacity = '1';
      statusEl.style.display = "none";
    }, 150); // Small delay for smooth transition

  } catch (e) {
    container.style.opacity = '1';
    statusEl.textContent = "Could not load document. Ensure you're running a local server.";
    console.error(e);
  }
}

/* ─── Sidebar ─── */
function buildSidebarItems(items, containerId, activeDocKey) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  items.forEach(item => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = item.title;
    a.dataset.path = item.path;
    a.dataset.title = item.title;
    if (item.title === activeDocKey) a.classList.add("active");
    a.addEventListener("click", async e => {
      e.preventDefault();
      // Remove active class from all links
      document.querySelectorAll(".sidebar-list a").forEach(l => l.classList.remove("active"));
      a.classList.add("active");
      const contentContainer = document.querySelector(".docs-content, #reportsContent");
      if (contentContainer) {
        await loadDocumentIntoContainer(item, contentContainer);
        // Scroll top on mobile when item selected
        if (window.innerWidth < 900) {
          contentContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
    li.appendChild(a);
    container.appendChild(li);
  });
}

function renderSidebars(pageType) {
  if (pageType === "docs") {
    buildSidebarItems(docs.filter(d=>d.category==="getting-started"), "docsSidebarGettingStarted");
    buildSidebarItems(docs.filter(d=>d.category==="dataset"), "docsSidebarDataset");
    buildSidebarItems(docs.filter(d=>d.category==="training"), "docsSidebarTraining");
    buildSidebarItems(docs.filter(d=>d.category==="models"), "docsSidebarModels");
    buildSidebarItems(docs.filter(d=>d.category==="project-planning"), "docsSidebarProjectPlanning");
  } else if (pageType === "reports") {
    buildSidebarItems(reports.filter(r=>r.category==="milestones"), "reportsSidebarMilestones");
    buildSidebarItems(reports.filter(r=>r.category==="status"), "reportsSidebarStatus");
    buildSidebarItems(reports.filter(r=>r.category==="research"), "reportsSidebarResearch");
  }
}

/* ─── Search ─── */
function initSearch(inputId, sidebarClass) {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.addEventListener("input", () => {
    const query = input.value.toLowerCase();
    document.querySelectorAll(`.${sidebarClass} a`).forEach(link => {
      const match = link.textContent.toLowerCase().includes(query);
      link.closest("li").style.display = match ? "block" : "none";
    });
  });
}

/* ─── Visuals ─── */
function renderVisuals(filter = "all") {
  const grid = document.getElementById("visualGrid");
  if (!grid) return;
  grid.innerHTML = "";

  // Fade out slightly
  grid.style.opacity = 0;

  setTimeout(() => {
    const items = filter === "all" ? visuals : visuals.filter(v => v.category === filter);
    items.forEach((item, i) => {
      const fig = document.createElement("figure");
      fig.className = "visual-card reveal";
      fig.style.animationDelay = `${i * 50}ms`;
      fig.innerHTML = `<div class="img-wrapper"><img src="${item.src}" alt="${item.title}" loading="lazy" /></div><figcaption>${item.title}</figcaption>`;
      fig.addEventListener("click", () => openImageViewer(item.src, item.title));
      grid.appendChild(fig);
    });
    grid.style.opacity = 1;
    grid.style.transition = 'opacity 0.3s ease';
    initReveal(); // Re-trigger reveal on new items
  }, 150);
}

function openImageViewer(src, title) {
  const viewer = document.getElementById("imageViewer");
  const img = document.getElementById("imageViewerImage");
  const cap = document.getElementById("imageViewerTitle");
  if (!viewer || !img || !cap) return;

  img.src = src;
  img.alt = title;
  cap.textContent = title;

  viewer.classList.add("open");
  viewer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden"; // Prevent background scroll
}

function closeImageViewer() {
  const viewer = document.getElementById("imageViewer");
  if (!viewer) return;

  viewer.classList.remove("open");
  viewer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function initImageViewer() {
  const closeBtn = document.getElementById("imageViewerClose");
  const viewer = document.getElementById("imageViewer");
  if (closeBtn) closeBtn.addEventListener("click", closeImageViewer);
  if (viewer) viewer.addEventListener("click", e => { if (e.target === viewer) closeImageViewer(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeImageViewer(); });
}

function initVisualFilter() {
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderVisuals(btn.dataset.filter);
    });
  });
}

/* ─── Score Cards ─── */
function renderScoreCards() {
  const container = document.getElementById("scoreCards");
  if (!container) return;
  metricRows.forEach((row, i) => {
    const card = document.createElement("article");
    card.className = "score-card reveal";
    card.style.animationDelay = `${i * 100}ms`;
    card.innerHTML = `
      <div class="score-title">${row.name}</div>
      <div class="score-values">
        <span>Best: ${row.best.toFixed(5)}</span>
        <span style="color: var(--muted)">Final: ${row.final.toFixed(5)}</span>
      </div>`;
    container.appendChild(card);
  });
}

/* ─── Chart ─── */
function drawMetricChart() {
  const canvas = document.getElementById("metricChart");
  if (!canvas) return;

  // Adjust canvas resolution for high DPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const w = rect.width, h = rect.height;
  const margin = { top: 30, right: 20, bottom: 40, left: 40 };
  const chartW = w - margin.left - margin.right;
  const chartH = h - margin.top - margin.bottom;

  const isLight = document.body.classList.contains("light-mode");
  const textColor = isLight ? "#0f172a" : "#fafafa";
  const gridColor = isLight ? "#e2e8f0" : "#27272a";
  const accent = isLight ? "#2563eb" : "#0ea5e9";
  const accentDim = isLight ? "#93c5fd" : "#0284c7";

  ctx.clearRect(0, 0, w, h);

  // Draw Grid
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(w - margin.right, y); ctx.stroke();
  }

  const groupW = chartW / metricRows.length;
  const barW = groupW * 0.25;

  metricRows.forEach((row, i) => {
    const center = margin.left + groupW * i + groupW / 2;
    // Normalize data visualization slightly so bars aren't clipped (assuming max is 1.0)
    const maxVal = 1.0;
    const bestH = (row.best / maxVal) * chartH;
    const finalH = (row.final / maxVal) * chartH;

    // Best Bar
    ctx.fillStyle = accent;
    ctx.beginPath();
    const bx = center - barW - 4, by = margin.top + chartH - bestH;
    ctx.roundRect(bx, by, barW, bestH, [4, 4, 0, 0]);
    ctx.fill();

    // Final Bar
    ctx.fillStyle = accentDim;
    ctx.beginPath();
    const fx = center + 4, fy = margin.top + chartH - finalH;
    ctx.roundRect(fx, fy, barW, finalH, [4, 4, 0, 0]);
    ctx.fill();

    // Text Label
    ctx.fillStyle = textColor;
    ctx.font = "600 12px Manrope";
    ctx.textAlign = "center";
    ctx.fillText(row.name, center, h - 14);
  });

  // Legend
  ctx.textAlign = "left";
  ctx.font = "600 12px Manrope";

  ctx.fillStyle = accent;
  ctx.beginPath(); ctx.roundRect(w - 140, 14, 12, 12, 2); ctx.fill();
  ctx.fillStyle = textColor;
  ctx.fillText("Best", w - 120, 24);

  ctx.fillStyle = accentDim;
  ctx.beginPath(); ctx.roundRect(w - 70, 14, 12, 12, 2); ctx.fill();
  ctx.fillStyle = textColor;
  ctx.fillText("Final", w - 50, 24);
}

/* ─── Theme ─── */
function initThemeToggle() {
  const toggle = document.getElementById("themeToggle");
  if (!toggle) return;
  const saved = localStorage.getItem("theme");
  if (saved === "light") {
    document.body.classList.add("light-mode");
    toggle.textContent = "🌙";
  } else {
    toggle.textContent = "☀️";
  }
  toggle.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    const isLight = document.body.classList.contains("light-mode");
    toggle.textContent = isLight ? "🌙" : "☀️";
    localStorage.setItem("theme", isLight ? "light" : "night");
    setTimeout(drawMetricChart, 50); // Redraw chart with new colors
  });
}

/* ─── Mobile Menu ─── */
function initMenu() {
  const toggle = document.getElementById("menuToggle");
  const menu = document.getElementById("menu");
  if (!toggle || !menu) return;
  toggle.addEventListener("click", () => menu.classList.toggle("open"));
  menu.querySelectorAll("a").forEach(l => l.addEventListener("click", () => menu.classList.remove("open")));
}

/* ─── Scroll Reveal ─── */
function initReveal() {
  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.animationPlayState = 'running';
          e.target.classList.add("in");
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
  );

  document.querySelectorAll(".reveal:not(.in)").forEach((item, i) => {
    // Give natural cascade delay if not explicitly set
    if (!item.style.animationDelay) {
      item.style.animationDelay = `${i * 100}ms`;
    }
    observer.observe(item);
  });
}

/* ─── Init ─── */
function init() {
  initMenu();
  initThemeToggle();
  initImageViewer();

  const path = window.location.pathname;

  if (path.includes("docs.html")) {
    renderSidebars("docs");
    initSearch("docSearch", "sidebar-list");
  } else if (path.includes("reports.html")) {
    renderSidebars("reports");
    initSearch("reportSearch", "sidebar-list");
  } else if (path.includes("visuals.html")) {
    renderVisuals();
    initVisualFilter();
  } else if (path.includes("detect.html")) {
    const slider = document.getElementById("confSlider");
    const val = document.getElementById("confValue");
    if (slider && val) slider.addEventListener("input", () => val.textContent = slider.valueAsNumber.toFixed(2));
  } else {
    // Home Page
    renderScoreCards();

    // Ensure chart renders correctly on load and resize
    const chartCard = document.querySelector('.chart-card');
    if (chartCard) {
      // Small timeout to allow CSS layout to settle before calculating sizes
      setTimeout(drawMetricChart, 50);
      window.addEventListener("resize", () => {
        requestAnimationFrame(drawMetricChart);
      });
    }
  }

  // Delay reveal to ensure styles are loaded
  setTimeout(initReveal, 100);
}

// Run when DOM is fully loaded
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
