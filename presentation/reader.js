function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let inCode = false;
  let inUl = false;
  let inOl = false;
  let inTable = false;

  const closeLists = () => {
    if (inUl) {
      html += "</ul>";
      inUl = false;
    }
    if (inOl) {
      html += "</ol>";
      inOl = false;
    }
  };

  const closeTable = () => {
    if (inTable) {
      html += "</tbody></table>";
      inTable = false;
    }
  };

  lines.forEach((rawLine, idx) => {
    const line = rawLine;
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeLists();
      closeTable();
      if (!inCode) {
        inCode = true;
        html += "<pre><code>";
      } else {
        inCode = false;
        html += "</code></pre>";
      }
      return;
    }

    if (inCode) {
      html += `${escapeHtml(line)}\n`;
      return;
    }

    if (!trimmed) {
      closeLists();
      closeTable();
      html += "<br />";
      return;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)/);
    if (heading) {
      closeLists();
      closeTable();
      const level = heading[1].length;
      const content = inlineMarkdown(escapeHtml(heading[2]));
      const id = heading[2]
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .trim()
        .replace(/\s+/g, "-");
      html += `<h${level} id="${id}">${content}</h${level}>`;
      return;
    }

    if (/^>\s+/.test(trimmed)) {
      closeLists();
      closeTable();
      html += `<blockquote>${inlineMarkdown(escapeHtml(trimmed.replace(/^>\s+/, "")))}</blockquote>`;
      return;
    }

    if (/^\|(.+)\|$/.test(trimmed)) {
      closeLists();
      if (!inTable) {
        html += "<table><tbody>";
        inTable = true;
      }

      const nextLine = lines[idx + 1] ? lines[idx + 1].trim() : "";
      if (/^\|?\s*[-:]+/.test(nextLine)) {
        const headers = trimmed
          .split("|")
          .map((cell) => cell.trim())
          .filter(Boolean)
          .map((cell) => `<th>${inlineMarkdown(escapeHtml(cell))}</th>`)
          .join("");
        html += `<thead><tr>${headers}</tr></thead><tbody>`;
        return;
      }

      const cols = trimmed
        .split("|")
        .map((cell) => cell.trim())
        .filter(Boolean)
        .map((cell) => `<td>${inlineMarkdown(escapeHtml(cell))}</td>`)
        .join("");
      html += `<tr>${cols}</tr>`;
      return;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      closeTable();
      if (!inUl) {
        closeLists();
        html += "<ul>";
        inUl = true;
      }
      html += `<li>${inlineMarkdown(escapeHtml(trimmed.replace(/^[-*]\s+/, "")))}</li>`;
      return;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      closeTable();
      if (!inOl) {
        closeLists();
        html += "<ol>";
        inOl = true;
      }
      html += `<li>${inlineMarkdown(escapeHtml(trimmed.replace(/^\d+\.\s+/, "")))}</li>`;
      return;
    }

    closeLists();
    closeTable();
    html += `<p>${inlineMarkdown(escapeHtml(trimmed))}</p>`;
  });

  closeLists();
  closeTable();

  if (inCode) {
    html += "</code></pre>";
  }

  return html;
}

function buildToc(container) {
  const toc = document.getElementById("toc");
  toc.innerHTML = "";

  const headings = container.querySelectorAll("h1, h2, h3");
  if (!headings.length) {
    toc.innerHTML = "<p>No headings found.</p>";
    return;
  }

  headings.forEach((heading) => {
    const a = document.createElement("a");
    a.href = `#${heading.id}`;
    a.textContent = heading.textContent;
    a.style.marginLeft = heading.tagName === "H3" ? "14px" : heading.tagName === "H2" ? "7px" : "0";
    toc.appendChild(a);
  });
}

async function loadDocument() {
  const params = new URLSearchParams(window.location.search);
  const title = params.get("title") || "Document";
  const src = params.get("src");

  const titleEl = document.getElementById("docTitle");
  const statusEl = document.getElementById("status");
  const contentEl = document.getElementById("docContent");

  titleEl.textContent = title;

  if (!src) {
    statusEl.textContent = "No document source provided.";
    return;
  }

  try {
    const response = await fetch(src);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const markdown = await response.text();
    contentEl.innerHTML = markdownToHtml(markdown);
    statusEl.style.display = "none";
    buildToc(contentEl);
  } catch (error) {
    statusEl.textContent = "Could not load document. Run this site with a local server (for example: python3 -m http.server).";
    console.error(error);
  }
}

loadDocument();
