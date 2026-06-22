/* DermaScan AI - client interactions (vanilla JS, no framework)
   - theme toggle (persisted)
   - drag & drop + paste + live preview
   - export medical advice: Save as PDF / download report / copy
*/
(function () {
  "use strict";

  /* -------------------- Theme toggle -------------------- */
  var root = document.documentElement;
  var saved = null;
  try {
    saved = localStorage.getItem("dermascan-theme");
  } catch (e) {}
  if (saved) root.setAttribute("data-theme", saved);

  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("dermascan-theme", next);
      } catch (e) {}
    });
  }

  /* -------------------- Toast helper -------------------- */
  var toastEl = null;
  var toastTimer = null;
  function toast(msg) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    requestAnimationFrame(function () {
      toastEl.classList.add("is-visible");
    });
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.remove("is-visible");
    }, 2400);
  }

  /* -------------------- Upload: drag/drop, paste, preview -------------------- */
  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");

  function showPreview(file) {
    if (!file || !/^image\//.test(file.type)) return;
    var reader = new FileReader();
    reader.onload = function (ev) {
      var img = document.getElementById("preview-img");
      var hint = document.getElementById("dropzone-hint");
      if (img) {
        img.src = ev.target.result;
        img.hidden = false;
      }
      if (hint) hint.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  function assignFile(file) {
    if (!file || !fileInput) return;
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    } catch (e) {
      /* DataTransfer unsupported - the native input value still applies on click */
    }
    showPreview(file);
  }

  if (dropzone && fileInput) {
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) assignFile(files[0]);
    });

    // Paste an image from the clipboard anywhere on the page.
    document.addEventListener("paste", function (e) {
      var items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (var i = 0; i < items.length; i++) {
        if (items[i].type && items[i].type.indexOf("image") === 0) {
          assignFile(items[i].getAsFile());
          toast("Image pasted from clipboard");
          break;
        }
      }
    });
  }

  /* -------------------- Export medical advice -------------------- */
  var dataEl = document.getElementById("report-data");
  var report = null;
  if (dataEl) {
    try {
      report = JSON.parse(dataEl.textContent);
    } catch (e) {
      report = null;
    }
  }

  function buildAdviceText(r) {
    var res = r.result || {};
    var c = res.cancer_assessment || {};
    var p = res.prediction || {};
    var a = res.advice || {};
    var lines = [];
    lines.push("DermaScan AI - Skin Lesion Screening Report");
    lines.push("Generated: " + (r.generated_at || ""));
    lines.push("");
    lines.push("PATIENT DETAILS");
    lines.push("  Age: " + (r.age || "n/a"));
    lines.push("  Sex: " + (r.sex || "n/a"));
    lines.push("  Body site: " + (r.localization || "n/a"));
    lines.push("");
    lines.push("SCREENING SUMMARY");
    lines.push("  Cancer-risk score: " + c.malignant_percent + "% (benign " + c.benign_percent + "%)");
    lines.push("  Assessment: " + (c.message || ""));
    lines.push("  Most likely type: " + (p.label || "") + " (" + p.confidence_percent + "% confidence)");
    lines.push("");
    lines.push("PROBABILITIES");
    (res.probabilities || []).forEach(function (item) {
      lines.push("  " + item.label + ": " + item.percent + "%" + (item.malignant ? " [cancerous]" : ""));
    });
    lines.push("");
    lines.push("RECOMMENDED NEXT STEPS");
    lines.push("  " + (a.headline || "") + " (timing: " + (a.urgency || "") + ")");
    if (a.about) lines.push("  About: " + a.about);
    if (a.recommended_action) lines.push("  Action: " + a.recommended_action);
    (a.steps || []).forEach(function (s, i) {
      lines.push("  " + (i + 1) + ". " + s);
    });
    lines.push("");
    lines.push("ABCDE WARNING SIGNS");
    (a.abcde || []).forEach(function (s) {
      lines.push("  - " + s);
    });
    lines.push("");
    lines.push(res.disclaimer || "");
    return lines.join("\n");
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function buildReportHtml(r) {
    var res = r.result || {};
    var c = res.cancer_assessment || {};
    var p = res.prediction || {};
    var a = res.advice || {};
    var rows = (res.probabilities || [])
      .map(function (item) {
        return (
          "<tr><td>" +
          esc(item.label) +
          (item.malignant ? ' <span style="color:#e11d48">(cancerous)</span>' : "") +
          '</td><td style="text-align:right">' +
          item.percent +
          "%</td></tr>"
        );
      })
      .join("");
    var steps = (a.steps || [])
      .map(function (s) {
        return "<li>" + esc(s) + "</li>";
      })
      .join("");
    var abcde = (a.abcde || [])
      .map(function (s) {
        return "<li>" + esc(s) + "</li>";
      })
      .join("");
    return (
      "<!doctype html><html><head><meta charset='utf-8'><title>DermaScan AI Report</title>" +
      "<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#10233f;max-width:760px;margin:32px auto;padding:0 20px;line-height:1.5}" +
      "h1{margin:0 0 4px}h2{margin:24px 0 8px;font-size:1.05rem;border-bottom:1px solid #e3e9f3;padding-bottom:6px}" +
      ".muted{color:#5a6b8c;font-size:.9rem}.score{font-size:2rem;font-weight:800}" +
      "table{width:100%;border-collapse:collapse;font-size:.92rem}td{padding:6px 0;border-bottom:1px solid #eef2f8}" +
      ".note{margin-top:24px;font-size:.8rem;color:#5a6b8c}.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2f8;font-size:.75rem}</style></head><body>" +
      "<h1>DermaScan AI</h1><div class='muted'>Skin lesion screening report &middot; " +
      esc(r.generated_at) +
      "</div>" +
      "<h2>Patient details</h2><div class='muted'>Age: " +
      esc(r.age || "n/a") +
      " &middot; Sex: " +
      esc(r.sex || "n/a") +
      " &middot; Body site: " +
      esc(r.localization || "n/a") +
      "</div>" +
      "<h2>Screening summary</h2><div class='score'>" +
      esc(c.malignant_percent) +
      "% <span class='muted' style='font-size:.9rem;font-weight:400'>estimated cancer risk</span></div>" +
      "<p>" +
      esc(c.message) +
      "<br><strong>Most likely type:</strong> " +
      esc(p.label) +
      " (" +
      esc(p.confidence_percent) +
      "% confidence)</p>" +
      "<h2>All probabilities</h2><table>" +
      rows +
      "</table>" +
      "<h2>Recommended next steps</h2><p><strong>" +
      esc(a.headline) +
      "</strong> <span class='tag'>" +
      esc(a.urgency) +
      "</span></p><p>" +
      esc(a.about) +
      "</p><p>" +
      esc(a.recommended_action) +
      "</p><ol>" +
      steps +
      "</ol>" +
      "<h2>ABCDE warning signs</h2><ul>" +
      abcde +
      "</ul>" +
      "<p class='note'>" +
      esc(res.disclaimer) +
      "</p></body></html>"
    );
  }

  function download(filename, content, mime) {
    var blob = new Blob([content], { type: mime });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  document.querySelectorAll("[data-export]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var kind = btn.getAttribute("data-export");
      if (kind === "pdf") {
        window.print();
        return;
      }
      if (!report) {
        toast("No report data available");
        return;
      }
      var stamp = new Date().toISOString().slice(0, 10);
      if (kind === "download") {
        download("dermascan-report-" + stamp + ".html", buildReportHtml(report), "text/html");
        toast("Report downloaded");
      } else if (kind === "copy") {
        var text = buildAdviceText(report);
        var label = btn.querySelector("[data-copy-label]");
        function done() {
          toast("Advice copied to clipboard");
          if (label) {
            var prev = label.textContent;
            label.textContent = "Copied!";
            setTimeout(function () {
              label.textContent = prev;
            }, 1600);
          }
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            download("dermascan-advice-" + stamp + ".txt", text, "text/plain");
          });
        } else {
          download("dermascan-advice-" + stamp + ".txt", text, "text/plain");
        }
      }
    });
  });

  // If a result is present, bring it into view on load.
  var result = document.getElementById("result");
  if (result && window.location.hash !== "#analyze") {
    setTimeout(function () {
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
  }
})();
