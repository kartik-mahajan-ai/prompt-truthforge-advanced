/* TruthForge UI application logic. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const ACTION_GLYPH = {
    accept: "✓", rewrite: "✎", reretrieve: "⟳", abstain: "🤐",
  };
  const OUTCOME_GLYPH = { better: "▲", same: "■", worse: "▼" };
  const KIND_LABEL = {
    info_loss: "info loss", hedging: "hedging", new_error: "new error",
  };

  async function api(path, opts) {
    const res = await fetch(path, opts);
    let data;
    try {
      data = await res.json();
    } catch (e) {
      throw new Error(`${res.status} ${res.statusText} (response was not JSON)`);
    }
    if (!res.ok) {
      const err = new Error(data.error || `${res.status} ${res.statusText}`);
      err.traceback = data.traceback;
      throw err;
    }
    return data;
  }

  function errorBox(err) {
    return `<div class="err"><div class="t">${esc(err.message)}</div>
      ${err.traceback ? `<pre>${esc(err.traceback)}</pre>` : ""}</div>`;
  }

  /* ------------------------------------------------------------- theme */

  const savedTheme = (() => {
    try { return localStorage.getItem("tf-theme"); } catch (e) { return null; }
  })();
  if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);

  $("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const isDark = cur
      ? cur === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("tf-theme", next); } catch (e) { /* private mode */ }
    if (state.lastRun) renderRun(state.lastRun);
    if (state.lastCompare) renderCompare(state.lastCompare);
  });

  /* -------------------------------------------------------------- tabs */

  const TABS = ["ask", "pipeline", "runs", "annotate"];

  function showTab(name, pushHash) {
    if (!TABS.includes(name)) name = "pipeline";
    document.querySelectorAll("nav button").forEach((b) =>
      b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = true));
    $("panel-" + name).hidden = false;
    if (pushHash && location.hash.slice(1) !== name) location.hash = name;
    if (name === "runs") loadRuns();
    if (name === "annotate") loadAnnotRuns();
    if (name === "ask") loadDocs();
  }

  document.querySelectorAll("nav button").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab, true));
  });

  // Deep-linking: /#runs opens the comparison view directly, which is handy
  // for bookmarking and for screenshotting a specific view.
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));

  const state = {
    examples: [],
    lastRun: null,
    lastCompare: null,
    annot: { items: [], idx: 0, run: null, labelled: {}, pending: null },
  };

  /* ------------------------------------------------------------ status */

  async function loadStatus() {
    try {
      const s = await api("/api/status");
      const dot = $("gpu-dot"), text = $("gpu-text");
      if (s.cuda && s.cuda.available) {
        dot.className = "dot ok";
        const usedGb = (s.cuda.used_mb / 1024).toFixed(1);
        const totalGb = (s.cuda.total_mb / 1024).toFixed(1);
        text.textContent =
          `${s.cuda.device_name.replace(/NVIDIA GeForce /, "")} · ` +
          `${usedGb} / ${totalGb} GB used`;
        $("gpu-pill").title =
          "Graphics card memory, not system RAM. Models load here when you run.";
      } else if (s.torch_available) {
        dot.className = "dot warn";
        text.textContent = "CPU only — latency numbers not meaningful";
      } else {
        dot.className = "dot bad";
        text.textContent = "torch not installed — baselines only";
      }

      fillSelect($("detector"), s.available_detectors, "nli");
      fillSelect($("router"), s.available_routers, "rule");
      fillSelect($("generator"), s.available_generators, "local");
    } catch (e) {
      $("gpu-dot").className = "dot bad";
      $("gpu-text").textContent = "server unreachable";
    }
  }

  function fillSelect(sel, values, preferred) {
    if (sel.options.length) return;
    sel.innerHTML = values
      .map((v) => `<option value="${esc(v)}"${v === preferred ? " selected" : ""}>${esc(v)}</option>`)
      .join("");
  }

  /* ---------------------------------------------------------- passages */

  function addPassage(text) {
    const row = document.createElement("div");
    row.className = "passage-row";
    const ta = document.createElement("textarea");
    ta.rows = 3;
    ta.value = text || "";
    ta.placeholder = "Paste an evidence passage…";
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "×";
    rm.title = "Remove this passage";
    rm.addEventListener("click", () => row.remove());
    row.append(ta, rm);
    $("passages").appendChild(row);
  }

  function getPassages() {
    return Array.from($("passages").querySelectorAll("textarea"))
      .map((t) => t.value.trim()).filter(Boolean);
  }

  $("add-passage").addEventListener("click", () => addPassage(""));

  /* ------------------------------------------------------ strategy */
  /* Three buttons replace the router dropdown. The hidden <select> stays so
     that the rest of the app, the query-parameter handler and the tests all
     keep reading the router from one place. */

  function setStrategy(router) {
    document.querySelectorAll(".strategy").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.router === router)));
    const sel = $("router");
    if (sel && [...sel.options].some((o) => o.value === router)) sel.value = router;
  }

  document.querySelectorAll(".strategy").forEach((btn) =>
    btn.addEventListener("click", () => setStrategy(btn.dataset.router)));

  const bandInput = $("th-band");
  if (bandInput) {
    const sync = () => { $("band-label").textContent = Number(bandInput.value).toFixed(2); };
    bandInput.addEventListener("input", sync);
    sync();
  }

  /* ---------------------------------------------------------- examples */

  async function loadExamples() {
    // Only the bundled fixture for now. The server still serves
    // ?source=ragtruth, so restoring the picker is a markup change --
    // see loadExamples() and applyQueryParams().
    const sourceEl = $("example-source");
    const source = sourceEl ? sourceEl.value : "ragtruth";
    const picker = $("example-picker");
    picker.innerHTML = "<option value=''>loading…</option>";
    $("example-note").innerHTML = "";
    try {
      const data = await api(`/api/examples?source=${encodeURIComponent(source)}&limit=60`);
      if (!data.available) {
        picker.innerHTML = "<option value=''>— unavailable —</option>";
        $("example-note").innerHTML =
          `<div class="warn-note">${esc(data.reason)}</div>`;
        return;
      }
      state.examples = data.examples;
      picker.innerHTML =
        "<option value=''>— none —</option>" +
        data.examples.map((ex, i) =>
          `<option value="${i}">${esc(ex.id)} · ${esc(ex.question.slice(0, 58))}` +
          `${ex.n_gold_spans ? ` [${ex.n_gold_spans} gold span${ex.n_gold_spans > 1 ? "s" : ""}]` : ""}</option>`
        ).join("");
      $("example-note").innerHTML = data.synthetic
        ? `<div class="warn-note"><strong>Built-in demo examples.</strong> Hand-written,
             not real data. Fine for checking the pipeline; never report numbers from
             them. Switch to RAGTruth above for the real thing.</div>`
        : `<div class="callout-ok"><strong>RAGTruth test split.</strong>
             ${data.examples.length} real answers from GPT-4, Llama-2 and Mistral, with
             hallucinated spans marked by human annotators. The <em>[n gold spans]</em>
             tag shows how many each one contains.</div>`;
    } catch (e) {
      picker.innerHTML = "<option value=''>— error —</option>";
      $("example-note").innerHTML = errorBox(e);
    }
  }

  if ($("example-source")) {
    $("example-source").addEventListener("change", loadExamples);
  }

  $("example-picker").addEventListener("change", (e) => {
    const ex = state.examples[Number(e.target.value)];
    if (!ex) return;
    $("question").value = ex.question;
    $("answer").value = ex.response;
    $("passages").innerHTML = "";
    ex.passages.forEach(addPassage);
  });

  /* --------------------------------------------------------- word diff */

  function wordDiff(a, b) {
    const A = String(a).split(/(\s+)/), B = String(b).split(/(\s+)/);
    const n = A.length, m = B.length;
    // LCS table. Answers are short; O(nm) is fine and exact.
    const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1
          : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    let i = 0, j = 0, out = "";
    const flush = (tag, buf) => (buf ? `<${tag}>${esc(buf)}</${tag}>` : "");
    let delBuf = "", insBuf = "";
    while (i < n && j < m) {
      if (A[i] === B[j]) {
        out += flush("del", delBuf) + flush("ins", insBuf);
        delBuf = insBuf = "";
        out += esc(A[i]); i++; j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        delBuf += A[i++];
      } else {
        insBuf += B[j++];
      }
    }
    while (i < n) delBuf += A[i++];
    while (j < m) insBuf += B[j++];
    return out + flush("del", delBuf) + flush("ins", insBuf);
  }

  /* --------------------------------------------------------------- run */

  $("run-btn").addEventListener("click", runPipeline);

  async function runPipeline() {
    const btn = $("run-btn");
    const passages = getPassages();
    if (!passages.length) {
      $("results").innerHTML =
        `<div class="card">${errorBox(new Error("Add at least one evidence passage."))}</div>`;
      return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> running…';
    $("results").innerHTML =
      '<div class="card"><div class="empty"><span class="spinner"></span> ' +
      'running the pipeline — first use of a model also loads it (~10-20 s)</div></div>';

    const num = (id) => {
      const el = $(id);
      return el && el.value !== "" ? Number(el.value) : undefined;
    };
    const body = {
      question: $("question").value,
      answer: $("answer").value,
      passages,
      detector: $("detector").value,
      router: $("router").value,
      generator: $("generator").value,
      abstain_style: $("abstain-style") ? $("abstain-style").value : undefined,
      max_iterations: num("max-iter") ?? 2,
      // Thresholds not shown in the UI are simply omitted; the server keeps
      // its configured defaults for those.
      accept_support_prob: num("th-accept"),
      rewrite_support_prob: num("th-rewrite"),
      uncertain_band: num("th-band"),
      thin_evidence_coverage: num("th-thin"),
      contradiction_margin: num("th-contra"),
    };
    Object.keys(body).forEach((k) => body[k] === undefined && delete body[k]);

    try {
      const data = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      state.lastRun = data;
      renderRun(data);
    } catch (e) {
      $("results").innerHTML = `<div class="card">${errorBox(e)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Run";
    }
  }

  /* Only the final decision per claim. Showing one chip per loop pass
     doubled every row without adding information. */
  function decisionChips(decs) {
    if (!decs.length) return "";
    const x = decs[decs.length - 1];
    const glyph = ACTION_GLYPH[x.action] || "";
    const extra = decs.length > 1
      ? `<span class="reason dim">after ${decs.length} passes</span>` : "";
    return `<span class="badge ${x.action}">${glyph} ${x.action}</span>` +
           `<span class="reason">${esc(x.reason)}</span>${extra}`;
  }

  function renderRun(d) {
    const out = [];

    // Conditions that make the numbers below untrustworthy. These go first
    // and are not dismissable: an uncalibrated judge still produces a
    // confident-looking faithfulness score.
    (d.warnings || []).forEach((w) => {
      const cls = w.level === "warn" ? "warn-note" : "card-note";
      out.push(`<div class="${cls}" style="margin-bottom:12px;">
        <strong>${esc(w.title)}</strong><br>${esc(w.body)}</div>`);
    });
    const m = d.metrics || {};
    const cm = m.correction_n_attempts || 0;
    const totalTokens = (d.costs || []).reduce((s, c) => s + (c.total_tokens || 0), 0);

    /* --- headline stats --- */
    const supported = Math.round((m.faithfulness ?? 0) * (d.claims || []).length);
    out.push(`<div class="card"><h2>Result</h2><div class="stat-row">
      <div class="stat" title="Sentences the answer was split into"><div class="k">claims</div><div class="v">${(d.claims || []).length}</div></div>
      <div class="stat" title="Share of final claims the fact-checker says the sources back up"><div class="k">backed by sources</div><div class="v">${fmt(m.faithfulness)}</div><div class="u">${supported} of ${(d.claims || []).length}</div></div>
      <div class="stat" title="How many times it tried to fix a claim"><div class="k">fix attempts</div><div class="v">${cm}</div></div>
      <div class="stat" title="Text processed by the local models on your GPU. Nothing is sent to any API."><div class="k">model tokens</div><div class="v">${totalTokens}</div></div>
      <div class="stat"><div class="k">time</div><div class="v">${(d.wall_clock_s || 0).toFixed(2)}<span class="u">s</span></div></div>
      <div class="stat"><div class="k">passes</div><div class="v">${d.iterations_used}</div></div>
    </div></div>`);

    /* --- answer before / after --- */
    const changed = d.initial_answer !== d.final_answer;
    out.push(`<div class="card"><h2>Answer ${changed ? "— changed" : "— unchanged"}</h2>
      ${changed
        ? `<div class="diff">
             <div class="diff-row"><div class="diff-label">diff</div>
               <div>${wordDiff(d.initial_answer, d.final_answer)}</div></div>
           </div>`
        : `<div class="diff"><div class="diff-row"><div class="diff-label">answer</div>
             <div>${esc(d.final_answer)}</div></div></div>`}
      ${d.abstained ? '<p class="card-note" style="margin-top:9px;">🤐 This answer abstains on at least one claim.</p>' : ""}
    </div>`);

    /* --- claims --- */
    const lastVerdict = {};
    (d.verdicts || []).forEach((v) => (lastVerdict[v.claim_id] = v));
    const decisionsByClaim = {};
    (d.decisions || []).forEach((x) =>
      (decisionsByClaim[x.claim_id] = decisionsByClaim[x.claim_id] || []).push(x));

    const claimHtml = (d.claims || []).map((c, i) => {
      const v = lastVerdict[c.claim_id];
      const decs = decisionsByClaim[c.claim_id] || [];
      const ex = (d.explain || {})[c.claim_id] || {};
      const p = v ? v.support_prob : null;
      const cls = p == null ? "" : p >= 0.7 ? "" : p >= 0.4 ? "mid" : "low";

      return `<div class="claim">
        <div class="claim-head">
          <span class="claim-idx">${i + 1}</span>
          ${v ? `<span class="badge ${v.label}">${v.label === "supported" ? "✓ supported" : "✗ unsupported"}</span>` : ""}
          ${decisionChips(decs)}
        </div>
        <div class="claim-text">${esc(c.text)}</div>
        <div class="claim-meta">
          <span class="meter">
            <span class="meter-track">
              <span class="meter-fill ${cls}" style="width:${((p ?? 0) * 100).toFixed(1)}%"></span>
              <span class="meter-tick" style="left:30%"></span>
              <span class="meter-tick" style="left:70%"></span>
            </span>
            <span class="meter-value">${p == null ? "—" : p.toFixed(3)}</span>
          </span>
          <span>coverage ${ex.coverage != null ? ex.coverage.toFixed(2) : "—"}</span>
          ${ex.source_conflict != null ? `<span>source conflict ${ex.source_conflict.toFixed(2)}</span>` : ""}
        </div>
        ${ex.best_evidence ? `<div class="evidence-quote"><span class="lbl">best supporting evidence</span>${esc(ex.best_evidence)}</div>` : ""}
        ${ex.conflicting_evidence && ex.contradiction > 0.4
          ? `<div class="evidence-quote conflict"><span class="lbl">conflicting evidence</span>${esc(ex.conflicting_evidence)}</div>` : ""}
      </div>`;
    }).join("");

    out.push(`<div class="card"><h2>Claims <span class="count">— the bar is how confident the fact-checker is that the sources back each one</span></h2>
      ${claimHtml || '<div class="empty">The splitter produced no claims.</div>'}</div>`);

    /* --- attempts --- */
    if ((d.attempts || []).length) {
      const rows = d.attempts.map((a) => `
        <div class="claim">
          <div class="claim-head">
            <span class="badge ${a.action}">${ACTION_GLYPH[a.action] || ""} ${a.action}</span>
            <span class="claim-idx">iter ${a.iteration}</span>
            ${a.outcome ? `<span class="badge ${a.outcome}">${OUTCOME_GLYPH[a.outcome]} ${a.outcome}${a.worse_kind ? " · " + KIND_LABEL[a.worse_kind] : ""}</span>` : ""}
            <span class="reason">P(sup) ${a.before_support_prob.toFixed(3)} → ${a.after_support_prob != null ? a.after_support_prob.toFixed(3) : "—"}</span>
          </div>
          ${a.before_text === a.after_text
            ? `<div class="diff"><div class="diff-row"><div class="diff-label">no change</div><div>${esc(a.before_text)}</div></div></div>`
            : `<div class="diff">
                 <div class="diff-row"><div class="diff-label">before</div><div>${esc(a.before_text)}</div></div>
                 <div class="diff-row"><div class="diff-label">after</div><div>${esc(a.after_text)}</div></div>
                 <div class="diff-row"><div class="diff-label">diff</div><div>${wordDiff(a.before_text, a.after_text)}</div></div>
               </div>`}
        </div>`).join("");
      out.push(`<div class="card"><h2>Fix attempts <span class="count">— better / same / worse labels are automatic guesses, not human judgement</span></h2>${rows}</div>`);
    }

    /* --- routing reasons + cost --- */
    const reasons = m.routing_reasons || {};
    out.push(`<div class="split" style="grid-template-columns:1fr 1fr;">
      <div class="card"><h2>Routing decisions</h2>
        ${Object.keys(reasons).length
          ? `<table class="data"><thead><tr><th>reason</th><th style="text-align:right">n</th></tr></thead><tbody>
             ${Object.entries(reasons).map(([k, n]) =>
               `<tr><td class="name mono" style="font-size:11.5px">${esc(k)}</td><td class="num">${n}</td></tr>`).join("")}
             </tbody></table>`
          : '<div class="empty">No decisions recorded.</div>'}
      </div>
      <div class="card"><h2>Where the compute went</h2><div id="stage-chart"></div>
        <p class="card-note" style="margin:10px 0 0;">Local models on your GPU. Fact-checking is counted too — leaving it out would make every option look cheaper than it is.</p>
      </div>
    </div>`);

    $("results").innerHTML = out.join("");
    window.TFCharts.stageCostBars($("stage-chart"), d.cost_by_stage || []);
  }

  const fmt = (v) => (v == null ? "—" : Number(v).toFixed(3));

  /* -------------------------------------------------------------- runs */

  async function loadRuns() {
    const host = $("run-list");
    host.innerHTML = '<div class="empty"><span class="spinner"></span> loading…</div>';
    try {
      const data = await api("/api/runs");
      $("runs-count").textContent = `— ${data.runs.length} in ${data.runs_dir}`;
      if (!data.runs.length) {
        host.innerHTML =
          `<div class="empty">No runs yet. Produce some:<br>
           <span class="mono">python -m truthforge.cli --sample --router noop --run-name vanilla</span><br>
           <span class="mono">python -m truthforge.cli --sample --router always_correct --run-name always</span><br>
           <span class="mono">python -m truthforge.cli --sample --router rule --run-name rule</span></div>`;
        return;
      }
      host.innerHTML = `<table class="data">
        <thead><tr><th></th><th>run</th><th>router</th><th>detector</th>
          <th style="text-align:right">turns</th>
          <th style="text-align:right">faithfulness</th>
          <th style="text-align:right">worse rate</th>
          <th style="text-align:right">tokens/turn</th></tr></thead>
        <tbody>${data.runs.map((r) => `<tr>
          <td><input type="checkbox" class="run-cb" value="${esc(r.name)}" checked></td>
          <td class="name">${esc(r.name)}</td>
          <td>${esc(r.router || "—")}</td>
          <td>${esc(r.detector || "—")}</td>
          <td class="num">${r.n_turns}</td>
          <td class="num">${fmt(r.metrics.faithfulness)}</td>
          <td class="num">${r.metrics.correction_worse_rate ?? "—"}</td>
          <td class="num">${r.metrics.tokens_per_turn ?? "—"}</td>
        </tr>`).join("")}</tbody></table>
        <button class="ghost" id="compare-btn" style="margin-top:12px;">Compare selected</button>`;
      $("compare-btn").addEventListener("click", doCompare);
      doCompare();
    } catch (e) {
      host.innerHTML = errorBox(e);
    }
  }

  async function doCompare() {
    const names = Array.from(document.querySelectorAll(".run-cb:checked")).map((c) => c.value);
    if (!names.length) {
      $("compare-out").innerHTML = '<div class="card"><div class="empty">Select at least one run.</div></div>';
      return;
    }
    try {
      const data = await api("/api/compare?runs=" + encodeURIComponent(names.join(",")));
      state.lastCompare = data;
      renderCompare(data);
    } catch (e) {
      $("compare-out").innerHTML = `<div class="card">${errorBox(e)}</div>`;
    }
  }

  /* The head-to-head verdict. This is the card that says whether a claim is
     licensed at all, so it leads with that and shows the numbers under it. */
  function verdictCard(data) {
    const v = data.verdict;
    if (!v) return "";
    if (v.error) return `<div class="card"><h2>Head to head</h2>${esc(v.error)}</div>`;

    const q = v.quality_delta, c = v.cost_delta;
    const win = v.verdict === "dominates" || v.verdict === "equal quality, cheaper"
                || v.verdict === "better quality, not cheaper";
    const sig = (d) => d.significant
      ? '<span class="badge better">significant</span>'
      : '<span class="badge muted">not significant</span>';

    return `<div class="card">
      <h2>Head to head — ${esc(v.ours)} vs ${esc(v.baseline)}</h2>
      <div class="${win ? "callout-ok" : "warn-note"}" style="margin-bottom:12px;">
        <strong>${esc(v.verdict)}</strong><br>${esc(v.claim)}
      </div>
      <table class="data">
        <thead><tr><th>measure</th><th style="text-align:right">difference</th>
          <th style="text-align:right">95% CI</th><th style="text-align:right">p</th><th></th></tr></thead>
        <tbody>
          <tr><td class="name">net quality</td>
            <td class="num">${q.mean_diff >= 0 ? "+" : ""}${q.mean_diff.toFixed(4)}</td>
            <td class="num">[${q.ci_low.toFixed(4)}, ${q.ci_high.toFixed(4)}]</td>
            <td class="num">${q.p_value.toFixed(4)}</td><td>${sig(q)}</td></tr>
          <tr><td class="name">tokens / turn</td>
            <td class="num">${c.mean_diff >= 0 ? "+" : ""}${c.mean_diff.toFixed(1)}</td>
            <td class="num">[${c.ci_low.toFixed(1)}, ${c.ci_high.toFixed(1)}]</td>
            <td class="num">${c.p_value.toFixed(4)}</td><td>${sig(c)}</td></tr>
        </tbody>
      </table>
      <p class="card-note" style="margin-top:10px;">
        Arms run on the same examples, so differences are paired — a paired bootstrap over
        turns, not a comparison of two means. A win is only declared when the relevant
        difference is significant.
      </p>
    </div>`;
  }

  /* The six-cell ledger: three wins, three distinct ways to lose. */
  function ledgerCard(data) {
    const L = data.ledger;
    if (!L || !Object.keys(L).length) return "";
    const arms = Object.values(L);
    // The two gold sources are not interchangeable and the panel has to say
    // which one is in play. The annotated case needs its own caveat, not
    // silence: RAGTruth's spans under-annotate, so "missed" is a floor.
    const warn = data.gold_source === "detector_on_original"
      ? `<div class="warn-note" style="margin-bottom:12px;">
           <strong>No annotated labels.</strong> "Was supported" is the fact-checker's own
           verdict on the original answer, so the judge is grading work its own flags caused.
           Indicative only — run on RAGTruth before quoting these.
         </div>`
      : `<div class="warn-note" style="margin-bottom:12px;">
           <strong>Scored against RAGTruth's human annotations.</strong> Those spans
           under-annotate — a re-annotation pass found hallucinations the release marks
           clean — so <em>missed</em> is a lower bound and <em>over-corrected</em> an upper one.
         </div>`;

    const rows = arms.map((a) => `<tr>
        <td class="name">${esc(a.arm)}</td>
        <td class="num">${a.true_keep}</td><td class="num">${a.false_fix}</td>
        <td class="num">${a.false_drop}</td><td class="num">${a.false_keep}</td>
        <td class="num">${a.true_fix}</td><td class="num">${a.true_drop}</td>
        <td class="num">${a.error_removal.toFixed(3)}</td>
        <td class="num">${a.retention.toFixed(3)}</td>
        <td class="num"><strong>${a.net_quality.toFixed(3)}</strong></td>
      </tr>`).join("");

    return `<div class="card">
      <h2>Claim-fate ledger</h2>
      <p class="card-note">
        Every claim in the original answer, crossed with what happened to it.
        <strong>false fix</strong> is over-correction, <strong>false drop</strong> is information loss,
        <strong>false keep</strong> is a missed hallucination. Net quality is the harmonic mean of
        error removal and retention — abstaining on everything and changing nothing both score 0.
      </p>
      ${warn}
      <table class="data">
        <thead><tr>
          <th>arm</th><th style="text-align:right">true keep</th>
          <th style="text-align:right">false fix</th><th style="text-align:right">false drop</th>
          <th style="text-align:right">false keep</th><th style="text-align:right">true fix</th>
          <th style="text-align:right">true drop</th>
          <th style="text-align:right">removal</th><th style="text-align:right">retention</th>
          <th style="text-align:right">net quality</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${data.pareto_front && data.pareto_front.length
        ? `<p class="card-note" style="margin-top:10px;">Pareto front (nothing beats these on both
             quality and cost): <strong>${data.pareto_front.map(esc).join(", ")}</strong></p>` : ""}
    </div>`;
  }

  function renderCompare(data) {
    const rows = data.rows || [];
    if (!rows.length) {
      $("compare-out").innerHTML = '<div class="card"><div class="empty">Nothing to compare.</div></div>';
      return;
    }
    const cols = ["arm", "accuracy", "faithfulness", "over_correction_rate",
      "futile_rate", "rejection_rate", "abstention_rate", "tokens",
      "latency_s", "n_turns"];

    $("compare-out").innerHTML = `
      <div class="card"><h2>Results table</h2>
        <table class="data"><thead><tr>${cols.map((c, i) =>
          `<th${i ? ' style="text-align:right"' : ""}>${c.replace(/_/g, " ")}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((r) => `<tr>${cols.map((c, i) =>
          i === 0 ? `<td class="name">${esc(r[c])}</td>`
                  : `<td class="num">${r[c] == null ? "—" : r[c]}</td>`).join("")}</tr>`).join("")}
        </tbody></table>
        <p class="card-note" style="margin-top:10px;">
          <strong>accuracy</strong> is blank on RAGTruth: the dataset has no gold answer to score against,
          so these rows are faithfulness-only. RERETRIEVE is also unavailable there — every RAGTruth number
          is a three-action result.<br>
          <strong>rejection rate</strong> is the share of attempts whose corrector output was malformed and
          thrown away (usually: it echoed the whole answer instead of one sentence). That is a
          corrector-capability confound, not evidence about whether correcting was worth trying —
          a high value here means the <em>futile rate</em> beside it is partly the model's fault.</p>
      </div>
      <div class="split" style="grid-template-columns:1fr 1fr;">
        <div class="card"><h2>Quality vs compute</h2>
          <p class="card-note">The headline plot. Up and to the left is better.</p>
          <div id="scatter"></div></div>
        <div class="card"><h2>Correction outcomes</h2>
          <p class="card-note">Share of attempts by outcome. Blue/red rather than green/red: green vs red fails colour-blind separation.</p>
          <div id="outcomes"></div></div>
      </div>
      <div class="card"><h2>Kinds of damage</h2>
        <p class="card-note">Only counted when the outcome is WORSE. This taxonomy is the project's contribution.</p>
        <div id="kinds"></div></div>
      ${verdictCard(data)}
      ${ledgerCard(data)}`;

    const detail = data.detail || {};
    const arms = rows.map((r) => {
      const m = detail[r.arm] || {};
      return {
        name: r.arm,
        better: m.correction_better || 0,
        same: m.correction_same || 0,
        worse: m.correction_worse || 0,
        kinds: m.correction_worse_kinds || {},
      };
    });

    window.TFCharts.costQualityScatter($("scatter"), rows.map((r) => ({
      name: r.arm, tokens: r.tokens, faithfulness: r.faithfulness,
      over_correction_rate: r.over_correction_rate,
    })));
    window.TFCharts.outcomeBars($("outcomes"), arms);
    window.TFCharts.worseKindBars($("kinds"), arms);
  }

  /* ---------------------------------------------------------- annotate */

  async function loadAnnotRuns() {
    const sel = $("annot-run");
    if (sel.options.length) return;
    try {
      const data = await api("/api/runs");
      sel.innerHTML = data.runs.map((r) =>
        `<option value="${esc(r.name)}">${esc(r.name)} (${r.n_turns} turns)</option>`).join("")
        || "<option value=''>— no runs —</option>";
    } catch (e) {
      sel.innerHTML = "<option value=''>— error —</option>";
    }
  }

  $("annot-load").addEventListener("click", async () => {
    const run = $("annot-run").value;
    const who = $("annot-who").value.trim();
    if (!run || !who) {
      $("annot-area").innerHTML =
        `<div class="card">${errorBox(new Error("Pick a run and enter your initials."))}</div>`;
      return;
    }
    try {
      const data = await api("/api/annotate/batch?run=" + encodeURIComponent(run));
      state.annot = {
        items: data.items, idx: 0, run, who,
        labelled: data.labelled || {}, pending: null,
      };
      // Resume where this annotator left off.
      const done = new Set(Object.entries(state.annot.labelled)
        .filter(([, by]) => by[who]).map(([k]) => k));
      const next = data.items.findIndex((it) => !done.has(it.attempt_key));
      state.annot.idx = next >= 0 ? next : 0;
      renderAnnot();
    } catch (e) {
      $("annot-area").innerHTML = `<div class="card">${errorBox(e)}</div>`;
    }
  });

  function renderAnnot() {
    const a = state.annot;
    if (!a.items.length) {
      $("annot-area").innerHTML =
        '<div class="card"><div class="empty">No correction attempts in this run to label.</div></div>';
      $("annot-progress").innerHTML = "";
      return;
    }
    const done = Object.entries(a.labelled).filter(([, by]) => by[a.who]).length;
    $("annot-progress").innerHTML = `
      <div class="progress-track"><div class="progress-fill" style="width:${(done / a.items.length) * 100}%"></div></div>
      <div class="dim" style="font-size:12px;">${done} / ${a.items.length} labelled by ${esc(a.who)}
      · item ${a.idx + 1}</div>`;

    if (a.idx >= a.items.length) {
      $("annot-area").innerHTML = `<div class="card"><div class="empty">
        Batch complete — ${done} labelled.<br>
        <span class="dim">Now have the second annotator do the same, then run
        <span class="mono">python scripts/agreement.py</span> BEFORE adjudicating anything.</span>
      </div></div>`;
      return;
    }

    const it = a.items[a.idx];
    const prev = (a.labelled[it.attempt_key] || {})[a.who];
    const sel = a.pending || (prev ? prev.outcome : null);
    const kind = a.pending === null && prev ? prev.worse_kind : null;

    $("annot-area").innerHTML = `<div class="annot-card">
      <div class="dim mono" style="font-size:11px;margin-bottom:10px;">
        ${esc(it.attempt_key)} · action=${esc(it.action)} · iteration ${it.iteration}
      </div>
      <div class="annot-evidence"><strong>Evidence — this is the only ground truth.</strong><br>${esc(it.evidence)}</div>
      <div class="annot-pair">
        <div class="annot-side"><span class="lbl">before</span>${esc(it.before_text)}</div>
        <div class="annot-side"><span class="lbl">after</span>${esc(it.after_text)}</div>
      </div>

      <div class="choice-grid">
        <button class="choice better" data-outcome="better" aria-pressed="${sel === "better"}">
          <span class="big">▲ Better</span><span class="dim">fixed something real</span><br><span class="key">1</span></button>
        <button class="choice same" data-outcome="same" aria-pressed="${sel === "same"}">
          <span class="big">■ Same</span><span class="dim">nothing meaningful changed</span><br><span class="key">2</span></button>
        <button class="choice worse" data-outcome="worse" aria-pressed="${sel === "worse"}">
          <span class="big">▼ Worse</span><span class="dim">damaged the answer</span><br><span class="key">3</span></button>
      </div>

      <div id="kind-row" ${sel === "worse" ? "" : "hidden"}>
        <div class="dim" style="font-size:12px;margin-bottom:7px;">Which kind of damage? Tie-break order: new error → info loss → hedging.</div>
        <div class="choice-grid">
          <button class="choice" data-kind="info_loss" aria-pressed="${kind === "info_loss"}">
            <span class="big">🗑 Info loss</span><span class="dim">dropped supported detail</span><br><span class="key">Q</span></button>
          <button class="choice" data-kind="hedging" aria-pressed="${kind === "hedging"}">
            <span class="big">🌫 Hedging</span><span class="dim">clear → vague</span><br><span class="key">W</span></button>
          <button class="choice" data-kind="new_error" aria-pressed="${kind === "new_error"}">
            <span class="big">🆕 New error</span><span class="dim">invented something</span><br><span class="key">E</span></button>
        </div>
      </div>

      <label class="field"><span>Notes (borderline cases are worth more with a note)</span>
        <input type="text" id="annot-notes" value="${esc(prev ? prev.notes || "" : "")}"></label>

      <div style="display:flex;gap:9px;margin-top:8px;">
        <button class="ghost" id="annot-prev">← previous</button>
        <button class="ghost" id="annot-skip">skip →</button>
        <span style="flex:1"></span>
        <span class="dim" style="font-size:11.5px;align-self:center;">1/2/3 choose · Q/W/E kind · ← → navigate</span>
      </div>
    </div>`;

    document.querySelectorAll("[data-outcome]").forEach((b) =>
      b.addEventListener("click", () => chooseOutcome(b.dataset.outcome)));
    document.querySelectorAll("[data-kind]").forEach((b) =>
      b.addEventListener("click", () => chooseKind(b.dataset.kind)));
    $("annot-prev").addEventListener("click", () => { move(-1); });
    $("annot-skip").addEventListener("click", () => { move(1); });
  }

  function chooseOutcome(outcome) {
    if (outcome === "worse") {
      state.annot.pending = "worse";
      renderAnnot();
      return;
    }
    saveLabel(outcome, "");
  }

  function chooseKind(kind) { saveLabel("worse", kind); }

  async function saveLabel(outcome, worse_kind) {
    const a = state.annot;
    const it = a.items[a.idx];
    const notesEl = $("annot-notes");
    try {
      await api("/api/annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run: a.run, annotator_id: a.who, attempt_key: it.attempt_key,
          claim_id: it.claim_id, action: it.action, iteration: it.iteration,
          before_text: it.before_text, after_text: it.after_text,
          evidence: it.evidence, outcome, worse_kind,
          notes: notesEl ? notesEl.value : "",
        }),
      });
      a.labelled[it.attempt_key] = a.labelled[it.attempt_key] || {};
      a.labelled[it.attempt_key][a.who] = { outcome, worse_kind, notes: notesEl ? notesEl.value : "" };
      a.pending = null;
      move(1);
    } catch (e) {
      $("annot-area").insertAdjacentHTML("afterbegin", errorBox(e));
    }
  }

  function move(delta) {
    const a = state.annot;
    a.pending = null;
    a.idx = Math.max(0, Math.min(a.items.length, a.idx + delta));
    renderAnnot();
  }

  document.addEventListener("keydown", (e) => {
    if ($("panel-annotate").hidden) return;
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
      if (e.key !== "Enter") return;
    }
    const keys = { 1: "better", 2: "same", 3: "worse" };
    const kinds = { q: "info_loss", w: "hedging", e: "new_error" };
    const k = e.key.toLowerCase();
    if (keys[e.key]) { e.preventDefault(); chooseOutcome(keys[e.key]); }
    else if (kinds[k] && state.annot.pending === "worse") { e.preventDefault(); chooseKind(kinds[k]); }
    else if (e.key === "ArrowRight") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); move(-1); }
  });


  /* ===================================================== ask your docs ===
     The refusal is the feature. An answer the documents do not support is
     dropped by the same router the research is about, so "not enough
     evidence" here is the ABSTAIN action surfacing to a user. */

  async function loadDocs() {
    try {
      const d = await api("/api/docs");
      renderDocs(d);
      if (!$("ask-detector").options.length) {
        const s = await api("/api/status");
        fillSelect($("ask-detector"), s.available_detectors, "nli");
      }
    } catch (e) { $("doc-list").innerHTML = errorBox(e); }
  }

  function renderDocs(d) {
    if (!d.files || !d.files.length) {
      $("doc-list").innerHTML =
        '<p class="card-note" style="margin:12px 0 0;">No documents yet.</p>';
      return;
    }
    $("doc-list").innerHTML = d.files.map((f) => `
      <div class="doc-row">
        <span class="name">${esc(f.name)}</span>
        <span class="meta">${f.chunks} chunks</span>
        <button data-file="${esc(f.name)}" title="Remove">&times;</button>
      </div>`).join("") +
      `<p class="card-note" style="margin:10px 0 0;">
         ${d.n_files} file${d.n_files === 1 ? "" : "s"} &middot; ${d.n_chunks} chunks &middot;
         ${(d.n_characters / 1000).toFixed(1)}k characters indexed</p>`;

    $("doc-list").querySelectorAll("button[data-file]").forEach((b) =>
      b.addEventListener("click", async () => {
        try {
          renderDocs(await api("/api/docs/remove", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: b.dataset.file }),
          }));
        } catch (e) { $("doc-list").insertAdjacentHTML("afterbegin", errorBox(e)); }
      }));
  }

  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onerror = () => reject(new Error(`could not read ${file.name}`));
      r.onload = () => resolve(String(r.result).split(",")[1]);
      r.readAsDataURL(file);
    });
  }

  async function uploadFiles(files) {
    const list = $("doc-list");
    for (const file of files) {
      list.insertAdjacentHTML("afterbegin",
        `<p class="card-note" id="up-msg"><span class="spinner"></span> reading ${esc(file.name)}…</p>`);
      try {
        const content_b64 = await readAsBase64(file);
        const d = await api("/api/docs/upload", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: file.name, content_b64 }),
        });
        renderDocs(d);
      } catch (e) {
        renderDocs(await api("/api/docs").catch(() => ({ files: [] })));
        list.insertAdjacentHTML("afterbegin", errorBox(e));
      }
    }
  }

  const dz = $("dropzone");
  if (dz) {
    dz.addEventListener("click", () => $("file-input").click());
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); }
    });
    $("file-input").addEventListener("change", (e) => {
      uploadFiles([...e.target.files]); e.target.value = "";
    });
    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
    dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  }

  if ($("ask-btn")) $("ask-btn").addEventListener("click", askQuestion);
  if ($("ask-q")) $("ask-q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askQuestion();
  });

  async function askQuestion() {
    const q = $("ask-q").value.trim();
    if (!q) return;
    const btn = $("ask-btn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> thinking…';
    $("ask-out").innerHTML =
      '<div class="card"><div class="empty"><span class="spinner"></span> ' +
      'retrieving, answering, then checking every sentence against your documents</div></div>';
    try {
      const d = await api("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          detector: $("ask-detector").value || undefined,
          top_k: Number($("ask-topk").value) || 5,
          dense: $("ask-dense") ? $("ask-dense").checked : false,
        }),
      });
      renderAnswer(d);
    } catch (e) {
      $("ask-out").innerHTML = `<div class="card">${errorBox(e)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask";
    }
  }

  function renderAnswer(d) {
    const out = [];
    (d.warnings || []).forEach((w) => {
      out.push(`<div class="${w.level === "warn" ? "warn-note" : "card-note"}"
        style="margin-bottom:12px;"><strong>${esc(w.title)}</strong><br>${esc(w.body)}</div>`);
    });

    out.push(`<div class="card"><h2>Answer</h2>
      ${d.insufficient_evidence
        ? `<div class="answer-box refused"><strong>Not enough evidence.</strong><br>
             ${esc(d.message)}</div>
           <p class="card-note" style="margin-top:10px;">
             ${d.refused_by === "question_not_covered_by_documents"
               ? `Nothing was drafted at all. The question's key terms barely appear in
                  your documents, so there was nothing to answer from &mdash; and checking
                  each sentence afterwards would not have caught this. If the retrieved
                  text is about a different topic, the model writes about that topic and
                  every sentence it writes is genuinely supported by it.`
               : `The model drafted an answer, but every sentence failed verification against
                  your documents, so all of them were removed. Refusing is the correct outcome
                  here &mdash; the alternative is a confident answer from the model's own memory.`}</p>`
        : `<div class="answer-box">${esc(d.answer)}</div>`}
      <div class="stat-row" style="margin-top:12px;">
        <div class="stat"><div class="k">sentences kept</div>
          <div class="v">${d.n_claims - d.n_dropped}<span class="u">of ${d.n_claims}</span></div></div>
        <div class="stat"><div class="k">dropped as unsupported</div><div class="v">${d.n_dropped}</div></div>
        <div class="stat"><div class="k">sources used</div><div class="v">${d.sources.length}</div></div>
        <div class="stat"><div class="k">time</div><div class="v">${d.seconds}<span class="u">s</span></div></div>
      </div></div>`);

    if (d.draft && d.draft.trim() !== (d.answer || "").trim()) {
      out.push(`<div class="card"><h2>What the model first wrote
        <span class="count">— before verification</span></h2>
        <div class="diff"><div class="diff-row"><div class="diff-label">draft</div>
          <div>${esc(d.draft)}</div></div></div></div>`);
    }

    const sup = Object.values(d.support || {});
    if (sup.length) {
      out.push(`<div class="card"><h2>Why each sentence is trusted</h2>
        <p class="card-note">Every sentence kept in the answer, and the source line that backs it.</p>
        ${sup.map((s) => `<div class="claim">
            <div class="claim-head">
              ${s.entailment != null
                ? `<span class="badge ${s.entailment >= 0.5 ? "supported" : "unsupported"}">${s.entailment.toFixed(3)}</span>` : ""}
              ${s.citation ? `<span class="reason">${esc(s.citation)}</span>` : ""}
              <span class="reason dim">coverage ${s.coverage}</span>
            </div>
            <div class="claim-text">${esc(s.text)}</div>
            ${s.supporting_sentence
              ? `<div class="evidence-quote"><span class="lbl">backed by</span>${esc(s.supporting_sentence)}</div>` : ""}
          </div>`).join("")}</div>`);
    }

    if (d.sources.length) {
      out.push(`<div class="card"><h2>Sources it used</h2>
        ${d.sources.map((s) => `<div class="source">
            <div class="cite">${esc(s.citation)}<span class="score">score ${s.score}</span></div>
            <div class="body">${esc(s.text.slice(0, 400))}${s.text.length > 400 ? "…" : ""}</div>
          </div>`).join("")}</div>`);
    }

    out.push(`<div class="card"><h2>Where the compute went</h2>
      <div id="ask-cost"></div></div>`);

    $("ask-out").innerHTML = out.join("");
    window.TFCharts.stageCostBars($("ask-cost"), d.cost_by_stage || []);
  }

  /* -------------------------------------------------------------- init */

  /* Query parameters make a specific pipeline run linkable, which is how you
     hand someone a reproduction rather than a description of one:
       ?example=3&detector=nli&router=rule&generator=local&autorun=1          */
  async function applyQueryParams() {
    const q = new URLSearchParams(location.search);
    if (!q.toString()) return;

    // ?ask=<question>&autorun=1 opens the Ask tab with that question (and runs it).
    if (q.get("ask") !== null && $("ask-q")) {
      showTab("ask", false);
      $("ask-q").value = q.get("ask");
      if (q.get("autorun") === "1") await askQuestion();
      return;
    }

    if (q.get("source") && $("example-source")) {
      $("example-source").value = q.get("source");
      await loadExamples();
    }
    for (const [param, id] of [["detector", "detector"], ["router", "router"],
                               ["generator", "generator"]]) {
      const v = q.get(param);
      if (v && $(id).querySelector(`option[value="${CSS.escape(v)}"]`)) $(id).value = v;
    }
    if (q.get("router")) setStrategy(q.get("router"));
    if (q.get("example") !== null) {
      const picker = $("example-picker");
      picker.value = q.get("example");
      picker.dispatchEvent(new Event("change"));
    }
    if (q.get("question")) $("question").value = q.get("question");
    if (q.get("band")) $("th-band").value = q.get("band");
    if (q.get("autorun") === "1") await runPipeline();
  }

  addPassage("");
  if (location.hash) showTab(location.hash.slice(1), false);

  (async () => {
    await loadStatus();
    await loadExamples();
    await applyQueryParams();
  })();
  window.addEventListener("resize", () => {
    clearTimeout(window.__tfResize);
    window.__tfResize = setTimeout(() => {
      if (state.lastCompare && !$("panel-runs").hidden) renderCompare(state.lastCompare);
      if (state.lastRun && !$("panel-pipeline").hidden) {
        window.TFCharts.stageCostBars($("stage-chart"), state.lastRun.cost_by_stage || []);
      }
    }, 180);
  });
})();
