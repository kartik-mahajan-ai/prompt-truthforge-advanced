/* Inline-SVG charts.
 *
 * Built by hand rather than with a library so the page stays dependency-free
 * and the mark specs are explicit.
 *
 * Conventions applied throughout (data-viz method):
 *   - 2px surface-coloured gap between adjacent stacked segments and bars
 *   - 4px rounded data-ends, anchored flat to the baseline
 *   - >=8px markers, 2px surface ring where marks can overlap
 *   - recessive grid and axes; text in ink tokens, never in the series color
 *   - a legend whenever there are >=2 series, plus selective direct labels
 *   - a table view beside every chart, which is also the relief for the
 *     light-mode slots that sit below 3:1 contrast
 *   - hover tooltips on every mark
 */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const GAP = 2;          // surface gap between adjacent fills
  const END_R = 4;        // rounded data-end radius

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) {
        node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* Rounded on one end only, flat against the baseline. */
  function barPath(x, y, w, h, r, side) {
    if (h <= 0 || w <= 0) return "";
    const rr = Math.max(0, Math.min(r, h / 2, w / 2));
    if (side === "top") {
      return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y}
              L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr}
              L${x + w},${y + h} Z`;
    }
    if (side === "right") {
      return `M${x},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr}
              L${x + w},${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h}
              L${x},${y + h} Z`;
    }
    if (side === "left") {
      return `M${x + w},${y} L${x + rr},${y} Q${x},${y} ${x},${y + rr}
              L${x},${y + h - rr} Q${x},${y + h} ${x + rr},${y + h}
              L${x + w},${y + h} Z`;
    }
    return `M${x},${y} h${w} v${h} h${-w} Z`;
  }

  /* -------------------------------------------------------------- tooltip */

  const tip = {
    node: null,
    ensure() {
      if (!this.node) this.node = document.getElementById("tooltip");
      return this.node;
    },
    show(html, evt, host) {
      const n = this.ensure();
      if (!n) return;
      n.innerHTML = html;
      n.classList.add("on");
      const hb = host.getBoundingClientRect();
      const nb = n.getBoundingClientRect();
      let left = evt.clientX - hb.left + 12;
      let top = evt.clientY - hb.top - nb.height - 10;
      if (left + nb.width > hb.width) left = evt.clientX - hb.left - nb.width - 12;
      if (top < 0) top = evt.clientY - hb.top + 16;
      n.style.left = Math.max(0, left) + "px";
      n.style.top = top + "px";
    },
    hide() {
      const n = this.ensure();
      if (n) n.classList.remove("on");
    },
  };

  function hoverable(node, html, host) {
    node.style.cursor = "default";
    node.addEventListener("mousemove", (e) => tip.show(html, e, host));
    node.addEventListener("mouseleave", () => tip.hide());
  }

  function legend(items) {
    const wrap = document.createElement("div");
    wrap.className = "legend";
    items.forEach((it) => {
      const span = document.createElement("span");
      span.className = "item";
      span.innerHTML =
        `<span class="swatch" style="background:${it.color}"></span>${it.label}`;
      wrap.appendChild(span);
    });
    return wrap;
  }

  function table(headers, rows) {
    const t = document.createElement("table");
    t.className = "data";
    const thead = document.createElement("thead");
    thead.innerHTML =
      "<tr>" + headers.map((h, i) =>
        `<th${i ? ' style="text-align:right"' : ""}>${h}</th>`).join("") + "</tr>";
    t.appendChild(thead);
    const tb = document.createElement("tbody");
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = r.map((c, i) =>
        i === 0 ? `<td class="name">${c}</td>` : `<td class="num">${c}</td>`).join("");
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }

  /* =======================================================================
   * Outcome breakdown: a 100% stacked bar per arm.
   *
   * better / same / worse is polarity around a neutral, so this uses the
   * DIVERGING pair (blue <-> red) with a gray midpoint, not green/red.
   * Green vs red fails CVD separation badly (deltaE 4.1 deutan, below even the
   * warn band); blue vs red measures 23.8 light / 19.2 dark. "same" is the
   * gray midpoint and never relies on hue -- it always carries its label.
   * ===================================================================== */
  function outcomeBars(host, arms) {
    host.innerHTML = "";
    if (!arms.length) {
      host.innerHTML = '<div class="empty">No correction attempts to break down.</div>';
      return;
    }

    const colors = {
      better: cssVar("--better"),
      same: cssVar("--same"),
      worse: cssVar("--worse"),
    };
    const surface = cssVar("--surface-1");

    host.appendChild(legend([
      { label: "▲ better", color: colors.better },
      { label: "■ same", color: colors.same },
      { label: "▼ worse", color: colors.worse },
    ]));

    const rowH = 32, gapY = 12, padL = 132, padR = 56, padT = 6;
    const w = Math.max(host.clientWidth || 560, 420);
    const plotW = w - padL - padR;
    const h = padT + arms.length * (rowH + gapY);

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, wrap);
    el("title", {}, svg).textContent =
      "Correction outcomes by arm, as a share of all attempts";

    arms.forEach((arm, i) => {
      const y = padT + i * (rowH + gapY);
      const total = arm.better + arm.same + arm.worse;

      el("text", {
        x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "ax-title",
      }, svg).textContent = arm.name;

      if (!total) {
        el("text", { x: padL, y: y + rowH / 2 + 4, class: "ax-label" }, svg)
          .textContent = "no attempts";
        return;
      }

      const segs = [
        { key: "better", n: arm.better },
        { key: "same", n: arm.same },
        { key: "worse", n: arm.worse },
      ].filter((s) => s.n > 0);

      let x = padL;
      segs.forEach((s, si) => {
        const frac = s.n / total;
        const segW = Math.max(1, frac * plotW - (si < segs.length - 1 ? GAP : 0));
        const side = segs.length === 1 ? "both"
          : si === 0 ? "left" : si === segs.length - 1 ? "right" : "flat";

        const p = el("path", {
          d: barPath(x, y, segW, rowH, END_R, side),
          fill: colors[s.key],
          stroke: surface,
          "stroke-width": 1,
        }, svg);

        hoverable(p, `<div class="t-title">${arm.name} · ${s.key}</div>
          <div class="t-row"><span>attempts</span><span>${s.n}</span></div>
          <div class="t-row"><span>share</span><span>${(frac * 100).toFixed(1)}%</span></div>`,
          wrap);

        // Direct label inside the segment when it fits; identity never rests
        // on color alone.
        if (segW > 46) {
          el("text", {
            x: x + segW / 2, y: y + rowH / 2 + 4, "text-anchor": "middle",
            class: "mark-label", fill: s.key === "same" ? cssVar("--text-primary") : "#fff",
          }, svg).textContent = `${Math.round(frac * 100)}%`;
        }
        x += frac * plotW;
      });

      el("text", {
        x: w - padR + 8, y: y + rowH / 2 + 4, class: "ax-label",
      }, svg).textContent = `n=${total}`;
    });

    host.appendChild(wrap);
    host.appendChild(table(
      ["arm", "better", "same", "worse", "worse rate"],
      arms.map((a) => {
        const t = a.better + a.same + a.worse;
        return [a.name, a.better, a.same, a.worse,
          t ? (a.worse / t).toFixed(3) : "—"];
      })
    ));
  }

  /* =======================================================================
   * Worse-kind breakdown: grouped columns, 3 categorical slots.
   * Validated all-pairs in both modes.
   * ===================================================================== */
  function worseKindBars(host, arms) {
    host.innerHTML = "";
    const kinds = [
      { key: "info_loss", label: "🗑 info loss", color: cssVar("--kind-1") },
      { key: "hedging", label: "🌫 hedging", color: cssVar("--kind-2") },
      { key: "new_error", label: "🆕 new error", color: cssVar("--kind-3") },
    ];

    const max = Math.max(1, ...arms.flatMap((a) => kinds.map((k) => a.kinds[k.key] || 0)));
    if (max <= 1 && !arms.some((a) => kinds.some((k) => a.kinds[k.key]))) {
      host.innerHTML =
        '<div class="empty">No WORSE outcomes recorded — nothing to break down.<br>' +
        '<span class="dim">Under Gate 2, a worse-rate below 5% means the efficiency angle is the story.</span></div>';
      return;
    }

    host.appendChild(legend(kinds.map((k) => ({ label: k.label, color: k.color }))));

    const padL = 44, padR = 12, padT = 10, padB = 42;
    const w = Math.max(host.clientWidth || 560, 420);
    const h = 210;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, wrap);
    el("title", {}, svg).textContent = "Kinds of damage, by arm";

    // recessive grid
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = (max / ticks) * i;
      const y = padT + plotH - (v / max) * plotH;
      el("line", {
        x1: padL, x2: w - padR, y1: y, y2: y,
        stroke: i === 0 ? cssVar("--axis") : cssVar("--grid"), "stroke-width": 1,
      }, svg);
      el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", class: "ax-label" }, svg)
        .textContent = Math.round(v);
    }

    const groupW = plotW / arms.length;
    const barW = Math.min(26, (groupW - 16) / kinds.length - GAP);

    arms.forEach((arm, gi) => {
      const gx = padL + gi * groupW;
      kinds.forEach((k, ki) => {
        const n = arm.kinds[k.key] || 0;
        const bh = (n / max) * plotH;
        const x = gx + (groupW - (barW + GAP) * kinds.length) / 2 + ki * (barW + GAP);
        const y = padT + plotH - bh;

        if (bh > 0) {
          const p = el("path", {
            d: barPath(x, y, barW, bh, END_R, "top"),
            fill: k.color, stroke: cssVar("--surface-1"), "stroke-width": 1,
          }, svg);
          hoverable(p, `<div class="t-title">${arm.name}</div>
            <div class="t-row"><span>${k.label}</span><span>${n}</span></div>`, wrap);
          el("text", {
            x: x + barW / 2, y: y - 5, "text-anchor": "middle", class: "mark-label",
          }, svg).textContent = n;
        }
      });
      el("text", {
        x: gx + groupW / 2, y: h - padB + 18, "text-anchor": "middle", class: "ax-title",
      }, svg).textContent = arm.name;
    });

    host.appendChild(wrap);
    host.appendChild(table(
      ["arm", "info loss", "hedging", "new error"],
      arms.map((a) => [a.name, a.kinds.info_loss || 0, a.kinds.hedging || 0,
        a.kinds.new_error || 0])
    ));
  }

  /* =======================================================================
   * The headline plot: quality gained vs compute spent.
   *
   * One series with a direct label on every point, rather than one hue per
   * arm: with seven arms a categorical scatter cannot clear the all-pairs
   * colour floors, and labels identify the points better than colour would.
   * ===================================================================== */
  function costQualityScatter(host, points, opts) {
    host.innerHTML = "";
    opts = opts || {};
    const yKey = opts.yKey || "faithfulness";
    const yLabel = opts.yLabel || "faithfulness →";

    const valid = points.filter((p) => p.tokens != null && p[yKey] != null);
    if (valid.length < 1) {
      host.innerHTML = '<div class="empty">Not enough runs to plot.</div>';
      return;
    }

    const padL = 52, padR = 76, padT = 14, padB = 44;
    const w = Math.max(host.clientWidth || 560, 420);
    const h = 268;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    const xs = valid.map((p) => p.tokens), ys = valid.map((p) => p[yKey]);
    const xMax = Math.max(...xs) * 1.08 || 1;
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (yMax - yMin < 0.02) { yMin -= 0.02; yMax += 0.02; }
    const yPad = (yMax - yMin) * 0.18;
    yMin = Math.max(0, yMin - yPad); yMax = Math.min(1, yMax + yPad);

    const sx = (v) => padL + (v / xMax) * plotW;
    const sy = (v) => padT + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, wrap);
    el("title", {}, svg).textContent =
      "Quality against compute spent. Up and to the left is better.";

    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      el("line", { x1: padL, x2: w - padR, y1: y, y2: y,
        stroke: cssVar("--grid"), "stroke-width": 1 }, svg);
      el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", class: "ax-label" }, svg)
        .textContent = (yMax - ((yMax - yMin) / 4) * i).toFixed(3);
    }
    el("line", { x1: padL, x2: w - padR, y1: padT + plotH, y2: padT + plotH,
      stroke: cssVar("--axis"), "stroke-width": 1 }, svg);
    for (let i = 0; i <= 4; i++) {
      const v = (xMax / 4) * i;
      el("text", { x: sx(v), y: h - padB + 16, "text-anchor": "middle", class: "ax-label" }, svg)
        .textContent = Math.round(v);
    }
    el("text", { x: padL + plotW / 2, y: h - 6, "text-anchor": "middle", class: "ax-title" }, svg)
      .textContent = "tokens per turn (all stages, detector included) →";
    el("text", { x: 12, y: padT + plotH / 2, class: "ax-title",
      transform: `rotate(-90 12 ${padT + plotH / 2})`, "text-anchor": "middle" }, svg)
      .textContent = yLabel;

    // Direct labels carry identity here (one hue, N arms), so they must not
    // collide or run off the plot. Flip to the left when the label would
    // overflow the right edge, and stack vertically when two points sit close
    // enough that their labels would overlap.
    const CHAR_W = 6.2, LINE_H = 13, RIGHT_EDGE = w - 4;
    const placed = [];

    valid.slice().sort((a, b) => sy(a[yKey]) - sy(b[yKey])).forEach((p) => {
      const cx = sx(p.tokens), cy = sy(p[yKey]);
      const labelW = p.name.length * CHAR_W;
      const flip = cx + 11 + labelW > RIGHT_EDGE;

      let ly = cy + 4;
      // Nudge down while this label would sit on top of an earlier one.
      for (let guard = 0; guard < 8; guard++) {
        const clash = placed.some(
          (q) => Math.abs(q.y - ly) < LINE_H && Math.abs(q.x - cx) < labelW + 24
        );
        if (!clash) break;
        ly += LINE_H;
      }
      placed.push({ x: cx, y: ly });

      // 2px surface ring so overlapping points stay separable.
      el("circle", { cx, cy, r: 7, fill: "none",
        stroke: cssVar("--surface-1"), "stroke-width": 4 }, svg);
      const dot = el("circle", { cx, cy, r: 5.5, fill: cssVar("--series-1") }, svg);
      hoverable(dot, `<div class="t-title">${p.name}</div>
        <div class="t-row"><span>${yKey}</span><span>${Number(p[yKey]).toFixed(4)}</span></div>
        <div class="t-row"><span>tokens/turn</span><span>${Math.round(p.tokens)}</span></div>
        ${p.over_correction_rate != null
          ? `<div class="t-row"><span>worse rate</span><span>${p.over_correction_rate}</span></div>` : ""}`,
        wrap);

      // A leader line when the label had to move away from its point.
      if (Math.abs(ly - (cy + 4)) > 2) {
        el("line", {
          x1: cx + (flip ? -8 : 8), y1: cy, x2: cx + (flip ? -10 : 10), y2: ly - 4,
          stroke: cssVar("--axis"), "stroke-width": 1,
        }, svg);
      }

      el("text", {
        x: cx + (flip ? -11 : 11), y: ly, class: "point-label",
        "text-anchor": flip ? "end" : "start",
      }, svg).textContent = p.name;
    });

    host.appendChild(wrap);
    host.appendChild(table(
      ["arm", yKey, "tokens/turn", "worse rate"],
      valid.map((p) => [p.name, Number(p[yKey]).toFixed(4), Math.round(p.tokens),
        p.over_correction_rate ?? "—"])
    ));
  }

  /* Cost by stage: a single-series horizontal bar. No legend -- one series. */
  function stageCostBars(host, stages) {
    host.innerHTML = "";
    const rows = stages.filter((s) => s.tokens > 0 || s.latency_s > 0);
    if (!rows.length) {
      host.innerHTML = '<div class="empty">No cost recorded.</div>';
      return;
    }
    const max = Math.max(...rows.map((s) => s.tokens), 1);
    const padL = 92, padR = 62, rowH = 20, gapY = 7;
    const w = Math.max(host.clientWidth || 420, 340);
    const h = rows.length * (rowH + gapY);
    const plotW = w - padL - padR;

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, wrap);
    el("title", {}, svg).textContent = "Tokens spent per pipeline stage";

    rows.forEach((s, i) => {
      const y = i * (rowH + gapY);
      const bw = Math.max(2, (s.tokens / max) * plotW);
      el("text", { x: padL - 9, y: y + rowH / 2 + 4, "text-anchor": "end", class: "ax-title" }, svg)
        .textContent = s.stage;
      const p = el("path", {
        d: barPath(padL, y, bw, rowH, END_R, "right"), fill: cssVar("--series-1"),
      }, svg);
      hoverable(p, `<div class="t-title">${s.stage}</div>
        <div class="t-row"><span>tokens</span><span>${s.tokens}</span></div>
        <div class="t-row"><span>latency</span><span>${s.latency_s.toFixed(3)}s</span></div>
        <div class="t-row"><span>calls</span><span>${s.calls}</span></div>`, wrap);
      el("text", { x: padL + bw + 8, y: y + rowH / 2 + 4, class: "ax-label" }, svg)
        .textContent = `${s.tokens} tok · ${s.latency_s.toFixed(2)}s`;
    });

    host.appendChild(wrap);
  }

  global.TFCharts = {
    outcomeBars, worseKindBars, costQualityScatter, stageCostBars, table, legend, cssVar,
  };
})(window);
