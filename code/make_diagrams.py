"""Diagrams for the Milestone 2 PDF, drawn from what the code actually does."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#22303C", "#5B6670"
FILL = {"io": "#EEF1F4", "llm": "#FDE6DE", "check": "#E3EEF5", "decide": "#FFF2D9", "out": "#E4F2EA"}
EDGE = {"io": "#8A96A3", "llm": "#E4572E", "check": "#3A6B8C", "decide": "#D99A2B", "out": "#4C9A73"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, text, kind="io", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label="", dashed=False, color=MUTED, lx=0, ly=0.12, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center", fontsize=8.5,
                color=color, style="italic")


def legend(ax, x, y, items):
    for i, (k, t) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y - i * 0.32), 0.28, 0.2, boxstyle="round,pad=0.01", fc=FILL[k], ec=EDGE[k]))
        ax.text(x + 0.38, y - i * 0.32 + 0.1, t, va="center", fontsize=8.5, color=MUTED)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1. Prompt workflow: where the two prompts run
fig, ax = canvas(13, 5.2)
box(ax, 1.1, 3.6, 1.7, 0.8, "User question\n+ documents", "io")
box(ax, 3.3, 3.6, 1.8, 0.8, "Retrieve\n(BM25 + dense)", "check")
box(ax, 5.6, 3.6, 1.9, 0.9, "P1 Generator\nQwen2.5-1.5B", "llm", bold=True)
box(ax, 7.9, 3.6, 1.7, 0.8, "Split into\nclaims (rules)", "check")
box(ax, 10.1, 3.6, 1.8, 0.8, "Verify each claim\n(NLI DeBERTa-v3)", "check")
box(ax, 12.1, 3.6, 1.5, 0.8, "Router\n(rule cascade)", "decide")
box(ax, 12.1, 1.5, 1.6, 0.8, "ACCEPT\nkeep claim", "out")
ax.text(12.1, 0.9, "kept and corrected claims\ngo into the final answer", ha="center", va="top", fontsize=8, color=MUTED)
box(ax, 9.8, 1.5, 1.9, 0.9, "P2 Corrector\nQwen2.5-1.5B", "llm", bold=True)
box(ax, 7.5, 1.5, 1.8, 0.8, "Output guard\n(one sentence?)", "check")
box(ax, 5.2, 1.5, 1.7, 0.8, "ABSTAIN\nremove claim", "decide")
box(ax, 2.6, 1.5, 2.2, 0.8, "Final answer\n+ citations + cost", "out", bold=True)
arrow(ax, 1.95, 3.6, 2.4, 3.6)
arrow(ax, 4.2, 3.6, 4.65, 3.6, "passages", ly=0.22)
arrow(ax, 6.55, 3.6, 7.05, 3.6, "draft", ly=0.22)
arrow(ax, 8.75, 3.6, 9.2, 3.6, "claims", ly=0.22)
arrow(ax, 11.0, 3.6, 11.35, 3.6)
arrow(ax, 12.1, 3.2, 12.1, 1.9, "supported", lx=0.55, ly=0)
arrow(ax, 11.6, 3.25, 10.4, 1.95, "REWRITE", lx=0.45, ly=0.1)
arrow(ax, 8.85, 1.5, 8.4, 1.5)
arrow(ax, 7.5, 1.9, 9.6, 3.2, "re-verify", dashed=True, lx=-0.5, ly=0.1)
arrow(ax, 11.4, 3.25, 5.6, 1.9, "ABSTAIN", color="#D99A2B", lx=1.6, ly=0.35, rad=0.08)
arrow(ax, 4.35, 1.5, 3.7, 1.5)
ax.text(6.5, 4.75, "Prompt workflow: P1 drafts the answer once; P2 runs only for claims the router sends to REWRITE",
        ha="center", fontsize=11, color=INK, weight="bold")
legend(ax, 0.2, 0.75, [("llm", "LLM prompt"), ("check", "deterministic step / model check")])
save(fig, "1_prompt_workflow.png")

# 2. Task decomposition
fig, ax = canvas(14.4, 4.4)
box(ax, 7.2, 3.9, 6.4, 0.75, "Answer a question truthfully from the user's documents", "io", fs=11, bold=True)
steps = [("1. Retrieve\nevidence", "check"), ("2. Question\ncovered?\n(gate)", "decide"),
         ("3. Draft answer\n(P1)", "llm"), ("4. Split into\nclaims", "check"), ("5. Verify each\nclaim", "check"),
         ("6. Decide\nper claim\n(router)", "decide"), ("7. Correct (P2)\nor withhold", "llm"),
         ("8. Assemble\nanswer, cita-\ntions, cost", "out")]
xs = [0.95 + i * 1.78 for i in range(8)]
for x, (t, k) in zip(xs, steps):
    box(ax, x, 2.2, 1.6, 1.05, t, k, fs=9)
    arrow(ax, 7.2, 3.5, x, 2.75, color="#B8C0C8")
subs = {0: "BM25 + bge-small\nfused by RRF", 1: "coverage ≥ 0.55,\nelse widen, refuse", 2: "Qwen2.5-1.5B,\ngreedy",
        3: "line + sentence\nboundaries, offsets", 4: "max entailment\nover premises", 5: "6-rule cascade\n(δ, κ, ρ, γ)",
        6: "guard rejects\necho / long / off-subject", 7: "per-claim citation,\ntoken ledger"}
for i, t in subs.items():
    ax.text(xs[i], 1.45, t, ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.25)
save(fig, "2_task_decomposition.png")

# 3. Input-output flow
fig, ax = canvas(13, 4.2)
box(ax, 1.6, 2.9, 2.6, 0.8, "Documents\n(.txt .md .pdf .docx)", "io")
box(ax, 1.6, 1.4, 2.6, 0.8, "Question\n(free text)", "io")
box(ax, 6.0, 2.15, 3.2, 1.4, "TruthForge\nretrieve → gate → P1 → split →\nverify → route → P2", "decide", bold=True)
outs = [("Answer: only sentences that\npassed verification", 3.4), ("or: \"Not enough evidence\"\n(declined)", 2.5),
        ("Per-sentence citation\n(supporting sentence)", 1.6), ("Dropped sentences, tokens,\ncalls and time per stage", 0.7)]
for t, y in outs:
    box(ax, 11.0, y, 3.4, 0.7, t, "out", fs=9)
    arrow(ax, 7.6, 2.15, 9.3, y)
arrow(ax, 2.9, 2.9, 4.4, 2.35)
arrow(ax, 2.9, 1.4, 4.4, 1.95)
ax.text(1.6, 3.75, "INPUT", ha="center", fontsize=10, color=MUTED, weight="bold")
ax.text(11.0, 4.05, "OUTPUT", ha="center", fontsize=10, color=MUTED, weight="bold")
save(fig, "3_input_output_flow.png")

# 4. RAG pipeline
fig, ax = canvas(13, 4.6)
box(ax, 1.1, 2.6, 1.6, 0.8, "Question", "io")
box(ax, 3.4, 3.5, 1.9, 0.75, "BM25\n(keyword)", "check")
box(ax, 3.4, 1.7, 1.9, 0.75, "Dense retriever\nbge-small-en-v1.5", "check")
box(ax, 5.8, 2.6, 1.9, 0.9, "Reciprocal rank\nfusion (k = 60)\ntop-5 passages", "check")
box(ax, 8.2, 2.6, 2.0, 0.9, "Answerability gate\ncoverage of question\nterms ≥ 0.55?", "decide")
box(ax, 11.0, 3.5, 2.4, 0.8, "P1 Generator\n(answer from passages)", "llm")
box(ax, 11.0, 1.7, 2.4, 0.8, "Widen to top-10;\nstill low → refuse", "decide")
arrow(ax, 1.9, 2.75, 2.45, 3.35)
arrow(ax, 1.9, 2.45, 2.45, 1.85)
arrow(ax, 4.35, 3.4, 4.85, 2.8)
arrow(ax, 4.35, 1.8, 4.85, 2.4)
arrow(ax, 6.75, 2.6, 7.2, 2.6)
arrow(ax, 9.2, 2.8, 9.8, 3.4, "yes", lx=-0.2)
arrow(ax, 9.2, 2.4, 9.8, 1.8, "no", lx=-0.2, ly=-0.12)
arrow(ax, 11.0, 2.1, 11.0, 3.1, "re-check", dashed=True, lx=0.5, ly=0)
ax.text(6.5, 4.35, "Retrieval (RAG) pipeline", ha="center", fontsize=11, color=INK, weight="bold")
save(fig, "4_rag_pipeline.png")

# 5. System architecture: the figure from the IMD report (same system)
shutil.copy(HERE.parent / "paper/figs/fig_architecture.png", OUT / "5_system_architecture.png")
print("wrote 5_system_architecture.png")
