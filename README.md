# TruthForge: Verified Answers from Your Documents

SCSE4041 Mini Project · Milestone 2 · Team TruthForge (PE-B01-34)

TruthForge answers a question **only from the documents you give it**, checks every sentence of its own
answer against those documents, removes or corrects sentences that are not supported, and says
**"Not enough evidence"** when the documents do not cover the question.

**Prototype:** [TO FILL: Colab link to code/TruthForge_Demo.ipynb, shared as 'Anyone with the link']

## Team
- Kartik Mahajan (S24CSEU0640) – team leader
- Aaditya Goel (S24CSEU0614)

## How to open and run it
- **Colab (easiest):** open `code/TruthForge_Demo.ipynb` in Google Colab, choose Runtime → Change runtime type → T4 GPU,
  then Runtime → Run all. The last cell opens a form where you can type your own question.
- **Locally (Windows/Linux, NVIDIA GPU):** the TruthForge code is in `code/truthforge/`.
  ```bash
  cd code/truthforge
  pip install -e ".[models]"
  python scripts/download_ragtruth.py      # RAGTruth data (35 MB) into code/truthforge/data/raw/
  python -m truthforge.ui --open           # web app; use the Ask tab
  ```
- **Re-running the tests and prompt experiments** (from the top of this folder, after the steps above):
  `python code/run_tests.py`, `python code/prompt_experiments.py`, `python code/screenshots.py T1 T2 T8`.
- **Rebuilding the report:** fill `submit/team_info.json`, run `python code/build_m2_docs.py`, then
  `powershell -File code/export_pdfs.ps1` (needs Microsoft Word).

## Prompts
Two final prompts on Qwen2.5-1.5B-Instruct (greedy, 64 tokens): `prompts/final/P1_generator.txt` (drafts the
answer) and `prompts/final/P2_corrector.txt` (rewrites one unsupported claim). Older versions are in
`prompts/v1`, `v2`, `v3`.

## Folder contents
| Folder | Contents |
|---|---|
| `prompts/` | v1, v2, v3 and final prompt versions, one `.txt` per prompt |
| `code/` | Colab notebook (prototype), `run_tests.py`, `prompt_experiments.py`, `screenshots.py`, diagram and report builders |
| `code/truthforge/` | the TruthForge system itself (Python package, scripts, unit tests) |
| `test-data/` | the 84 test documents, `results.csv`, `verdicts.json`, one JSON per test in `outputs/`, prompt-version runs in `prompt_runs/` |
| `docs/` | diagrams, screenshots, the Milestone 2 PDF |
| `submit/` | the two PDFs to upload, form answers, reel script |

## Results
- 12 tests (6 edge, 2 invalid): 9 pass, 3 fail (see `test-data/verdicts.json`).
- Prompt iteration: corrector v1 → v2 removed all format violations (5 → 0 of 40); v3 gave no
  further measurable gain.
