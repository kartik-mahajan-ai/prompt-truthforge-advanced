# Changelog

Dates are when each change was made. [TO FILL: mark where Milestone 1 ended.]

## 2026-11-17
- Prompt iteration runs: corrector P2 v1/v2/v3 and generator P1 v1/v2 re-run on fixed inputs; results in `test-data/prompt_runs/`.
- Test table: 12 inputs (6 edge, 2 invalid) run through the web app's Ask feature; outputs, verdicts and screenshots saved.
- Found and documented: the NLI verifier accepts general-knowledge sentences against unrelated passages (tests T1, T9).
- Fixed the test document set: 7 of 25 "answerable" questions had lost their passages because the same question had also
  been drawn as unanswerable; only answerable questions whose passages are present are now used.
- Web app: `?ask=<question>&autorun=1` link opens the Ask tab with a question (used for the screenshots).
- Colab prototype notebook `code/TruthForge_Demo.ipynb`; diagrams for the report.

## 2026-11-02
- Abstract shortened; LaTeX version of the research report; README.

## 2026-10-29
- Second verifier (MiniCheck) tested; withholding a claim only when both verifiers agree cut withheld correct claims from 272 to 146.

## 2026-10-27
- Human annotation of 50 rewrites: verifier-judged "damage" was right only 6.7% of the time, so rewrites are scored neutrally.
- Claim splitter bug fixed (wrong subject prefixes such as "(The) It ..."), all runs repeated.

## 2026-10-03
- Self-RAG-style and CRAG-style comparison policies; open-corpus test with unanswerable questions; answerability gate
  (refuse before generating when the question's key terms are missing from the retrieved text).
- Refusal sentences no longer treated as claims.

## 2026-10-01
- Verifier tested on real RAGTruth data; thresholds chosen on the training split.

## 2026-09-20
- Pipeline built: retrieval, generation, claim splitting, verification, per-claim router, corrector with output guard.
