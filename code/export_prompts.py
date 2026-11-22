"""Write every prompt version to prompts/ (final versions are taken from the code itself)."""

import inspect
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code" / "truthforge" / "src"))

from truthforge.correct.rewrite import REWRITE_PROMPT  # noqa: E402
from truthforge.loop import TruthForgeLoop  # noqa: E402

P = HERE / "prompts"

# P1 final: the template inside TruthForgeLoop._generate_answer, with its f-string
# variables turned into {evidence} / {question} placeholders.
src = inspect.getsource(TruthForgeLoop._generate_answer)
body = re.search(r"prompt = \((.*?)\n\s*\)", src, re.S).group(1)
parts = re.findall(r'f?"((?:[^"\\]|\\.)*)"', body)
P1_FINAL = "".join(parts).encode().decode("unicode_escape")
assert "{evidence}" in P1_FINAL and "{question}" in P1_FINAL, P1_FINAL

P1_V1 = """Answer the question using the passages.

PASSAGES:
{evidence}

QUESTION: {question}

ANSWER:"""

P2_V1 = """Correct this sentence so that it matches the evidence.

EVIDENCE:
{evidence}

ANSWER:
{answer}

SENTENCE:
{claim}

CORRECTED SENTENCE:"""

P2_V2 = """Correct ONE sentence so that it matches the evidence.

EVIDENCE:
{evidence}

ANSWER:
{answer}

SENTENCE TO CORRECT:
{claim}

Rules:
- Output exactly ONE sentence: the corrected version of SENTENCE TO CORRECT.
- No preamble, no explanation, no quotes, no bullet points.
- Do not add facts absent from the evidence.

CORRECTED SENTENCE:"""

files = {
    "v1/P1_generator.txt": P1_V1,
    "v2/P1_generator.txt": P1_FINAL,
    "final/P1_generator.txt": P1_FINAL,
    "v1/P2_corrector.txt": P2_V1,
    "v2/P2_corrector.txt": P2_V2,
    "v3/P2_corrector.txt": REWRITE_PROMPT,
    "final/P2_corrector.txt": REWRITE_PROMPT,
}
for name, text in files.items():
    (P / name).write_text(text.strip("\n") + "\n", encoding="utf-8")
    print("wrote prompts/" + name)
