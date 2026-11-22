"""Fetch the RAGTruth release into data/raw/ragtruth/.

    python scripts/download_ragtruth.py

RAGTruth (Niu et al., ACL 2024) is ~18k RAG responses annotated at case and
word level. The release lives in the authors' GitHub repo; this script pulls
the two jsonl files from raw.githubusercontent.com.

If the URLs have moved, the script says so and prints manual instructions
rather than silently writing a broken file.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "ParticleMedia/RAGTruth"
BRANCH = "main"
FILES = ["source_info.jsonl", "response.jsonl"]
BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/dataset"

MANUAL = f"""
Automatic download failed. To do it by hand:

  1. git clone https://github.com/{REPO}.git
  2. copy {REPO}/dataset/source_info.jsonl and response.jsonl
     into data/raw/ragtruth/

Then re-run your command. Check the repo first -- the dataset path or branch
may have changed since this script was written (2026-09-20).
"""


def fetch(url: str, dest: Path) -> None:
    print(f"  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "truthforge/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def verify(path: Path) -> int:
    """Confirm the file is jsonl and return its row count."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {i + 1} is not valid JSON") from exc
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/ragtruth")
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"fetching RAGTruth into {out}")
    for name in FILES:
        dest = out / name
        if dest.exists() and not args.force:
            print(f"  {name} already present ({verify(dest)} rows), skipping")
            continue
        try:
            fetch(f"{BASE}/{name}", dest)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"\nfailed on {name}: {exc}")
            dest.unlink(missing_ok=True)
            print(MANUAL)
            return 1

    print("\nverifying:")
    for name in FILES:
        print(f"  {name}: {verify(out / name)} rows")

    print("\nNext: python scripts/gate1_detector_vs_baseline.py --split test --task QA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
