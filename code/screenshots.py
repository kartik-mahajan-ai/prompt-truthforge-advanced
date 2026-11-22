"""Screenshots of real runs in the TruthForge web app (Ask tab), for the test table.

    python "Milestone 2/code/screenshots.py" T1 T5 T8

Starts the app on port 8431, uploads test-data/documents/, then opens
http://127.0.0.1:8431/?ask=<question>&autorun=1 in headless Edge for each test
and saves docs/screenshots/<test>.png. Run run_tests.py first.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE / "code" / "truthforge"   # this folder's copy of TruthForge
PORT = 8431
BASE = f"http://127.0.0.1:{PORT}"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main(tests) -> int:
    import os
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))   # run this folder's copy, not an installed one
    server = subprocess.Popen([sys.executable, "-m", "truthforge.ui", "--port", str(PORT)], cwd=REPO, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/api/status", timeout=2)
                break
            except OSError:
                time.sleep(1)
        for f in sorted((HERE / "test-data/documents").glob("*.txt")):
            post("/api/docs/upload", {"filename": f.name, "content_b64": base64.b64encode(f.read_bytes()).decode()})
        # warm the models so the screenshot run is not a cold start
        post("/api/ask", {"question": "warm up"})
        out = HERE / "docs/screenshots"
        out.mkdir(parents=True, exist_ok=True)
        for t in tests:
            q = json.loads((HERE / f"test-data/outputs/{t}.json").read_text(encoding="utf-8"))["question"]
            url = f"{BASE}/?ask={urllib.parse.quote(q)}&autorun=1#ask"
            png = out / f"{t}.png"
            subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--window-size=1280,1500", "--virtual-time-budget=90000",
                            f"--screenshot={png}", url], timeout=300, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(t, "->", png if png.exists() else "FAILED")
    finally:
        server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["T1", "T5", "T8"]))
