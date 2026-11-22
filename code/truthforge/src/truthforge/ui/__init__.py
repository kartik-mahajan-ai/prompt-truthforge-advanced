"""Local web UI for TruthForge.

Stdlib only -- no Flask, no FastAPI -- so the UI runs wherever the core package
runs. Start it with:

    python -m truthforge.ui

The server binds to 127.0.0.1 by design. It loads local models and reads and
writes files under the project directory, so it is not something to expose on a
network.
"""

from .server import create_server, serve

__all__ = ["create_server", "serve"]
