"""Run the real app against a scripted fake OpenRouter (no API key, no network).

For UI development and demos of the trace view only: replies are keyword-routed, not intelligent.
    python scripts/dev_mock_server.py [--port 8000] [--latency 0.4]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--latency", type=float, default=0.4, help="simulated seconds per LLM call")
ap.add_argument("--db", default=os.path.join(tempfile.gettempdir(), "portfolio_mock.db"))
args = ap.parse_args()

os.environ.setdefault("DB_PATH", args.db)
os.environ.setdefault("OPENROUTER_API_KEY", "mock")
os.environ.setdefault("ADMIN_PASSWORD", "admin")

import uvicorn  # noqa: E402

from app import agent  # noqa: E402
from tests.fake_openrouter import FakeOpenRouter  # noqa: E402

agent.TEST_TRANSPORT = FakeOpenRouter(latency_s=args.latency).transport
print(f"MOCK LLM in use. Admin login: any user / password '{os.environ['ADMIN_PASSWORD']}'. DB: {os.environ['DB_PATH']}")
uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")
