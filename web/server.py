#!/usr/bin/env python3
"""
Lightweight Web Server & API for Hiver AI Customer Support Agent Interactive UI.
Provides REST endpoint `/api/predict` for live inference and serves static dashboard assets.
"""

import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_clean_logging, get_device
from src.agent import UnifiedSupportAgent

setup_clean_logging()

# Global cached agent instance
_AGENT_CACHE = {}


def get_agent(brand_name: str = "amazonhelp", mode: str = "track_b_deberta", use_reranker: bool = True):
    key = f"{brand_name}_{mode}_{use_reranker}"
    if key not in _AGENT_CACHE:
        classifier_mode = "track_b_deberta" if "deberta" in mode else ("track_b_setfit" if "setfit" in mode else "track_a")
        try:
            agent = UnifiedSupportAgent(brand_name=brand_name, classifier_mode=classifier_mode, use_reranker=use_reranker)
            _AGENT_CACHE[key] = agent
        except Exception as e:
            print(f"[WARNING] Could not load live neural agent ({e}). Falling back to lightweight mode.")
            agent = UnifiedSupportAgent(brand_name=brand_name, classifier_mode="track_a", use_reranker=False)
            _AGENT_CACHE[key] = agent
    return _AGENT_CACHE[key]


class AgentDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "web"), **kwargs)

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                payload = json.loads(body)
                query = payload.get("query", "").strip()
                mode = payload.get("mode", "deberta")
                use_reranker = payload.get("use_reranker", True)

                if not query:
                    self.send_error(400, "Query cannot be empty")
                    return

                # Execute agent inference
                agent = get_agent(brand_name="amazonhelp", mode=mode, use_reranker=use_reranker)
                out = agent.process(query)

                response_data = {
                    "query": out.query,
                    "brand": out.brand,
                    "predicted_intent": out.predicted_intent,
                    "classification_confidence": out.classification_confidence,
                    "classification_method": out.classification_method,
                    "should_escalate": out.should_escalate,
                    "escalation_reason": out.escalation_reason,
                    "risk_level": out.risk_level,
                    "draft_reply": out.draft_reply,
                    "retrieved_context": [
                        {
                            "passage_id": c.passage_id,
                            "customer_query": c.customer_query,
                            "historical_resolution": c.historical_resolution,
                            "similarity_score": c.similarity_score
                        }
                        for c in out.retrieved_context[:3]
                    ],
                    "latency_ms": out.latency_ms
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AgentDashboardHandler)
    print("\n" + "=" * 80)
    print(f"🚀 HIVER AI SUPPORT AGENT — INTERACTIVE DASHBOARD & DEMO SERVER")
    print(f"Server URL:    http://localhost:{port}")
    print(f"Compute:       {get_device()}")
    print("=" * 80 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Interactive Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
