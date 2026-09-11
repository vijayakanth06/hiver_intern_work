#!/usr/bin/env python3
"""
Interactive CLI Demo for Hiver AI Customer Support Agent.
Run live queries against the fine-tuned DeBERTa classifier, RAG retrieval engine,
and conformal risk-aware escalation policy.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_clean_logging, get_device
from src.agent import UnifiedSupportAgent

setup_clean_logging()


def display_result(result):
    print("\n" + "─" * 70)
    print(f"💬 CUSTOMER QUERY:       \"{result.query}\"")
    print("─" * 70)
    print(f"🎯 PREDICTED INTENT:      {result.predicted_intent} (Confidence: {result.classification_confidence:.1%})")
    print(f"⚙️  CLASSIFIER METHOD:     {result.classification_method}")
    
    if result.should_escalate:
        print(f"🚨 ESCALATION DECISION:   ⚠️  ESCALATE TO HUMAN AGENT")
        print(f"📌 ESCALATION REASON:     {result.escalation_reason}")
        print(f"🛡️ RISK LEVEL:            {result.risk_level.upper()}")
    else:
        print(f"✅ ESCALATION DECISION:   🤖 AUTO-HANDLE (Safe to dispatch)")
        print(f"🛡️ RISK LEVEL:            {result.risk_level.upper()}")

    print(f"\n📝 GROUNDED REPLY DRAFT:")
    print(f"   \"{result.draft_reply}\"")
    
    if result.retrieved_context:
        print(f"\n📚 TOP RETRIEVED CONTEXT:")
        top_ctx = result.retrieved_context[0]
        print(f"   - Match: \"{top_ctx.customer_query[:60]}...\" (Sim: {top_ctx.similarity_score:.3f})")
        print(f"   - Hist. Resolution: \"{top_ctx.historical_resolution[:80]}...\"")

    print(f"\n⏱️  LATENCY:               {result.latency_ms:.1f} ms")
    print("─" * 70 + "\n")


def run_demo(brand: str = "amazonhelp", interactive: bool = False):
    print("\n" + "=" * 75)
    print(f"🚀 HIVER AI SUPPORT AGENT — LIVE DEMO (@{brand.upper()})")
    print(f"Compute Device: {get_device()} | Classifier: Fine-Tuned DeBERTa-v3 LoRA")
    print("=" * 75)

    print("\n[INFO] Loading agent neural models onto GPU VRAM...")
    agent = UnifiedSupportAgent(brand_name=brand, classifier_mode="track_b_deberta")
    print("[INFO] Agent loaded and ready!\n")

    test_queries = [
        "My package was supposed to arrive yesterday and tracking hasn't updated. Where is it?",
        "You charged my credit card twice for Prime renewal and refuse to refund. I will sue and contact the FTC!",
        "My account is locked and someone unauthorized is buying gift cards on my account help!",
        "How do I cancel my Amazon Music subscription? I don't want to be billed next month.",
        "The item I received is broken in pieces and the box was crushed by the delivery driver."
    ]

    if not interactive:
        print("Running 5 representative customer support scenarios:\n")
        for q in test_queries:
            res = agent.process(q)
            display_result(res)
    else:
        print("Entering interactive mode. Type your customer query (or 'exit' to quit):\n")
        while True:
            try:
                user_input = input("Customer Query > ").strip()
                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    print("\nExiting demo. Goodbye!")
                    break
                res = agent.process(user_input)
                display_result(res)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting demo. Goodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="Run Live Interactive Demo")
    parser.add_argument("--brand", type=str, default="amazonhelp", help="Target brand handle")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive terminal mode")
    args = parser.parse_args()

    run_demo(brand=args.brand, interactive=args.interactive)


if __name__ == "__main__":
    main()
