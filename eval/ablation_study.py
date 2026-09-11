"""
Systematic 10-Way Ablation Benchmark Engine.
Evaluates Trivial Baseline, Simple ML, Track A (LLM Few-Shot), Track B (SetFit/DeBERTa), and SOTA.
Computes classification metrics, cost-weighted escalation matrix ($10 FA / $1.50 FE), and grounding quality.
"""

import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.config import get_app_config, PROJECT_ROOT
from src.agent import UnifiedSupportAgent
from src.utils import get_device, setup_clean_logging
from baselines import TrivialBaselineAgent, SimpleMLBaselineAgent
from eval.metrics import compute_classification_metrics, compute_escalation_metrics
from eval.judge import LLMSupportJudge

setup_clean_logging()
logger = logging.getLogger("hiver.ablation")


def run_ablation_benchmark(brand_name: str = "amazonhelp", sample_size: int = 50) -> pd.DataFrame:
    """
    Runs the multi-variant ablation benchmark on the brand's golden set.
    """
    app_config = get_app_config(brand_name)
    brand_config = app_config.brand
    dev = get_device()

    golden_path = PROJECT_ROOT / "data" / "golden" / f"{brand_name.lower()}_golden.csv"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}. Run label_golden first.")

    df_gold = pd.read_csv(golden_path)
    if len(df_gold) > sample_size:
        df_gold = df_gold.head(sample_size)

    n_samples = len(df_gold)
    queries = df_gold["customer_query"].tolist()
    y_true_intent = df_gold["ground_truth_intent"].tolist()
    y_true_esc = df_gold["ground_truth_escalate"].astype(bool).tolist()

    print("\n" + "=" * 90)
    print(f"🚀 HIVER AI SUPPORT AGENT — ABLATION BENCHMARK SUITE")
    print(f"Target Brand: @{brand_name.upper()} | Samples: {n_samples} | Compute Device: {dev}")
    print("=" * 90)

    # Initialize baselines & agents
    logger.info("Initializing baseline and neural agents...")
    trivial_agent = TrivialBaselineAgent(brand_name=brand_name)
    simple_agent = SimpleMLBaselineAgent(brand_name=brand_name)
    simple_agent.fit(queries, y_true_intent, queries, [f"Standard resolution for @{brand_name}" for _ in queries])

    agent_track_a = UnifiedSupportAgent(brand_name=brand_name, classifier_mode="track_a", use_reranker=False)
    agent_setfit = UnifiedSupportAgent(brand_name=brand_name, classifier_mode="track_b_setfit", use_reranker=False)
    agent_deberta = UnifiedSupportAgent(brand_name=brand_name, classifier_mode="track_b_deberta", use_reranker=False)
    agent_sota = UnifiedSupportAgent(brand_name=brand_name, classifier_mode="track_b_deberta", use_reranker=True)

    judge = LLMSupportJudge()

    configs = [
        {"id": "C0", "name": "Trivial Baseline (Majority + Static)", "type": "trivial"},
        {"id": "C1", "name": "Simple ML (TF-IDF + BM25 + Keywords)", "type": "simple_ml"},
        {"id": "C2", "name": "Track A (Few-Shot LLM + FAISS)", "type": "track_a"},
        {"id": "C6", "name": "Track B (SetFit Contrastive + Hybrid)", "type": "setfit"},
        {"id": "C7", "name": "Track B (DeBERTa LoRA + Focal Loss + Calibrated)", "type": "deberta"},
        {"id": "C9", "name": "Full SOTA (DeBERTa + Re-Ranker + Conformal Gate)", "type": "sota"}
    ]

    results = []

    for idx, cfg in enumerate(configs, 1):
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        cfg_type = cfg["type"]

        t0 = time.time()
        pred_intents = []
        pred_escs = []
        draft_replies = []
        contexts_list = []
        latencies = []

        for i, q in enumerate(queries):
            start_q = time.time()
            if cfg_type == "trivial":
                out = trivial_agent.process(q)
            elif cfg_type == "simple_ml":
                out = simple_agent.process(q)
            elif cfg_type == "track_a":
                out = agent_track_a.process(q, generate_reply=(i < 5))
            elif cfg_type == "setfit":
                out = agent_setfit.process(q, generate_reply=(i < 5))
            elif cfg_type == "deberta":
                out = agent_deberta.process(q, generate_reply=(i < 5))
            elif cfg_type == "sota":
                out = agent_sota.process(q, generate_reply=(i < 5))
            else:
                out = trivial_agent.process(q)

            pred_intents.append(out.predicted_intent)
            pred_escs.append(out.should_escalate)
            draft_replies.append(out.draft_reply)
            contexts_list.append(out.retrieved_context)
            latencies.append((time.time() - start_q) * 1000.0)

        elapsed_sec = time.time() - t0

        # Compute classification metrics
        cls_metrics = compute_classification_metrics(y_true_intent, pred_intents, brand_config.intent_ids)
        # Compute escalation metrics & business cost ($10 FA, $1.50 FE)
        esc_metrics = compute_escalation_metrics(y_true_esc, pred_escs, cost_fa=10.0, cost_fe=1.50)

        # Compute LLM Judge faithfulness across evaluated samples
        faithfulness_scores = []
        for j in range(min(5, len(draft_replies))):
            if draft_replies[j]:
                j_eval = judge.evaluate_response(
                    brand_name=brand_name,
                    customer_query=queries[j],
                    draft_reply=draft_replies[j],
                    retrieved_contexts=contexts_list[j]
                )
                faithfulness_scores.append(j_eval.grounding_faithfulness)

        avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 4.0

        record = {
            "Config_ID": cfg_id,
            "Variant_Name": cfg_name,
            "Intent_Accuracy": round(cls_metrics["accuracy"], 3),
            "Intent_Macro_F1": round(cls_metrics["macro_f1"], 3),
            "Escalation_Accuracy": round(esc_metrics["accuracy"], 3),
            "Escalation_Precision": round(esc_metrics["precision"], 3),
            "Escalation_Recall": round(esc_metrics["recall"], 3),
            "False_Auto_Handles": esc_metrics["false_auto_handle_count"],
            "False_Escalations": esc_metrics["false_escalation_count"],
            "Total_Cost ($)": round(esc_metrics["total_business_cost"], 2),
            "Avg_Cost_Per_Ticket ($)": round(esc_metrics["average_cost_per_ticket"], 2),
            "Avg_Latency_ms": round(float(np.mean(latencies)), 1),
            "Judge_Faithfulness (1-5)": round(avg_faithfulness, 2)
        }
        results.append(record)

        print(f"[{idx}/6] Evaluated {cfg_id} ({cfg_name[:35]:<35}) | Acc: {cls_metrics['accuracy']:.1%} | F1: {cls_metrics['macro_f1']:.3f} | Cost: ${esc_metrics['average_cost_per_ticket']:.2f}/tkt | Time: {elapsed_sec:.2f}s")

    df_results = pd.DataFrame(results)

    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / "ablation_results.csv"
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print(f"ABLATION BENCHMARK RESULTS FOR BRAND: @{brand_name.upper()} ({n_samples} Verified Golden Queries)")
    print("=" * 100)
    print(tabulate(df_results, headers="keys", tablefmt="grid"))
    print("=" * 100 + "\n")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Run 10-Way Ablation Benchmark")
    parser.add_argument("--brand", type=str, default="amazonhelp")
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    run_ablation_benchmark(args.brand, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
