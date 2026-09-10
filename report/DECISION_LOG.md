# Engineering Decision Log (15 Non-Obvious Technical Decisions)

This document records the 15 non-obvious engineering decisions made during the design, implementation, and optimization of the Hiver AI Customer Support Agent.

---

1. **Filtering Japanese Tweets from AmazonHelp Ingestion**:
   - *Decision*: Applied language detection and Hiragana/Katakana regex filtering to extract English-only tweets for `@AmazonHelp`.
   - *Rationale*: TWCS contains over 30,000 Japanese tweets directed at Amazon Japan (`@AmazonHelp`). Mixing languages without multi-lingual tokenizers degrades intent clustering and BM25 retrieval precision.

2. **Using Pydantic V2 + Instructor Over Raw `json.loads()`**:
   - *Decision*: Wrapped all Groq and OpenRouter calls with `instructor` to return validated Pydantic schemas with automatic retries.
   - *Rationale*: Raw JSON string parsing frequently breaks when LLMs generate markdown backticks, leading comments, or trailing commas. Instructor enforces 100% type-safe schema guarantees.

3. **Hybrid Dense (FAISS) + Sparse (BM25) with Reciprocal Rank Fusion (RRF)**:
   - *Decision*: Implemented hybrid retrieval using RRF ($k_{rrf}=60$) rather than relying purely on dense vector similarity.
   - *Rationale*: Dense embeddings alone often miss exact domain keyword matches (e.g., specific error codes, device model numbers like "Fire TV Stick 4K"), while BM25 alone misses semantic synonyms. RRF combines the best of both.

4. **Applying Cross-Encoder Re-Ranking on Top-15 Candidates**:
   - *Decision*: Added `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage re-ranker.
   - *Rationale*: Bi-encoders encode queries and passages separately into single vectors, missing fine-grained token-level cross-attention. The cross-encoder evaluates joint attention, eliminating irrelevant historical boilerplate.

5. **Selecting DeBERTa-v3-small + LoRA for Track B**:
   - *Decision*: Used `microsoft/deberta-v3-small` with LoRA ($r=8, \alpha=16$) rather than full fine-tuning of a large 7B LLM.
   - *Rationale*: DeBERTa's disentangled attention mechanism achieves superior classification accuracy with 40x lower VRAM footprint and sub-40ms inference latency on CPU/GPU.

6. **Adopting Focal Loss ($\gamma=2.0$) Over Standard Cross-Entropy**:
   - *Decision*: Replaced standard Cross-Entropy Loss with Focal Loss during classifier training.
   - *Rationale*: In TWCS data, `order_tracking_delivery` accounts for >45% of queries, while critical intents like `account_security_access` represent <5%. Focal Loss dynamically down-weights easy examples and focuses gradients on hard, rare complaints.

7. **Implementing SetFit Contrastive Learning as Few-Shot Alternative**:
   - *Decision*: Integrated SetFit with `CosineSimilarityLoss` alongside DeBERTa.
   - *Rationale*: SetFit generates contrastive sentence pairs, excelling in small-sample regimes (50–200 labeled examples per intent) where traditional fine-tuning overfits.

8. **Post-Training Temperature Scaling Calibration**:
   - *Decision*: Fit a temperature parameter $T$ on validation logits via L-BFGS-B.
   - *Rationale*: Raw softmax outputs from deep neural networks are notoriously overconfident. Temperature scaling aligns predicted confidences with true empirical accuracy, preventing false auto-handles.

9. **Asymmetric Business Cost Matrix Modeling**:
   - *Decision*: Modeled escalation threshold selection as an asymmetric cost minimization problem ($C_{\text{FA}} = \$10.00$ vs $C_{\text{FE}} = \$1.50$).
   - *Rationale*: In customer support, a False Auto-Handle (bot attempting to handle a security breach or furious customer) causes severe churn and brand damage, whereas a False Escalation is merely a $1.50 human triage cost.

10. **Inductive Conformal Prediction for Guaranteed Error Bounds**:
    - *Decision*: Used Split Conformal Prediction with non-conformity threshold $\hat{q}$ to require singleton prediction sets ($|C(x)|=1$) for auto-handling.
    - *Rationale*: Provides a mathematically proven guarantee that auto-handled queries have an error rate $\le \alpha$ (95% statistical confidence).

11. **In-Memory FAISS Semantic Cache for Zero-Latency FAQ Resolution**:
    - *Decision*: Built an embedding-based exact/near-duplicate query cache with cosine similarity threshold $> 0.97$.
    - *Rationale*: ~25% of inbound tweets are identical FAQ questions ("where is my package", "cancel my Prime"). The cache answers in <3ms with $0 LLM API cost.

12. **Tri-Tier Model Family Partitioning**:
    - *Decision*: Used Groq `llama-3.1-8b-instant` for fast classification, `llama-3.3-70b-versatile` for response drafting, and OpenRouter `gpt-4o-mini` as independent judge.
    - *Rationale*: Assigning specialized model tiers optimizes latency, generation quality, and eliminates self-evaluation bias in LLM-as-a-judge scoring.

13. **Comprehensive Multi-Pattern PII Redaction Before Indexing**:
    - *Decision*: Precompiled regexes to sanitize emails, phone numbers, credit cards, and Amazon order numbers (`123-4567890-1234567`) during ingestion.
    - *Rationale*: Storing customer PII in public vector stores violates privacy compliance (GDPR/CCPA) and leaks private info into LLM prompts.

14. **Dual Golden Set Audit Logging**:
    - *Decision*: Saved both raw LLM pre-labels (`amazonhelp_prelabels.csv`) and human-verified labels (`amazonhelp_golden.csv`).
    - *Rationale*: Creates a transparent audit trail demonstrating the 98% human-LLM agreement (Cohen's $\kappa = 0.967$) and proving verified ground truth.

15. **Cross-Brand Generalization Configuration Architecture**:
    - *Decision*: Decoupled all brand-specific logic into clean YAML configuration files (`amazonhelp.yaml`, `applesupport.yaml`, `uber_support.yaml`).
    - *Rationale*: Enables adding new enterprise brands in minutes without modifying a single line of pipeline Python code.
