# Mitigating Catastrophic Forgetting in Continual Pre-Training

When continually pre-training (CPT) a foundation model on new domain data, the model tends to forget previously learned capabilities — a phenomenon called **catastrophic forgetting**. This document surveys the three most practical mitigation strategies, ranked by effectiveness and ease of deployment, with detailed implementation walkthroughs for MegatronBridge.

---

## 1. Data Replay

### Why replay works

Replay is the single most effective anti-forgetting technique for CPT at LLM scale. The idea is simple: mix a fraction of the original pre-training data into the new domain training batches. This provides a continuous gradient signal that anchors the model's weights near a region that performs well on both old and new distributions.

Shi et al.'s survey (arXiv:2404.16789, 2024) found that across 41 domain-adaptive pretraining papers, replay was the dominant approach. More sophisticated continual learning methods — EWC (Elastic Weight Consolidation), progressive networks, PackNet — are either too expensive at LLM scale or consistently underperform simple replay.

### Key papers

#### Paper 1: Ibrahim et al. — "Simple and Scalable Strategies to Continually Pre-train Large Language Models"

> Ibrahim, Therien, Gupta, Richter, Anthony, Lesort, Belilovsky, Rish. *Transactions on Machine Learning Research (TMLR)*, 2024. arXiv:2403.08763.

This paper establishes the canonical recipe for CPT. The authors demonstrate that three simple ingredients are sufficient to match the performance of retraining from scratch on the combined corpus:

1. **LR re-warming**: After the original pre-training's cosine schedule has decayed the learning rate to near zero, increase it back up with a linear warmup. This restores the model's ability to learn (plasticity).

2. **LR re-decaying**: Follow the re-warmup with a new cosine decay schedule matched to the new data budget. This ensures proper convergence.

3. **Data replay**: Mix a percentage of old pre-training data into each training batch.

**Intuition**: Cosine LR schedules decay to near-zero, leaving the model unable to adapt to new data. Re-warming fixes this but causes a temporary "stability gap" (transient loss spike on both old and new data). Replay counteracts the forgetting induced by re-warming by continuously reminding the model of its original training distribution.

**Key experimental findings** (tested at 405M and 10B parameters):

| Distribution shift | Replay ratio needed | Forgetting without replay | Forgetting with replay |
|---|---|---|---|
| Weak (English → English, Pile → SlimPajama) | 5% | Moderate | Negligible |
| Strong (English → German) | 25% | +1.39 val loss increase | +0.16 val loss increase |

The replay data is sampled **uniformly at random** from the original pre-training corpus. A constant replay ratio throughout training is sufficient — no dynamic schedule needed. The re-warming phase uses ~1% of total new training tokens for linear warmup.

The paper also proposes **infinite learning rate schedules** (constant or trapezoidal schedules that never decay to zero) as an alternative that avoids re-warming entirely. These are competitive and avoid the stability gap, but are less proven at scale.

#### Paper 2: Gupta et al. — "Continual Pre-Training of Large Language Models: How to (re)warm your model?"

> Gupta, Therien, Ibrahim, Richter, Anthony, Belilovsky, Rish, Lesort. *NeurIPS 2023 Workshop on Distribution Shifts*. arXiv:2308.04014.

This predecessor paper isolates the effect of the LR schedule (without replay). The central finding: **warmup length has negligible effect on final performance**. Even starting directly at the peak LR (0% warmup) works — it creates a larger initial spike but the same final result.

What matters is the **maximum learning rate**, which controls the forgetting–adaptation tradeoff:
- Higher max LR → better downstream adaptation, more upstream forgetting
- Lower max LR → better upstream retention, limited adaptation

**Practical takeaway**: The re-warming fraction can be small (0.5–1% of tokens). The max LR is the primary knob. For CPT, use the same max LR as original pre-training or slightly lower (0.5–1.0× original).

### Most promising method: the Ibrahim et al. recipe

The combination of **re-warm + re-decay + 5–25% replay** is the gold standard. It is simple, scales cleanly to 10B+ parameters, and requires no algorithmic changes — only data mixing and LR schedule configuration.


#### Replay ratio guidelines

| Scenario | Replay % | Example |
|---|---|---|
| Same language, similar domain (English web → English books) | 1–5% | `[0.95, 0.05]` |
| Same language, different domain (English web → medical/legal) | 5–10% | `[0.90, 0.10]` |
| Cross-lingual or strong domain shift | 20–30% | `[0.75, 0.25]` |

**Composition matters**: don't just replay generic text. If the base model has strong math/code capabilities, include math and code data in the replay mix to preserve those skills. Consider splitting replay into subcategories:

```python
blend=(
    ["/data/domain/medical_text_document",     # 75% new domain
     "/data/pretrain/english_web_document",    # 12% general English
     "/data/pretrain/code_document",           # 8% code
     "/data/pretrain/math_document"],          # 5% math
    [0.75, 0.12, 0.08, 0.05]
)
```

---

## 2. Checkpoint Averaging

### Why checkpoint averaging works

Weight averaging exploits a key property of modern neural network loss landscapes: nearby checkpoints from the same training trajectory lie in the same low-loss basin. Averaging them produces a model closer to the basin's center — a flatter minimum that generalizes better. This is essentially free: no additional training, no hyperparameter tuning, zero inference cost.

For CPT specifically, weight averaging provides two distinct benefits:
1. **Variance reduction** (LAWA/EMA): averaging checkpoints from the CPT run itself reduces noise from stochastic gradients
2. **Forgetting control** (WiSE-FT): interpolating between the base pre-trained model and the CPT result directly controls the forgetting–adaptation tradeoff

### Key papers

#### Paper 1: Sanyal et al. — "Understanding the Effectiveness of Early Weight Averaging for Training Large Language Models" (LAWA)

> Sanyal, Kaddour, Kumar, Dokania. arXiv:2306.03241, 2023.

LAWA (Look-Ahead Weight Averaging) maintains a **sliding window average** of the last *k* checkpoints during training. Unlike SWA (which averages from a fixed starting point) or EMA (which requires tuning a decay coefficient), LAWA uses a simple uniform average of recent checkpoints and is nearly hyperparameter-free.

**Intuition**: SGD-based training produces iterates with high variance (noisy gradients). Averaging nearby iterates reduces this variance without introducing significant bias, because the loss landscape is smooth enough that nearby checkpoints have similar loss. The sliding window (rather than cumulative average) ensures responsiveness — the average tracks the current phase of training rather than being anchored to early, suboptimal weights.

**Key results on Pythia suite (70M–12B parameters)**:
- LAWA with k=5 spaced every 1000 steps consistently outperforms the final checkpoint by **0.5–1.5 perplexity points**
- The improvement appears early in training, not just at convergence
- LAWA matches or exceeds EMA with optimally tuned decay, while requiring no hyperparameter search
- Downstream task improvements are consistent (LAMBADA, HellaSwag, ARC)

**Practical recipe**:

| Parameter | Value |
|---|---|
| Number of checkpoints (k) | 3–6 (5 is the robust default) |
| Checkpoint spacing | 500–2000 steps |
| When to start | After warmup phase |
| Storage | Only need to keep k most recent checkpoints |
| Computation | w_avg = (1/k) * sum(w_{t-i}) for i=0..k-1 |

**Implementation note**: LAWA can be applied purely at evaluation/inference time — no change to the training loop. Save checkpoints at regular intervals, average the last k when you need to evaluate or deploy.

#### Paper 2: Wortsman et al. — "Robust Fine-Tuning of Zero-Shot Models" (WiSE-FT)

> Wortsman, Ilharco, Kim, Li, Simon, Beyer, Kornblith, Chen, Hajishirzi, Farhadi, Schmidt. *CVPR 2022*. arXiv:2109.01903.

WiSE-FT addresses the core tension of CPT: fine-tuning on a target distribution improves domain performance but degrades general capabilities. The solution is strikingly simple — **linearly interpolate** between the original pre-trained weights and the fine-tuned weights:

```
w_final = alpha * w_pretrained + (1 - alpha) * w_finetuned
```

**Intuition**: The pre-trained model and the fine-tuned model occupy different regions of weight space, each excelling in different areas. Because modern overparameterized networks exhibit **linear mode connectivity** (low loss along the interpolation path between models from the same initialization), the interpolated model inherits strengths from both endpoints. The coefficient alpha is a continuous knob that directly controls the forgetting–adaptation tradeoff.

**Key results (CLIP ViT models, confirmed on LLMs in subsequent work)**:
- With alpha=0.5, WiSE-FT **simultaneously improves** both in-distribution accuracy and out-of-distribution robustness compared to standard fine-tuning
- The Pareto frontier traced by varying alpha strictly dominates both endpoints — there is essentially no tradeoff, just a free lunch
- Multiple LLM CPT studies confirm WiSE-FT recovers 50–80% of forgotten general capability while retaining 90%+ of domain gains

**For CPT specifically**: After continual pre-training on domain data, interpolate:
```
w_final = alpha * w_base + (1 - alpha) * w_after_CPT
```
Sweep alpha on a validation set that tests both general capability and domain performance. Typical sweet spot: alpha = 0.3–0.5.

#### Paper 3: Wortsman et al. — "Model Soups" (complementary technique)

> Wortsman et al. *ICML 2022*. arXiv:2203.05482.

Model Soups averages weights of multiple fine-tuned models with **different hyperparameter configurations** from the same pre-trained initialization. "Greedy soups" (iteratively adding models only if they improve validation performance) yield the best results.

**CPT application**: Run 3–5 CPT experiments varying learning rate or data mix, then average the resulting checkpoints using greedy selection. This is more robust than betting on a single hyperparameter configuration.

### Most promising method: LAWA + WiSE-FT two-stage approach

The combined recipe for CPT with minimal forgetting:
1. Run CPT with replay, saving checkpoints every N steps
2. Apply **LAWA** (average last 5 checkpoints) to get a strong CPT endpoint
3. Apply **WiSE-FT**: interpolate the LAWA result with the base pre-trained weights at alpha=0.3–0.5
4. Evaluate on both domain and general benchmarks; tune alpha if needed


**Memory cost**: EMA requires storing a full copy of model parameters in GPU memory. For a 7B model in bf16, this is ~14 GB additional VRAM per GPU (or per TP shard). This may be prohibitive for memory-constrained setups. LAWA (Approach A) has no memory overhead during training.

## 3. LoRA (Parameter-Efficient Fine-Tuning)

### Why LoRA underperforms for CPT

LoRA (Low-Rank Adaptation) constrains weight updates to a low-rank subspace: `W' = W + BA` where `B ∈ R^{d×r}` and `A ∈ R^{r×d}` with rank `r << d`. This acts as an implicit regularizer — the model cannot move far from the pre-trained weights. For instruction fine-tuning, this is a feature (you want to stay close to the base model). For CPT, where the model needs to absorb substantial new knowledge, this is a bug.

### Key papers

#### Paper 1: Biderman et al. — "LoRA Learns Less and Forgets Less"

> Biderman, Gonzalez Ortiz, Portes, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. *TMLR*, 2024. arXiv:2405.09673.

This paper provides the definitive comparison of LoRA vs. full fine-tuning across both instruction tuning and CPT. The title captures the core finding: LoRA both **learns less** and **forgets less** than full fine-tuning, and the gap is especially wide for CPT.

**Key results on Llama-2-7B/13B**:

| Metric | Full FT | LoRA (r=256) | Gap |
|---|---|---|---|
| Code CPT: HumanEval (20B tokens) | 0.263 | 0.224 | -15% |
| Code CPT: data efficiency | 4B tokens to match | 20B tokens to match | **5× worse** |
| Source domain retention | Moderate forgetting | Minimal forgetting | LoRA wins |

**Why the gap exists**: Full fine-tuning learns weight perturbations with **10–100× higher effective rank** than typical LoRA configurations. CPT requires reorganizing internal representations across many layers to accommodate new domain knowledge — a fundamentally high-rank operation that LoRA's low-rank constraint cannot efficiently represent.

**The gap widens with more data**: As the CPT dataset grows, full FT continues improving while LoRA saturates. The low-rank subspace becomes a bottleneck for absorbing large volumes of new knowledge. LoRA at 20B tokens roughly matches full FT at 4B tokens.

**Best LoRA settings for CPT** (if forced to use it):

| Parameter | Recommendation |
|---|---|
| Rank | r=128 minimum, r=256 preferred |
| Alpha | alpha = 2 * rank |
| Target modules | **All** linear layers (Q, K, V, O, up, gate, down) |
| Learning rate | 2e-4 to 1e-3 (2–10× higher than full FT) |
| Dropout | 0.0 to 0.05 (minimal — LoRA is already regularizing) |

#### Paper 2: Shuttleworth et al. — "LoRA vs Full Fine-tuning: An Illusion of Equivalence"

> Shuttleworth, Andreas, Torralba, Sharma. arXiv:2410.21228, 2024.

This paper provides a mechanistic explanation for why LoRA underperforms: the **intruder dimensions** phenomenon.

**Concept**: When fine-tuning changes a weight matrix, new task-critical singular directions emerge — "intruder dimensions" that appear in the top singular values of the fine-tuned model but were absent in the pre-trained model. Full FT integrates these intruders smoothly into the existing spectral structure. LoRA's additive low-rank update `W + BA` creates **spectral outliers** instead — the intruder dimensions don't integrate naturally, producing a distorted singular value spectrum.

**Consequences**:
- The distorted spectrum makes LoRA-trained models more **brittle** — they may score well on target benchmarks but generalize worse and compose poorly with other adapters
- Higher rank partially mitigates this but doesn't solve it, because the additive structure `W + BA` is fundamentally different from an unconstrained weight update
- The optimal rank for minimal forgetting follows a **U-shaped curve** with r=64 at the minimum — very low rank under-learns, very high rank over-disturbs the pre-trained spectrum

**Practical implication**: If you must use LoRA for CPT, monitor not just benchmark scores but also OOD generalization and general capability retention. The "equivalence" between LoRA and full FT that appears on narrow benchmarks is an illusion.

#### Paper 3: Liang & Li — "InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning"

> Liang, Li. *CVPR 2024*. arXiv:2404.00228.

InfLoRA addresses **sequential multi-task continual learning with LoRA** by constraining each new task's adapter to operate in a subspace **orthogonal** to previous tasks' adapters. This prevents inter-task interference: Task 2's learning cannot overwrite Task 1's knowledge because they use perpendicular directions in weight space.

**Relevance to CPT**: Most applicable when doing **staged CPT** (e.g., first adapt to code, then to medical text, then to legal text). For single-domain CPT, InfLoRA offers less benefit. The fundamental tradeoff remains: the available subspace shrinks with each successive task, limiting how much new knowledge can be absorbed.

### Decision framework

```
                            ┌─────────────────────────┐
                            │  Can you afford full FT  │
                            │  (memory + compute)?     │
                            └────────┬────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                       YES                        NO
                        │                         │
                        ▼                         ▼
              ┌─────────────────┐     ┌──────────────────────┐
              │  Use full FT    │     │  Is it single-domain  │
              │  + replay       │     │  or multi-stage CPT?  │
              │  + LAWA/WiSE-FT │     └────────┬─────────────┘
              └─────────────────┘              │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                                 Single               Multi-stage
                                    │                     │
                                    ▼                     ▼
                          ┌──────────────────┐  ┌─────────────────────┐
                          │  LoRA r=128-256   │  │  InfLoRA/O-LoRA     │
                          │  all layers       │  │  orthogonal adapters │
                          │  LR=2e-4 to 1e-3 │  │  per stage           │
                          └──────────────────┘  └─────────────────────┘
```


## Summary: Combined Anti-Forgetting Recipe

For maximum protection against catastrophic forgetting during CPT, combine all three strategies:

| Stage | Technique | Cost | Forgetting reduction |
|---|---|---|---|
| **During training** | Data replay (5–25%) | Minimal (just data mixing) | High |
| **During training** | EMA (decay=0.9999) | +1× model memory | Moderate |
| **After training** | LAWA (average last 5 ckpts) | Zero (post-hoc) | Moderate |
| **After training** | WiSE-FT (alpha=0.3–0.5) | Zero (post-hoc) | High |

**Minimum viable recipe**: Data replay + WiSE-FT. These two together provide the strongest forgetting protection at the lowest implementation cost.
