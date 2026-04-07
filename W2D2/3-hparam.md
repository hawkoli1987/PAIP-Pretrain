## LR and batch size sweeps

### Learning rate sweep ranges by model size


| Model size | Original peak LR | CPT sweep range  | Recommended grid               |
| ---------- | ---------------- | ---------------- | ------------------------------ |
| 7B–13B     | ~3e-4            | **1e-5 to 3e-4** | {1e-5, 3e-5, 5e-5, 1e-4, 2e-4} |
| 34B        | ~2e-4            | **1e-5 to 1e-4** | {1e-5, 3e-5, 5e-5, 1e-4}       |
| 70B        | ~1.5e-4          | **1e-5 to 5e-5** | {1e-5, 2e-5, 3e-5, 5e-5}       |


These ranges are derived from published CPT runs: Code Llama used 1e-4 across scales for 500B-token code CPT, Llemma 34B used 5e-5 for 50B-token math CPT, and Databricks swept {1e-5, 3e-6, 1e-5, 3e-5} for Llama-2-7B CPT on 14.5B tokens. DeepSeek's pretraining LR decreased from 4.2e-4 (7B) to 3.2e-4 (67B), roughly a 24% reduction for 10× more parameters.

A critical insight from "Scaling Optimal LR Across Token Horizons" (arXiv:2409.19913): **optimal LR decreases with training duration**. If you sweep LR using 1B-token pilot runs but train for 20B tokens, the optimal LR for the full run will be somewhat lower. Apply a ~20–30% reduction heuristic when extrapolating from short sweeps.

### Batch size and the critical batch size concept

McCandlish et al. ("An Empirical Model of Large-Batch Training," arXiv:1812.06162, 2018) defined the **gradient noise scale** B_noise = tr(Σ)/‖G‖², which predicts the critical batch size—the point where returns from increasing batch size begin to diminish. Below the critical batch size, doubling the batch roughly halves the required steps; above it, you get diminishing returns.

Practical batch sizes for CPT at this scale: **4M–16M tokens per batch**. Code Llama and Llemma both used 4M tokens. DeepSeek-V3 ramped from 12.6M to 63M tokens over training. A recent revisitation (Merrill et al., arXiv:2505.23971, 2025) found that critical batch size starts near zero, rises rapidly, then plateaus during training, and does not depend strongly on model size—supporting the common practice of **batch size warmup** (starting at ~50% of target and ramping over the first 5–10% of training).

### LR-batch size scaling for Adam

For Adam-family optimizers, the **square root scaling rule** applies: when doubling batch size, increase LR by ~1.4× (√2), not 2×. This was theoretically derived by Malladi et al. (NeurIPS 2022) via stochastic differential equation analysis. Li et al. (arXiv:2405.14578, NeurIPS 2024) discovered a "surge phenomenon" where optimal LR first rises then falls as batch size increases, with a sweet spot near the gradient noise scale. The practical implication: there is an optimal batch size beyond which even with the correct LR scaling, further increases hurt.

For a joint sweep, use **4–5 LR values × 3–4 batch size values = 12–20 grid points**, each trained for 1–3B tokens. This is sufficient to identify the Pareto-optimal LR-BS combination.

### μP for transferring hyperparameters across scales

**μP (Maximal Update Parameterization)** from Yang et al. ("Tensor Programs V," arXiv:2203.03466, 2022) enables zero-shot HP transfer across model widths. The protocol: parametrize your target model in μP, tune HPs on a small proxy (~40M params), and transfer to 7B–70B. Yang et al. demonstrated transfer from 40M→6.7B (GPT-3 scale), outperforming published GPT-3 numbers at just **7% of total tuning cost**.

The Cerebras/EleutherAI practitioner's guide recommends: choose a proxy model (~40M params, d_model≥256) with depth roughly matching the target; run 200–350 random HP samples; select top-10 by validation loss; validate at intermediate scale (256M–590M); then transfer to the target. Critical practical advice: **batch size must exceed the critical batch size even for the proxy model**, and **depth matters**—proxy depth should approximate target depth.

Everett et al. ("Scaling Exponents Across Parameterizations and Optimizers," arXiv:2407.05872, ICML 2024) showed that all parameterizations can achieve HP transfer with the right per-layer LR exponents, and proposed **Adam-atan2**, a scale-invariant variant that eliminates the epsilon parameter entirely. Their per-layer LR prescription for standard parameterization actually outperformed μP in some settings, validated up to **26.8B parameters**.

**Important limitation for CPT**: μP was designed for training from scratch. For CPT, the model already has learned weights and the optimization landscape differs from random initialization. No published work specifically validates μP for continued pretraining. Practitioners typically use μP-informed scaling heuristics (how LR scales with width) rather than full μP reformulation for CPT.

---

## Secondary hyperparameters and stability techniques

### Weight decay, gradient clipping, and warmup

**Weight decay = 0.1** is universal across Llama 3, DeepSeek-V3, and Qwen 2.5. For CPT sweeps, {0.01, 0.05, 0.1, 0.3} covers the relevant range. A recent finding from Han et al. (arXiv:2602.11137, 2025) suggests that higher weight decay (0.3–1.0) during pretraining makes models more plastic and better at adapting to downstream tasks—potentially relevant for CPT where adaptability is the goal. Always use **decoupled** weight decay (AdamW), never L2 regularization in Adam (Loshchilov & Hutter, 2017).

**Gradient clipping at 1.0 (max grad norm)** is standard and rarely needs adjustment for CPT. Monitor whether clipping activates frequently (>10% of steps)—if so, the LR is likely too high. For extra stability, AdaGC (arXiv:2502.11034, 2025) offers adaptive per-tensor clipping using EMA of historical gradient norms and eliminated all loss spikes on Llama-2 7B/13B.

**Warmup fraction**: 1–2% of total CPT steps (or ~2000 steps as a fixed count). Peak LR matters far more than warmup length. "Reuse, Don't Retrain" (arXiv:2407.07263) actually found that **no warmup achieved the best evaluation results** in their experiments on a 15B model, contradicting the conventional wisdom—suggesting that practitioners should include warmup={0, 0.5%, 1%, 2%} in their sweep.

### Data mixing ratios to prevent catastrophic forgetting

The replay ratio depends on the severity of domain shift:

- **Mild shift** (general→domain-specific English): **5% replay** is sufficient (Ibrahim et al. 2024's validated recipe).
- **Moderate shift** (e.g., math/code specialization): **10–20% general data**. Llemma used 95% math + 2% general + 3% code. GeoGalactica used 8:1:1 (domain:arXiv:code).
- **Strong shift** (e.g., new language): Up to **50% replay** needed. AMD/ROCm's Poro 2 multilingual CPT blog reported that with strong distribution shifts, 50% replay of original data may be necessary.

Gu et al. (EMNLP 2024) discovered a **Critical Mixture Ratio (CMR)** scaling law—a power-law relationship between loss, mixture ratio, and training tokens. This provides a principled formula for the optimal ratio rather than relying on rules of thumb. The "Reuse, Don't Retrain" paper advocates a **two-phase data strategy**: Phase 1 uses a general blend, Phase 2 shifts to a quality blend with aggressive upweighting of the target domain, with the switch point tuned as a hyperparameter.

When original pretraining data is unavailable (typical for open models), Llemma's approach of using **The Pile as a surrogate** for the unknown original distribution is a practical workaround. The "Efficient Continual Pre-training by Mitigating the Stability Gap" paper (arXiv:2406.14833) recommends following the original pretraining data mixture and replacing only the web-crawl portions with domain-specific data.

### Stability tricks

**Z-loss** (from PaLM, Chowdhery et al. 2024) adds an auxiliary penalty on the log-sum-exp of logits, encouraging the softmax normalizer to stay near zero. Coefficient of **~1e-4** is standard. Used in PaLM, OLMo, and Chameleon. **Logit soft-capping** (from Gemma 2) applies tanh to compress logits into a bounded range before softmax—Rybakov et al. (arXiv:2410.16682, 2024) showed it allows **1.5× higher learning rates** without divergence. **QK-norm** (Dehghani et al. 2023) applies LayerNorm to Q and K before attention, preventing attention logit explosion, though it can hurt long-context performance. If the base model was pretrained without these architectural features, **do not add them during CPT**—they require significant re-adaptation. Z-loss is the safest addition since it only modifies the loss function, not the architecture.

### Dropout: almost never for CPT

**Dropout = 0** is universal for modern LLM pretraining, and Liu et al. (arXiv:2505.24788, ACL 2025) confirmed that single-epoch training without dropout outperforms training with dropout on downstream tasks. The exception is **multi-epoch CPT on small domain corpora**: Muennighoff et al. ("To Repeat or Not To Repeat," arXiv:2305.13230, 2023) found dropout of **0.1** is highly effective at mitigating multi-epoch degradation. If your domain dataset is small enough to require multiple epochs, add dropout of 0.05–0.1. Otherwise, keep it at zero.

---

