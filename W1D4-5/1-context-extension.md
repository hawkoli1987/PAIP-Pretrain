# Context length extension: revised content for the guide chapter

**The production recipe for long-context LLMs in 2024–2026 has converged on a two-part approach — a large RoPE base frequency during pretraining (ABF) plus a wavelength-aware post-hoc extension method (YaRN or a close variant) — but the details matter more than the labels.** This revision consolidates the proliferation of method names into a single chronological lineage, provides a concrete YaRN walkthrough, confirms MegatronBridge defaults, and addresses the critical question of context-length regression during continued pretraining on short documents.

---

### The baseline: RoPE (Su et al., April 2021)

RoPE encodes position *m* by rotating each pair of hidden dimensions *d* by angle *m × θ_d*, where **θ_d = 10000^(−2d/D)**. Low-indexed dimensions rotate fast (high frequency, short wavelength ~6 tokens at d=0), encoding local position. High-indexed dimensions rotate slowly (low frequency, wavelength exceeding the training context), encoding global position. The model learns to decode these rotation angles into relative distance. The critical limitation: RoPE provides no guarantee for positions beyond the training context — extrapolation fails rapidly.

### Step 1 → Position Interpolation (Chen et al., June 2023, arXiv:2306.15595)

**What it changes:** Linearly compresses all position indices: *m → m/s* where *s = target_length / original_length*. This squeezes extended positions back into the trained range.

**Why it was superseded:** Uniform compression is the problem. When *s* = 32, adjacent tokens at the start of the sequence are only **1/32 of a rotation apart** in every dimension. High-frequency dimensions that previously distinguished token *m* from token *m+1* with a full step of ~1 radian now see a step of ~0.03 radians. **Local position resolution is destroyed**, causing degradation beyond ~8× extension. The fix required treating dimensions differently.

**Status:** ⛔ Phased out as a standalone method. The insight that interpolation beats extrapolation survives in every successor.

### Step 2 → NTK-aware interpolation (bloc97, Reddit, ~June 29, 2023)

**What it changes:** Instead of scaling positions, it increases the base: **b′ = b × s^(D/(D−2))**. This has a dimension-dependent effect: high-frequency dimensions (low *d*) barely change, while low-frequency dimensions (high *d*) are interpolated aggressively. The name comes from Neural Tangent Kernel theory — high-frequency features are harder for networks to learn, so preserving them is critical.

**Why it was superseded:** The base-change formula applies a smooth but rigid mathematical curve across all dimensions. Some dimensions still get partially extrapolated rather than cleanly classified. The fix required an explicit per-dimension decision boundary.

**Status:** ⛔ Phased out as a standalone formula. But the core idea — changing the base frequency — became ABF (see below).

### Step 2.5 → NTK-by-parts and Dynamic NTK (bloc97 and emozilla, July 2023)

Two intermediate contributions appeared within days. **NTK-by-parts** introduced the ramp function that classifies each dimension into "don't interpolate" (high-frequency), "fully interpolate" (low-frequency), and "blend" (medium), based on the ratio of the dimension's wavelength to the training context length. **Dynamic NTK** made the scale factor adjust dynamically at inference time based on current sequence length. NTK-by-parts was absorbed directly into YaRN. Dynamic NTK remains a useful inference-only trick but is not used in training pipelines.

### Step 3 → YaRN (Peng et al., August 31, 2023, arXiv:2309.00071, ICLR 2024)

**What it changes on top of NTK-by-parts:** Two things. First, it formalizes the ramp function with recommended parameters **α=1, β=32** (for LLaMA-family models). Second, it adds **attention temperature scaling**: √(1/t) = 0.1 × ln(s) + 1, applied by scaling Q and K vectors before the dot product. This corrects for the entropy increase that occurs when attention distributes over many more positions.

**Status:** ✅ Actively used. The most widely adopted post-hoc extension method. Used by DeepSeek V3, Qwen 2/2.5/3, and many open-source models. Integrated into HuggingFace Transformers.

### Step 4 → ABF / Adjusted Base Frequency (Code Llama, August 2023; Xiong et al., September 2023)

**What it is:** Simply set θ_base to a large empirical value (500K, 1M, 5M, 10M) during pretraining or continued pretraining. No formula, no post-hoc adjustment — just bake longer-wavelength RoPE frequencies into the model from the start.

**Relationship to NTK-aware:** ABF is the pretraining-time version of the NTK-aware idea. NTK-aware gives a formula for computing the new base from the extension ratio; **ABF practitioners skip the formula and pick a large number empirically**, then train long enough for the model to adapt. In practice, the results converge. Code Llama used θ=1,000,000 in August 2023, predating the formal "ABF" label from Xiong et al. a month later.

**Production values:** Llama 3.1 uses **θ = 500,000**. Qwen 2/2.5/3 uses **θ = 1,000,000**. Qwen 3.5 uses **θ = 10,000,000**. DeepSeek V3 notably kept the standard **θ = 10,000** and relied entirely on YaRN for extension — proving ABF is not strictly required if a strong post-hoc method is used.

**Status:** ✅ Dominant in production. Nearly every frontier model (except DeepSeek V3) uses an elevated base.

### Step 5 → LongRoPE (Microsoft, February 2024, arXiv:2402.13753, ICML 2024)

**What it changes:** Replaces formula-based per-dimension scaling with **evolutionary search** to find optimal rescale factors for each dimension independently. Also handles initial token positions specially (unscaled) and uses a progressive two-stage extension strategy. Achieves 2M+ token context.

**Status:** ✅ Used in Microsoft's Phi-3 and Phi-4 model families. Research frontier.

### Step 6 → LongRoPE2 (Microsoft, February 2025, arXiv:2502.20082, ICML 2025)

**What it changes over LongRoPE:** Introduces needle-driven perplexity evaluation (better search signal than standard perplexity), mixed context window training (original RoPE for short sequences, rescaled for long), and critical dimension boundary detection. Extends LLaMA-3-8B to 128K with only **10B tokens** — 80× fewer than Meta's approach.

**Status:** ✅ Cutting-edge research. The latest in the search-based paradigm.

### Sidebar: DCA is not a RoPE method

**Dual Chunk Attention** (Cheng et al., February 2024, arXiv:2402.17463, ICML 2024) does not modify RoPE frequencies at all. It redesigns the attention computation to chunk the sequence into segments no longer than the training context, remapping relative positions within chunks so the model never sees out-of-distribution position values. DCA decomposes attention into intra-chunk, inter-chunk, and successive-chunk components. It is training-free and orthogonal to RoPE scaling — Qwen 2.5 and Qwen 3 layer DCA on top of ABF + YaRN for inference-time extension to 1M+ tokens. Classify DCA as an **attention pattern modification**, not a positional encoding method.

### What is phased out, what survives

| Method | Status | Why |
|--------|--------|-----|
| Position Interpolation | ⛔ Phased out | Uniform compression destroys local resolution |
| NTK-aware (standalone) | ⛔ Phased out | Rigid formula; YaRN's by-parts ramp is strictly better |
| Dynamic NTK | ⚠️ Inference-only niche | Useful for dynamic inference; not used in training |
| NTK-by-parts | Absorbed into YaRN | Not used independently |
| **ABF** | ✅ Production standard | Simplest, most robust for pretraining |
| **YaRN** | ✅ Production standard | Best post-hoc method with theoretical grounding |
| **LongRoPE/2** | ✅ Active research | Better than YaRN in extreme settings; more complex |
| **DCA** | ✅ Inference complement | Orthogonal to RoPE methods; used by Qwen family |

---

## 2. The production recipe and a concrete YaRN walkthrough

### The two-part recipe

The dominant 2024–2025 approach combines two complementary techniques:

**Part 1 — ABF during pretraining.** Set θ to a large value (500K–10M) during pretraining or the long-context pretraining stage. This provides a native long context window by ensuring all RoPE dimensions have wavelengths well-suited to the target context length. Models typically pretrain at a shorter context (4K–8K) then progressively extend through multiple stages.

**Part 2 — YaRN for post-hoc extension.** When additional context beyond the pretrained length is needed, apply YaRN scaling. This requires only ~0.1% of original pretraining tokens (~1–2B tokens for a 7B model) to fine-tune, or can even be applied training-free with some degradation.

This recipe is not universal. Llama 3.1 uses a custom NTK-by-parts variant (called "llama3" rope type) that shares YaRN's intellectual lineage but uses different parameterization and omits the attention temperature correction. DeepSeek V3 skips ABF entirely (keeping θ=10,000) and relies on YaRN alone with a large factor of 40. But YaRN or a YaRN-like wavelength-aware method is the common thread across nearly all frontier models.

### YaRN step-by-step: extending from 4K to 128K

Here is a concrete worked example using LLaMA-2 7B parameters (head dimension D=128, base=10,000, original context L=4,096, target L′=131,072).

**Step 1 — Compute the scale factor.** s = L′/L = 131,072/4,096 = **32**.

**Step 2 — Compute wavelengths and ratios for each dimension.** For each of the 64 dimension pairs (d = 0 to 63):

- Wavelength: λ_d = 2π × 10000^(2d/128)
- Ratio: r(d) = L / λ_d = 4096 / λ_d

Concrete values at key dimensions: d=0 has λ≈6.3 tokens, r≈652 (extremely high frequency — encodes adjacent token relationships). d=24 has λ≈199 tokens, r≈20.6 (medium). d=44 has λ≈3,535 tokens, r≈1.16 (near the boundary). d=63 has λ≈54,710 tokens, r≈0.075 (very low frequency — wavelength exceeds the entire training context).

**Step 3 — Classify dimensions using the YaRN ramp function.** With α=1, β=32:

- **r(d) > 32 → High-frequency (d ≈ 0–20).** These ~20 dimensions encode local relationships. Their wavelengths fit many complete cycles within the training context. **Do not interpolate.** γ = 1, keep original frequency. Interpolating these would compress the fine position resolution the model relies on for syntax, coreference, and local coherence.

- **1 ≤ r(d) ≤ 32 → Medium (d ≈ 21–44).** These ~24 dimensions get a linear blend: γ = (r − 1)/31. For example, d=32 has r≈6.5, so γ≈0.18, meaning the frequency is scaled to **f′ = f × (0.82/32 + 0.18) = f × 0.206** — heavy interpolation but not complete. These dimensions encode mid-range context relationships and can tolerate partial compression.

- **r(d) < 1 → Low-frequency (d ≈ 45–63).** These ~19 dimensions have wavelengths exceeding the training context — the model never saw a full rotation during training. **Fully interpolate:** divide frequency by s=32. This is safe because these dimensions were never well-calibrated to begin with.

**Step 4 — Apply attention temperature scaling.** √(1/t) = 0.1 × ln(32) + 1 = 0.1 × 3.466 + 1 = **1.347**. This means t ≈ 0.551. Implementation: multiply the RoPE-embedded Q and K vectors by 1.347 before computing attention logits. The intuition: extending context means attention spreads across more tokens, producing flatter (higher-entropy) softmax distributions. The temperature correction **sharpens attention back** to the entropy level the model was trained with. Without this correction, the model's learned attention patterns become "blurry."

**Step 5 — Fine-tune.** The YaRN paper used **400 steps × batch 64 × 64K sequence length ≈ 1.6B tokens** on PG19 long-document data. Learning rate 2×10⁻⁵, AdamW, linear warmup of 20 steps. This represents roughly **0.1% of LLaMA-2's pretraining budget**. The s=32 model was fine-tuned on 64K sequences but successfully extrapolated to 128K, demonstrating YaRN's "train short, test long" capability.

---

## 3. Frontier model comparison with consistent terminology

| Model | θ base | Pretrain ctx | Extension method | Extension stages | Final ctx | Category |
|-------|--------|-------------|-----------------|-----------------|----------|----------|
| **Llama 3.1** | 500,000 | 8K | ABF + custom NTK-by-parts (factor=8, no temp scaling) | 8K → progressive → 128K (~800B tokens) | 128K | ABF + custom by-parts |
| **Llama 4** | Undisclosed | 256K | iRoPE (interleaved NoPE + RoPE + temp scaling) | 256K → 1M–10M mid-training | 1M–10M | Novel architecture |
| **DeepSeek V3** | 10,000 | 4K | YaRN only (factor=40, β_fast=32, β_slow=1, mscale=1.0) | 4K → 32K (1000 steps) → 128K (1000 steps) | 128K | YaRN post-hoc only |
| **Qwen 3** | 1,000,000 | 32K | ABF + YaRN (factor=4) | Pretrain to 32K; YaRN user-applied for 131K | 131K | ABF + YaRN post-hoc |
| **Qwen 3.5** | 10,000,000 | 262K | ABF + YaRN + DCA | Progressive to 262K; YaRN+DCA for 1M | 1M | ABF + YaRN + DCA |
| **Qwen 2.5-Turbo** | 10,000,000 | 32K→262K | ABF (progressive) + YaRN + DCA | 4 stages: 32K→64K→128K→262K; then 4× | 1M | ABF + YaRN + DCA |
| **Gemma 3** | 1,000,000 (global) / 10,000 (local) | 32K | ABF + Position Interpolation (factor=8), hybrid local/global attention | 32K → 128K via PI rescaling | 128K | ABF + PI + hybrid attn |
| **Kimi k1.5** | 1,000,000 | 4K | ABF + progressive training | 4K → 32K → 131K with upsampled long data | 131K | ABF + progressive |
| **Mistral Large** | 1,000,000 | 32K+ | ABF | Single-stage | 128K | ABF only |

Three patterns emerge. First, **ABF is nearly universal** — every model except DeepSeek V3 uses an elevated base. Second, **YaRN (or a close wavelength-aware variant) is the dominant post-hoc method**, used by DeepSeek V3, Qwen 2/2.5/3/3.5, and the community. Llama 3.1's custom "llama3" type shares YaRN's core NTK-by-parts logic but differs in parameterization and omits the temperature correction. Third, **progressive multi-stage training** has become standard — no major model jumps from short to long context in a single step.

---

## 4. MegatronBridge defaults: full causal attention across documents

### The default behavior

When Megatron-Core's GPTDataset packs multiple documents into a single training sequence, it applies **full causal attention across document boundaries by default** — there is no cross-document masking. Tokens from document B can attend to all tokens from document A within the same packed sequence.

This is controlled by two flags — `reset_attention_mask` and `reset_position_ids` — both of which **default to False**. In Megatron-LM's `arguments.py`, both use `action='store_true'`, making them opt-in. The GPTDatasetConfig dataclass defines both as `Optional[bool] = None`, but all official examples (QuickStart, `gpt_config.yaml`) explicitly pass `False`. Multiple GitHub issues confirm the runtime default prints `reset_attention_mask .... False`.

This means position IDs also increment continuously across document boundaries by default — position 0 of document B gets position ID = (length of document A), not 0.

### How to enable cross-document masking

**Method A: Dense mask (legacy).** Pass `--reset-attention-mask --reset-position-ids --eod-mask-loss` to the training script. This calls `get_ltor_masks_and_position_ids()`, which finds EOD tokens, constructs a block-diagonal causal attention mask (each document gets its own causal triangle), and resets position IDs to 0 at each boundary. The resulting mask is dense with shape `[batch, 1, seq_len, seq_len]`, making it **O(S²) in memory** — expensive for long sequences.

**Method B: THD format with cu_seqlens (modern, efficient).** Use FlashAttention's variable-length API via `PackedSeqParams`:

```python
from megatron.core.packed_seq_params import PackedSeqParams

packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,    # e.g., [0, 4096, 8192, 12288]
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format='thd',
)
```

When `qkv_format='thd'` is set, TransformerEngine dispatches to FlashAttention's varlen kernel, which **never computes attention across document boundaries**. Compute scales as O(Σ sᵢ²) rather than O((Σ sᵢ)²). This is the correct approach for long-sequence training.

### Megatron Bridge specifics

In Megatron Bridge, packed sequences for fine-tuning are configured via `PackedSequenceSpecs` in the `FinetuningDatasetConfig`. The `GPTSFTPackedDataset` class has a `return_cu_seqlen` flag, documented as: *"Having cu_seqlen in the model input enables THD attention kernel, which is the correct format for training with packed sequence to prevent cross-sequence attention. This flag should be True unless you have a specific use case."*

An important nuance from the Megatron Bridge documentation: **packed-sequence support is explicitly designed for fine-tuning, not pretraining.** For pretraining, GPTDataset already concatenates documents to fill sequences to the target length, "eliminating padding tokens without requiring the boundary-aware packing infrastructure." This means **standard Megatron-Core pretraining uses naive concatenation without cross-document masking.** This was flagged as an issue in NeMo GitHub Issue #9664 (July 2024), referencing that Llama 3 applied proper document masking during pretraining.

### Practical recommendation

For the CPT recipe in this guide, explicitly enable cross-document masking. Use the THD/cu_seqlens path (`PackedSeqParams` with `qkv_format='thd'`) when packing short documents into long sequences. If using the legacy pretraining path, set `--reset-attention-mask --reset-position-ids`. The ProLong paper (Gao et al., 2024) confirmed that "using document masks in continual long-context training leads to both better long-context results and short-context performance."

---

## 5. CPT on short documents: preserving 128K context capability

This section addresses the most critical practical question: when the base model supports 128K context but your CPT corpus averages only ~4K tokens per document, what context length should you train at?

### Short-context CPT will degrade long-context ability

The evidence is clear and consistent across multiple studies. **Training exclusively at short context lengths causes measurable regression in long-context capabilities**, even when the base model was originally trained for long context.

The mechanism is straightforward. During CPT at 4K, the model's attention weights and projection matrices receive gradient updates only for interactions within 4K-token windows. **RoPE dimensions encoding positions beyond 4K receive zero gradient signal**, and the model's learned attention patterns for long-range dependencies can drift. The LongRoPE2 paper (2025) formalized this: higher RoPE dimensions that don't complete full rotations during training become "insufficiently trained," with empirical rotation periods extending beyond the theoretical — exactly what happens when a 128K model is fine-tuned at 4K.

The ProLong study (Gao et al., 2024) found that short task performance decreases monotonically as long-context data increases in the training mix — but critically, **the inverse is also true**: training only on short data degrades long-context performance. Their optimal balance was **60% long-context data and 40% short-context data**. InternVL2 experiments showed models trained "exclusively on short-context data perform competitively on standard benchmarks but suffer significant degradation on long-context tasks."

A subtle but important caveat: the fundamental RoPE frequency parameters (θ values in `config.json`) are architectural constants that don't change during training. What drifts are the **learned weight matrices** that project into and out of the RoPE-encoded space. These weights encode the model's understanding of how to use positional information, and they can silently lose long-range capability if never exercised at long range.

### What the major model projects do

Every frontier model project uses progressive multi-stage training with careful data composition for long-context stages. The specifics are instructive.

**Llama 3.1** used approximately **800B training tokens** for its long-context extension from 8K to 128K, progressively increasing context length. Meta explicitly monitored two criteria at each stage: (1) short-context evaluation metrics fully recovered, and (2) needle-in-a-haystack tests passed perfectly at the new length. Their SFT data was **99.89% short sequences** (average <1K tokens) with only **0.11% long sequences** (average ~37K tokens) — demonstrating that SFT can be overwhelmingly short if the underlying model's long-context capability is already solid from pretraining.

**DeepSeek V3** extended from 4K to 128K in two stages (4K→32K at 1,000 steps, then 32K→128K at 1,000 steps) using YaRN scaling. The v3.1 update significantly expanded these stages — 10× more data for the 32K stage and 3.3× more for the 128K stage — suggesting the original allocation was insufficient.

**Qwen 2.5-Turbo** used four progressive stages (32K→64K→128K→262K) with a consistent data composition principle: **40% of sequences at the current maximum length, 60% shorter**. This ratio provides the long-range gradient signal while maintaining short-context stability.

**Fu et al. (2024, ICML)** found that **500M–5B tokens** of properly composed long-context data is sufficient to extend a 4K model to 128K retrieval capability. The critical insight: **per-source length upsampling** — increasing the proportion of long sequences while keeping the domain mixture balanced — outperforms naive global upsampling (which skews toward books and other naturally long genres).

### The recommended recipe for CPT at 128K with mostly-short documents

Given a 128K-capable base model (e.g., Qwen3-4B) and a domain corpus averaging ~4K tokens, the following approach preserves long-context ability while maximizing domain adaptation.

**Pack short documents into 128K sequences with cross-document attention masking.** Concatenate 16–32 domain documents per sequence to fill the 128K window. Apply intra-document masking (each document attends only to itself) using FlashAttention's varlen API via `cu_seqlens`. Reset position IDs at document boundaries so positions span the full 0–128K range. This is the single most important step — it ensures the model's RoPE dimensions and long-range attention patterns continue to receive gradient signal at every position up to 128K, even when individual documents are short. ProLong confirmed that cross-document masking during CPT improves both long-context and short-context performance.

**Mix in 15–25% replay data with natively long documents.** Include general-domain long-context data — code repositories, books, long web documents, or technical papers exceeding 32K tokens — as replay data. This serves dual purposes: catastrophic forgetting prevention and long-context maintenance. Code repositories are particularly valuable because they naturally contain long-range dependencies (function definitions referenced thousands of tokens later). Following ProLong's finding, aim for roughly **60% long sequences (including packed domain docs) and 40% naturally short sequences** by token count.

**If you have some natively long domain documents, upsample them aggressively.** If 5–10% of your domain corpus exceeds 32K tokens, upsample these to represent 20–40% of training tokens. Per-source upsampling (maintaining domain balance while increasing long-document proportion) is more effective than global upsampling, per Fu et al.

**Maintain diverse sequence lengths across the batch.** Target approximately 40% of sequences at 128K (packed domain docs + replay), 30% at 32K–64K, and 30% at natural document length (4K–16K). This mirrors Qwen 2.5-Turbo's proven 40/60 long/short ratio while ensuring the model sees its full context range.

**Do not modify the RoPE configuration.** Keep the same base frequency and scaling parameters as the original model. The model already supports 128K — changing RoPE parameters could destabilize existing capabilities without benefit.

**Monitor both short and long-context metrics during training.** Run needle-in-a-haystack evaluations at 4K, 16K, 32K, 64K, and 128K periodically. Monitor standard short-context benchmarks. If long-context performance degrades, increase the proportion of long-sequence training data. Do not rely solely on perplexity — ProLong showed perplexity is a misleading proxy for long-context capability; downstream task evaluation is necessary.

### What not to do

**Do not train exclusively at 4K.** This is the highest-risk choice. The model's attention patterns for positions >4K will receive zero gradient updates, causing silent regression that may only surface when users attempt long-context tasks. The RoPE dimensions encoding long-range position information will effectively be frozen while all surrounding weights shift.

**Do not assume packing alone preserves long-context ability without masking.** Packing documents into long sequences with full cross-document attention (the Megatron-Core default) is different from proper intra-document masking. Full cross-document attention allows the model to "cheat" by attending across document boundaries, which does not exercise the same long-range within-document attention patterns that real long-context use requires.

**Do not skip replay data.** Pure domain-specific CPT without any general data replay risks catastrophic forgetting of both general capabilities and long-context skills. The continual learning literature consistently shows 5–20% replay data prevents the worst degradation.

---

## Conclusion

The RoPE extension landscape simplifies dramatically when viewed as a single evolutionary chain: PI introduced interpolation, NTK-aware shifted it to the base frequency, NTK-by-parts added per-dimension treatment, and YaRN formalized the whole framework with attention temperature correction. In production, this distills to ABF (large base during pretraining) plus YaRN (post-hoc extension), with DCA as an orthogonal inference-time complement. The proliferation of names masks what is fundamentally one idea refined through five iterations.

For the CPT context length question, the evidence converges on a clear answer: **pack short documents into full-length sequences with cross-document masking, mix in replay data with natively long documents, and train at the model's full context length.** The cost of maintaining long-context capability during CPT is modest — proper data composition and packing strategy rather than additional compute. The cost of losing it is much harder to recover from, as it essentially requires repeating the long-context extension stage. The guiding principle from Qwen 2.5-Turbo's recipe — 40% of sequences at maximum length, 60% shorter — provides a practical starting point that can be adjusted based on needle-in-a-haystack monitoring during training.