
## AdamW remains the safe choice; Muon is unproven for CPT

### AdamW: the industry standard configuration

Every major lab converges on nearly identical AdamW settings for LLM training. **β₁=0.9, β₂=0.95, weight decay=0.1, gradient clipping=1.0** is the universal recipe used by Llama 2/3, DeepSeek-V3, Qwen 2.5, and OLMo 2. The only notable variation is epsilon: most use the default 1e-8, but Qwen uses **1e-6** for better mixed-precision stability. β₂=0.95 (rather than the Adam default of 0.999) provides faster adaptation to gradient distribution changes and is critical for large-scale training stability.

For CPT, these settings do not need to change. The optimizer states should be **reset** (fresh initialization) when starting from a public checkpoint—optimizer states are rarely released with open models. LR re-warming compensates for the missing optimizer state information. If you have access to the original optimizer states (e.g., continuing your own model), either continuing with existing states plus gentle re-warming or resetting states plus full re-warming both work, with the latter being more robust for large distribution shifts.

### Muon: promising but not validated for CPT

Muon (MomentUm Orthogonalized by Newton-Schulz), introduced by Keller Jordan (kellerjordan.github.io/posts/muon/), treats weight matrices as operators rather than collections of scalars. It orthogonalizes gradient updates via Newton-Schulz iteration, equalizing singular values to prevent optimization from being dominated by a few spectral directions. Moonshot AI's scaling paper ("Muon is Scalable for LLM Training," arXiv:2502.16982, 2025) demonstrated **~2× computational efficiency** over AdamW in scaling law experiments and trained Moonlight, a 3B/16B MoE model on 5.7T tokens.

However, three factors argue against Muon for CPT at this time:

- **No CPT-specific validation exists.** All published Muon results are for from-scratch pretraining. The Moonlight paper found that when the SFT optimizer differs from the pretraining optimizer, Muon SFT shows no significant advantage—a concerning signal for switching to Muon mid-training.
- **Diminishing returns at scale.** "Fantastic Pretraining Optimizers and Where to Find Them" (Wen et al., Stanford, arXiv:2509.02046, 2025) showed that Muon's speedup over **well-tuned** AdamW drops from ~1.4× at 0.1B to ~1.1× at 1.2B and may vanish at 7B+. Much of Muon's reported advantage comes from comparisons against poorly-tuned AdamW baselines.
- **Implementation complexity.** Muon only works on ≥2D weight matrices and requires AdamW fallback for embeddings, layer norms, and biases. It needs Newton-Schulz iteration in the forward pass, and distributed training integration (FSDP, DeepSpeed) requires careful handling of parameter shapes. DeepSpeed support was merged in PR #7509 with ZeRO Stage 1/2/3; FSDP2 implementations exist but are less mature.

**Recommendation: Use AdamW for CPT at 7B–70B scale.** The hyperparameter recipes from every major lab are directly applicable, the risk is lower, and the gains from Muon are likely marginal at this scale. Consider Muon only if you are doing very large-scale CPT (hundreds of billions of tokens) that resembles continued full pretraining, or if you are training from scratch with Muon and want to maintain optimizer consistency.

### Other optimizers worth knowing about

**SOAP** (arXiv:2409.11321, 2024) combines Shampoo's preconditioning with Adam and shows 35–40% wall-clock improvements over AdamW in large-batch settings. It outperforms Muon in overtraining scenarios (≥8× Chinchilla data ratio) but has higher memory overhead and implementation complexity. **Schedule-Free Adam** (Defazio et al., Meta, 2024) eliminates the LR schedule entirely through interpolation and iterate averaging—attractive for CPT because it removes the need to specify a stopping time, but lacks large-scale CPT validation. **Lion** (arXiv:2302.06675, 2023) offers 50% less optimizer memory than AdamW but requires 3–10× smaller LR and larger weight decay, with no clear advantage over well-tuned AdamW for text LLMs.

---
