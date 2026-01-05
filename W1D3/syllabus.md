# Week 1, Day 3: Data-Mixing, Stratification and Sampling

## Key Knowledge-Points

### 1. Data Mixing Strategies
- **Intuition**: Real-world training uses multiple data sources (different languages, domains, quality levels). Data mixing determines the proportion of each source in each batch. This is required because different sources have different value (e.g., high-quality code vs. low-quality web text). In practice, mixing can be uniform (equal probability) or weighted (by quality, token count, or domain expertise). The goal is to balance exposure to diverse data while prioritizing high-value sources.
- **Exercise Steps**:
  1. Load multiple datasets with different characteristics (language, domain, quality)
  2. Implement uniform mixing: sample equally from all sources
  3. Implement weighted mixing: sample proportionally to predefined weights
  4. Visualize the distribution of sources in generated batches
  5. Compare training dynamics: uniform vs. weighted mixing on a toy task

### 2. Stratification by Language and Quality
- **Intuition**: Data sources are stratified into groups (e.g., EN_CC, ZH_nonCC, CODE) to ensure balanced representation. Stratification prevents one dominant source from overwhelming others and allows fine-grained control over data composition. This is required because raw data distributions are highly skewed (e.g., English dominates web data). In practice, we assign each source to a stratum, calculate stratum-level proportions, then distribute those proportions among sources within each stratum.
- **Exercise Steps**:
  1. Categorize datasets into strata (language × quality combinations)
  2. Calculate token counts per stratum from data catalog
  3. Implement stratum-level proportion calculation (e.g., 60% EN_CC, 20% CODE, 20% SEA languages)
  4. Distribute stratum proportions to individual sources within each stratum
  5. Generate a mixing configuration file (YAML) with source-level proportions

### 3. Data Configuration Management
- **Intuition**: Data mixing configurations are complex (many sources, proportions, paths) and must be version-controlled and reproducible. Configuration files (YAML) centralize all data-related settings. This is required because manual configuration is error-prone and makes experiments irreproducible. In practice, we use YAML files that define datasets, their paths, token counts, and mixing proportions, with scripts to validate and expand configurations.
- **Exercise Steps**:
  1. Examine an existing `data_config.yaml` structure
  2. Write a script to validate configuration (check paths exist, proportions sum to 1)
  3. Implement configuration expansion: convert high-level proportions to source-level paths
  4. Generate a data mixing script that reads config and constructs Megatron data paths
  5. Practice updating configurations for different experiments (different language mixes, quality thresholds)

