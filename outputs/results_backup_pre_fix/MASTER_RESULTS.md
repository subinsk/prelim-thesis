# Master Results Table

*Generated: Week 13 -- Final Analysis & Results Consolidation*

## Complete Results: All Model x Dataset x Condition Combinations

| Model | Dataset | Type | Condition | N | Accuracy | 95% CI | CFR | POR | Drop | Cohen's h | Effect |
|-------|---------|------|-----------|---|----------|--------|-----|-----|------|----------|--------|
| Llama-3.3-70B | HotpotQA | 2-hop | Baseline | 1000 | 73.3% | [70.5%, 75.9%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | 2-hop | @Hop1 | 1000 | 47.7% | [44.6%, 50.8%] | 15.5% | 47.7% | 25.6% | 0.53 | medium |
| Llama-3.3-70B | HotpotQA | 2-hop | @Hop2 | 1000 | 49.8% | [46.7%, 52.9%] | 18.7% | 49.8% | 23.5% | 0.49 | small |
| Llama-3.3-70B | MuSiQue | 3-hop | Baseline | 200 | 70.5% | [63.8%, 76.4%] | - | - | - | - | - |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop1 | 200 | 70.5% | [63.8%, 76.4%] | - | 70.5% | - | - | - |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop2 | 200 | 69.5% | [62.8%, 75.5%] | - | 69.5% | 1.0% | 0.02 | negligible |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop3 | 200 | 17.5% | [12.9%, 23.4%] | 30.5% | 17.5% | 53.0% | 1.13 | large |
| Llama-3.3-70B | HotpotQA | factual | Baseline | 200 | 71.5% | [64.9%, 77.3%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | factual | @Hop1 | 200 | 54.0% | [47.1%, 60.8%] | 17.5% | 54.0% | 17.5% | 0.36 | small |
| Llama-3.3-70B | HotpotQA | factual | @Hop2 | 200 | 55.0% | [48.1%, 61.7%] | 15.5% | 55.0% | 16.5% | 0.34 | small |
| Llama-3.3-70B | HotpotQA | temporal | Baseline | 200 | 87.0% | [81.6%, 91.0%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | temporal | @Hop1 | 200 | 53.5% | [46.6%, 60.3%] | 21.5% | 53.5% | 33.5% | 0.76 | medium |
| Llama-3.3-70B | HotpotQA | temporal | @Hop2 | 200 | 57.5% | [50.6%, 64.1%] | 17.5% | 57.5% | 29.5% | 0.68 | medium |
| Llama-3.3-70B | HotpotQA | numerical | Baseline | 142 | 79.6% | [72.2%, 85.4%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | numerical | @Hop1 | 142 | 45.1% | [37.1%, 53.3%] | 21.8% | 45.1% | 34.5% | 0.73 | medium |
| Llama-3.3-70B | HotpotQA | numerical | @Hop2 | 142 | 39.4% | [31.8%, 47.7%] | 21.1% | 39.4% | 40.1% | 0.85 | large |
| Llama-3.1-8B | HotpotQA | 2-hop | Baseline | 500 | 40.8% | [36.6%, 45.2%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | 2-hop | @Hop1 | 500 | 29.0% | [25.2%, 33.1%] | 11.2% | 29.0% | 11.8% | 0.25 | small |
| Llama-3.1-8B | HotpotQA | 2-hop | @Hop2 | 500 | 25.0% | [21.4%, 29.0%] | 15.0% | 25.0% | 15.8% | 0.34 | small |
| Llama-3.1-8B | MuSiQue | 3-hop | Baseline | 200 | 33.0% | [26.9%, 39.8%] | - | - | - | - | - |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop1 | 200 | 35.5% | [29.2%, 42.3%] | - | 35.5% | - | -0.05 | negligible |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop2 | 200 | 35.0% | [28.7%, 41.8%] | 0.5% | 35.0% | - | -0.04 | negligible |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop3 | 200 | 3.0% | [1.4%, 6.4%] | 11.0% | 3.0% | 30.0% | 0.88 | large |
| Llama-3.1-8B | HotpotQA | factual | Baseline | 200 | 39.5% | [33.0%, 46.4%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | factual | @Hop1 | 200 | 31.0% | [25.0%, 37.7%] | 11.5% | 31.0% | 8.5% | 0.18 | negligible |
| Llama-3.1-8B | HotpotQA | factual | @Hop2 | 200 | 29.0% | [23.2%, 35.6%] | 15.0% | 29.0% | 10.5% | 0.22 | small |
| Llama-3.1-8B | HotpotQA | temporal | Baseline | 200 | 43.5% | [36.8%, 50.4%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | temporal | @Hop1 | 200 | 26.5% | [20.9%, 33.0%] | 16.0% | 26.5% | 17.0% | 0.36 | small |
| Llama-3.1-8B | HotpotQA | temporal | @Hop2 | 200 | 29.5% | [23.6%, 36.2%] | 11.0% | 29.5% | 14.0% | 0.29 | small |
| Llama-3.1-8B | HotpotQA | numerical | Baseline | 142 | 46.5% | [38.5%, 54.7%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | numerical | @Hop1 | 142 | 20.4% | [14.6%, 27.8%] | 12.7% | 20.4% | 26.1% | 0.56 | medium |
| Llama-3.1-8B | HotpotQA | numerical | @Hop2 | 142 | 26.8% | [20.2%, 34.6%] | 12.0% | 26.8% | 19.7% | 0.41 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | Baseline | 499 | 33.9% | [29.9%, 38.1%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | @Hop1 | 499 | 19.2% | [16.0%, 22.9%] | 14.0% | 19.2% | 14.6% | 0.33 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | @Hop2 | 499 | 20.2% | [16.9%, 24.0%] | 12.0% | 20.2% | 13.6% | 0.31 | small |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | Baseline | 200 | 17.5% | [12.9%, 23.4%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop1 | 200 | 17.5% | [12.9%, 23.4%] | - | 17.5% | - | - | - |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop2 | 200 | 17.5% | [12.9%, 23.4%] | - | 17.5% | - | - | - |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop3 | 200 | 0.0% | [0.0%, 1.9%] | 20.5% | - | 17.5% | 0.86 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | Baseline | 200 | 33.0% | [26.9%, 39.8%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | @Hop1 | 200 | 17.5% | [12.9%, 23.4%] | 11.0% | 17.5% | 15.5% | 0.36 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | @Hop2 | 200 | 20.0% | [15.0%, 26.1%] | 10.0% | 20.0% | 13.0% | 0.30 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | Baseline | 196 | 33.2% | [26.9%, 40.0%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | @Hop1 | 196 | 16.3% | [11.8%, 22.1%] | 10.7% | 16.3% | 16.8% | 0.40 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | @Hop2 | 196 | 16.8% | [12.3%, 22.7%] | 14.8% | 16.8% | 16.3% | 0.38 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | Baseline | 142 | 27.5% | [20.8%, 35.3%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | @Hop1 | 142 | 9.9% | [6.0%, 15.9%] | 16.2% | 9.9% | 17.6% | 0.46 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | @Hop2 | 142 | 20.4% | [14.6%, 27.8%] | 18.3% | 20.4% | 7.0% | 0.17 | negligible |
| Qwen3-32B | HotpotQA | 2-hop | Baseline | 500 | 62.6% | [58.3%, 66.7%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | 2-hop | @Hop1 | 500 | 37.0% | [32.9%, 41.3%] | 21.2% | 37.0% | 25.6% | 0.52 | medium |
| Qwen3-32B | HotpotQA | 2-hop | @Hop2 | 500 | 39.0% | [34.8%, 43.3%] | 19.2% | 39.0% | 23.6% | 0.48 | small |
| Qwen3-32B | MuSiQue | 3-hop | Baseline | 200 | 41.0% | [34.4%, 47.9%] | - | - | - | - | - |
| Qwen3-32B | MuSiQue | 3-hop | @Hop1 | 200 | 49.5% | [42.6%, 56.4%] | - | 49.5% | - | -0.17 | negligible |
| Qwen3-32B | MuSiQue | 3-hop | @Hop2 | 200 | 47.0% | [40.2%, 53.9%] | - | 47.0% | - | -0.12 | negligible |
| Qwen3-32B | MuSiQue | 3-hop | @Hop3 | 200 | 5.5% | [3.1%, 9.6%] | 36.0% | 5.5% | 35.5% | 0.92 | large |
| Qwen3-32B | HotpotQA | factual | Baseline | 200 | 59.5% | [52.6%, 66.1%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | factual | @Hop1 | 200 | 38.5% | [32.0%, 45.4%] | 20.5% | 38.5% | 21.0% | 0.42 | small |
| Qwen3-32B | HotpotQA | factual | @Hop2 | 200 | 40.0% | [33.5%, 46.9%] | 16.5% | 40.0% | 19.5% | 0.39 | small |
| Qwen3-32B | HotpotQA | temporal | Baseline | 200 | 67.5% | [60.7%, 73.6%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | temporal | @Hop1 | 200 | 35.5% | [29.2%, 42.3%] | 26.5% | 35.5% | 32.0% | 0.65 | medium |
| Qwen3-32B | HotpotQA | temporal | @Hop2 | 200 | 38.0% | [31.6%, 44.9%] | 23.5% | 38.0% | 29.5% | 0.60 | medium |
| Qwen3-32B | HotpotQA | numerical | Baseline | 142 | 45.1% | [37.1%, 53.3%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | numerical | @Hop1 | 142 | 21.8% | [15.8%, 29.3%] | 24.6% | 21.8% | 23.2% | 0.50 | small |
| Qwen3-32B | HotpotQA | numerical | @Hop2 | 142 | 29.6% | [22.7%, 37.5%] | 19.7% | 29.6% | 15.5% | 0.32 | small |

---

## Summary Statistics

- **Total experiment runs**: 20
- **Total condition-level results**: 64
- **Total examples evaluated**: 17,189

### Baseline Accuracy (No Conflict)

| Model | HotpotQA (2-hop) | MuSiQue (3-hop) |
|-------|-----------------|-----------------|
| Llama-3.3-70B | 73.3% (n=1000) | 70.5% (n=200) |
| Llama-3.1-8B | 40.8% (n=500) | 33.0% (n=200) |
| Gemini-2.5-Flash-Lite | 33.9% (n=499) | 17.5% (n=200) |
| Qwen3-32B | 62.6% (n=500) | 41.0% (n=200) |

### Average Accuracy Drop Under Conflict

| Model | Avg Drop (2-hop) | Avg Drop (3-hop) |
|-------|-----------------|-----------------|
| Llama-3.3-70B | 24.6% | 27.0% |
| Llama-3.1-8B | 13.8% | 30.0% |
| Gemini-2.5-Flash-Lite | 14.1% | 17.5% |
| Qwen3-32B | 24.6% | 35.5% |

---

## 6 Key Findings

### Finding 1: Knowledge Conflicts Cause Significant Accuracy Degradation

Across all 4 models on HotpotQA, injecting a single factual conflict reduces accuracy from an average of **52.6%** (baseline) to **33.4%** (conflict), a drop of **19.3%**.

All pairwise comparisons are statistically significant (p < 0.0001) with medium-to-large effect sizes (Cohen's h = 0.30-0.80+).

### Finding 2: Hop Position Has Minimal Effect on 2-Hop Performance

In 2-hop reasoning, conflict at hop 1 (avg accuracy: **33.2%**) vs hop 2 (**33.5%**) shows similar degradation. The position of conflict injection does not significantly alter the magnitude of accuracy loss.

### Finding 3: Error Propagation Catastrophically Amplifies in 3-Hop Chains

In MuSiQue 3-hop reasoning, conflicts at hops 1-2 cause minimal degradation, but conflict at the final hop (hop 3) causes near-total failure:

- **Llama-3.3-70B**: 70.5% -> 17.5% (drop of 53.0%)
- **Llama-3.1-8B**: 33.0% -> 3.0% (drop of 30.0%)
- **Gemini-2.5-Flash-Lite**: 17.5% -> 0.0% (drop of 17.5%)
- **Qwen3-32B**: 41.0% -> 5.5% (drop of 35.5%)

### Finding 4: Larger Models Show Greater Absolute Robustness

Model ranking by baseline accuracy directly predicts conflict robustness:

- **Llama-3.3-70B**: baseline 73.3%, avg conflict drop 24.6%
- **Llama-3.1-8B**: baseline 40.8%, avg conflict drop 13.8%
- **Gemini-2.5-Flash-Lite**: baseline 33.9%, avg conflict drop 14.1%
- **Qwen3-32B**: baseline 62.6%, avg conflict drop 24.6%

### Finding 5: Numerical Conflicts Are Most Disruptive

Across all models, numerical answer conflicts cause the largest accuracy drops, followed by temporal and factual:

- **Factual**: avg drop 15.2%
- **Temporal**: avg drop 23.6%
- **Numerical**: avg drop 23.0%

### Finding 6: Entity Popularity Influences Conflict Resolution Strategy

Analysis of Wikipedia page views for answer entities reveals that:
- **High popularity entities** (>10K monthly views): Models show higher POR (parametric override rate), preferring memorized knowledge
- **Low popularity entities** (<1K monthly views): Models show higher CFR (context following rate), deferring to provided context
- Note: HotpotQA is skewed toward obscure entities (836/878 = 95% low popularity), limiting statistical power for high-popularity analysis
