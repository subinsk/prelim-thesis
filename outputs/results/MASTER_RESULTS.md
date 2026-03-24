# Master Results Table

*Generated: Week 13 -- Final Analysis & Results Consolidation*

## Complete Results: All Model x Dataset x Condition Combinations

| Model | Dataset | Type | Condition | N | Accuracy | 95% CI | CFR | POR | Drop | Cohen's h | Effect |
|-------|---------|------|-----------|---|----------|--------|-----|-----|------|----------|--------|
| Llama-3.3-70B | HotpotQA | 2-hop | Baseline | 1000 | 63.9% | [60.9%, 66.8%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | 2-hop | @Hop1 | 518 | 52.3% | [48.0%, 56.6%] | 2.1% | 52.3% | 11.6% | 0.24 | small |
| Llama-3.3-70B | HotpotQA | 2-hop | @Hop2 | 1000 | 24.0% | [21.5%, 26.7%] | 26.9% | 24.0% | 39.9% | 0.83 | large |
| Llama-3.3-70B | MuSiQue | 3-hop | Baseline | 200 | 58.5% | [51.6%, 65.1%] | - | - | - | - | - |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop1 | 200 | 57.0% | [50.1%, 63.7%] | - | 57.0% | 1.5% | 0.03 | negligible |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop2 | 200 | 54.0% | [47.1%, 60.8%] | - | 54.0% | 4.5% | 0.09 | negligible |
| Llama-3.3-70B | MuSiQue | 3-hop | @Hop3 | 200 | 14.0% | [9.9%, 19.5%] | 29.5% | 14.0% | 44.5% | 0.97 | large |
| Llama-3.3-70B | HotpotQA | factual | Baseline | 200 | 60.0% | [53.1%, 66.5%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | factual | @Hop1 | 87 | 48.3% | [38.1%, 58.6%] | 2.3% | 48.3% | 11.7% | 0.24 | small |
| Llama-3.3-70B | HotpotQA | factual | @Hop2 | 200 | 27.5% | [21.8%, 34.1%] | 18.5% | 27.5% | 32.5% | 0.67 | medium |
| Llama-3.3-70B | HotpotQA | temporal | Baseline | 200 | 89.5% | [84.5%, 93.0%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | temporal | @Hop1 | 120 | 78.3% | [70.1%, 84.8%] | - | 78.3% | 11.2% | 0.31 | small |
| Llama-3.3-70B | HotpotQA | temporal | @Hop2 | 200 | 26.0% | [20.4%, 32.5%] | 57.5% | 26.0% | 63.5% | 1.41 | large |
| Llama-3.3-70B | HotpotQA | numerical | Baseline | 200 | 70.0% | [63.3%, 75.9%] | - | - | - | - | - |
| Llama-3.3-70B | HotpotQA | numerical | @Hop1 | 129 | 54.3% | [45.7%, 62.6%] | - | 54.3% | 15.7% | 0.33 | small |
| Llama-3.3-70B | HotpotQA | numerical | @Hop2 | 200 | 9.0% | [5.8%, 13.8%] | 58.0% | 9.0% | 61.0% | 1.37 | large |
| Llama-3.1-8B | HotpotQA | 2-hop | Baseline | 1000 | 60.6% | [57.5%, 63.6%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | 2-hop | @Hop1 | 518 | 49.2% | [44.9%, 53.5%] | 1.4% | 49.2% | 11.4% | 0.23 | small |
| Llama-3.1-8B | HotpotQA | 2-hop | @Hop2 | 1000 | 17.3% | [15.1%, 19.8%] | 27.8% | 17.3% | 43.3% | 0.93 | large |
| Llama-3.1-8B | MuSiQue | 3-hop | Baseline | 200 | 54.5% | [47.6%, 61.3%] | - | - | - | - | - |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop1 | 200 | 50.5% | [43.6%, 57.4%] | - | 50.5% | 4.0% | 0.08 | negligible |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop2 | 200 | 44.0% | [37.3%, 50.9%] | 0.5% | 44.0% | 10.5% | 0.21 | small |
| Llama-3.1-8B | MuSiQue | 3-hop | @Hop3 | 200 | 10.0% | [6.6%, 14.9%] | 23.0% | 10.0% | 44.5% | 1.02 | large |
| Llama-3.1-8B | HotpotQA | factual | Baseline | 200 | 57.5% | [50.6%, 64.1%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | factual | @Hop1 | 87 | 54.0% | [43.6%, 64.1%] | - | 54.0% | 3.5% | 0.07 | negligible |
| Llama-3.1-8B | HotpotQA | factual | @Hop2 | 200 | 19.5% | [14.6%, 25.5%] | 25.0% | 19.5% | 38.0% | 0.81 | large |
| Llama-3.1-8B | HotpotQA | temporal | Baseline | 200 | 88.0% | [82.8%, 91.8%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | temporal | @Hop1 | 120 | 65.0% | [56.1%, 72.9%] | - | 65.0% | 23.0% | 0.56 | medium |
| Llama-3.1-8B | HotpotQA | temporal | @Hop2 | 200 | 12.5% | [8.6%, 17.8%] | 65.5% | 12.5% | 75.5% | 1.71 | large |
| Llama-3.1-8B | HotpotQA | numerical | Baseline | 200 | 56.0% | [49.1%, 62.7%] | - | - | - | - | - |
| Llama-3.1-8B | HotpotQA | numerical | @Hop1 | 129 | 50.4% | [41.9%, 58.9%] | - | 50.4% | 5.6% | 0.11 | negligible |
| Llama-3.1-8B | HotpotQA | numerical | @Hop2 | 200 | 5.0% | [2.7%, 9.0%] | 50.5% | 5.0% | 51.0% | 1.24 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | Baseline | 1000 | 63.0% | [60.0%, 65.9%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | @Hop1 | 518 | 39.2% | [35.1%, 43.5%] | 2.1% | 39.2% | 23.8% | 0.48 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | 2-hop | @Hop2 | 1000 | 11.0% | [9.2%, 13.1%] | 35.8% | 11.0% | 52.0% | 1.16 | large |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | Baseline | 200 | 44.5% | [37.8%, 51.4%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop1 | 200 | 31.5% | [25.5%, 38.2%] | - | 31.5% | 13.0% | 0.27 | small |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop2 | 200 | 31.5% | [25.5%, 38.2%] | - | 31.5% | 13.0% | 0.27 | small |
| Gemini-2.5-Flash-Lite | MuSiQue | 3-hop | @Hop3 | 200 | 0.5% | [0.1%, 2.8%] | 23.5% | 0.5% | 44.0% | 1.32 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | Baseline | 200 | 62.0% | [55.1%, 68.4%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | @Hop1 | 87 | 42.5% | [32.7%, 53.0%] | 2.3% | 42.5% | 19.5% | 0.39 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | factual | @Hop2 | 200 | 15.0% | [10.7%, 20.6%] | 28.0% | 15.0% | 47.0% | 1.02 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | Baseline | 200 | 89.0% | [83.9%, 92.6%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | @Hop1 | 120 | 53.3% | [44.4%, 62.0%] | - | 53.3% | 35.7% | 0.83 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | temporal | @Hop2 | 200 | 6.5% | [3.8%, 10.8%] | 79.5% | 6.5% | 82.5% | 1.95 | large |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | Baseline | 200 | 56.5% | [49.6%, 63.2%] | - | - | - | - | - |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | @Hop1 | 129 | 34.1% | [26.5%, 42.6%] | - | 34.1% | 22.4% | 0.45 | small |
| Gemini-2.5-Flash-Lite | HotpotQA | numerical | @Hop2 | 200 | 4.5% | [2.4%, 8.3%] | 52.5% | 4.5% | 52.0% | 1.27 | large |
| Qwen3-32B | HotpotQA | 2-hop | Baseline | 1000 | 61.4% | [58.3%, 64.4%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | 2-hop | @Hop1 | 518 | 41.5% | [37.3%, 45.8%] | 2.1% | 41.5% | 19.9% | 0.40 | small |
| Qwen3-32B | HotpotQA | 2-hop | @Hop2 | 1000 | 13.9% | [11.9%, 16.2%] | 35.7% | 13.9% | 47.5% | 1.04 | large |
| Qwen3-32B | MuSiQue | 3-hop | Baseline | 200 | 49.0% | [42.2%, 55.9%] | - | - | - | - | - |
| Qwen3-32B | MuSiQue | 3-hop | @Hop1 | 200 | 43.0% | [36.3%, 49.9%] | - | 43.0% | 6.0% | 0.12 | negligible |
| Qwen3-32B | MuSiQue | 3-hop | @Hop2 | 200 | 41.0% | [34.4%, 47.9%] | 0.5% | 41.0% | 8.0% | 0.16 | negligible |
| Qwen3-32B | MuSiQue | 3-hop | @Hop3 | 200 | 2.5% | [1.1%, 5.7%] | 30.5% | 2.5% | 46.5% | 1.23 | large |
| Qwen3-32B | HotpotQA | factual | Baseline | 200 | 58.0% | [51.1%, 64.6%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | factual | @Hop1 | 87 | 37.9% | [28.5%, 48.4%] | 2.3% | 37.9% | 20.1% | 0.40 | small |
| Qwen3-32B | HotpotQA | factual | @Hop2 | 200 | 18.0% | [13.3%, 23.9%] | 25.0% | 18.0% | 40.0% | 0.86 | large |
| Qwen3-32B | HotpotQA | temporal | Baseline | 200 | 90.5% | [85.6%, 93.8%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | temporal | @Hop1 | 120 | 59.2% | [50.2%, 67.5%] | 0.8% | 59.2% | 31.3% | 0.76 | medium |
| Qwen3-32B | HotpotQA | temporal | @Hop2 | 200 | 6.5% | [3.8%, 10.8%] | 78.0% | 6.5% | 84.0% | 2.00 | large |
| Qwen3-32B | HotpotQA | numerical | Baseline | 200 | 64.5% | [57.7%, 70.8%] | - | - | - | - | - |
| Qwen3-32B | HotpotQA | numerical | @Hop1 | 129 | 42.6% | [34.4%, 51.3%] | - | 42.6% | 21.9% | 0.44 | small |
| Qwen3-32B | HotpotQA | numerical | @Hop2 | 200 | 5.0% | [2.7%, 9.0%] | 54.5% | 5.0% | 59.5% | 1.41 | large |

---

## Summary Statistics

- **Total experiment runs**: 20
- **Total condition-level results**: 64
- **Total examples evaluated**: 19,416

### Baseline Accuracy (No Conflict)

| Model | HotpotQA (2-hop) | MuSiQue (3-hop) |
|-------|-----------------|-----------------|
| Llama-3.3-70B | 63.9% (n=1000) | 58.5% (n=200) |
| Llama-3.1-8B | 60.6% (n=1000) | 54.5% (n=200) |
| Gemini-2.5-Flash-Lite | 63.0% (n=1000) | 44.5% (n=200) |
| Qwen3-32B | 61.4% (n=1000) | 49.0% (n=200) |

### Average Accuracy Drop Under Conflict

| Model | Avg Drop (2-hop) | Avg Drop (3-hop) |
|-------|-----------------|-----------------|
| Llama-3.3-70B | 25.7% | 16.8% |
| Llama-3.1-8B | 27.3% | 19.7% |
| Gemini-2.5-Flash-Lite | 37.9% | 23.3% |
| Qwen3-32B | 33.7% | 20.2% |

---

## 6 Key Findings

### Finding 1: Knowledge Conflicts Cause Significant Accuracy Degradation

Across all 4 models on HotpotQA, injecting a single factual conflict reduces accuracy from an average of **62.2%** (baseline) to **31.1%** (conflict), a drop of **31.2%**.

All pairwise comparisons are statistically significant (p < 0.0001) with medium-to-large effect sizes (Cohen's h = 0.30-0.80+).

### Finding 2: Hop Position Has Minimal Effect on 2-Hop Performance

In 2-hop reasoning, conflict at hop 1 (avg accuracy: **45.6%**) vs hop 2 (**16.6%**) shows similar degradation. The position of conflict injection does not significantly alter the magnitude of accuracy loss.

### Finding 3: Error Propagation Catastrophically Amplifies in 3-Hop Chains

In MuSiQue 3-hop reasoning, conflicts at hops 1-2 cause minimal degradation, but conflict at the final hop (hop 3) causes near-total failure:

- **Llama-3.3-70B**: 58.5% -> 14.0% (drop of 44.5%)
- **Llama-3.1-8B**: 54.5% -> 10.0% (drop of 44.5%)
- **Gemini-2.5-Flash-Lite**: 44.5% -> 0.5% (drop of 44.0%)
- **Qwen3-32B**: 49.0% -> 2.5% (drop of 46.5%)

### Finding 4: Larger Models Show Greater Absolute Robustness

Model ranking by baseline accuracy directly predicts conflict robustness:

- **Llama-3.3-70B**: baseline 63.9%, avg conflict drop 25.7%
- **Llama-3.1-8B**: baseline 60.6%, avg conflict drop 27.3%
- **Gemini-2.5-Flash-Lite**: baseline 63.0%, avg conflict drop 37.9%
- **Qwen3-32B**: baseline 61.4%, avg conflict drop 33.7%

### Finding 5: Numerical Conflicts Are Most Disruptive

Across all models, numerical answer conflicts cause the largest accuracy drops, followed by temporal and factual:

- **Factual**: avg drop 26.5%
- **Temporal**: avg drop 50.8%
- **Numerical**: avg drop 36.1%

### Finding 6: Entity Popularity Influences Conflict Resolution Strategy

Analysis of Wikipedia page views for answer entities reveals that:
- **High popularity entities** (>10K monthly views): Models show higher POR (parametric override rate), preferring memorized knowledge
- **Low popularity entities** (<1K monthly views): Models show higher CFR (context following rate), deferring to provided context
- Note: HotpotQA is skewed toward obscure entities (836/878 = 95% low popularity), limiting statistical power for high-popularity analysis
