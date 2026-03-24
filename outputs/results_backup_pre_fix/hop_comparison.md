# 2-Hop vs 3-Hop Comparison

Generated: 2026-03-05 18:16

## Overview

| Model | Dataset | Hops | Baseline Acc | Avg Conflict Acc | Max Drop (pp) | Avg CFR | Avg POR |
|-------|---------|------|-------------|-----------------|--------------|---------|--------|
| Llama-3.3-70B | HotpotQA | 2 | 71.2% | 42.4% | 29.8 | 17.7% | 42.4% |
| Llama-3.3-70B | MuSiQue | 3 | 70.5% | 52.5% | 53.0 | 10.2% | 52.5% |
| Llama-3.1-8B | HotpotQA | 2 | 40.8% | 27.0% | 15.8 | 13.1% | 27.0% |
| Llama-3.1-8B | MuSiQue | 3 | 33.0% | 24.5% | 30.0 | 3.8% | 24.5% |
| Gemini-2.5-Flash-Lite | HotpotQA | 2 | 33.9% | 19.7% | 14.6 | 13.0% | 19.7% |
| Gemini-2.5-Flash-Lite | MuSiQue | 3 | 17.5% | 11.7% | 17.5 | 6.8% | 11.7% |
| Qwen3-32B | HotpotQA | 2 | 62.6% | 38.0% | 25.6 | 20.2% | 38.0% |
| Qwen3-32B | MuSiQue | 3 | 41.0% | 34.0% | 35.5 | 12.0% | 34.0% |

## Cross-Hop Comparison (Baseline Accuracy: 2-hop vs 3-hop)

| Model | 2-Hop Baseline | 3-Hop Baseline | Drop (pp) | Chi2 | p-value | Significant |
|-------|---------------|---------------|-----------|------|---------|-------------|
| Llama-3.3-70B | 71.2% | 70.5% | +0.7 | 0.01 | 0.9265 | No |
| Llama-3.1-8B | 40.8% | 33.0% | +7.8 | 3.35 | 0.0674 | No |
| Gemini-2.5-Flash-Lite | 33.9% | 17.5% | +16.4 | 17.72 | 0.0000 | Yes |
| Qwen3-32B | 62.6% | 41.0% | +21.6 | 26.24 | 0.0000 | Yes |

## Error Propagation (3-Hop): Accuracy Drop by Conflict Position

| Model | Baseline | Drop@Hop1 | Drop@Hop2 | Drop@Hop3 | Most Vulnerable |
|-------|----------|-----------|-----------|-----------|----------------|
| Llama-3.3-70B | 70.5% | +0.0pp | +1.0pp | +53.0pp | Hop 3 (+53.0pp) |
| Llama-3.1-8B | 33.0% | -2.5pp | -2.0pp | +30.0pp | Hop 3 (+30.0pp) |
| Gemini-2.5-Flash-Lite | 17.5% | +0.0pp | +0.0pp | +17.5pp | Hop 3 (+17.5pp) |
| Qwen3-32B | 41.0% | -8.5pp | -6.0pp | +35.5pp | Hop 3 (+35.5pp) |
