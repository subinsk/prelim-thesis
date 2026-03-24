# Model Comparison Results

Generated: 2026-03-24 23:22

| Model | Condition | N | Accuracy | CFR | POR |
|-------|-----------|---|----------|-----|-----|
| Llama-3.3-70B | No Conflict | 1000 | 63.9% | - | - |
| Llama-3.3-70B | Conflict@Hop1 | 518 | 52.3% | 2.1% | 52.3% |
| Llama-3.3-70B | Conflict@Hop2 | 1000 | 24.0% | 26.9% | 24.0% |
| Llama-3.1-8B | No Conflict | 1000 | 60.6% | - | - |
| Llama-3.1-8B | Conflict@Hop1 | 518 | 49.2% | 1.4% | 49.2% |
| Llama-3.1-8B | Conflict@Hop2 | 1000 | 17.3% | 27.8% | 17.3% |
| Gemini-2.5-Flash-Lite | No Conflict | 1000 | 63.0% | - | - |
| Gemini-2.5-Flash-Lite | Conflict@Hop1 | 518 | 39.2% | 2.1% | 39.2% |
| Gemini-2.5-Flash-Lite | Conflict@Hop2 | 1000 | 11.0% | 35.8% | 11.0% |
| Qwen3-32B | No Conflict | 1000 | 61.4% | - | - |
| Qwen3-32B | Conflict@Hop1 | 518 | 41.5% | 2.1% | 41.5% |
| Qwen3-32B | Conflict@Hop2 | 1000 | 13.9% | 35.7% | 13.9% |

## Statistical Comparison: Llama-3.3-70B vs Llama-3.1-8B

| Condition | Llama-3.3-70B | Llama-3.1-8B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 63.9% | 60.6% | 2.18 | 0.1399 | No |
| Conflict@Hop1 | 52.3% | 49.2% | 0.87 | 0.3513 | No |
| Conflict@Hop2 | 24.0% | 17.3% | 13.29 | 0.0003 | Yes |

## Statistical Comparison: Llama-3.3-70B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.3-70B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 63.9% | 63.0% | 0.14 | 0.7103 | No |
| Conflict@Hop1 | 52.3% | 39.2% | 17.46 | 0.0000 | Yes |
| Conflict@Hop2 | 24.0% | 11.0% | 57.63 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.3-70B vs Qwen3-32B

| Condition | Llama-3.3-70B | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 63.9% | 61.4% | 1.23 | 0.2673 | No |
| Conflict@Hop1 | 52.3% | 41.5% | 11.72 | 0.0006 | Yes |
| Conflict@Hop2 | 24.0% | 13.9% | 32.55 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.1-8B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.1-8B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 60.6% | 63.0% | 1.12 | 0.2898 | No |
| Conflict@Hop1 | 49.2% | 39.2% | 10.18 | 0.0014 | Yes |
| Conflict@Hop2 | 17.3% | 11.0% | 15.82 | 0.0001 | Yes |

## Statistical Comparison: Llama-3.1-8B vs Qwen3-32B

| Condition | Llama-3.1-8B | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 60.6% | 61.4% | 0.10 | 0.7483 | No |
| Conflict@Hop1 | 49.2% | 41.5% | 5.92 | 0.0149 | Yes |
| Conflict@Hop2 | 17.3% | 13.9% | 4.14 | 0.0420 | Yes |

## Statistical Comparison: Gemini-2.5-Flash-Lite vs Qwen3-32B

| Condition | Gemini-2.5-Flash-Lite | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 63.0% | 61.4% | 0.48 | 0.4891 | No |
| Conflict@Hop1 | 39.2% | 41.5% | 0.49 | 0.4860 | No |
| Conflict@Hop2 | 11.0% | 13.9% | 3.60 | 0.0579 | No |
