# Model Comparison Results

Generated: 2026-03-06 20:22

| Model | Condition | N | Accuracy | CFR | POR |
|-------|-----------|---|----------|-----|-----|
| Llama-3.3-70B | No Conflict | 1000 | 73.3% | - | - |
| Llama-3.3-70B | Conflict@Hop1 | 1000 | 47.7% | 15.5% | 47.7% |
| Llama-3.3-70B | Conflict@Hop2 | 1000 | 49.8% | 18.7% | 49.8% |
| Llama-3.1-8B | No Conflict | 500 | 40.8% | - | - |
| Llama-3.1-8B | Conflict@Hop1 | 500 | 29.0% | 11.2% | 29.0% |
| Llama-3.1-8B | Conflict@Hop2 | 500 | 25.0% | 15.0% | 25.0% |
| Gemini-2.5-Flash-Lite | No Conflict | 499 | 33.9% | - | - |
| Gemini-2.5-Flash-Lite | Conflict@Hop1 | 499 | 19.2% | 14.0% | 19.2% |
| Gemini-2.5-Flash-Lite | Conflict@Hop2 | 499 | 20.2% | 12.0% | 20.2% |
| Qwen3-32B | No Conflict | 500 | 62.6% | - | - |
| Qwen3-32B | Conflict@Hop1 | 500 | 37.0% | 21.2% | 37.0% |
| Qwen3-32B | Conflict@Hop2 | 500 | 39.0% | 19.2% | 39.0% |

## Statistical Comparison: Llama-3.3-70B vs Llama-3.1-8B

| Condition | Llama-3.3-70B | Llama-3.1-8B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 73.3% | 40.8% | 148.79 | 0.0000 | Yes |
| Conflict@Hop1 | 47.7% | 29.0% | 47.26 | 0.0000 | Yes |
| Conflict@Hop2 | 49.8% | 25.0% | 83.41 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.3-70B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.3-70B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 73.3% | 33.9% | 214.34 | 0.0000 | Yes |
| Conflict@Hop1 | 47.7% | 19.2% | 112.99 | 0.0000 | Yes |
| Conflict@Hop2 | 49.8% | 20.2% | 120.01 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.3-70B vs Qwen3-32B

| Condition | Llama-3.3-70B | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 73.3% | 62.6% | 17.58 | 0.0000 | Yes |
| Conflict@Hop1 | 47.7% | 37.0% | 15.05 | 0.0001 | Yes |
| Conflict@Hop2 | 49.8% | 39.0% | 15.21 | 0.0001 | Yes |

## Statistical Comparison: Llama-3.1-8B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.1-8B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 40.8% | 33.9% | 4.84 | 0.0278 | Yes |
| Conflict@Hop1 | 29.0% | 19.2% | 12.47 | 0.0004 | Yes |
| Conflict@Hop2 | 25.0% | 20.2% | 2.97 | 0.0850 | No |

## Statistical Comparison: Llama-3.1-8B vs Qwen3-32B

| Condition | Llama-3.1-8B | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 40.8% | 62.6% | 46.71 | 0.0000 | Yes |
| Conflict@Hop1 | 29.0% | 37.0% | 6.88 | 0.0087 | Yes |
| Conflict@Hop2 | 25.0% | 39.0% | 21.88 | 0.0000 | Yes |

## Statistical Comparison: Gemini-2.5-Flash-Lite vs Qwen3-32B

| Condition | Gemini-2.5-Flash-Lite | Qwen3-32B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 33.9% | 62.6% | 81.43 | 0.0000 | Yes |
| Conflict@Hop1 | 19.2% | 37.0% | 38.10 | 0.0000 | Yes |
| Conflict@Hop2 | 20.2% | 39.0% | 41.26 | 0.0000 | Yes |
