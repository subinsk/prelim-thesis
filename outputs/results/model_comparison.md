# Model Comparison Results

Generated: 2026-03-05 13:44

| Model | Condition | N | Accuracy | CFR | POR |
|-------|-----------|---|----------|-----|-----|
| Llama-3.3-70B | No Conflict | 500 | 71.2% | - | - |
| Llama-3.3-70B | Conflict@Hop1 | 500 | 41.4% | 17.6% | 41.4% |
| Llama-3.3-70B | Conflict@Hop2 | 500 | 43.4% | 17.8% | 43.4% |
| Llama-3.1-8B | No Conflict | 500 | 40.8% | - | - |
| Llama-3.1-8B | Conflict@Hop1 | 500 | 29.0% | 11.2% | 29.0% |
| Llama-3.1-8B | Conflict@Hop2 | 500 | 25.0% | 15.0% | 25.0% |
| Gemini-2.5-Flash-Lite | No Conflict | 499 | 33.9% | - | - |
| Gemini-2.5-Flash-Lite | Conflict@Hop1 | 499 | 19.2% | 14.0% | 19.2% |
| Gemini-2.5-Flash-Lite | Conflict@Hop2 | 499 | 20.2% | 12.0% | 20.2% |

## Statistical Comparison: Llama-3.3-70B vs Llama-3.1-8B

| Condition | Llama-3.3-70B | Llama-3.1-8B | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 71.2% | 40.8% | 92.54 | 0.0000 | Yes |
| Conflict@Hop1 | 41.4% | 29.0% | 16.31 | 0.0001 | Yes |
| Conflict@Hop2 | 43.4% | 25.0% | 36.80 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.3-70B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.3-70B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 71.2% | 33.9% | 138.10 | 0.0000 | Yes |
| Conflict@Hop1 | 41.4% | 19.2% | 57.00 | 0.0000 | Yes |
| Conflict@Hop2 | 43.4% | 20.2% | 60.67 | 0.0000 | Yes |

## Statistical Comparison: Llama-3.1-8B vs Gemini-2.5-Flash-Lite

| Condition | Llama-3.1-8B | Gemini-2.5-Flash-Lite | Chi2 | p-value | Significant |
|-----------|---------|--------|------|---------|-------------|
| No Conflict | 40.8% | 33.9% | 4.84 | 0.0278 | Yes |
| Conflict@Hop1 | 29.0% | 19.2% | 12.47 | 0.0004 | Yes |
| Conflict@Hop2 | 25.0% | 20.2% | 2.97 | 0.0850 | No |
