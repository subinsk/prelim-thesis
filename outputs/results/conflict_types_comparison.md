# Conflict Type Comparison Results

Generated: 2026-03-06 18:31

## Accuracy by Conflict Type and Condition

| Model | Conflict Type | Condition | N | Accuracy | CFR | POR |
|-------|--------------|-----------|---|----------|-----|-----|
| Llama-3.3-70B | Factual | No Conflict | 200 | 71.5% | - | - |
| Llama-3.3-70B | Factual | Conflict@Hop1 | 200 | 54.0% | 17.5% | 54.0% |
| Llama-3.3-70B | Factual | Conflict@Hop2 | 200 | 55.0% | 15.5% | 55.0% |
| Llama-3.3-70B | Temporal | No Conflict | 200 | 87.0% | - | - |
| Llama-3.3-70B | Temporal | Conflict@Hop1 | 200 | 53.5% | 21.5% | 53.5% |
| Llama-3.3-70B | Temporal | Conflict@Hop2 | 200 | 57.5% | 17.5% | 57.5% |
| Llama-3.3-70B | Numerical | No Conflict | 142 | 79.6% | - | - |
| Llama-3.3-70B | Numerical | Conflict@Hop1 | 142 | 45.1% | 21.8% | 45.1% |
| Llama-3.3-70B | Numerical | Conflict@Hop2 | 142 | 39.4% | 21.1% | 39.4% |
| Llama-3.1-8B | Factual | No Conflict | 200 | 39.5% | - | - |
| Llama-3.1-8B | Factual | Conflict@Hop1 | 200 | 31.0% | 11.5% | 31.0% |
| Llama-3.1-8B | Factual | Conflict@Hop2 | 200 | 29.0% | 15.0% | 29.0% |
| Llama-3.1-8B | Temporal | No Conflict | 200 | 43.5% | - | - |
| Llama-3.1-8B | Temporal | Conflict@Hop1 | 200 | 26.5% | 16.0% | 26.5% |
| Llama-3.1-8B | Temporal | Conflict@Hop2 | 200 | 29.5% | 11.0% | 29.5% |
| Llama-3.1-8B | Numerical | No Conflict | 142 | 46.5% | - | - |
| Llama-3.1-8B | Numerical | Conflict@Hop1 | 142 | 20.4% | 12.7% | 20.4% |
| Llama-3.1-8B | Numerical | Conflict@Hop2 | 142 | 26.8% | 12.0% | 26.8% |
| Gemini-2.5-Flash-Lite | Factual | No Conflict | 200 | 33.0% | - | - |
| Gemini-2.5-Flash-Lite | Factual | Conflict@Hop1 | 200 | 17.5% | 11.0% | 17.5% |
| Gemini-2.5-Flash-Lite | Factual | Conflict@Hop2 | 200 | 20.0% | 10.0% | 20.0% |
| Gemini-2.5-Flash-Lite | Numerical | No Conflict | 142 | 27.5% | - | - |
| Gemini-2.5-Flash-Lite | Numerical | Conflict@Hop1 | 142 | 9.9% | 16.2% | 9.9% |
| Gemini-2.5-Flash-Lite | Numerical | Conflict@Hop2 | 142 | 20.4% | 18.3% | 20.4% |
| Qwen3-32B | Factual | No Conflict | 200 | 59.5% | - | - |
| Qwen3-32B | Factual | Conflict@Hop1 | 200 | 38.5% | 20.5% | 38.5% |
| Qwen3-32B | Factual | Conflict@Hop2 | 200 | 40.0% | 16.5% | 40.0% |
| Qwen3-32B | Temporal | No Conflict | 200 | 67.5% | - | - |
| Qwen3-32B | Temporal | Conflict@Hop1 | 200 | 35.5% | 26.5% | 35.5% |
| Qwen3-32B | Temporal | Conflict@Hop2 | 200 | 38.0% | 23.5% | 38.0% |
| Qwen3-32B | Numerical | No Conflict | 142 | 45.1% | - | - |
| Qwen3-32B | Numerical | Conflict@Hop1 | 142 | 21.8% | 24.6% | 21.8% |
| Qwen3-32B | Numerical | Conflict@Hop2 | 142 | 29.6% | 19.7% | 29.6% |

## Summary: Accuracy Drop by Conflict Type

| Model | Conflict Type | Baseline | Avg Conflict Acc | Drop (pp) |
|-------|--------------|----------|-----------------|----------|
| Llama-3.3-70B | Factual | 71.5% | 54.5% | 17.0 |
| Llama-3.3-70B | Temporal | 87.0% | 55.5% | 31.5 |
| Llama-3.3-70B | Numerical | 79.6% | 42.3% | 37.3 |
| Llama-3.1-8B | Factual | 39.5% | 30.0% | 9.5 |
| Llama-3.1-8B | Temporal | 43.5% | 28.0% | 15.5 |
| Llama-3.1-8B | Numerical | 46.5% | 23.6% | 22.9 |
| Gemini-2.5-Flash-Lite | Factual | 33.0% | 18.8% | 14.3 |
| Gemini-2.5-Flash-Lite | Numerical | 27.5% | 15.1% | 12.3 |
| Qwen3-32B | Factual | 59.5% | 39.2% | 20.2 |
| Qwen3-32B | Temporal | 67.5% | 36.8% | 30.8 |
| Qwen3-32B | Numerical | 45.1% | 25.7% | 19.4 |
