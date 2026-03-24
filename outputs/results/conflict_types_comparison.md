# Conflict Type Comparison Results

Generated: 2026-03-24 23:18

## Accuracy by Conflict Type and Condition

| Model | Conflict Type | Condition | N | Accuracy | CFR | POR |
|-------|---------------|-----------|---|----------|-----|-----|
| Llama-3.3-70B | Factual | No Conflict | 200 | 60.0% | - | - |
| Llama-3.3-70B | Factual | Conflict@Hop1 | 87 | 48.3% | 2.3% | 48.3% |
| Llama-3.3-70B | Factual | Conflict@Hop2 | 200 | 27.5% | 18.5% | 27.5% |
| Llama-3.3-70B | Temporal | No Conflict | 200 | 89.5% | - | - |
| Llama-3.3-70B | Temporal | Conflict@Hop1 | 120 | 78.3% | - | 78.3% |
| Llama-3.3-70B | Temporal | Conflict@Hop2 | 200 | 26.0% | 57.5% | 26.0% |
| Llama-3.3-70B | Numerical | No Conflict | 200 | 70.0% | - | - |
| Llama-3.3-70B | Numerical | Conflict@Hop1 | 129 | 54.3% | - | 54.3% |
| Llama-3.3-70B | Numerical | Conflict@Hop2 | 200 | 9.0% | 58.0% | 9.0% |
| Llama-3.1-8B | Factual | No Conflict | 200 | 57.5% | - | - |
| Llama-3.1-8B | Factual | Conflict@Hop1 | 87 | 54.0% | - | 54.0% |
| Llama-3.1-8B | Factual | Conflict@Hop2 | 200 | 19.5% | 25.0% | 19.5% |
| Llama-3.1-8B | Temporal | No Conflict | 200 | 88.0% | - | - |
| Llama-3.1-8B | Temporal | Conflict@Hop1 | 120 | 65.0% | - | 65.0% |
| Llama-3.1-8B | Temporal | Conflict@Hop2 | 200 | 12.5% | 65.5% | 12.5% |
| Llama-3.1-8B | Numerical | No Conflict | 200 | 56.0% | - | - |
| Llama-3.1-8B | Numerical | Conflict@Hop1 | 129 | 50.4% | - | 50.4% |
| Llama-3.1-8B | Numerical | Conflict@Hop2 | 200 | 5.0% | 50.5% | 5.0% |
| Gemini-2.5-Flash-Lite | Factual | No Conflict | 200 | 62.0% | - | - |
| Gemini-2.5-Flash-Lite | Factual | Conflict@Hop1 | 87 | 42.5% | 2.3% | 42.5% |
| Gemini-2.5-Flash-Lite | Factual | Conflict@Hop2 | 200 | 15.0% | 28.0% | 15.0% |
| Gemini-2.5-Flash-Lite | Temporal | No Conflict | 200 | 89.0% | - | - |
| Gemini-2.5-Flash-Lite | Temporal | Conflict@Hop1 | 120 | 53.3% | - | 53.3% |
| Gemini-2.5-Flash-Lite | Temporal | Conflict@Hop2 | 200 | 6.5% | 79.5% | 6.5% |
| Gemini-2.5-Flash-Lite | Numerical | No Conflict | 200 | 56.5% | - | - |
| Gemini-2.5-Flash-Lite | Numerical | Conflict@Hop1 | 129 | 34.1% | - | 34.1% |
| Gemini-2.5-Flash-Lite | Numerical | Conflict@Hop2 | 200 | 4.5% | 52.5% | 4.5% |
| Qwen3-32B | Factual | No Conflict | 200 | 58.0% | - | - |
| Qwen3-32B | Factual | Conflict@Hop1 | 87 | 37.9% | 2.3% | 37.9% |
| Qwen3-32B | Factual | Conflict@Hop2 | 200 | 18.0% | 25.0% | 18.0% |
| Qwen3-32B | Temporal | No Conflict | 200 | 90.5% | - | - |
| Qwen3-32B | Temporal | Conflict@Hop1 | 120 | 59.2% | 0.8% | 59.2% |
| Qwen3-32B | Temporal | Conflict@Hop2 | 200 | 6.5% | 78.0% | 6.5% |
| Qwen3-32B | Numerical | No Conflict | 200 | 64.5% | - | - |
| Qwen3-32B | Numerical | Conflict@Hop1 | 129 | 42.6% | - | 42.6% |
| Qwen3-32B | Numerical | Conflict@Hop2 | 200 | 5.0% | 54.5% | 5.0% |

## Summary: Accuracy Drop by Conflict Type

| Model | Conflict Type | Baseline | Avg Conflict Acc | Drop (pp) |
|-------|---------------|----------|------------------|-----------|
| Llama-3.3-70B | Factual | 60.0% | 37.9% | 22.1 |
| Llama-3.3-70B | Temporal | 89.5% | 52.2% | 37.3 |
| Llama-3.3-70B | Numerical | 70.0% | 31.6% | 38.4 |
| Llama-3.1-8B | Factual | 57.5% | 36.8% | 20.7 |
| Llama-3.1-8B | Temporal | 88.0% | 38.8% | 49.2 |
| Llama-3.1-8B | Numerical | 56.0% | 27.7% | 28.3 |
| Gemini-2.5-Flash-Lite | Factual | 62.0% | 28.8% | 33.2 |
| Gemini-2.5-Flash-Lite | Temporal | 89.0% | 29.9% | 59.1 |
| Gemini-2.5-Flash-Lite | Numerical | 56.5% | 19.3% | 37.2 |
| Qwen3-32B | Factual | 58.0% | 28.0% | 30.0 |
| Qwen3-32B | Temporal | 90.5% | 32.8% | 57.7 |
| Qwen3-32B | Numerical | 64.5% | 23.8% | 40.7 |
