# Knowledge Conflicts in Multi-Hop Reasoning

Characterizing how knowledge conflicts propagate through multi-hop reasoning chains when retrieved context contradicts an LLM's parametric memory.

## Key Findings

| Model | No Conflict | Conflict@Hop1 | Conflict@Hop2 |
|-------|-------------|---------------|---------------|
| Llama-3.3-70B | 71.0% | 42.0% (↓29pp) | 45.0% (↓26pp) |
| Llama-3.1-8B | 53.0% | 37.0% (↓16pp) | 30.0% (↓23pp) |

- Conflicts cause **16-29 percentage point accuracy drops** (p < 0.05)
- Hop 1 conflicts are more damaging for larger models
- Smaller models are more vulnerable overall but show different conflict patterns

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env (https://console.groq.com/)
python test_setup.py
```

## Run Experiments

You can run experiments either via **Python scripts** (CLI) or **Jupyter notebooks** (interactive). Pick whichever suits your workflow.

### Option A — Python Scripts

```bash
# Run primary experiment (single model, 100 examples, 3 conditions)
python experiments/run_conflict_experiment.py

# Run multi-model comparison (skips models with existing results)
python experiments/run_model_comparison.py

# Extract qualitative examples
python experiments/extract_qualitative.py

# Generate figures from results
python src/analysis/visualize.py
```

### Option B — Jupyter Notebooks

```bash
jupyter notebook
```

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_explore_results.ipynb` | Browse results, regenerate charts, inspect individual predictions |
| `notebooks/02_run_model_comparison.ipynb` | Run new model experiments and generate comparison figures interactively |

## Project Structure

```
src/
  data/
    hotpotqa_loader.py       # Load HotpotQA, extract bridge questions
    conflict_injector.py     # Entity substitution for conflict injection
  inference/
    groq_client.py           # Groq API wrapper with rate limit handling
    prompt_templates.py      # CoT prompts and answer extraction
  evaluation/
    metrics.py               # Answer checking (correct, CFR, POR)
  analysis/
    visualize.py             # Publication-quality figures

experiments/
  run_conflict_experiment.py # Main experiment (single model)
  run_model_comparison.py    # Multi-model comparison runner
  extract_qualitative.py     # Extract examples for slides

notebooks/
  01_explore_results.ipynb       # Interactive result explorer
  02_run_model_comparison.ipynb  # Run models & generate comparisons

outputs/
  results/<model-id>/        # Per-model experiment results (JSON)
  figures/                   # Generated charts (PNG)
  qualitative_examples.md    # Slide-ready example breakdowns
```

## Adding a New Model

Edit `MODELS` in `experiments/run_model_comparison.py`:

```python
MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B"},
    {"id": "your-new-model-id",       "label": "Display Name"},  # add here
]
```

Then run `python experiments/run_model_comparison.py` — it only runs models without existing results.

## Methodology

1. **Dataset**: HotpotQA bridge questions (2-hop reasoning)
2. **Conflict Injection**: Entity substitution — replace correct answer with plausible fake in one supporting document
3. **Conditions**: No conflict (baseline), conflict at hop 1, conflict at hop 2
4. **Metrics**: Accuracy, Context-Following Rate (CFR), Parametric-Override Rate (POR)
5. **Inference**: Chain-of-Thought prompting, temperature=0
