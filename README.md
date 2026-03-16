# openresearch

Openresearch is an autonomous ML research assistant that turns a dataset and a goal into a tested set of baselines, clear comparisons, and a research starter pack you can build on. It is designed for researchers and builders who want credible experiments without spending days wiring the pipeline.

## Why openresearch

Most research time is not spent on modeling. It is spent on wiring data, running repeated baselines, chasing errors, and formatting results. openresearch automates that first pass so you can focus on ideas and analysis instead of plumbing.

Use it when you need:
- Fast, reproducible baselines
- Clear tradeoffs across methods
- A structured report with artifacts you can share
- A research draft you can refine

## What you get

After a run, openresearch writes a research starter pack to the output directory:
- `data_report.html` for EDA, diagnostics, and dataset risks
- `method_comparison.html` to compare metrics and runtime
- `notebooks/` with one notebook per method for reproducibility
- `best_model/` with saved artifacts and metadata
- `paper_draft.docx` as an editable draft outline

## How it works

```
Problem statement
  -> Problem Analyst
  -> EDA Agent
  -> Method Formulator
  -> Code Generator
  -> Execution Agent (Kaggle kernels)
  -> Evaluator
  -> Paper Writer
```

## Features

- Problem analysis that infers task type and target
- Dataset diagnostics for missing values, leakage, imbalance, and size risks
- Automatic method selection based on dataset characteristics
- Parallelized Kaggle execution with retries and timeouts
- Robust error capture and failure summaries
- Clear scoring across performance, speed, interpretability, and robustness

## Requirements

- Python 3.11+
- Kaggle account and API token
- LLM API key (Anthropic, OpenAI, or [MiniMax](https://platform.minimax.io))

## Quickstart

### 1. Install

```bash
pip install autoresearch
```

### 2. Configure

Edit `config.yaml` and add your keys plus your problem and data source.

```yaml
api_keys:
  anthropic: "YOUR_KEY"
  openai: "YOUR_KEY"
  minimax: "YOUR_KEY"           # optional — https://platform.minimax.io
  kaggle_username: "YOUR_USERNAME"
  kaggle_key: "YOUR_KAGGLE_KEY"

problem:
  statement: "Predict churn from customer features"
  data_source:
    type: "kaggle"
    identifier: "username/dataset"
```

### 3. Run

```bash
autoresearch run
```

## Typical workflow

1. Provide a plain English problem statement
2. Point to a dataset (Kaggle, local CSV, or HuggingFace)
3. openresearch runs EDA and proposes methods
4. Kernels execute in parallel with progress tracking
5. You receive a report, notebooks, and a best model

## Use cases

- Academic baselines for new datasets
- Internal model selection with limited engineering time
- Fast feasibility checks before investing in large experiments

## Examples

See `examples/` for sample configs and expected outputs.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT
