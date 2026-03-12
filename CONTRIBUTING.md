# Contributing to OpenResearch

AutoResearch exists for researchers who were told they didn't have the
resources to do research. If that resonates with you, we'd love your help.

---

## Where to start

The best first contributions are ones that make AutoResearch more useful
for researchers who are new to ML. That means:

- **Clearer error messages** — if something fails and the message is confusing, fix it
- **New knowledge base entries** — adding algorithms that `ml_methods.json` doesn't cover yet
- **New notebook templates** — improving the Jinja2 templates in `templates/notebooks/`
- **More tests** — especially tests that catch real-world data edge cases
- **Documentation** — examples, walkthroughs, translations

Before writing a new feature, open an issue and describe what you want to build.
This saves everyone time.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourorg/autoresearch
cd autoresearch

# Install in development mode
pip install -e ".[dev]"

# Run tests (no API keys needed)
pytest tests/ -v -m "not integration"
```

The dev dependencies are defined in `pyproject.toml` under `[project.optional-dependencies]`.

---

## Running the tests

```bash
# Unit tests only (no API keys, fast)
pytest tests/ -v -m "not integration"

# With coverage
pytest tests/ -v --cov=autoresearch --cov-report=term-missing

# All tests including integration (needs keys in env)
ANTHROPIC_API_KEY=sk-... KAGGLE_USERNAME=... KAGGLE_KEY=... \
  pytest tests/ -v
```

---

## Adding a new algorithm to the knowledge base

Edit `knowledge_base/ml_methods.json`. Add an entry under the appropriate task type:

```json
{
  "id": "catboost",
  "name": "CatBoost",
  "family": "Gradient Boosting",
  "requires_gpu": false,
  "min_rows": 500,
  "max_rows": 5000000,
  "strengths": [
    "Handles categorical features natively without encoding",
    "Often works well with default parameters",
    "Built-in overfitting detection"
  ],
  "weaknesses": [
    "Slower to train than LightGBM",
    "Large model files"
  ],
  "good_when": [
    "many high-cardinality categorical features",
    "you want strong defaults without tuning"
  ],
  "bad_when": [
    "dataset is very small",
    "training speed is critical"
  ],
  "hyperparam_space": {
    "n_estimators": [100, 1000],
    "depth": [4, 10],
    "learning_rate": [0.01, 0.3],
    "l2_leaf_reg": [1, 10]
  },
  "feature_engineering": ["SimpleImputer(median)"],
  "complexity": "medium",
  "estimated_runtime_minutes": 15
}
```

Then add a test to `tests/test_autoresearch.py` verifying the required fields.

---

## Mentor voice guidelines

Every user-facing string in AutoResearch should follow these principles.
Before merging, read your error messages and progress updates out loud.
Ask: "Would this make sense to someone who has never used ML before?"

**Don't:**
```python
raise ValueError("InvalidShapeError: expected (n, d) got (n,)")
```

**Do:**
```python
raise ValueError(
    "The model expected 2D input but got 1D data. "
    "This usually means the target column is included in the features. "
    "Check that TARGET_COL is correctly set to your label column."
)
```

---

## Pull request checklist

Before opening a PR:

- [ ] `pytest tests/ -v -m "not integration"` passes
- [ ] `ruff check autoresearch/` passes
- [ ] New functions have docstrings
- [ ] Error messages are in plain English
- [ ] `knowledge_base/ml_methods.json` changes have all required fields

---

## Code of conduct

Be kind. This project is for people who are learning. If someone asks a
question that seems basic, remember that expertise is unevenly distributed
and curiosity is a gift.

---

## License

MIT. Everything you contribute becomes part of the commons.
