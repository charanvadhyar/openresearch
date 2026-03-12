# Contributing to openresearch

openresearch exists to make applied ML research accessible. If that aligns with your goals, contributions are welcome.

## Where to start

Good first contributions improve usability and reliability:
- Clearer errors and diagnostics
- More robust notebook templates
- Tests for real-world data edge cases
- Documentation and examples

Please open an issue before large changes to align on scope.

## Setup

```bash
# Clone the repo
git clone https://github.com/charanvadhyar/openresearch.git
cd openresearch

# Install dev deps
pip install -e ".[dev]"

# Run tests (no API keys needed)
pytest tests/ -v -m "not integration"
```

## Running tests

```bash
# Unit tests only
pytest tests/ -v -m "not integration"

# With coverage
pytest tests/ -v --cov=autoresearch --cov-report=term-missing

# Integration tests (needs keys in env)
OPENAI_API_KEY=sk-... KAGGLE_USERNAME=... KAGGLE_KEY=... \
  pytest tests/ -v
```

## Adding a new algorithm

Edit `knowledge_base/ml_methods.json` and add a method entry under the right task type. Then add a test to `tests/test_autoresearch.py` verifying required fields.

## Style

User-facing messages should be plain English and actionable. Avoid jargon unless you explain it.

## PR checklist

- `pytest tests/ -v -m "not integration"` passes
- New functions include docstrings where appropriate
- Error messages are clear and user-focused
- Changes to `ml_methods.json` include required fields

## License

MIT
