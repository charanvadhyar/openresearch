# AutoResearch

**Autonomous ML Research Agent — research for everyone.**

You have a problem and some data. AutoResearch does the first 3–5 days of
research work in a few hours — EDA, method selection, running experiments,
comparing results, and writing a structured draft you can build on.

**Requirements:**
- A Kaggle account (free) — this is where your code runs
- An Anthropic API key — this powers the AI reasoning
- Python 3.11+

That's it. No GPU. No cloud account. No ML expertise required.

---

## What you get

After a run, you'll find a **Research Starter Pack** in `./autoresearch_output/`:

| File | What it is |
|------|-----------|
| `data_report.html` | Everything AutoResearch found about your data |
| `method_comparison.html` | Side-by-side comparison of all methods tried |
| `best_model/` | Your best model, saved and ready to use |
| `notebooks/` | One reproducible notebook per method |
| `paper_draft.docx` | Structured draft — yours to edit and build on |
| `literature_snapshot.md` | Relevant papers to start from |
| `next_steps.md` | Concrete recommendations for your research |

> ⚠️ The paper draft is a **starting point**, not a finished paper.
> Every AI-generated section is marked `[DRAFT — YOUR REVIEW NEEDED]`.
> The experiment results are real. The written sections need your voice.

---

## Quickstart

### 1. Install

```bash
pip install autoresearch
```

### 2. Set up your credentials

```bash
autoresearch init
```

This walks you through setting up your Kaggle token and Anthropic key.
They're stored locally and never shared with anyone.

### 3. Configure your problem

Open `config.yaml` and fill in two things:

```yaml
problem:
  statement: "I want to predict whether a patient will be readmitted to
              hospital within 30 days, using 20 features from their record."
  data_source:
    type: "kaggle"
    identifier: "your-competition-name"
```

### 4. Run

```bash
autoresearch run
```

AutoResearch will talk you through what it's doing at each step.
You can watch it work, or come back in a few hours.

---

## Who is this for?

- A student who wants to do research but has no lab, no GPU, no advisor
- A doctor or economist who understands the problem deeply but not the ML
- A self-taught programmer ready to move from tutorials to original work
- A small research team that can't afford to spend a week on baselines

**The only requirements are curiosity and patience.**

---

## How it works

```
Your problem statement
        ↓
Problem Analyst     — figures out what kind of ML problem this is
        ↓
EDA Agent           — looks at your data, finds patterns and issues
        ↓
Method Formulator   — proposes 3–5 approaches (explains why each one)
        ↓
Code Generator      — writes clean, runnable notebooks for each method
        ↓
Execution Agent     — runs everything on YOUR free Kaggle GPU
        ↓
Evaluator           — compares results, picks the best, flags risks
        ↓
Paper Writer        — builds your Research Starter Pack
```

Every step explains its reasoning in plain English.
If AutoResearch is unsure about something, it asks you before continuing.

---

## Cost

| What | Who pays | How much |
|------|----------|----------|
| GPU compute | You (Kaggle free tier) | $0 |
| Claude API calls | You (your Anthropic key) | ~$0.50–$3 per run |
| AutoResearch itself | — | Free, open source (MIT) |

A typical run uses 3 Kaggle kernels (EDA + 3 methods) and costs about $1–2
in Claude API calls.

---

## Example runs

Coming soon:
- `examples/heart_disease/` — medical classification, small dataset
- `examples/house_prices/` — regression, Kaggle competition
- `examples/sentiment/` — NLP text classification

---

## Contributing

AutoResearch is built for researchers who were told they didn't have the
resources to do research. If that resonates with you, contributions are welcome.

See `CONTRIBUTING.md` for how to get started.

---

## License

MIT — use it, build on it, share it.
