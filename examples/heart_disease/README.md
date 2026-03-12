# Example: Heart Disease Prediction

This example walks through a complete AutoResearch run on the
[UCI Heart Disease dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci).

It's a good first run because:
- Dataset is small (303 rows) — fast, no GPU needed
- Problem is well-understood — you can verify AutoResearch's reasoning
- Interpretability matters — shows how the constraint system works

---

## What this example does

**Input:** 303 patients, 13 cardiac measurements, binary target (heart disease yes/no)

**AutoResearch will:**
1. Classify this as a binary classification problem (confidence ~0.95)
2. Notice the dataset is small and flag deep learning as inappropriate
3. Propose 3 interpretable methods (Logistic Regression, Random Forest, XGBoost)
4. Run all 3 on Kaggle CPU — takes ~20 minutes total
5. Pick the winner using your weighted criteria (interpretability weighted higher)
6. Write a paper draft explaining what was done

**Expected results:**
| Method | AUC-ROC | Train time |
|---|---|---|
| Logistic Regression | ~0.82 | ~30 sec |
| Random Forest | ~0.85 | ~2 min |
| XGBoost | ~0.87 | ~3 min |

Winner: XGBoost (or Random Forest depending on your interpretability weight)

---

## Setup

### 1. Get the dataset

Go to https://www.kaggle.com/datasets/ronitf/heart-disease-uci and accept the terms.

### 2. Fill in your credentials

Open `config.yaml` and fill in:
```yaml
api_keys:
  anthropic: "sk-ant-..."
  kaggle_username: "your-username"
  kaggle_key: "your-key"
```

### 3. Run

```bash
cd examples/heart_disease
autoresearch run --config config.yaml --output ./heart_disease_output
```

Or from the repo root:
```bash
autoresearch run \
  --config examples/heart_disease/config.yaml \
  --output examples/heart_disease/heart_disease_output
```

---

## What you'll get

After ~45 minutes, `./heart_disease_output/` will contain:

```
heart_disease_output/
├── data_report.html          — EDA: what AutoResearch found in the data
├── comparison.html           — Side-by-side method comparison
├── paper_draft.md            — Paper skeleton (review required)
├── paper_draft.docx          — Word version of the draft
├── notebooks/
│   ├── logistic_regression.ipynb
│   ├── random_forest.ipynb
│   └── xgboost.ipynb
└── run_XXXXXXXX_state.json   — Saved state (for resuming)
```

---

## What to do next

After the run, open `comparison.html` in your browser. Look at:

1. **The winner's AUC score** — is it in the expected range (~0.85–0.88)?
2. **The train/val gap** — should be under 0.05 for all three methods on this data
3. **The risk flags** — probably none for this clean dataset

Then open `paper_draft.md` and find all the `[RESEARCHER TO FILL]` sections.
The most important ones:
- Why does heart disease prediction matter in your specific context?
- What's novel about your approach vs existing literature?
- What do you plan to do next that AutoResearch can't do for you?

---

## Understanding the output

### Why XGBoost wins

XGBoost wins on raw performance (~0.87 AUC). With interpretability weighted at
0.30 in this config, it's slightly penalized compared to Logistic Regression
(perfect interpretability score), but its performance advantage is large enough
to win overall.

If you increase interpretability weight to 0.50, Logistic Regression will win.
That might be the right call if doctors need to explain each decision to patients.

### Reading the data report

The data report will show:
- **Age**: right-skewed, range 29–77 — expected
- **Cholesterol**: some outliers (values over 400) — flagged as warning
- **Thal**: categorical with 4 values — high mutual information with target
- **Health score**: ~85/100 — clean dataset, minor issues only

### The paper draft

Every AI-generated section is marked `[DRAFT — RESEARCHER REVIEW NEEDED]`.
The methodology section will describe what was actually run (accurate).
The introduction and related work sections need your knowledge of the clinical
literature (AutoResearch doesn't know which papers matter for your specific work).

---

## Modifying this example

**To try more methods:** change `max_methods: 3` to `max_methods: 5`

**To prioritize interpretability even more:**
```yaml
evaluation_weights:
  performance:      0.25
  speed:            0.10
  interpretability: 0.50
  robustness:       0.15
```

**To try a different dataset:** change `identifier` to another Kaggle tabular
classification dataset. The rest of the config works for any binary classification problem.
