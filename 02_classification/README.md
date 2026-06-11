# 02 — Classification: Probability of Default (PD)

**Goal:** understand binary classification fully — sigmoid, log loss, the metrics
that matter under class imbalance, threshold choice, and calibration — using a
**real, open dataset**.

## The problem
Will a credit-card customer default next month? The probability of "yes" is the
**PD** in `Expected Loss ≈ PD × LGD × EAD`. Module 01 modelled LGD with
regression; this models PD with classification.

## The dataset
**Default of Credit Card Clients** (Taiwan, 2005), UCI repository — 30,000 real
customers, 23 features, a 0/1 default label, ~22% default rate. On UCI and Kaggle.

## Importing data (covered in the notebook, Step 0)
- **Method A — direct URL** with `pandas.read_csv(url)` (no account; what the notebook runs).
- **Method B — Kaggle API**: create `kaggle.json` token, `pip install kaggle`,
  `kaggle datasets download -d uciml/default-of-credit-card-clients-dataset --unzip`.
- **Method C — kagglehub**: `kagglehub.dataset_download(...)`.
- **Bonus — UCI package**: `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=350)`.

## What's in here
A single self-contained notebook, `02_classification.ipynb`: data import, EDA,
logistic regression **from scratch** (sigmoid + log loss + gradient descent),
the scikit-learn pipeline with a gradient-boosting benchmark, the full metric
suite (confusion matrix, precision/recall/F1, ROC, PR), **threshold selection by
cost**, **calibration**, odds-ratio interpretation, and interview Q&A. Outputs
and charts are pre-run.

### Run it
```bash
# Anaconda: open the file in Jupyter and Shift+Enter through it.
pip install -r ../requirements.txt   # only if something is missing
jupyter notebook 02_classification.ipynb
```

## Headline results (on the real data)
- From-scratch logistic AUC **0.716** == scikit-learn logistic **0.716** (implementation verified).
- Gradient boosting AUC **0.778** — beats logistic, the kind of uplift you must justify.
- Cost-minimising threshold **0.39** vs the naive 0.50.

## Key ideas / interview points
Accuracy is a trap under imbalance; use precision, recall, F1, ROC-AUC, PR-AUC.
The 0.5 threshold is a business choice, not a law. For a PD model, **calibration**
(do the probabilities mean what they say?) matters as much as discrimination,
because PDs feed capital. Prefer a transparent logistic model unless a complex
one's uplift is real and explainable.
