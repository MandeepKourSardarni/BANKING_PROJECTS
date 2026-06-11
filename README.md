# Banking_projects — Learn ML algorithms by building them, in a banking context

A hands-on curriculum: one algorithm at a time, each as a small banking project,
as a self-contained Jupyter notebook (.ipynb) with **every line of code commented in plain English**. The aim is to learn the
algorithm, understand it deeply, and prepare for a Data Science / AI Model Risk
interview — all from the same code.

## How to use this repo
- Work through the numbered folders **in order**.
- In each folder: read the `README.md` first (the theory + interview Q&A), then
  read the heavily-commented code top to bottom, then run it and read the output.
- Each module is self-contained and runs on synthetic data — no downloads needed.

## Roadmap

| # | Module | Banking project | Status |
|---|--------|-----------------|--------|
| 01 | **Regression** | Predict expected loss on defaulted loans (the "L" in PD/LGD/EAD) | ✅ done |
| 02 | Classification | Probability of Default (PD) — will this loan go bad? | ⬜ next |
| 03 | Anomaly detection | Unusual transactions / accounts | ⬜ |
| 04 | Natural Language Processing | Classify customer complaints / extract info | ⬜ |
| 05 | Computer vision | Cheque / document image processing | ⬜ |
| 06 | Reinforcement learning | Sequential decisions (e.g. collections strategy) | ⬜ |
| 07 | Recommendation systems | Next-best product for a customer | ⬜ |
| 08 | Dimensionality reduction | PCA on customer/portfolio features | ⬜ |
| 09 | LLMs (generative + agentic AI) | Document Q&A / agentic workflows + their risks | ⬜ |



## Setup (once)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Banking_projects: regression module"
git remote add origin https://github.com/<your-username>/Banking_projects.git
git branch -M main
git push -u origin main
```
