# 01 — Regression: Predicting Expected Loss on Defaulted Loans

**Goal of this module:** understand regression *fully* — the maths, the code, the
assumptions, and the way a Model Risk reviewer interrogates it — using a real
banking problem.

---

## The problem in plain words

When a borrower defaults, the bank rarely loses the whole loan. It recovers some
money. The amount it *fails* to recover is the **loss**. Predicting that dollar
loss is a **regression** problem (the answer is a number, not a yes/no).

This is the **"L" in the PD / LGD / EAD** credit-risk framework:
- **PD** – Probability of Default (a *classification* problem — module 02)
- **LGD** – Loss Given Default, the % of the exposure lost (regression)
- **EAD** – Exposure At Default, the amount owed when default happens

Expected Loss ≈ **PD × LGD × EAD**. This module models the loss dollars directly.

---

## What's in here

A single self-contained notebook, `01_regression.ipynb`. Open it in Jupyter
(Anaconda), Colab, or VS Code and run the cells top to bottom. It walks through:
data generation, linear regression **from scratch** (normal equation + gradient
descent), the scikit-learn pipeline (Linear/Ridge/Lasso, cross-validation,
metrics), and the model-risk **diagnostics** (residuals, QQ normality, VIF,
coefficient interpretation). Charts render inline; outputs are already saved so
you can read it before running.

### Run it
```bash
# Anaconda users: just open the file in Jupyter and press Shift+Enter through it.
pip install -r ../requirements.txt   # only if something is missing
jupyter notebook 01_regression.ipynb
```

---

## The core idea

Linear regression predicts the target as a **weighted sum** of the inputs:

```
loss_hat = w0 + w1·loan_amount + w2·collateral_value + ... 
```

"Training" = finding the weights that make predictions closest to reality.
"Closest" is measured by **Mean Squared Error (MSE)** — the average squared miss.
Two ways to find those weights:

1. **Normal Equation:** `w = (Xᵀ X)⁻¹ Xᵀ y`. One exact calculation. Great for
   small/medium data; slow when there are very many features (matrix inverse).
2. **Gradient Descent:** start random, repeatedly nudge the weights *downhill* on
   the error surface. Scales to huge data and is how neural nets train too.

Our from-scratch file shows **both give the same answer** — and that they match
scikit-learn. That's how you know you understand it.

---

## What the model found (and why it's "correct")

Because we *built* the data, we know the true relationships, and the fitted
coefficients recover them with the right signs:

| Feature | Coefficient sign | Makes sense? |
|---------|------------------|--------------|
| loan_amount | **+** (largest) | bigger loan → bigger dollar loss ✔ |
| collateral_value | **−** | more collateral → recover more → less loss ✔ |
| debt_to_income | **+** | more debt burden → more loss ✔ |
| num_prior_defaults | **+** | repeat defaulters lose more ✔ |
| interest_rate | **+** | riskier (pricier) loans lose more ✔ |
| annual_income, borrower_age | **≈ 0** | we never built these into the loss ✔ |

Recovering the right signs is the first thing a reviewer checks: *do the
coefficients agree with business intuition?*

---

## The diagnostics (this is the Model Risk part)

A model that scores well can still be **wrong for the wrong reasons**. We check:

- **Residuals vs Fitted** (`artifacts/residuals_vs_fitted.png`): the leftover
  errors should look like a random cloud around zero. A funnel shape =
  *heteroscedasticity* (error grows with size) → standard errors untrustworthy.
- **Normal QQ plot** (`artifacts/qq_plot.png`): residuals should fall on the
  diagonal if they're roughly normal. Big departures break the assumptions
  behind confidence intervals and p-values.
- **VIF (Variance Inflation Factor):** measures *multicollinearity* — whether a
  feature is predictable from the others. Here `loan_amount`, `monthly_payment`
  and `collateral_value` are correlated (payment is derived from the amount), so
  their VIFs are elevated. High VIF (>5, definitely >10) makes individual
  coefficients unstable and hard to trust.

---

## Linear vs Ridge vs Lasso

All three are linear models; the difference is the **penalty** on big weights:

- **Linear (OLS):** no penalty. Can overfit and gets shaky with correlated features.
- **Ridge (L2):** penalises the *square* of weights → shrinks them smoothly,
  stabilises correlated features. Keeps all features.
- **Lasso (L1):** penalises the *absolute* value → can push weak weights to
  **exactly zero**, so it also does *feature selection*.

The penalty strength is `alpha` — a **hyperparameter** you tune (e.g. with
cross-validation), not something the model learns.

---

## Interview prep — questions you should be able to answer

**Conceptual**
1. *Why not use accuracy for regression?* Accuracy is for classes. Regression uses
   error magnitude: RMSE, MAE, MAPE, R².
2. *RMSE vs MAE?* RMSE squares errors so it punishes large misses more (sensitive
   to outliers); MAE is the plain average error (more robust). RMSE ≫ MAE signals
   a few big errors — true here because the log back-transform amplifies large losses.
3. *What does R² mean?* The fraction of the target's variance the model explains;
   1.0 is perfect, 0 is no better than predicting the mean. Can be misleading —
   always pair it with an error metric in real units.
4. *OLS assumptions?* Linearity, independent errors, constant error variance
   (homoscedasticity), roughly normal residuals, low multicollinearity.
5. *Why log-transform the target?* Loss is right-skewed and strictly positive;
   logging makes it more symmetric so the linear model fits better. Remember to
   exponentiate predictions back to dollars before reporting.
6. *Ridge vs Lasso — when?* Ridge when you want to keep all (correlated) features
   but stabilise them; Lasso when you also want automatic feature selection.
7. *Normal Equation vs Gradient Descent?* Normal eq is exact but costs a matrix
   inverse (bad for very wide data); gradient descent scales and generalises to
   non-linear models, but needs a learning rate and feature scaling.

**Model Risk flavoured**
8. *How would you validate this model?* Out-of-sample/out-of-time testing, check
   coefficient signs vs intuition, residual diagnostics, stability over time,
   benchmark against a simpler model, and document limitations.
9. *What's the risk of multicollinearity in a regulatory model?* Coefficients
   become unstable, so attributions ("which factor drives loss") become
   unreliable — a problem when you must *explain* decisions.
10. *Why scale features?* So regularisation penalises them fairly and gradient
    descent converges; and so coefficient magnitudes are comparable.

---

## Limitations (state these — reviewers love honesty)
- Synthetic data — do **not** draw real conclusions; swap in real data.
- LGD is bounded [0,1]; we modelled dollars and logged them. A stricter approach
  is *beta regression* or a fractional-response model — a good "what next" answer.
- A single train/test split was used for the headline numbers; CV gives the
  steadier estimate. Out-of-*time* testing matters even more in banking.
