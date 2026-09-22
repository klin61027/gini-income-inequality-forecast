# Forecasting U.S. Income Inequality with Autoregressive Models

> A senior capstone that builds a ladder of time-series models — from a simple trend line up to ARIMAX — to forecast the U.S. Gini index, and finds that the *simplest* autoregressive model wins.

---

## The Problem

The **Gini index** is the standard summary of income inequality: 0 means everyone earns the same, 1 means one person earns everything. U.S. inequality has drifted upward for decades, and economists, policymakers, and journalists all want to know: *where is it headed?*

Forecasting it is harder than it looks. We only get **one Gini value per year**, so the dataset is tiny (about 60 points). With so little data, it is dangerously easy to build a model that looks brilliant on the years it was trained on and then falls apart on the years it has never seen.

So the real question of this project was not just *"what will inequality be in 2030?"* but a methodological one:

**Does adding model complexity actually improve out-of-sample forecasts of inequality — or does it just fit noise?**

---

## How It Works

We built a **three-layer ladder of models**, each more expressive than the last, and judged them all the same way: by how well they predict years they were never trained on.

```
Layer 1  —  Rolling trend line          (where is inequality drifting?)
              ↓
Layer 2  —  AR(p) autoregression        (how much does last year predict this year?)
              ↓
Layer 3  —  ARX / ARMA / ARIMA / ARIMAX (does the economy — GDP, CPI, unemployment — help?)
```

Every model is scored with an **expanding-window rolling forecast**: train on 1963→year *t*, predict year *t*+1, step forward, repeat. Nothing the model sees when it predicts a year comes from that year or later, so there is **no look-ahead bias** — the error you see is the error you would have gotten forecasting in real time.

- **Train window:** 1963–1993
- **Test window:** 1994–2023 (30 years of true out-of-sample forecasts)

Model complexity inside Layers 2 and 3 is chosen with **AIC and BIC**, which reward fit but penalize extra parameters — the formal way of asking "is this added term earning its keep?"

---

## What We Built

### Layer 1 — Rolling OLS Trend
An expanding-window least-squares fit of `Gini = b + a·(year − t₀)`. This captures the long upward drift but nothing about year-to-year dynamics.
**Out-of-sample: MAE 0.0081 · RMSE 0.0106 · MAPE 2.01%**

### Layer 2 — Autoregression, AR(p)
We let the Gini index predict itself from its own recent history and used **BIC to choose the order p**. BIC picked **p = 1** — a single lag.

The fitted model:

```
Giniₜ = 0.01230 + 0.97038 · Giniₜ₋₁ + εₜ
```

That 0.97 coefficient is the whole story in one number: **about 97% of last year's inequality carries straight into this year.** Inequality is enormously persistent.
**Out-of-sample: MAE 0.0046 · RMSE 0.0066 · MAPE 1.13%** — a big jump over the trend line.

### Layer 3 — Adding the Economy (ARX, ARMA, ARIMA, ARIMAX)
If inequality is driven by the economy, then GDP, inflation, and unemployment should sharpen the forecast. We tested them. Exogenous inputs were year-over-year changes in **GDP, CPI, and unemployment** (FRED data), at lags of 1–3 years, with **BIC choosing which to keep**.

The best exogenous specification (ARX) was:

```
Giniₜ = c + φ₁·Giniₜ₋₁ + β₁·ΔGDPₜ₋₂ + β₂·ΔCPIₜ₋₁ + εₜ
```

We also fit ARMA(2,2), ARIMA(3,1,3), and a full ARIMAX. The MA-containing models were fit by **conditional maximum likelihood** (`scipy.optimize`), since OLS can't estimate moving-average terms.

---

## The Result — Simpler Is Better

Here is every model on the same 30-year out-of-sample test:

| Model | MAE | RMSE | MAPE |
|-------|------|------|------|
| Layer 1 — Rolling trend | 0.0081 | 0.0106 | 2.01% |
| **Layer 2 — AR(1)** | **0.0046** | **0.0066** | **1.13%** |
| 3A — ARX (GDP + CPI) | 0.0048 | 0.0059 | 1.17% |
| 3B — ARMA(2,2) | 0.0053 | 0.0074 | 1.31% |
| 3C — ARIMA(3,1,3) | 0.0057 | 0.0082 | 1.40% |
| 3D — ARIMAX | 0.0057 | 0.0082 | 1.40% |

**The plain AR(1) is the best forecaster.** Every richer model — adding macro variables, adding moving-average terms, adding differencing — either matched it or did *worse* out of sample. ARX shaved RMSE by a hair but lost on MAE and MAPE, and none of it was a real improvement.

> **Key takeaway:** Adding complexity does not improve out-of-sample prediction of inequality. The Gini index is dominated by its own persistence — knowing last year's value gets you almost all the way there, and the macroeconomy adds little on top. AIC and BIC pointed at the simple model, and the out-of-sample test confirmed it. This is a clean illustration of the bias–variance tradeoff: on a 60-point series, extra parameters buy in-sample fit and pay for it in forecast error.

---

## Tech Stack

| Tool | Used for |
|------|----------|
| Python 3 | Everything |
| pandas / numpy | Data wrangling, deltas, expanding windows |
| statsmodels | `AutoReg` (AR selection), `ARIMA` |
| scipy.optimize | Custom conditional-MLE fit for MA-containing models |
| matplotlib | All figures |

Data: U.S. Gini index plus GDP, CPIAUCSL, and UNRATE from **FRED**.

---

## Try It Yourself

```bash
git clone https://github.com/klin61027/gini-forecasting.git
cd gini-forecasting
pip install -r requirements.txt
python src/senior_project.py
```

The script reads the CSVs from `data/`, runs all three layers, prints the results table, and writes every figure to `results/figures/`.

---

## Honest Tradeoffs

- **Tiny sample.** ~60 annual points is the fundamental constraint. It is exactly *why* the complex models overfit, and it caps how much any method can do.
- **Annual data only.** The Gini index is published yearly, so we can't exploit higher-frequency dynamics.
- **Single-country.** The models are fit on the U.S. alone; a panel across countries would test whether "persistence dominates" generalizes.
- **The forecast assumes the past regime holds.** A structural break (a policy shock, a crisis) is exactly what a self-referential AR(1) can't see coming.

---

## Repository Layout

```
gini-forecasting/
├── README.md                   # You are here
├── requirements.txt
├── data/
│   ├── us_gini.csv
│   ├── GDP.csv
│   ├── CPIAUCSL.csv
│   └── UNRATE.csv
├── src/
│   └── senior_project.py       # Full three-layer pipeline
├── results/
│   ├── figures/                # All generated plots (.png)
│   └── layer2_results.csv
└── docs/
    ├── Senior_Presentation.pdf # Final talk
    └── capstone_paper.pdf      # Written synthesis
```

---

## Team

**Kevin Lin (林敬智)** and **Bruce Chen** · Seattle University · Spring 2026
Advised by Dr. Henrich
