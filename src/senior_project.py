# -*- coding: utf-8 -*-
"""
Forecasting U.S. Income Inequality with Autoregressive Models
Senior capstone - Kevin Lin & Bruce Chen, Seattle University, Spring 2026

Builds a three-layer ladder of time-series models (rolling OLS trend ->
AR(p) -> ARX/ARMA/ARIMA/ARIMAX) and evaluates each with an expanding-window
out-of-sample forecast of the U.S. Gini index (1994-2023).

Run from the repo root:  python src/senior_project.py
Data CSVs go in data/ ; figures are written to results/figures/.
"""

# ════════════════════════════════════════════════════════════════
#  SETUP — run this first
#  Defines local data/output folders (relative to the repo root, so
#  this runs anywhere after `git clone`) and imports every library
#  used below ONCE so no later cell has to re-import.
# ════════════════════════════════════════════════════════════════
import os

# Put the FRED CSVs in data/ ; every figure is written to results/figures/.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def figpath(name):
    """Route every figure into results/figures/ regardless of the bare filename."""
    return os.path.join(FIG_DIR, name)

print("Data dir:", DATA_DIR)
print("CSVs found:",
      [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
      if os.path.isdir(DATA_DIR) else "(data/ not found - add the CSVs)")

# --- shared imports for the whole notebook ---
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

"""# Data"""

# ════════════════════════════════════════════════════════════════
#  SHARED FIGURE STYLE — Seattle U house theme (maroon / gray)
#  Run this ONCE. It sets GLOBAL rcParams, so every figure below
#  (even cells you don't edit) picks up the same fonts, maroon
#  titles, grid, and color cycle automatically.
# ════════════════════════════════════════════════════════════════
# --- Core colors ---
SU_MAROON = "#960000"   # primary (Seattle U red, RGB 150,0,0)
SU_GRAY   = "#6b6b6b"
INK       = "#1a1a1a"   # near-black: "actual"/historical lines & text
GRID      = "#cfcfcf"

# Qualitative palette for multi-model comparisons (maroon leads)
PALETTE = ["#960000", "#3a6ea5", "#c8861d", "#4a7c59", "#6a4a7c", "#6b6b6b"]

# Diverging colors for +/- residual & delta bars
POS = "#4a7c59"   # sage  (increase / positive)
NEG = "#c0504d"   # brick (decrease / negative)

# Per-model color + marker, so the SAME model looks the same in EVERY figure
MODEL_STYLE = {
    "Layer 1": dict(color=PALETTE[5], marker="o"),  # gray
    "AR":      dict(color=SU_MAROON,  marker="s"),  # maroon = the winner
    "ARX":     dict(color=PALETTE[1], marker="^"),  # slate
    "ARMA":    dict(color=PALETTE[2], marker="D"),  # ochre
    "ARIMA":   dict(color=PALETTE[3], marker="v"),  # sage
    "ARIMAX":  dict(color=PALETTE[4], marker="P"),  # plum
}
def mstyle(name):
    """Return {'color','marker'} for a model, robust to labels like 'Layer 2: AR(1)'."""
    n = name.upper()
    for key in ("ARIMAX", "ARIMA", "ARMA", "ARX"):
        if key in n:
            return dict(MODEL_STYLE[key])
    if "AR(" in n:                       # AR(1), AR(p)
        return dict(MODEL_STYLE["AR"])
    if "LAYER 1" in n or "TREND" in n:
        return dict(MODEL_STYLE["Layer 1"])
    return dict(color=SU_GRAY, marker="o")

# --- One look for every figure ---
mpl.rcParams.update({
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.family":       "serif",       # matches an academic paper
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   SU_MAROON,
    "axes.labelsize":    11,
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        GRID,
    "grid.alpha":        0.6,
    "grid.linewidth":    0.6,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   9,
    "lines.linewidth":   1.8,
    "lines.markersize":  4,
    "axes.prop_cycle":   mpl.cycler(color=PALETTE),
})

def style_axis(ax, title=None, xlabel=None, ylabel=None):
    if title  is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax

def suptitle(fig, text):
    fig.suptitle(text, fontsize=15, fontweight="bold", color=SU_MAROON, y=1.02)

GINI_PATH   = os.path.join(DATA_DIR, "us_gini.csv")
GDP_PATH    = os.path.join(DATA_DIR, "GDP.csv")
CPI_PATH    = os.path.join(DATA_DIR, "CPIAUCSL.csv")
UNRATE_PATH = os.path.join(DATA_DIR, "UNRATE.csv")

TRAIN_START = 1963
TEST_START  = 1994
TEST_END    = 2023

# ── Gini ──
gini = pd.read_csv(GINI_PATH)
gini.columns = gini.columns.str.strip()
gini = gini.sort_values("year").reset_index(drop=True)

# ── GDP (quarterly → annual → growth → delta) ──
tmp = pd.read_csv(GDP_PATH)
tmp.columns = tmp.columns.str.strip()
tmp["year"] = pd.to_datetime(tmp["observation_date"]).dt.year
gdp = (tmp.groupby("year")["GDP"].mean()
       .reset_index().rename(columns={"GDP": "v"}))
gdp["growth"] = gdp["v"].pct_change() * 100
gdp["d_gdp"]  = gdp["growth"].diff()
d_gdp_map = gdp.set_index("year")["d_gdp"].to_dict()

# ── CPI (monthly → annual → inflation → delta) ──
tmp = pd.read_csv(CPI_PATH)
tmp.columns = tmp.columns.str.strip()
tmp["year"] = pd.to_datetime(tmp["observation_date"]).dt.year
cpi = (tmp.groupby("year")["CPIAUCSL"].mean()
       .reset_index().rename(columns={"CPIAUCSL": "v"}))
cpi["inflation"] = cpi["v"].pct_change() * 100
cpi["d_cpi"]     = cpi["inflation"].diff()
d_cpi_map = cpi.set_index("year")["d_cpi"].to_dict()

# ── Unemployment (monthly → annual → delta) ──
tmp = pd.read_csv(UNRATE_PATH)
tmp.columns = tmp.columns.str.strip()
tmp["year"] = pd.to_datetime(tmp["observation_date"]).dt.year
ur = (tmp.groupby("year")["UNRATE"].mean()
      .reset_index().rename(columns={"UNRATE": "v"}))
ur["d_unrate"] = ur["v"].diff()
d_unrate_map = ur.set_index("year")["d_unrate"].to_dict()

print(f"Gini: {gini['year'].min()}-{gini['year'].max()}")
print(f"GDP delta: {gdp.dropna()['year'].min()}-{gdp['year'].max()}")
print(f"CPI delta: {cpi.dropna()['year'].min()}-{cpi['year'].max()}")
print(f"URATE delta: {ur.dropna()['year'].min()}-{ur['year'].max()}")

"""# Layer 1: Linear Trend (Rolling OLS)"""

# ══════════════ LAYER 1: Linear Trend (Rolling OLS) ══════════════
# Reuse the `gini` frame and TRAIN_START/TEST_START/TEST_END from the Data cell.
df = gini.copy().dropna()
df.columns = [c.strip().lower() for c in df.columns]
df = df.sort_values("year").reset_index(drop=True)
T0 = TRAIN_START

results = []
for t in range(TEST_START, TEST_END + 1):
    tr = df[(df["year"] >= TRAIN_START) & (df["year"] <= t - 1)]
    x = (tr["year"].values - T0).astype(float)
    y = tr["gini"].values.astype(float)
    X = np.column_stack([np.ones_like(x), x])
    b_hat, a_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_pred = b_hat + a_hat * (t - T0)
    y_true = df.loc[df["year"] == t, "gini"].values[0]
    results.append({"year": t, "actual": y_true,
                    "pred": y_pred, "resid": y_true - y_pred,
                    "a": a_hat, "b": b_hat})

fc1 = pd.DataFrame(results)
mae  = np.mean(np.abs(fc1["resid"]))
rmse = np.sqrt(np.mean(fc1["resid"]**2))
mape = np.mean(np.abs(fc1["resid"] / fc1["actual"])) * 100

print(f"Layer 1: Linear Trend | MAE={mae:.4f}, "
      f"RMSE={rmse:.4f}, MAPE={mape:.2f}%")

"""# Layer2: AR Model"""

# ══════════════ LAYER 2: AR(p) with AIC/BIC Selection ══════════════
train_vals = df[(df["year"] >= TRAIN_START)
                & (df["year"] <= TEST_START - 1)]["gini"].values

# ── Step 1: Test all p, show AIC and BIC ──
best_p_aic, best_aic_val = 1, np.inf
best_p_bic, best_bic_val = 1, np.inf

print(f"{'p':<5} {'AIC':>12} {'BIC':>12}")
print("-" * 32)
for p in range(1, 11):
    try:
        mod = AutoReg(train_vals, lags=p, trend='c').fit()
        tag = ""
        if mod.aic < best_aic_val:
            best_aic_val = mod.aic
            best_p_aic = p
        if mod.bic < best_bic_val:
            best_bic_val = mod.bic
            best_p_bic = p
        print(f"p={p:<4} {mod.aic:>12.2f} {mod.bic:>12.2f}")
    except: pass

print(f"\n>>> AIC selects p = {best_p_aic} (AIC = {best_aic_val:.2f})")
print(f">>> BIC selects p = {best_p_bic} (BIC = {best_bic_val:.2f})")

# Use BIC (more conservative)
best_p = best_p_bic

# ── Step 2: Rolling one-step-ahead forecast ──
results2 = []
for t in range(TEST_START, TEST_END + 1):
    tr = df[(df["year"] >= TRAIN_START)
            & (df["year"] <= t - 1)]
    y = tr["gini"].values
    y_lag = np.column_stack([y[best_p - i - 1 : len(y) - i - 1]
                             for i in range(best_p)])
    y_curr = y[best_p:]
    X = np.column_stack([np.ones(len(y_curr)), y_lag])
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y_curr)

    x_new = np.array([1.0] + [y[-(i+1)] for i in range(best_p)])
    y_pred = x_new @ beta
    y_true = df.loc[df["year"] == t, "gini"].values[0]
    results2.append({"year": t, "actual": y_true,
                     "pred": y_pred, "resid": y_true - y_pred})

fc2 = pd.DataFrame(results2)
mae2  = np.mean(np.abs(fc2["resid"]))
rmse2 = np.sqrt(np.mean(fc2["resid"]**2))
mape2 = np.mean(np.abs(fc2["resid"] / fc2["actual"])) * 100

print(f"\nLayer 2: AR({best_p}) | MAE={mae2:.4f}, "
      f"RMSE={rmse2:.4f}, MAPE={mape2:.2f}%")
print(f"Coefficients: c={beta[0]:.6f}, "
      + ", ".join([f"phi{i+1}={beta[i+1]:.6f}"
                   for i in range(best_p)]))
fc2.to_csv(os.path.join(RESULTS_DIR, "layer2_results.csv"), index=False)
fc2[["year", "actual", "pred", "resid"]]

"""

```
# This is formatted as code
```

# Layer3: ARX, ARMA, ARIMA, ARIMAX"""

# ══════════════ LAYER 3: Shared Utilities ══════════════
import csv, math, itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
from scipy.optimize import minimize

@dataclass
class Series:
    years: List[int]
    values: np.ndarray
    def slice(self, start, end):
        mask = np.array([(start <= y <= end) for y in self.years])
        return Series([y for y,m in zip(self.years,mask) if m], self.values[mask])

@dataclass
class ModelResult:
    name: str; order: str; bic: float; mae: float; rmse: float; mape: float
    years: List[int]; actuals: List[float]; preds: List[float]
    coefficients: Dict[str,float] = field(default_factory=dict)

def read_gini(path):
    years, vals = [], []
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            y=int(float(row["year"].strip())); v=float(row["gini"].strip())
            if not(math.isnan(v)or math.isinf(v)): years.append(y); vals.append(v)
    pairs=sorted(zip(years,vals))
    return Series([p[0] for p in pairs], np.array([p[1] for p in pairs]))

def _read_fred(path, val_col):
    rows=[]
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            parts=row["observation_date"].strip().split("-")
            yr,mo=int(parts[0]),int(parts[1]); v=float(row[val_col].strip())
            if not(math.isnan(v)or math.isinf(v)): rows.append((yr,mo,v))
    rows.sort(); return rows

def _annualise(rows):
    accum=defaultdict(list)
    for yr,mo,v in rows: accum[yr].append(v)
    return {yr:sum(vs)/len(vs) for yr,vs in accum.items()}

def load_macro(gdp_path, cpi_path, unrate_path, year_start, year_end):
    gdp_ann=_annualise(_read_fred(gdp_path,"GDP"))
    cpi_ann=_annualise(_read_fred(cpi_path,"CPIAUCSL"))
    unr_ann=_annualise(_read_fred(unrate_path,"UNRATE"))
    result={"dGDP":{},"dCPI":{},"dUNRATE":{}}
    for yr in range(year_start, year_end+1):
        prev=yr-1
        if yr in gdp_ann and prev in gdp_ann: result["dGDP"][yr]=gdp_ann[yr]-gdp_ann[prev]
        if yr in cpi_ann and prev in cpi_ann: result["dCPI"][yr]=cpi_ann[yr]-cpi_ann[prev]
        if yr in unr_ann and prev in unr_ann: result["dUNRATE"][yr]=unr_ann[yr]-unr_ann[prev]
    return result

def build_exog_matrix(macro, years, selected):
    if not selected: return None
    X=np.full((len(years),len(selected)),np.nan)
    for j,(var,lag) in enumerate(selected):
        for i,yr in enumerate(years):
            target=yr-lag
            if target in macro[var]: X[i,j]=macro[var][target]
    return None if np.any(np.isnan(X)) else X

def fit_arima(y_raw, p, d, q, exog=None, max_p_in_grid=0):
    y=y_raw.copy()
    for _ in range(d): y=np.diff(y)
    exog_d=None
    if exog is not None:
        exog_d=exog.copy()
        for _ in range(d): exog_d=exog_d[1:]
    trim=max(max_p_in_grid,max(p,q)) if max(p,q)>0 else max_p_in_grid
    n_exog=0 if exog_d is None else exog_d.shape[1]
    n_params=1+p+q+n_exog
    def css_obj(params):
        n=len(y); c_v=params[0]; phi=params[1:1+p]
        th=params[1+p:1+p+q]; beta=params[1+p+q:1+p+q+n_exog]
        resid=np.zeros(n); start=max(p,q) if max(p,q)>0 else 0
        for t in range(start,n):
            pred=c_v
            for i in range(p): pred+=phi[i]*y[t-1-i]
            for j in range(q): pred+=th[j]*resid[t-1-j]
            if exog_d is not None: pred+=np.dot(beta,exog_d[t])
            resid[t]=y[t]-pred
        r=resid[trim:] if trim>0 else resid; n_eff=len(r)
        if n_eff==0: return 1e18
        s2=np.sum(r**2)/n_eff
        if s2<=1e-30: s2=1e-30
        return 0.5*n_eff*(np.log(2*np.pi*s2)+1.0)
    x0=np.zeros(n_params)
    if len(y)>0: x0[0]=np.mean(y)
    res=minimize(css_obj,x0,method="Nelder-Mead",options={"maxiter":20000,"xatol":1e-10,"fatol":1e-10})
    params=res.x
    n=len(y); resid=np.zeros(n); start=max(p,q) if max(p,q)>0 else 0
    for t in range(start,n):
        pred=params[0]
        for i in range(p): pred+=params[1+i]*y[t-1-i]
        for j in range(q): pred+=params[1+p+j]*resid[t-1-j]
        if exog_d is not None: pred+=np.dot(params[1+p+q:1+p+q+n_exog],exog_d[t])
        resid[t]=y[t]-pred
    r=resid[trim:] if trim>0 else resid; n_eff=len(r)
    sigma2=np.sum(r**2)/n_eff
    if sigma2<=1e-30: sigma2=1e-30
    logL=-0.5*n_eff*(np.log(2*np.pi*sigma2)+1.0)
    bic=(n_params+1)*np.log(n_eff)-2*logL
    return params, sigma2, bic, n_eff

def forecast_arima_one_step(params, y_raw, p, d, q, exog_history=None, exog_new=None):
    y=y_raw.copy()
    for _ in range(d): y=np.diff(y)
    exog_d=None
    if exog_history is not None:
        exog_d=exog_history.copy()
        for _ in range(d): exog_d=exog_d[1:]
    c=params[0]; phi=params[1:1+p]; theta=params[1+p:1+p+q]
    n_exog=0 if exog_d is None else exog_d.shape[1]
    beta=params[1+p+q:1+p+q+n_exog]
    n=len(y); resid=np.zeros(n); start=max(p,q) if max(p,q)>0 else 0
    for t in range(start,n):
        pred=c
        for i in range(p): pred+=phi[i]*y[t-1-i]
        for j in range(q): pred+=theta[j]*resid[t-1-j]
        if exog_d is not None: pred+=np.dot(beta,exog_d[t])
        resid[t]=y[t]-pred
    pred=c
    for i in range(p):
        if n-1-i>=0: pred+=phi[i]*y[n-1-i]
    for j in range(q):
        if n-1-j>=0: pred+=theta[j]*resid[n-1-j]
    if exog_new is not None: pred+=np.dot(beta,exog_new)
    forecast=pred
    if d>=1: forecast+=y_raw[-1]
    return float(forecast)

def ols_fit_np(X, y):
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    resid=y-X@beta; sigma2=float(np.sum(resid**2)/len(y))
    return beta, sigma2

def ols_bic(sigma2, n, k):
    if sigma2<=1e-30: sigma2=1e-30
    logL=-0.5*n*(np.log(2*np.pi*sigma2)+1.0)
    return (k+1)*np.log(n)-2*logL

def calc_mae(yt,yp): return float(np.mean(np.abs(yt-yp)))
def calc_rmse(yt,yp): return float(np.sqrt(np.mean((yt-yp)**2)))
def calc_mape(yt,yp):
    d=np.where(np.abs(yt)>1e-12,np.abs(yt),1e-12)
    return float(100.0*np.mean(np.abs((yt-yp)/d)))

def print_result(r):
    print(f"\n{'='*60}\n  {r.name}\n{'='*60}")
    print(f"  Order: {r.order}\n  BIC: {r.bic:.4f}")
    print(f"  MAE={r.mae:.6f}, RMSE={r.rmse:.6f}, MAPE={r.mape:.2f}%")
    if r.coefficients:
        print("  Coefficients:", ", ".join(f"{k}={v:.6g}" for k,v in r.coefficients.items()))

print("Utilities loaded.")

#ARX
# ══════════════ LAYER 3A: ARX ══════════════
gini_s = read_gini(GINI_PATH)
macro = load_macro(GDP_PATH, CPI_PATH, UNRATE_PATH, 1960, 2023)
train_s = gini_s.slice(1963, 1993)
AR_P = 1; LAGS = [1,2,3]; VARIABLES = ["dGDP","dCPI","dUNRATE"]

def build_ar_exog(values, years, macro, selected, p):
    rows_X, rows_y, yr_labels = [], [], []
    for t in range(p, len(values)):
        row = [1.0]
        for lag in range(1, p+1): row.append(values[t-lag])
        yr = years[t]; valid = True
        for var, lag_val in selected:
            target = yr - lag_val
            if target in macro[var]: row.append(macro[var][target])
            else: valid = False; break
        if not valid: continue
        rows_X.append(row); rows_y.append(values[t]); yr_labels.append(yr)
    return np.array(rows_X), np.array(rows_y), yr_labels

# BIC subset selection
all_pairs = [(v,l) for v in VARIABLES for l in LAGS]
combos = [[]]
for r in range(1, len(all_pairs)+1):
    for c in itertools.combinations(all_pairs, r): combos.append(list(c))

earliest = 1963 + AR_P + max(LAGS)
best_bic_arx, best_combo = float("inf"), []
for combo in combos:
    X,y,yrl = build_ar_exog(train_s.values, train_s.years, macro, combo, AR_P)
    if len(y)==0: continue
    mask = [yr>=earliest for yr in yrl]; Xt,yt = X[mask], y[mask]
    if len(yt)<=Xt.shape[1]: continue
    beta,sigma2 = ols_fit_np(Xt, yt)
    bic_val = ols_bic(sigma2, len(yt), Xt.shape[1])
    if bic_val < best_bic_arx: best_bic_arx=bic_val; best_combo=combo

# Rolling forecast
hist_yrs, hist_vals = list(train_s.years), list(train_s.values)
preds_arx = []
for yr in range(1994, 2024):
    X,y,_ = build_ar_exog(np.array(hist_vals), hist_yrs, macro, best_combo, AR_P)
    beta,_ = ols_fit_np(X, y)
    row = [1.0]
    for lag in range(1, AR_P+1): row.append(hist_vals[-lag])
    for var, lag_val in best_combo: row.append(macro[var].get(yr-lag_val, 0.0))
    preds_arx.append(float(np.dot(beta, row)))
    hist_yrs.append(yr); hist_vals.append(gini_s.values[gini_s.years.index(yr)])

actuals_arx = [gini_s.values[gini_s.years.index(yr)] for yr in range(1994,2024)]
a_arx, p_arx = np.array(actuals_arx), np.array(preds_arx)
order_arx = f"AR({AR_P}) + " + " + ".join(f"{v}_lag{l}" for v,l in best_combo) if best_combo else f"AR({AR_P}) only"

arx = ModelResult("Layer 3A: ARX", order_arx, best_bic_arx,
    calc_mae(a_arx,p_arx), calc_rmse(a_arx,p_arx), calc_mape(a_arx,p_arx),
    list(range(1994,2024)), actuals_arx, preds_arx)
print_result(arx)

# ══════════════ LAYER 3C: ARIMA(p,1,q) ══════════════
train_arima = gini_s.slice(1963, 1993)
D = 1; MAX_PQ = 3
best_bic_arima, best_p_i, best_q_i = float("inf"), 0, 0

print(f"{'(p,1,q)':<10} {'BIC':>12}")
print("-"*25)
for p in range(0,4):
    for q in range(0,4):
        if p==0 and q==0: continue
        try:
            _,_,bic_val,_ = fit_arima(train_arima.values, p, D, q, max_p_in_grid=MAX_PQ)
            print(f"({p},1,{q}){'':<4} {bic_val:>12.2f}")
            if bic_val < best_bic_arima: best_bic_arima=bic_val; best_p_i=p; best_q_i=q
        except: pass
print(f"\n>>> BIC selects ARIMA({best_p_i},1,{best_q_i})")

# Rolling forecast
hist_arima = list(train_arima.values)
preds_arima, last_params = [], None
for yr in range(1994, 2024):
    y_h = np.array(hist_arima)
    try:
        params,_,_,_ = fit_arima(y_h, best_p_i, D, best_q_i)
        last_params = params
        pred = forecast_arima_one_step(params, y_h, best_p_i, D, best_q_i)
    except: pred = hist_arima[-1]
    preds_arima.append(pred)
    hist_arima.append(gini_s.values[gini_s.years.index(yr)])

actuals_arima = [gini_s.values[gini_s.years.index(yr)] for yr in range(1994,2024)]
a_ai, p_ai = np.array(actuals_arima), np.array(preds_arima)

coefs_arima = {}
if last_params is not None:
    coefs_arima["c"] = last_params[0]
    for i in range(best_p_i): coefs_arima[f"phi{i+1}"] = last_params[1+i]
    for j in range(best_q_i): coefs_arima[f"theta{j+1}"] = last_params[1+best_p_i+j]

arima = ModelResult("Layer 3C: ARIMA", f"ARIMA({best_p_i},1,{best_q_i})", best_bic_arima,
    calc_mae(a_ai,p_ai), calc_rmse(a_ai,p_ai), calc_mape(a_ai,p_ai),
    list(range(1994,2024)), actuals_arima, preds_arima, coefs_arima)
print_result(arima)

# ══════════════ LAYER 3D: ARIMAX(p,1,q) + exogenous ══════════════
# Phase 1: reuse ARIMA order from Cell 3
bp, bq = best_p_i, best_q_i
print(f"Using ARIMA({bp},1,{bq}) from Phase 1")

# Phase 2: exog subset selection
all_pairs_ax = [(v,l) for v in ["dGDP","dCPI","dUNRATE"] for l in [1,2,3]]
train_ax = gini_s.slice(1963, 1993)
best_bic_ax = best_bic_arima  # baseline = no exog
best_combo_ax = []

for r in range(1, min(5, len(all_pairs_ax)+1)):
    for combo in itertools.combinations(all_pairs_ax, r):
        combo_list = list(combo)
        exog = build_exog_matrix(macro, train_ax.years, combo_list)
        if exog is None: continue
        try:
            _,_,bic_val,_ = fit_arima(train_ax.values, bp, 1, bq, exog=exog, max_p_in_grid=MAX_PQ)
            if bic_val < best_bic_ax: best_bic_ax=bic_val; best_combo_ax=combo_list
        except: continue

if best_combo_ax:
    print(f"Selected exog: {', '.join(f'{v}_lag{l}' for v,l in best_combo_ax)}")
else:
    print("No exogenous variables selected (ARIMA alone is best)")

# Rolling forecast
hist_yrs_ax, hist_vals_ax = list(train_ax.years), list(train_ax.values)
preds_ax, last_p_ax = [], None

for yr in range(1994, 2024):
    y_h = np.array(hist_vals_ax)
    exog_h, exog_new = None, None
    if best_combo_ax:
        exog_h = build_exog_matrix(macro, hist_yrs_ax, best_combo_ax)
        exog_new = np.array([macro[v].get(yr-l, 0.0) for v,l in best_combo_ax])
    try:
        params,_,_,_ = fit_arima(y_h, bp, 1, bq, exog=exog_h)
        last_p_ax = params
        pred = forecast_arima_one_step(params, y_h, bp, 1, bq,
                                       exog_history=exog_h, exog_new=exog_new)
    except: pred = hist_vals_ax[-1]
    preds_ax.append(pred)
    hist_yrs_ax.append(yr); hist_vals_ax.append(gini_s.values[gini_s.years.index(yr)])

actuals_ax = [gini_s.values[gini_s.years.index(yr)] for yr in range(1994,2024)]
a_ax, p_ax_arr = np.array(actuals_ax), np.array(preds_ax)
order_ax = f"ARIMA({bp},1,{bq})"
if best_combo_ax: order_ax += " + " + " + ".join(f"{v}_lag{l}" for v,l in best_combo_ax)

coefs_ax = {}
if last_p_ax is not None:
    coefs_ax["c"]=last_p_ax[0]
    for i in range(bp): coefs_ax[f"phi{i+1}"]=last_p_ax[1+i]
    for j in range(bq): coefs_ax[f"theta{j+1}"]=last_p_ax[1+bp+j]
    for idx,(v,l) in enumerate(best_combo_ax): coefs_ax[f"{v}_lag{l}"]=last_p_ax[1+bp+bq+idx]

arimax = ModelResult("Layer 3D: ARIMAX", order_ax, best_bic_ax,
    calc_mae(a_ax,p_ax_arr), calc_rmse(a_ax,p_ax_arr), calc_mape(a_ax,p_ax_arr),
    list(range(1994,2024)), actuals_ax, preds_ax, coefs_ax)
print_result(arimax)

# ══════════════ FULL COMPARISON ══════════════
all_models = [
    ("Layer 1: Linear Trend", mae, rmse, mape),
    ("Layer 2: AR(1)", mae2, rmse2, mape2),
    (f"3A: ARX", arx.mae, arx.rmse, arx.mape),
    (f"3C: {arima.order}", arima.mae, arima.rmse, arima.mape),
    (f"3D: ARIMAX", arimax.mae, arimax.rmse, arimax.mape),
]
print("="*70)
print(f"  {'Model':<35} {'MAE':>10} {'RMSE':>10} {'MAPE%':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
for n,ma,rm,mp in all_models:
    print(f"  {n:<35} {ma:>10.6f} {rm:>10.6f} {mp:>8.2f}")
print("="*70)

names=[m[0] for m in all_models]; mapes=[m[3] for m in all_models]
fig,ax=plt.subplots(figsize=(10,4))
colors=["#2ecc71" if m==min(mapes) else "#3498db" for m in mapes]
bars=ax.barh(names,mapes,color=colors)
for b,v in zip(bars,mapes): ax.text(b.get_width()+0.02,b.get_y()+b.get_height()/2,f"{v:.2f}%",va="center")
ax.set_xlabel("MAPE (%)"); ax.invert_yaxis()
ax.set_title("All Models: Out-of-Sample MAPE"); plt.tight_layout(); plt.show()

from IPython.display import display, Markdown

g_1963 = df.loc[df["year"]==1963, "gini"].values[0]
g_1964 = df.loc[df["year"]==1964, "gini"].values[0]
g_1993 = df.loc[df["year"]==1993, "gini"].values[0]
g_1994 = df.loc[df["year"]==1994, "gini"].values[0]

display(Markdown(rf"""
# Worked Examples: How Each Model Makes a Prediction

All examples predict **1994** using data up to **1993**.

---

## Example 1: AR(1) — "Use last year's Gini"

$$G_t = c + \phi_1 G_{{t-1}} + \varepsilon_t$$

From training (1963–1993), OLS gives $c = -0.0180$, $\phi_1 = 1.0524$.

$$\hat{{G}}_{{1994}} = -0.0180 + 1.0524 \times \underbrace{{{g_1993:.4f}}}_{{G_{{1993}}}} = 0.4068$$

Actual $G_{{1994}} = {g_1994:.4f}$, so the **error** is:

$$\varepsilon_{{1994}} = {g_1994:.4f} - 0.4068 = {g_1994 - 0.4068:.4f}$$

The AR(1) **ignores** this error — it won't use it next year.

---

## Example 2: ARMA(1,1) — "Use last year's Gini AND last year's error"

$$G_t = c + \phi_1 G_{{t-1}} + \theta_1 \varepsilon_{{t-1}} + \varepsilon_t$$

The new term $\theta_1 \varepsilon_{{t-1}}$ is the **MA correction**. Suppose $\theta_1 = -0.3$:

**Predicting 1995**, knowing we over-predicted 1994 by 0.0069:

$$\hat{{G}}_{{1995}} = c + \phi_1 \times G_{{1994}} + \theta_1 \times \varepsilon_{{1994}}$$
$$= c + \phi_1 \times {g_1994:.4f} + (-0.3) \times (-0.0069)$$
$$= \text{{AR prediction}} + 0.0021$$

The MA term **pushes the forecast up** because we under-predicted last year.
Without MA, the model keeps making the same mistake. With MA, it **learns from errors**.

---

## Example 3: ARIMA(1,1,1) — "Model the CHANGES, not the levels"

First, **difference** the series:
$$\Delta G_t = G_t - G_{{t-1}}$$
$$\Delta G_{{1993}} = G_{{1993}} - G_{{1992}} = {g_1993:.4f} - {df.loc[df["year"]==1992,"gini"].values[0]:.4f} = {g_1993 - df.loc[df["year"]==1992,"gini"].values[0]:.4f}$$

Then model the changes:
$$\Delta G_t = \delta + \phi_1 \Delta G_{{t-1}} + \theta_1 \varepsilon_{{t-1}} + \varepsilon_t$$

Predict the **change** for 1994:
$$\widehat{{\Delta G}}_{{1994}} = \delta + \phi_1 \times \Delta G_{{1993}} + \theta_1 \times \varepsilon_{{1993}}$$

**Convert back** to a level:
$$\hat{{G}}_{{1994}} = G_{{1993}} + \widehat{{\Delta G}}_{{1994}} = {g_1993:.4f} + \widehat{{\Delta G}}_{{1994}}$$

**Why this helps:** Instead of estimating $\phi \approx 0.97$ on a trending series
(where OLS is unreliable), we estimate $\phi$ on the *changes* (which fluctuate around zero).

---

## Example 4: ARIMAX — "ARIMA + macro shocks"

$$\Delta G_t = \delta + \phi_1 \Delta G_{{t-1}} + \beta_1 \Delta\text{{GDP}}_{{t-1}} + \theta_1 \varepsilon_{{t-1}} + \varepsilon_t$$

Same as ARIMA, but now:
- If GDP **growth accelerated** last year ($\Delta\text{{GDP}}_{{t-1}} > 0$), and $\beta_1 > 0$, the model predicts inequality **rises**
- If GDP **growth slowed** ($\Delta\text{{GDP}}_{{t-1}} < 0$), it predicts inequality **falls**

This captures **external shocks** (2008 crisis, COVID) that pure time-series models miss.

---

## Visual Summary

| Model | Uses past Gini? | Uses past errors? | Differences? | Uses macro data? |
|-------|:-:|:-:|:-:|:-:|
| AR(1) | ✓ | ✗ | ✗ | ✗ |
| ARMA | ✓ | ✓ | ✗ | ✗ |
| ARIMA | ✓ | ✓ | ✓ | ✗ |
| ARIMAX | ✓ | ✓ | ✓ | ✓ |

Each row adds one capability to fix a limitation of the row above.
"""))

# ══════════════ AIC/BIC HEATMAP ══════════════ every layer needs this....
train_sel = gini_s.slice(1963, 1993)

bic_grid = np.full((4,4), np.nan)
aic_grid = np.full((4,4), np.nan)
for p in range(0,4):
    for q in range(0,4):
        if p==0 and q==0: continue
        try:
            _,_,bic_val,_ = fit_arima(train_sel.values, p, 1, q, max_p_in_grid=3)
            # Compute AIC too
            params,sigma2,_,n_eff = fit_arima(train_sel.values, p, 1, q, max_p_in_grid=3)
            k = 1+p+q+1
            logL = -0.5*n_eff*(np.log(2*np.pi*sigma2)+1.0)
            aic_val = 2*k - 2*logL
            bic_grid[p,q] = bic_val
            aic_grid[p,q] = aic_val
        except: pass
# (Plotting handled by the house-style heatmap cell further down.)

# ══════════════ PUBLICATION-STYLE METRICS TABLE ══════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")

headers = ["Model", "Equation", "MAE", "RMSE", "MAPE (%)"]
rows_data = [
    ["Layer 1", r"$G_t = a + b(t - t_0)$",
     f"{mae:.4f}", f"{rmse:.4f}", f"{mape:.2f}"],
    ["Layer 2: AR(1)", r"$G_t = c + \phi G_{t-1}$",
     f"{mae2:.4f}", f"{rmse2:.4f}", f"{mape2:.2f}"],
    ["3A: ARX", r"$G_t = c + \phi G_{t-1} + \beta \Delta X_{t-1}$",
     f"{arx.mae:.4f}", f"{arx.rmse:.4f}", f"{arx.mape:.2f}"],
    [f"3C: ARIMA({best_p_i},1,{best_q_i})",
     r"$\Delta G_t = \delta + \phi \Delta G_{t-1} + \theta \varepsilon_{t-1}$",
     f"{arima.mae:.4f}", f"{arima.rmse:.4f}", f"{arima.mape:.2f}"],
    ["3D: ARIMAX", r"$\Delta G_t = \delta + \phi \Delta G + \beta \Delta X + \theta \varepsilon$",
     f"{arimax.mae:.4f}", f"{arimax.rmse:.4f}", f"{arimax.mape:.2f}"],
]

# Find best MAPE row
mape_vals = [float(r[4]) for r in rows_data]
best_idx = mape_vals.index(min(mape_vals))

table = ax.table(cellText=rows_data, colLabels=headers,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(10)
table.scale(1, 1.8)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best row
for j in range(len(headers)):
    table[best_idx + 1, j].set_facecolor("#d5f5e3")

# Alternate row colors
for i in range(1, len(rows_data) + 1):
    if i != best_idx + 1:
        color = "#f8f9fa" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

ax.set_title("Table: Out-of-Sample Forecast Comparison (1994–2023)",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(figpath("layer3_metrics_table.png"), dpi=150, bbox_inches="tight")
plt.show()

# ══════════════ FORECAST: 2024-2030 ══════════════
# Use all data (1963-2023) to forecast forward
all_gini = gini_s.values.copy()
forecast_years = list(range(2024, 2031))

# AR(1) forecast
hist_ar = list(all_gini)
preds_ar_future = []
for yr in forecast_years:
    h = np.array(hist_ar)
    X_h = np.column_stack([np.ones(len(h)-1), h[:-1]])
    b, _ = ols_fit_np(X_h, h[1:])
    pred = float(b[0] + b[1]*h[-1])
    preds_ar_future.append(pred)
    hist_ar.append(pred)  # feed prediction back

# ARIMA forecast
hist_arima_f = list(all_gini)
preds_arima_future = []
for yr in forecast_years:
    y_h = np.array(hist_arima_f)
    try:
        params,_,_,_ = fit_arima(y_h, best_p_i, 1, best_q_i)
        pred = forecast_arima_one_step(params, y_h, best_p_i, 1, best_q_i)
    except:
        pred = hist_arima_f[-1]
    preds_arima_future.append(pred)
    hist_arima_f.append(pred)
# (Plotting handled by the house-style forecast cell further down.)

"""# Presentation Outline
#(Kevin)
#1: Intro: summary of what we have done for this project, motivation
#2: Outline
#3: terminnology: gini index, OLS, MAPE
#4: Layer1: Model, graph to explain model, how we approach the math
#5: terminnology: AIC, BIC, error, AR MODEL
#6: Layer2: Model, graph to explain model, how we approach the math
---------------------------------------------------------------------
#(Bruce)
#7: Layer3: Model, graph to explain model, how we approach the matth
#8: terminology(X) -> ARX -> terminology(MA) -> ARMA -> terminology(I) -> ARIMA -> ARIMAX
#9: Compare
#10: Predict
#11: Future work, weakness
#12: QA, thank you
"""

# ═══ Raw Gini series + train/test split (house style) ═══
TEST_START = 1994
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.axvspan(df["year"].min(), TEST_START, color="#f4f4f4")
ax.axvspan(TEST_START, df["year"].max(), color="#fbeeee")
ax.plot(df["year"], df["gini"], "o-", color=SU_MAROON, markersize=3.5)
ax.axvline(TEST_START, color=SU_GRAY, ls="--", lw=1.2)
ax.text(1978, ax.get_ylim()[1]*0.985, "Training", color=SU_GRAY,
        style="italic", ha="center", va="top")
ax.text(2009, ax.get_ylim()[1]*0.985, "Test", color=SU_MAROON,
        style="italic", ha="center", va="top")
style_axis(ax, title="U.S. Gini Coefficient (1963–2023)",
           ylabel="Gini Index", xlabel="Year")
fig.tight_layout(); fig.savefig(figpath("fig_gini_series.png")); plt.show()

# ═══ LAYER 1: Visualization (house style) ═══
fig, axes = plt.subplots(2, 1, figsize=(11, 7),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

ax = axes[0]
ax.plot(df["year"], df["gini"], "o-", color=INK, label="Actual Gini")
ax.plot(fc1["year"], fc1["pred"], "s--", color=SU_MAROON, label="Linear-trend forecast")
ax.axvline(TEST_START, color=SU_GRAY, ls=":", lw=1.2, label=f"Train/test split ({TEST_START})")
style_axis(ax, title=f"Layer 1: Rolling OLS Linear Trend  |  MAPE = {mape:.2f}%",
           ylabel="Gini Index")
ax.legend()

ax2 = axes[1]
ax2.bar(fc1["year"], fc1["resid"],
        color=[NEG if r < 0 else POS for r in fc1["resid"]], alpha=0.85)
ax2.axhline(0, color=INK, lw=0.6)
style_axis(ax2, ylabel="Residual", xlabel="Year")

fig.tight_layout(); fig.savefig(figpath("fig_layer1.png")); plt.show()

# ═══ LAYER 2: Visualization (house style) ═══
fig, axes = plt.subplots(2, 1, figsize=(11, 7),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

ax = axes[0]
ax.plot(fc2["year"], fc2["actual"], "o-", color=INK, label="Actual Gini")
ax.plot(fc1["year"], fc1["pred"], "s--", color=PALETTE[5], alpha=0.7,
        label=f"Layer 1  (MAPE={mape:.2f}%)")
ax.plot(fc2["year"], fc2["pred"], "^--", color=SU_MAROON,
        label=f"Layer 2: AR({best_p})  (MAPE={mape2:.2f}%)")
style_axis(ax, title="Layer 1 vs. Layer 2", ylabel="Gini Index")
ax.legend()

ax2 = axes[1]
w = 0.35
ax2.bar(fc1["year"] - w/2, fc1["resid"], w, color=PALETTE[5], alpha=0.6, label="Layer 1")
ax2.bar(fc2["year"] + w/2, fc2["resid"], w, color=SU_MAROON, alpha=0.7, label="Layer 2")
ax2.axhline(0, color=INK, lw=0.6)
style_axis(ax2, ylabel="Residual", xlabel="Year")
ax2.legend()

fig.tight_layout(); fig.savefig(figpath("fig_layer2.png")); plt.show()

# ── keep your existing print summary block (MAE/RMSE/MAPE) below, unchanged ──

# ═══ FULL COMPARISON bar (house style) ═══
# ── keep your print table (all_models loop) above, unchanged ──
names = [m[0] for m in all_models]; mapes = [m[3] for m in all_models]
fig, ax = plt.subplots(figsize=(10, 4))
colors = [SU_MAROON if m == min(mapes) else PALETTE[5] for m in mapes]
bars = ax.barh(names, mapes, color=colors)
for b, v in zip(bars, mapes):
    ax.text(b.get_width() + 0.02, b.get_y() + b.get_height()/2, f"{v:.2f}%",
            va="center", fontsize=9)
ax.invert_yaxis()
style_axis(ax, title="All Models: Out-of-Sample MAPE", xlabel="MAPE (%)")
fig.tight_layout(); fig.savefig(figpath("fig_mape_bars.png")); plt.show()

# ═══ LAYER 3: Forecast comparison (house style) ═══
fig, axes = plt.subplots(3, 1, figsize=(12, 12),
    gridspec_kw={"height_ratios": [3, 2, 2]})
yrs = list(range(1994, 2024))

ax = axes[0]
ax.plot(yrs, actuals_arx, "o-", color=INK, label="Actual Gini")
ax.plot(fc1["year"], fc1["pred"], "--", **mstyle("Layer 1"), alpha=0.5, label=f"L1 Trend ({mape:.2f}%)")
ax.plot(fc2["year"], fc2["pred"], "--", **mstyle("AR(1)"),   label=f"L2 AR(1) ({mape2:.2f}%)")
ax.plot(yrs, preds_arx,   "--", **mstyle("ARX"),    label=f"3A ARX ({arx.mape:.2f}%)")
ax.plot(yrs, preds_arima, "--", **mstyle("ARIMA"),  label=f"3C {arima.order} ({arima.mape:.2f}%)")
ax.plot(yrs, preds_ax,    "--", **mstyle("ARIMAX"), label=f"3D ARIMAX ({arimax.mape:.2f}%)")
style_axis(ax, title="Layer 1 → Layer 3: Forecast Comparison", ylabel="Gini Index")
ax.legend(loc="upper left", ncol=2)

ax2 = axes[1]; w = 0.2
r_ar1 = fc2["resid"].values
r_arx = a_arx - p_arx; r_arima = a_ai - p_ai; r_arimax = a_ax - p_ax_arr
for off, (nm, r) in zip([-1.5, -0.5, 0.5, 1.5],
        [("AR(1)", r_ar1), ("ARX", r_arx), ("ARIMA", r_arima), ("ARIMAX", r_arimax)]):
    ax2.bar(np.array(yrs) + off*w, r, w, color=mstyle(nm)["color"], alpha=0.75, label=nm)
ax2.axhline(0, color=INK, lw=0.6)
style_axis(ax2, ylabel="Residual"); ax2.legend(ncol=4)

ax3 = axes[2]
for nm, r in [("AR(1)", r_ar1), ("ARX", r_arx), ("ARIMA", r_arima), ("ARIMAX", r_arimax)]:
    ax3.plot(yrs, np.cumsum(np.abs(r)), "-", **mstyle(nm), label=nm)
style_axis(ax3, title="Cumulative Absolute Error Over Time",
           ylabel="Cumulative |Error|", xlabel="Year"); ax3.legend()

fig.tight_layout(); fig.savefig(figpath("fig_layer3_compare.png")); plt.show()

# ── keep your bic_grid / aic_grid compute loop above, unchanged ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, grid, title in [(axes[0], aic_grid, "AIC"), (axes[1], bic_grid, "BIC")]:
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"q={i}" for i in range(4)])
    ax.set_yticklabels([f"p={i}" for i in range(4)])
    ax.set_title(f"{title} for ARIMA(p,1,q)")
    for i in range(4):
        for j in range(4):
            val = grid[i, j]
            if not np.isnan(val):
                best = val == np.nanmin(grid)
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if best else "normal",
                        color="white" if best else INK)
    plt.colorbar(im, ax=ax, shrink=0.8)
suptitle(fig, r"ARIMA$(p,1,q)$ Order Selection — lower = better")
fig.tight_layout(); fig.savefig(figpath("fig_aicbic_heatmap.png")); plt.show()
print(f"BIC selects: ARIMA({best_p_i},1,{best_q_i})")

# ═══ INDIVIDUAL MODEL PANELS (house style) ═══
yrs=list(range(1994,2024))
models_plot=[
    ("Layer 2: AR(1)",            fc2["pred"].values, fc2["resid"].values),
    ("3A: ARX",                   p_arx,    a_arx - p_arx),
    (f"3C: ARIMA({best_p_i},1,{best_q_i})", p_ai, a_ai - p_ai),
    ("3D: ARIMAX",                p_ax_arr, a_ax - p_ax_arr),
]
fig,axes=plt.subplots(4,2,figsize=(14,14))
for row,(name,preds_arr,resids) in enumerate(models_plot):
    mape_val=np.mean(np.abs(resids/actuals_arx))*100
    ax=axes[row,0]
    ax.plot(yrs,actuals_arx,"o-",color=INK,label="Actual")
    ax.plot(yrs,preds_arr,"s--",color=SU_MAROON,label="Predicted")
    ax.fill_between(yrs,actuals_arx,preds_arr,alpha=0.12,color=SU_MAROON)
    style_axis(ax,f"{name}  |  MAPE = {mape_val:.2f}%",ylabel="Gini"); ax.legend(fontsize=8)
    ax2=axes[row,1]
    ax2.hist(resids,bins=12,color=PALETTE[1],edgecolor="white",alpha=0.85,density=True)
    ax2.axvline(0,color=INK,ls="--",lw=1.2)
    ax2.axvline(np.mean(resids),color=SU_MAROON,ls="--",lw=1.5,label=f"Mean={np.mean(resids):.4f}")
    style_axis(ax2,f"Residual Distribution (std={np.std(resids):.4f})",xlabel="Residual"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(figpath("fig_layer3_panels.png")); plt.show()
print("saved fig_layer3_panels.png")

# ═══ ROLLING MAPE + ABSOLUTE ERROR (house style) ═══
yrs_arr = np.array(yrs); window = 5
fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
for nm, resids in [("AR(1)", fc2["resid"].values), ("ARX", a_arx - p_arx),
                   (f"ARIMA({best_p_i},1,{best_q_i})", a_ai - p_ai),
                   ("ARIMAX", a_ax - p_ax_arr)]:
    pct_err = np.abs(resids / actuals_arx) * 100
    rolling = pd.Series(pct_err).rolling(window, min_periods=1).mean().values
    ax.plot(yrs, rolling, "-", **mstyle(nm), label=nm)
style_axis(ax, title="Where Do Models Struggle?",
           ylabel=f"{window}-Year Rolling MAPE (%)")
ax.legend()

ax2 = axes[1]; w = 0.2
abs_errs = {"AR(1)": np.abs(fc2["resid"].values), "ARX": np.abs(a_arx - p_arx),
            "ARIMA": np.abs(a_ai - p_ai), "ARIMAX": np.abs(a_ax - p_ax_arr)}
x = np.arange(len(yrs))
for i, (nm, errs) in enumerate(abs_errs.items()):
    ax2.bar(x + i*w - 1.5*w, errs, w, color=mstyle(nm)["color"], alpha=0.8, label=nm)
ax2.set_xticks(x[::2]); ax2.set_xticklabels(yrs_arr[::2], rotation=45)
style_axis(ax2, ylabel="|Error|", xlabel="Year"); ax2.legend(fontsize=7, ncol=4)

fig.tight_layout(); fig.savefig(figpath("fig_abs_error.png")); plt.show()

# ═══ Why we difference (house style) ═══
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
yrs_ex  = list(range(1990, 2000))
gini_ex = [df.loc[df["year"] == y, "gini"].values[0] for y in yrs_ex]
delta_ex = [gini_ex[i] - gini_ex[i-1] for i in range(1, len(gini_ex))]

ax = axes[0]
ax.plot(yrs_ex, gini_ex, "o-", color=SU_MAROON, markersize=5)
style_axis(ax, title=r"Raw $y_t$ (trending)", ylabel="Gini")

ax = axes[1]
for i in range(1, len(yrs_ex)):
    ax.annotate("", xy=(yrs_ex[i], gini_ex[i]), xytext=(yrs_ex[i], gini_ex[i-1]),
                arrowprops=dict(arrowstyle="->", color=SU_MAROON, lw=1.8))
ax.plot(yrs_ex, gini_ex, "o", color=INK, markersize=5)
style_axis(ax, title=r"Compute $\Delta y_t = y_t - y_{t-1}$", ylabel="Gini")

ax = axes[2]
ax.bar(yrs_ex[1:], delta_ex, color=[POS if d > 0 else NEG for d in delta_ex], alpha=0.85)
ax.axhline(0, color=INK, lw=1)
style_axis(ax, title=r"$\Delta y_t$ (stationary)", ylabel=r"$\Delta$ Gini")

suptitle(fig, "Why We Difference: Making the Series Stationary")
fig.tight_layout(); fig.savefig(figpath("fig_differencing.png")); plt.show()

# ── keep your AR(1) + ARIMA forecast compute loops above, unchanged ──
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["year"], df["gini"], "o-", color=INK, markersize=3, label="Historical Gini")
ax.plot(forecast_years, preds_ar_future,    "s--", **mstyle("AR(1)"),  markersize=6, label="AR(1) forecast")
ax.plot(forecast_years, preds_arima_future, "^--", **mstyle("ARIMA"), markersize=6,
        label=f"ARIMA({best_p_i},1,{best_q_i}) forecast")
ax.plot([2023, 2024], [all_gini[-1], preds_ar_future[0]],    "--", color=SU_MAROON, alpha=0.5)
ax.plot([2023, 2024], [all_gini[-1], preds_arima_future[0]], "--", color=PALETTE[3], alpha=0.5)

ax.axvspan(2023.5, 2030.5, color=SU_MAROON, alpha=0.06)
ax.text(2027, ax.get_ylim()[1]*0.99, "FORECAST", ha="center", va="top",
        fontsize=12, color=SU_GRAY, fontweight="bold")
# label only the final point of each line (avoids the overlapping text)
ax.annotate(f"{preds_ar_future[-1]:.3f}",    (forecast_years[-1], preds_ar_future[-1]),
            textcoords="offset points", xytext=(6, 6),  fontsize=8, color=SU_MAROON)
ax.annotate(f"{preds_arima_future[-1]:.3f}", (forecast_years[-1], preds_arima_future[-1]),
            textcoords="offset points", xytext=(6, -12), fontsize=8, color=PALETTE[3])

style_axis(ax, title="U.S. Gini Coefficient: Historical + Forecast (2024–2030)",
           xlabel="Year", ylabel="Gini Index")
ax.legend()
fig.tight_layout(); fig.savefig(figpath("fig_forecast_2030.png")); plt.show()

# ── keep your print table below, unchanged ──