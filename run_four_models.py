# -*- coding: utf-8 -*-
"""
run_four_models.py
Train and compare RF / XGBoost / LightGBM / MLR
Same 7:3 split (seed 42), standardized features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_png   = r"C:\Users\Administrator\Desktop\Fig3_model_comparison.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig3_model_comparison.pdf"
out_csv   = r"C:\Users\Administrator\Desktop\Table4b_four_models.csv"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
random_state, test_ratio = 42, 0.30

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
assert df["dem"].max() > 1000, "dem 列异常"

X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)

rng = np.random.RandomState(random_state)
idx = rng.permutation(len(df))
n_test = int(len(df) * test_ratio)
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

models = {
    "RF": RandomForestRegressor(n_estimators=100, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        max_features=1.0, random_state=random_state, n_jobs=-1),
    "MLR": LinearRegression(),
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, tree_method="hist")
if HAS_LGB:
    models["LightGBM"] = LGBMRegressor(n_estimators=500, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbose=-1)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "pred": pred,
        "r2":   r2_score(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "mae":  mean_absolute_error(y_test, pred),
        "bias": np.mean(pred - y_test),
    }
    print(f"{name:<10} R²={results[name]['r2']:.4f}  RMSE={results[name]['rmse']:.3f}")

order = [k for k in ["MLR","RF","XGBoost","LightGBM"] if k in results]
table = pd.DataFrame([{
    "Model": n, "R²": round(results[n]["r2"],4),
    "RMSE": round(results[n]["rmse"],3), "MAE": round(results[n]["mae"],3),
    "Bias": round(results[n]["bias"],3), "N_test": len(y_test)
} for n in order])
table.to_csv(out_csv, index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
axes = axes.flatten()
lim = [min(y_test.min(), 0), max(y_test.max(), 75)]
for i, name in enumerate(order):
    ax = axes[i]
    pred = results[name]["pred"]
    ax.scatter(y_test, pred, s=3, alpha=0.25, color='steelblue', rasterized=True)
    ax.plot(lim, lim, 'k--', lw=1.5, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('ERA5 PWV (kg/m²)'); ax.set_ylabel('Retrieved PWV (kg/m²)')
    ax.set_title(f'({chr(97+i)}) {name}')
    ax.grid(alpha=0.3); ax.set_aspect('equal', adjustable='box')
    ax.text(0.05, 0.95,
            f"$R^2$ = {results[name]['r2']:.4f}\nRMSE = {results[name]['rmse']:.2f} kg/m²",
            transform=ax.transAxes, fontsize=13, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right')
for j in range(len(order), len(axes)):
    axes[j].axis('off')

fig.suptitle("Fig. 3. Comparison of the four retrieval models on the held-out test subset.",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")