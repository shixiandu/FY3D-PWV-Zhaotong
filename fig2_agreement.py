# -*- coding: utf-8 -*-
"""
fig2_agreement.py
Fig. 2: Retrieved vs ERA5 PWV + residual distribution (N = 273,805).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_png   = r"C:\Users\Administrator\Desktop\Fig2_agreement.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig2_agreement.pdf"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
random_state, test_ratio = 42, 0.30

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
df = df[(df["dem"] > 100) & (df["dem"] < 5000)].reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)

rng = np.random.RandomState(random_state)
idx = rng.permutation(len(df))
n_test = int(len(df) * test_ratio)
test_idx, train_idx = idx[:n_test], idx[n_test:]

rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=random_state, n_jobs=-1)
rf.fit(X[train_idx], y[train_idx])

y_pred = rf.predict(X[test_idx])
y_test = y[test_idx]

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
bias = np.mean(y_pred - y_test)
resid = y_pred - y_test

print(f"R²={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}  Bias={bias:+.3f}  N={len(y_test):,}")

coef = np.polyfit(y_test, y_pred, 1)
fit_x = np.linspace(y_test.min(), y_test.max(), 100)
fit_y = coef[0] * fit_x + coef[1]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

ax = axes[0]
ax.scatter(y_test, y_pred, s=3, alpha=0.25, color='steelblue', rasterized=True)
lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lim, lim, 'k--', lw=1.5, label='1:1 line')
ax.plot(fit_x, fit_y, 'r-', lw=1.8, label=f'Fit: y={coef[0]:.3f}x+{coef[1]:.2f}')
ax.set_xlabel('ERA5 PWV (kg/m²)'); ax.set_ylabel('Retrieved PWV (kg/m²)')
ax.set_title('(a) Retrieved vs ERA5 PWV')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.grid(alpha=0.3); ax.set_aspect('equal', adjustable='box')
textstr = (f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f} kg/m²\n"
           f"MAE = {mae:.2f} kg/m²\nN = {len(y_test):,}")
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='gray', alpha=0.9))
ax.legend(loc='lower right')

ax2 = axes[1]
ax2.hist(resid, bins=100, color='steelblue', alpha=0.75, edgecolor='none')
ax2.axvline(0, color='black', ls='--', lw=1.5, label='Zero line')
ax2.axvline(bias, color='red', ls='-', lw=1.8, label=f'Mean = {bias:+.3f}')
lo, hi = np.percentile(resid, [2.5, 97.5])
ax2.axvline(lo, color='gray', ls=':', lw=1.2, label=f'2.5% = {lo:.2f}')
ax2.axvline(hi, color='gray', ls=':', lw=1.2, label=f'97.5% = {hi:.2f}')
ax2.set_xlabel('Residual (Retrieved − ERA5) (kg/m²)')
ax2.set_ylabel('Frequency')
ax2.set_title('(b) Residual Distribution')
ax2.grid(alpha=0.3); ax2.legend(loc='upper right')

fig.suptitle("Fig. 2. Agreement between retrieved PWV and ERA5 PWV, "
             "and the corresponding residual distribution.",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")