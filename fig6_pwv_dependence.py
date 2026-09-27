# -*- coding: utf-8 -*-
"""
fig6_pwv_dependence.py
Fig. 6: Error vs PWV magnitude + annual mean comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_png   = r"C:\Users\Administrator\Desktop\Fig6_pwv_dependence.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig6_pwv_dependence.pdf"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
pwv_bins = [0,15,25,35,45,100]
pwv_labels = ['<15','15–25','25–35','35–45','>45']

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)

rng = np.random.RandomState(42)
idx = rng.permutation(len(df))
n_test = int(len(df) * 0.30)
test_idx = idx[:n_test]

rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=42, n_jobs=-1)
rf.fit(X[idx[n_test:]], y[idx[n_test:]])

test_df = df.iloc[test_idx].copy()
test_df["pred"] = rf.predict(X[test_idx])
test_df["err"] = test_df["pred"] - test_df[ref_col]
test_df["date"] = pd.to_datetime(test_df["date"].astype(str))
test_df["year"] = test_df["date"].dt.year

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

ax = axes[0]
test_df["band"] = pd.cut(test_df[ref_col], bins=pwv_bins, labels=pwv_labels, right=False)
g = test_df.groupby("band", observed=True)["err"]
rmse_vals = g.apply(lambda x: np.sqrt(np.mean(x**2)))
bias_vals = g.apply(lambda x: np.mean(x))
x_pos = np.arange(len(rmse_vals))
ax.bar(x_pos-0.2, rmse_vals.values, width=0.4, color='steelblue', label='RMSE')
ax.bar(x_pos+0.2, bias_vals.values, width=0.4, color='#EF5350', label='Bias')
for i, (r, b) in enumerate(zip(rmse_vals.values, bias_vals.values)):
    ax.text(i-0.2, r+0.10, f'{r:.2f}', ha='center', va='bottom', fontsize=11)
    off = 0.12 if b >= 0 else -0.40
    ax.text(i+0.2, b+off, f'{b:+.2f}', ha='center',
            va='bottom' if b>=0 else 'top', fontsize=11, color='darkred')
ax.axhline(0, color='black', ls='--', lw=1)
ax.set_xticks(x_pos); ax.set_xticklabels(rmse_vals.index)
ax.set_xlabel('PWV Range (kg/m²)'); ax.set_ylabel('Error (kg/m²)')
ax.set_title('(a) RMSE and Bias vs PWV')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

ax = axes[1]
p90 = test_df[ref_col].quantile(0.90); p10 = test_df[ref_col].quantile(0.10)
scen = {
    'High\nPWV':  test_df[test_df[ref_col] > p90],
    'Normal':     test_df[(test_df[ref_col] >= p10) & (test_df[ref_col] <= p90)],
    'Low\nPWV':   test_df[test_df[ref_col] < p10],
}
names = list(scen.keys())
rmse_s = [np.sqrt(np.mean(s["err"]**2)) for s in scen.values()]
bias_s = [np.mean(s["err"]) for s in scen.values()]
x_pos = np.arange(len(names))
ax.bar(x_pos-0.2, rmse_s, width=0.4, color=['#EF5350','#66BB6A','#42A5F5'], label='RMSE')
ax.bar(x_pos+0.2, bias_s, width=0.4, color='#B0BEC5', label='Bias')
for i, (r, b) in enumerate(zip(rmse_s, bias_s)):
    ax.text(i-0.2, r+0.15, f'{r:.2f}', ha='center', va='bottom', fontsize=11)
    off = 0.15 if b >= 0 else -0.45
    ax.text(i+0.2, b+off, f'{b:+.2f}', ha='center',
            va='bottom' if b>=0 else 'top', fontsize=11, color='darkred')
ax.axhline(0, color='black', ls='--', lw=1)
ax.set_xticks(x_pos); ax.set_xticklabels(names)
ax.set_xlabel('Scenario'); ax.set_ylabel('Error (kg/m²)')
ax.set_title('(b) Error in Extreme Scenarios')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

ax = axes[2]
ys = test_df.groupby("year").agg(era5=(ref_col,'mean'), rf=('pred','mean')).reset_index()
x_pos = np.arange(len(ys))
ax.bar(x_pos-0.2, ys["era5"], width=0.4, color='steelblue', label='ERA5 PWV')
ax.bar(x_pos+0.2, ys["rf"], width=0.4, color='#FFA726', label='Retrieved PWV')
for i, (a, b) in enumerate(zip(ys["era5"], ys["rf"])):
    ax.text(i-0.2, a+0.25, f'{a:.2f}', ha='center', va='bottom', fontsize=11)
    ax.text(i+0.2, b+0.25, f'{b:.2f}', ha='center', va='bottom', fontsize=11)
ax.set_xticks(x_pos); ax.set_xticklabels(ys["year"].astype(int))
ax.set_xlabel('Year'); ax.set_ylabel('Mean PWV (kg/m²)')
ax.set_title('(c) Annual Mean PWV')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

fig.suptitle("Fig. 6. Dependence of retrieval error on PWV magnitude "
             "and comparison of annual mean PWV.",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.subplots_adjust(wspace=0.35, bottom=0.18)
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")