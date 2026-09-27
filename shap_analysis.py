# -*- coding: utf-8 -*-
"""
shap_analysis.py
SHAP feature contribution analysis for the RF PWV model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_dir   = r"C:\Users\Administrator\Desktop"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
feature_labels = {"bt_ch1":"10.65V","bt_ch2":"10.65H","bt_ch3":"18.7V","bt_ch4":"18.7H",
                  "bt_ch5":"23.8V","bt_ch6":"23.8H","bt_ch7":"36.5V","bt_ch8":"36.5H",
                  "bt_ch9":"89.0V","bt_ch10":"89.0H",
                  "lon":"lon","lat":"lat","dem":"elevation","month":"month"}
n_sample = 2000

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)

rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=42, n_jobs=-1)
rf.fit(X, y)

rng = np.random.RandomState(42)
idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
X_sample = X[idx]
df_sample = df.iloc[idx].copy()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
print(f"SHAP shape: {shap_values.shape}")

X_df = pd.DataFrame(X_sample, columns=[feature_labels[c] for c in feature_cols])

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_df, show=False, max_display=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=600, bbox_inches='tight')
plt.close()

def plot_dependence(feat_name, fname):
    j = X_df.columns.get_loc(feat_name)
    x = X_sample[:, j]
    s = shap_values[:, j]
    if feat_name != "month":
        c = df_sample["month"].values; cbar_label = "month"
    else:
        c = df_sample["pwv"].values; cbar_label = "PWV"
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, s, c=c, cmap='Spectral_r', s=6, alpha=0.6)
    cb = plt.colorbar(sc); cb.set_label(cbar_label)
    plt.axhline(0, color='gray', ls='--', lw=1)
    plt.xlabel(feat_name); plt.ylabel("SHAP value (impact on PWV)")
    plt.title(f"SHAP dependence: {feat_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=600, bbox_inches='tight')
    plt.close()

plot_dependence("23.8H",     "shap_dependence_23.8H.png")
plot_dependence("month",     "shap_dependence_month.png")
plot_dependence("elevation", "shap_dependence_dem.png")

mean_abs = np.abs(shap_values).mean(axis=0)
rank = pd.DataFrame({"feature":[feature_labels[c] for c in feature_cols],
                     "mean_abs_SHAP": mean_abs}).sort_values("mean_abs_SHAP", ascending=False)
rank.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False, encoding="utf-8-sig")
print(rank.to_string(index=False))