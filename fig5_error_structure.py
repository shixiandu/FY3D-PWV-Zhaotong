# -*- coding: utf-8 -*-
"""
fig5_error_structure.py
Fig. 5: Error by elevation, season, month, and location.
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
out_png   = r"C:\Users\Administrator\Desktop\Fig5_error_structure.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig5_error_structure.pdf"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
elev_bins   = [0,500,1000,1500,2000,2500,3000,5000]
elev_labels = ['<500','500–1000','1000–1500','1500–2000',
               '2000–2500','2500–3000','>3000']

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
test_df["month"] = test_df["date"].dt.month
test_df["season"] = test_df["month"].map(
    {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
     6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0,0]
test_df["elev_band"] = pd.cut(test_df["dem"], bins=elev_bins,
                              labels=elev_labels, right=False)
elev_rmse = test_df.groupby("elev_band", observed=True)["err"].apply(
    lambda x: np.sqrt(np.mean(x**2)))
cnt = test_df.groupby("elev_band", observed=True).size()
ax.bar(range(len(elev_rmse)), elev_rmse.values, color='steelblue', edgecolor='white')
for i, (v, c) in enumerate(zip(elev_rmse.values, cnt.values)):
    ax.text(i, v+0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
ax.set_xticks(range(len(elev_rmse)))
ax.set_xticklabels(elev_rmse.index, rotation=30, ha='right')
ax.set_xlabel('Elevation (m)'); ax.set_ylabel('RMSE (kg/m²)')
ax.set_title('(a) RMSE by Elevation Band')
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

ax = axes[0,1]
season_order = ['Spring','Summer','Autumn','Winter']
colors = ['#66BB6A','#EF5350','#FFA726','#42A5F5']
season_rmse = test_df.groupby("season")["err"].apply(
    lambda x: np.sqrt(np.mean(x**2))).reindex(season_order)
ax.bar(range(len(season_rmse)), season_rmse.values, color=colors, edgecolor='white')
for i, v in enumerate(season_rmse.values):
    ax.text(i, v+0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
ax.set_xticks(range(len(season_rmse))); ax.set_xticklabels(season_order)
ax.set_xlabel('Season'); ax.set_ylabel('RMSE (kg/m²)')
ax.set_title('(b) RMSE by Season')
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

ax = axes[1,0]
data = [test_df[test_df["month"]==m]["err"].values for m in range(1,13)]
ax.boxplot(data, positions=range(1,13), widths=0.6,
           patch_artist=True, showfliers=False,
           medianprops=dict(color='red', lw=1.8),
           boxprops=dict(facecolor='steelblue', alpha=0.6, edgecolor='navy'),
           whiskerprops=dict(color='navy'), capprops=dict(color='navy'))
ax.axhline(0, color='black', ls='--', lw=1)
ax.set_xticks(range(1,13))
ax.set_xlabel('Month'); ax.set_ylabel('Residual (kg/m²)')
ax.set_title('(c) Monthly Residual Distribution')
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

ax = axes[1,1]
plot_df = test_df.sample(n=min(30000, len(test_df)), random_state=42)
sc = ax.scatter(plot_df["lon"], plot_df["lat"], c=plot_df["err"],
                cmap='RdBu_r', vmin=-5, vmax=5, s=2, alpha=0.6, rasterized=True)
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Residual (kg/m²)')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_title('(d) Spatial Distribution of Residual')
ax.grid(alpha=0.3, linestyle='--')
ax.set_aspect('equal', adjustable='box')

fig.suptitle("Fig. 5. Retrieval error by elevation, season, month, "
             "and geographic location.",
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")