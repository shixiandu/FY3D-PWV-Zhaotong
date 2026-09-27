# -*- coding: utf-8 -*-
"""
fig4_feature_importance.py
Fig. 4: RF predictor importance (highlight 23.8H, month, elevation).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":13,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_png   = r"C:\Users\Administrator\Desktop\Fig4_feature_importance.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig4_feature_importance.pdf"

ref_col = "pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
labels = {"bt_ch1":"10.65V","bt_ch2":"10.65H","bt_ch3":"18.7V","bt_ch4":"18.7H",
          "bt_ch5":"23.8V","bt_ch6":"23.8H","bt_ch7":"36.5V","bt_ch8":"36.5H",
          "bt_ch9":"89.0V","bt_ch10":"89.0H",
          "lon":"lon","lat":"lat","dem":"elevation","month":"month"}
highlight = {"23.8H":"#D32F2F","month":"#F57C00","elevation":"#1976D2"}
default_color = "#B0BEC5"

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)

rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=42, n_jobs=-1)
rf.fit(X, y)

imp = rf.feature_importances_ * 100
rank = pd.DataFrame({"feature":[labels[c] for c in feature_cols],
                     "importance": imp}).sort_values("importance").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = [highlight.get(f, default_color) for f in rank["feature"]]
bars = ax.barh(rank["feature"], rank["importance"],
               color=colors, edgecolor='white', height=0.75)
for bar, val in zip(bars, rank["importance"]):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{val:.2f}%', va='center', ha='left', fontsize=11)

ax.set_xlabel('Importance (%)'); ax.set_ylabel('Predictor')
ax.set_title('Random-Forest Predictor Importance')
ax.set_xlim(0, max(rank["importance"])*1.18)
ax.grid(axis='x', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)

legend_elements = [Patch(facecolor=highlight["23.8H"], label='23.8 GHz H'),
                   Patch(facecolor=highlight["month"], label='month'),
                   Patch(facecolor=highlight["elevation"], label='elevation'),
                   Patch(facecolor=default_color, label='Others')]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
print(rank.sort_values("importance", ascending=False).to_string(index=False))