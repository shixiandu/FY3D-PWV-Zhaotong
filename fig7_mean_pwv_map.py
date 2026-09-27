# -*- coding: utf-8 -*-
"""
fig7_mean_pwv_map.py
Fig. 7: Three-year mean retrieved PWV across Zhaotong.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
dem_file  = r"C:\Users\Administrator\WorkBuddy\2026-09-07-07-08-35\zhaotong_dem\zhaotong_city_dem_30m.tif"
out_png   = r"C:\Users\Administrator\Desktop\Fig7_mean_pwv_map.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig7_mean_pwv_map.pdf"

ref_col = "pwv"
rf_col = "rf_pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
lon_min, lon_max, lat_min, lat_max = 102.8, 105.3, 26.9, 28.6
grid_step = 0.1

cities = {"Zhaoyang":(103.72,27.34),"Zhenxiong":(104.87,27.44),
          "Yiliang":(104.05,27.63),"Weixin":(105.05,27.84),
          "Shuifu":(104.42,28.63),"Qiaojia":(102.92,26.91),
          "Yanjin":(104.23,28.11),"Daguan":(103.89,27.75),
          "Yongshan":(103.63,28.18),"Suijiang":(103.97,28.60)}

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)
rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=42, n_jobs=-1).fit(X, y)
df[rf_col] = rf.predict(X)

df["lon_g"] = (df["lon"] / grid_step).round() * grid_step
df["lat_g"] = (df["lat"] / grid_step).round() * grid_step
g = df.groupby(["lon_g","lat_g"])[rf_col].mean().reset_index()

lons = np.sort(g["lon_g"].unique()); lats = np.sort(g["lat_g"].unique())
Z = np.full((len(lats), len(lons)), np.nan)
li = {v:i for i,v in enumerate(lons)}; la = {v:i for i,v in enumerate(lats)}
for _, r in g.iterrows():
    Z[la[r["lat_g"]], li[r["lon_g"]]] = r[rf_col]

dem = None
if os.path.exists(dem_file):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(dem_file) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        dem = src.read(1, window=window)

fig, ax = plt.subplots(figsize=(12, 9))
if dem is not None:
    cmap_g = LinearSegmentedColormap.from_list("g", ["#F5F5F5","#BDBDBD","#757575"])
    ax.imshow(dem, extent=[lon_min,lon_max,lat_min,lat_max],
              origin='upper', cmap=cmap_g, alpha=0.7, aspect='auto', zorder=1)

im = ax.pcolormesh(lons, lats, Z, cmap=plt.cm.Spectral_r,
                   shading='auto', alpha=0.85, zorder=2)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('PWV (kg/m²)')

for name, (lon, lat) in cities.items():
    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
        ax.plot(lon, lat, 'o', color='black', markersize=5,
                markeredgecolor='white', markeredgewidth=1, zorder=3)
        t = ax.text(lon+0.03, lat+0.03, name, fontsize=10,
                    color='black', fontweight='bold', zorder=3)
        t.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.grid(alpha=0.3, linestyle='--', zorder=4)
ax.set_aspect('equal', adjustable='box')

fig.suptitle("Fig. 7. Three-year mean retrieved PWV across Zhaotong.",
             fontsize=15, fontweight='bold', y=1.02)
plt.subplots_adjust(top=0.94)
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")