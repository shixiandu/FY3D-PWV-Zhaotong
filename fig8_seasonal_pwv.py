# -*- coding: utf-8 -*-
"""
fig8_seasonal_pwv.py
Fig. 8: Seasonal mean PWV fields 2022-2024.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
dem_file  = r"C:\Users\Administrator\WorkBuddy\2026-09-07-07-08-35\zhaotong_dem\zhaotong_city_dem_30m.tif"
out_png   = r"C:\Users\Administrator\Desktop\Fig8_seasonal_pwv.png"
out_pdf   = r"C:\Users\Administrator\Desktop\Fig8_seasonal_pwv.pdf"

ref_col = "pwv"; rf_col = "rf_pwv"
feature_cols = ["bt_ch1","bt_ch2","bt_ch3","bt_ch4","bt_ch5",
                "bt_ch6","bt_ch7","bt_ch8","bt_ch9","bt_ch10",
                "lon","lat","dem","month"]
lon_min, lon_max, lat_min, lat_max = 102.8, 105.3, 26.9, 28.6
grid_step = 0.1
season_map = {"Spring (MAM)":[3,4,5],"Summer (JJA)":[6,7,8],
              "Autumn (SON)":[9,10,11],"Winter (DJF)":[12,1,2]}

df = pd.read_csv(data_file).dropna(subset=feature_cols+[ref_col]).reset_index(drop=True)
X = df[feature_cols].values.astype(np.float32)
y = df[ref_col].values.astype(np.float32)
rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           max_features=1.0, random_state=42, n_jobs=-1).fit(X, y)
df[rf_col] = rf.predict(X)

df["date"] = pd.to_datetime(df["date"].astype(str))
df["month"] = df["date"].dt.month
df["lon_g"] = (df["lon"] / grid_step).round() * grid_step
df["lat_g"] = (df["lat"] / grid_step).round() * grid_step

lons = np.sort(df["lon_g"].unique()); lats = np.sort(df["lat_g"].unique())
li = {v:i for i,v in enumerate(lons)}; la = {v:i for i,v in enumerate(lats)}

def make_grid(s):
    g = s.groupby(["lon_g","lat_g"])[rf_col].mean().reset_index()
    Z = np.full((len(lats), len(lons)), np.nan)
    for _, r in g.iterrows():
        Z[la[r["lat_g"]], li[r["lon_g"]]] = r[rf_col]
    return Z

dem = None
if os.path.exists(dem_file):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(dem_file) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        dem = src.read(1, window=window)

vmin = df[rf_col].quantile(0.02); vmax = df[rf_col].quantile(0.98)
cmap = plt.cm.Spectral_r

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
im = None
for i, (season, months) in enumerate(season_map.items()):
    ax = axes[i]
    sub = df[df["month"].isin(months)]
    if dem is not None:
        cmap_g = LinearSegmentedColormap.from_list("g", ["#F5F5F5","#BDBDBD","#757575"])
        ax.imshow(dem, extent=[lon_min,lon_max,lat_min,lat_max],
                  origin='upper', cmap=cmap_g, alpha=0.7, aspect='auto', zorder=1)
    Z = make_grid(sub)
    im = ax.pcolormesh(lons, lats, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', alpha=0.85, zorder=2)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'({chr(97+i)}) {season}')
    ax.grid(alpha=0.3, linestyle='--', zorder=3)
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.03, 0.05, f"Mean = {sub[rf_col].mean():.2f} kg/m²",
            transform=ax.transAxes, fontsize=11, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9), zorder=4)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
cbar.set_label('PWV (kg/m²)')

fig.suptitle("Fig. 8. Seasonal mean PWV fields for 2022–2024.",
             fontsize=15, fontweight='bold', y=1.00)
plt.subplots_adjust(top=0.95, wspace=0.28, hspace=0.30)
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")