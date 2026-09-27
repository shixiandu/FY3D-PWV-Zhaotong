# -*- coding: utf-8 -*-
"""
case_study_2023.py
July 2023 heavy-rainfall case over Zhaotong: PWV timeline + rainfall event.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})

folder    = r"C:\Users\Administrator\Desktop\2345"
data_file = os.path.join(folder, "training_data_3years_full.csv")
out_png   = r"C:\Users\Administrator\Desktop\case_study_2023.png"
out_csv   = r"C:\Users\Administrator\Desktop\case_study_2023.csv"

case_start = "2023-07-01"
case_end   = "2023-07-06"
PRECIP_MAX_MM = 123.9
PRECIP_MAX_HOURLY = 60.7

df = pd.read_csv(data_file)
df["date"] = pd.to_datetime(df["date"].astype(str))
df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

mask = (df["datetime"] >= case_start) & (df["datetime"] <= case_end)
case_df = df[mask].copy()

if len(case_df) == 0:
    raise ValueError("时间窗口内无数据")

ts = case_df.groupby("datetime").agg({"pwv":"mean","lat":"count"}) \
    .rename(columns={"lat":"n_pixels"}).reset_index()
ts.to_csv(out_csv, index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(ts["datetime"], ts["pwv"], "o-", color="steelblue",
        markersize=4, linewidth=1.2, label="Regional mean PWV")

storm_start = pd.Timestamp("2023-07-03 20:00")
storm_end   = pd.Timestamp("2023-07-04 07:00")
ax.axvspan(storm_start, storm_end, color="red", alpha=0.15, label="Heavy rain period")
ax.axvline(storm_start, color="red", ls="--", lw=1)
ax.axvline(storm_end, color="red", ls="--", lw=1)

peak_idx = ts["pwv"].idxmax()
peak_time = ts.loc[peak_idx, "datetime"]
peak_pwv  = ts.loc[peak_idx, "pwv"]
ax.annotate(f"Peak PWV = {peak_pwv:.1f} kg/m²",
            xy=(peak_time, peak_pwv), xytext=(10, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=10)

ax.set_ylabel("PWV (kg/m²)")
ax.set_title("PWV Evolution during the 3-4 July 2023 Zhaotong Rainstorm",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

ax2 = axes[1]
ax2.axvspan(storm_start, storm_end, color="red", alpha=0.15)
ax2.text(storm_start + (storm_end-storm_start)/2, 0.5,
         f"Max 24h rain: {PRECIP_MAX_MM} mm\nMax hourly: {PRECIP_MAX_HOURLY} mm/h",
         ha="center", va="center", fontsize=10, color="darkred")
ax2.set_ylim(0, 1); ax2.set_yticks([])
ax2.set_xlabel("Date")
ax2.set_title("Rainfall Event Timeline (3-4 July 2023)", fontsize=11)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")