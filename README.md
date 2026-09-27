# FY3D-PWV-Zhaotong

Terrain-aware retrieval of precipitable water vapor over Zhaotong,
southwestern China, using FY-3D MWRI and a random forest.

## Data sources
- FY-3D MWRI Level-1 brightness temperature: http://data.nsmc.org.cn
- ERA5 hourly total column water vapor: https://cds.climate.copernicus.eu
- Copernicus GLO-30 DEM: https://dataspace.copernicus.eu

## Environment
- Python 3.10
- scikit-learn
- xgboost
- lightgbm
- shap
- rasterio
- pandas, numpy, matplotlib

## Scripts
- run_four_models.py            : Train and compare RF, XGBoost, LightGBM, MLR
- fig2_agreement.py             : Fig. 2 (retrieved vs ERA5 PWV + residuals)
- fig4_feature_importance.py    : Fig. 4 (RF predictor importance)
- shap_analysis.py              : SHAP feature contribution analysis
- fig5_error_structure.py       : Fig. 5 (error by elevation/season/month)
- fig6_pwv_dependence.py        : Fig. 6 (error vs PWV, annual means)
- fig7_mean_pwv_map.py          : Fig. 7 (three-year mean PWV map)
- fig8_seasonal_pwv.py          : Fig. 8 (seasonal PWV fields)
- validation_by_orbit.py        : Cross-orbit validation (Table 6)
- case_study_2023.py            : July 2023 heavy-rainfall case study

## Contact
Shixian Du, Yongshan Meteorological Bureau, Yunnan, China
Email: 2670966672@qq.com