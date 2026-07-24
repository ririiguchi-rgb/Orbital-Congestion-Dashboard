# Validation Summary

The master notebook was executed from top to bottom against the committed real CelesTrak SATCAT snapshot.

## Data

- Raw SATCAT rows loaded: 70,006
- Current Earth-centred LEO rows retained: 28,489
- PAY rows: 17,295
- DEB rows: 10,134
- R/B rows: 1,011
- UNK rows: 49

## Unsupervised workflow

- Final features: PERIOD, INCLINATION, MEAN_ALTITUDE, ORBIT_SPREAD, ALTITUDE_RATIO
- Tested K-Means configurations: k = 2 through 8
- Selected cluster count: k = 5
- Best sampled silhouette score: 0.518
- PCA variance retained in two displayed components: 83.83%
- Nearest neighbours per object: 5
- Congestion score: percentile-based, 0 to 100

## Exploratory Random Forest

- Accuracy: 95.71%
- Balanced accuracy: 95.67%
- Macro F1: 95.42%
- PAY precision: 97.32%
- PAY recall: 95.84%
- DEB precision: 93.08%
- DEB recall: 95.50%

## Technical checks

- Notebook code cells executed: 28
- Notebook execution errors: 0
- Streamlit data contract: passed
- Dashboard-ready rows: 28,489
- `app.py` Python syntax compilation: passed

These results supersede all earlier 28-record demonstration metrics.
