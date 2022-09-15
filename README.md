# Walking predictions

machine learning that predicts walking and non-walking segments within timeseries data

## Installation
```bash
conda env create -f cfb_env.yaml
```

## Usage

```python
python cfb_ml.py
```
### Current prediction accuracies
```bash
features to drop (coef > 0.75): ['position_z', 'angular_acceleration_y', 'orientation_y']

# Base case accuracies (no tuning) - classification
kneighbor acc:  0.9960367905481193
logisticRegression acc:  0.9179939679453626
gradientBoostingClassifier acc:  0.9900545876018844
mlpClassifier acc:  0.9994516313965951
randomForestClassifier acc:  0.9995513347790324

check the amount of Walks (1) and Non-Walks (0) training  data:  labels
0         106087
1          54388
dtype: int64
check the amount of Walks (1) and Non-Walks (0) test  data:  labels
0         26386
1         13733
```
### Correlation Matrix
![](https://github.com/bszek213/Walking-Classification/blob/main/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
