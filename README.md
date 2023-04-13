# Customer Personality Analysis and Churn

- This is a quickly whipped up, well structured project using a **Customer Personality** dataset.
- I have conducted a quite in-depth feature extraction (as outlined in `feature_extraction.ipynb`).
- Models were tinkered with in `train.ipynb`.
- Currently implemented models:
  1. Logistic Regression
  2. Random Forest Classifiers

## Required Libraries

1. `sklearn`
2. `matplotlib`
3. `scipy`
4. `pandas`
5. `numpy`
6. `seaborn`

## Model metrics:

### Logistic Regression

1. Confusion Matrix - <br /> <br />
   ![logistic-confusion](models/logistic/logistic-ratio%3D0.8/confusion.png)

2. ROC Curve - <br /> <br />
   ![logistic-roc](models/logistic/logistic-ratio%3D0.8/roc-curve.png)

3. Test Accuracy - $0.775$

### Random Forest

1. Confusion Matrix - <br /> <br />
   ![forest-confusion](models/random_forest/random_forest-ratio%3D0.8/confusion.png)

2. ROC Curve - <br /> <br />
   ![forest-confusion](models/random_forest/random_forest-ratio%3D0.8/roc-curve.png)

3. Test Accuracy - $0.99$
