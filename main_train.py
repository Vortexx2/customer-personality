from typing import List, Optional

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as ss
import seaborn as sns
from sklearn import model_selection, linear_model, metrics, preprocessing
import sklearn
# IMPORTS ABOVE


def load_and_process_data(path: str):
  df = pd.read_csv('data/processed/extracted.csv')
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

  X = df.drop(columns='Accepted')
  Y = df['Accepted']

  return X, Y


def get_accuracy(preds: np.ndarray, y: np.ndarray):
  num_correct = np.equal(preds, y, dtype=np.int16).mean()

  return num_correct


def train(X_train: np.ndarray, y_train: np.ndarray, model: str, train_ratio: float = 0.8):
  if model == 'logistic':
    clf = linear_model.LogisticRegression(random_state=42)

  else:
    raise KeyError("Invalid model named used")

  clf = clf.fit(X_train, y_train)


def main(path: str, model: str, train_ratio: float = 0.8):
  assert 0 < train_ratio < 1, "Enter valid train ratio"

  X, y = load_and_process_data(path)
  X, y = X.to_numpy(), y.to_numpy()

  scaler = preprocessing.StandardScaler()
  X = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=42,
                                                                      shuffle=True, stratify=y)

  clf = train(X, y, 'logistic', train_ratio=0.8)
  train_preds = clf.predict(X_train)
  test_preds = clf.predict(X_test)

  train_accuracy, test_accuracy = get_accuracy(train_preds, y_train), get_accuracy(test_preds, y_test)
  print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

  
