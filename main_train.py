from typing import List, Optional, Literal

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as ss
import seaborn as sns
from sklearn import model_selection, linear_model, metrics, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn
# IMPORTS ABOVE


def load_and_process_data(path: str):
  df = pd.read_csv('data/processed/extracted.csv')
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

  X = df.drop(columns='Accepted')
  Y = df['Accepted']

  return X, Y


def get_accuracy(preds: np.ndarray, y: np.ndarray):
  num_correct = np.equal(preds, y).mean()

  return num_correct


def train(X_train: np.ndarray, y_train: np.ndarray, model: str, train_ratio: float = 0.8):
  if model == 'logistic':
    clf = linear_model.LogisticRegression(random_state=42)

  else:
    raise KeyError("Invalid model named used")

  clf = clf.fit(X_train, y_train)

  return clf


def save_confusion(y_test: np.ndarray, test_preds: np.ndarray, file_path: str):
  """
    Creates and saves a confusion matrix in a folder indicated by save_path under name `confusion.png`.
  """
  fig, ax = plt.subplots()
  confusion_matrix = metrics.confusion_matrix(y_test, test_preds)
  sns.heatmap(confusion_matrix, fmt='3d', annot=True, ax=ax)

  fig.savefig(file_path)
  return confusion_matrix


def save_roc_curve(test_probs: np.ndarray, y_test: np.ndarray, file_path: str):
  """
    Creates a ROC curve and saves it to `file_path`.
  """
  # generate a no skill prediction (majority class)
  ns_probs = [0 for _ in range(len(y_test))]

  # keep probabilities for the positive outcome only
  test_probs = test_probs[:, 1]
  # calculate scores
  ns_auc = roc_auc_score(y_test, ns_probs)
  lr_auc = roc_auc_score(y_test, test_probs)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Logistic: ROC AUC=%.3f' % (lr_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
  lr_fpr, lr_tpr, _ = roc_curve(y_test, test_probs)

  fig, ax = plt.subplots()
  # plot the roc curve for the model
  ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
  ax.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
  # axis labels
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  # show the legend
  ax.legend()

  fig.savefig(file_path)


def main(csv_path: str, model: Literal['logistic'], save_path: str, train_ratio: float = 0.8):
  """
    csv_path: path to csv with extracted features
    model: One of ['logistic']
    save_path: Folder where model related files will be saved
  """
  assert 0 < train_ratio < 1, "Enter valid train ratio"
  assert model in ['logistic'], f"Enter valid `model`. Invalid `model` {model}"

  X, y = load_and_process_data(csv_path)
  X, y = X.to_numpy(), y.to_numpy()

  scaler = preprocessing.StandardScaler()
  X = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=42,
                                                                      shuffle=True, stratify=y)

  print(
      f"No. training examples: {len(X_train)}, No. testing examples: {len(X_test)}")

  clf = train(X, y, model, train_ratio=0.8)
  train_preds = clf.predict(X_train)
  test_preds = clf.predict(X_test)

  train_accuracy, test_accuracy = get_accuracy(
      train_preds, y_train), get_accuracy(test_preds, y_test)
  print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

  save_confusion(y_test, test_preds, os.path.join(save_path, 'confusion.png'))
  print(f"Confusion Matrix saved at {save_path}")

  test_probs = clf.predict_proba(X_test)
  save_roc_curve(test_probs, y_test, os.path.join(save_path, 'roc-curve.png'))


CSV_PATH = 'data/processed/extracted.csv'
MODEL = 'logistic'
SAVE_PATH = 'models/logistic'
os.makedirs(SAVE_PATH, exist_ok=True)
main(CSV_PATH, MODEL, SAVE_PATH, train_ratio=0.8)
