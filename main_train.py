from typing import Literal, Tuple

import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import model_selection, linear_model, ensemble, metrics, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
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

  elif model == 'random_forest':
    clf = ensemble.RandomForestClassifier(n_estimators=100)

  else:
    raise KeyError("Invalid model named used")

  clf = clf.fit(X_train, y_train)

  return clf


def calculate_confusion_matrix(y_test: np.ndarray, test_preds: np.ndarray):
  return metrics.confusion_matrix(y_test, test_preds)


def calculate_metrics(confusion_matrix: np.ndarray):
  """
    Returns a dict with metrics
  """
  tn, fp, fn, tp = confusion_matrix.ravel()
  specificity = tn / (tn + fp)
  sensitivity = tp / (tp + fn)
  fpr = fp / (fp + tn)
  fnr = fn / (fn + tn)
  precision = tp / (tp + fp)

  # F1 = 2 * (precision * recall) / (precision + recall)
  f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

  return {
      "Sensitivity": sensitivity,
      "Specificity": specificity,
      "FPR": fpr,
      "FNR": fnr,
      "NPV": 1 - fnr,
      "Precision": precision,
      "Recall": sensitivity,
      "F1 Score": f1
  }


def save_confusion(confusion_matrix: np.ndarray, file_path: str):
  """
    Saves a confusion matrix in a folder indicated by save_path under name `confusion.png`.
    Returns the confusion matrix.
  """
  fig, ax = plt.subplots()
  sns.heatmap(confusion_matrix, fmt='3d', annot=True, ax=ax)

  fig.savefig(file_path)
  return confusion_matrix


def save_roc_curve(test_probs: np.ndarray, y_test: np.ndarray, file_path: str, model: str):
  """
    Creates a ROC curve and saves it to `file_path`.
  """
  # generate a no skill prediction (majority class)
  ns_probs = [0 for _ in range(len(y_test))]

  # keep probabilities for the positive outcome only
  test_probs = test_probs[:, 1]
  # calculate scores
  ns_auc = roc_auc_score(y_test, ns_probs)
  model_auc = roc_auc_score(y_test, test_probs)

  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Logistic: ROC AUC=%.3f' % model_auc)

  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
  model_fpr, model_tpr, _ = roc_curve(y_test, test_probs)

  fig, ax = plt.subplots()
  # plot the roc curve for the model
  ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
  ax.plot(model_fpr, model_tpr, marker='.', label=str.capitalize(model))
  # axis labels
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  # show the legend
  ax.legend()

  fig.savefig(file_path)

  return model_fpr, model_tpr


def main_train(X: np.ndarray, y: np.ndarray, model: Literal['logistic', 'random_forest'],
               save_path: str, train_ratio: float = 0.8):
  """  
  Args:
      X (np.ndarray): _description_
      y (np.ndarray): _description_
      model (Literal[&#39;logistic&#39;, &#39;random_forest&#39;]): _description_
      save_path (str): path where we will save files
      train_ratio (float, optional): _description_. Defaults to 0.8.
  """


  assert 0 < train_ratio < 1, "Enter valid train ratio"

  scaler = preprocessing.StandardScaler()
  X = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=42,
                                                                      shuffle=True, stratify=y)

  print(
      f"No. training examples: {len(X_train)}, No. testing examples: {len(X_test)}")


  # TRAIN AND GET ACCURACIES ON MODEL
  clf = train(X, y, model, train_ratio=0.8)
  train_preds = clf.predict(X_train)
  test_preds = clf.predict(X_test)

  train_accuracy, test_accuracy = get_accuracy(
      train_preds, y_train), get_accuracy(test_preds, y_test)
  print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")


  # PLOT CONFUSION MATRIX
  confusion_matrix = calculate_confusion_matrix(y_test, test_preds)
  confusion_matrix = save_confusion(
      confusion_matrix, os.path.join(save_path, 'confusion.png'))
  print(f"Confusion Matrix saved at {save_path}")

  # PLOT ROC CURVE
  test_probs = clf.predict_proba(X_test)
  save_roc_curve(test_probs, y_test, os.path.join(
      save_path, 'roc-curve.png'), model)

  # PRINT ALL REQUIRED METRICS
  print("\n\n---------- ALL METRICS BELOW -------------")
  calc_metrics = calculate_metrics(confusion_matrix)
  for metric_name, metr in calc_metrics.items():
    print(f"{metric_name}: { metr }")

  mcc = metrics.matthews_corrcoef(y_test, test_preds)
  print(f"MCC: {mcc}")

  print(f"Test Accuracy: {test_accuracy}")


def main(csv_path: str, model: Literal['logistic', 'random_forest'], save_path: str, train_ratios: Tuple[float]):
  assert model in [
      'logistic', 'random_forest'], f"Enter valid `model`. Invalid `model` {model}"

  X, y = load_and_process_data(csv_path)
  X, y = X.to_numpy(), y.to_numpy()

  print(f"MODEL: {str.capitalize(model)}")

  for train_ratio in train_ratios:
    # create current folder if does not exist
    curr_save_path = os.path.join(
        save_path, f"{model}-ratio={train_ratio}")
    os.makedirs(curr_save_path, exist_ok=True)

    print(f"CURRENT TRAINING RATIO: {train_ratio}")
    main_train(X, y, model, curr_save_path, train_ratio)
    print("\n\n")


CSV_PATH = 'data/processed/extracted.csv'
SAVE_PATH = 'models/random_forest'
os.makedirs(SAVE_PATH, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='logistic',
                    help='Model to use (one of ["logistic", "random_forest"])')

MODEL = parser.parse_args().model

main(CSV_PATH, MODEL, SAVE_PATH, train_ratios=(0.7, 0.8, 0.6))
