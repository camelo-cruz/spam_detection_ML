#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 07:52:48 2023

@author: alejandracamelocruz
"""
#Preprocessing and plotting packages
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint
from matplotlib import pyplot as plt

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def pie_plot(data, title):
    unique, counts = np.unique(data, return_counts=True)
    plt.figure(figsize=(8,8))

    plt.pie(
        x=counts,
        labels=['ham', 'spam'],
        autopct='%1.1f%%',
        explode=[0.05, 0.05],
        colors=sns.color_palette('Set2'))

    plt.title(
        label=title,
        pad=20)

    plt.savefig(f'plots/{title}.png')
    plt.show()


def evaluation(y_test, y_pred, title):
  accuracy = accuracy_score(y_test, y_pred)
  matrix = confusion_matrix(y_test, y_pred)
  matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

  plt.figure(figsize=(16,7))
  sns.set(font_scale=1.4)
  sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
  class_names = ['ham', 'spam']
  tick_marks = np.arange(len(class_names))
  tick_marks2 = tick_marks + 0.5
  plt.xticks(tick_marks, class_names, rotation=0)
  plt.yticks(tick_marks2, class_names, rotation=0)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.title(title)
  plt.savefig(f'plots/{title}.png')

  print("Accuracy:", accuracy)
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  plt.show()


def parameter_test(x_train, x_test, y_train, y_test, max_depth=False):
  random = 4
  test = []
  if max_depth:
    title = 'max_depth'
    for num in range(1,100):
      clf = RandomForestClassifier(max_depth = num, random_state=random)
      clf.fit(x_train, y_train)
      y_pred = clf.predict(x_test)
      accuracy = accuracy_score(y_test, y_pred)
      test.append(accuracy)

    plt.plot(test)
    plt.savefig(f'plots/{title}.png')
    plt.show()

