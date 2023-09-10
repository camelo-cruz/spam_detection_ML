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
from scipy.stats import randint
from matplotlib import pyplot as plt

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


def pie_plot(data, title):
    unique, counts = np.unique(data, return_counts=True)
    plt.figure(figsize=(8,8))

    plt.pie(
        x=counts, 
        labels=['ham', 'spam'],
        autopct='%1.1f%%',
        explode=[0.05, 0.05],
        colors=sns.color_palette('Set2'))
    
    # Add Title 
    plt.title(
        label=title,
        pad=20)
    
    plt.savefig(f'plots/{title}.png')
    
def visualize_features(X, Y, num_bins):
    df = pd.DataFrame(X[Y == 1].sum(),  columns = ['spam'])
    df['ham'] = X[Y == -1].sum()
    
    df1 = df.sample(frac =.005)
    
    axes = df1.plot.bar(rot=0, subplots=True)
    axes[1].legend(loc=2)
    
    axes.set_ylim(0, 10000)


# def main():

# Load MATLAB file
mat_data = scipy.io.loadmat('emails.mat')

# Extract data
X = pd.DataFrame.sparse.from_spmatrix(mat_data['X'])
X = X.T
Y = mat_data['Y'][0]

# How do data look like?
missing_values = X.isnull().any()
no_missing = missing_values[missing_values.isin([True])].empty
print(f'There are no missing values: {no_missing}')

# Shape of dataframe
print(X.head())
X.shape


# Plot how data are distributed
# pie_plot(Y, 'complete data (unbalanced)')

# Data is mildly unbalanced but model can be tried
#split train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    random_state=4)

#Try a first model
clf = RandomForestClassifier(max_depth=2, random_state=4)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#tune hyperparameters
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

rand_search = RandomizedSearchCV(clf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

search = rand_search.fit(x_train, y_train)
search.best_params_


print('Best hyperparameters:',  search.best_params_)


# pie_plot(y_train, 'train data (unbalanced)')
# pie_plot(y_test, 'test data')


# main()