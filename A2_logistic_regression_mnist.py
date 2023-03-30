#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:22:28 2023

@author:  Elias Stenhede Johansson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from A2_logistic_regression_simulation import logistic, logistic_regression_NR

def logistic_forecast(features,beta): 
    """Predicts target as 0 or 1 given some features and parameter estimates"""   
    signal_hat = np.dot(features, beta)
    y_hat=np.sign(signal_hat)
    y_hat[y_hat<0] = 0
    return y_hat
  
def prediction_accuracy(y_predicted,y_observed):
    """computes the prediction error by comparing the predicted and the observed labels in the test set"""
    errors= np.abs(y_predicted-y_observed)
    total_errors=sum(errors)
    acc=1-total_errors/len(y_predicted)    
    return acc

def logistic_regression_NR_penalized(features, target, num_steps, tolerance, lambda_0):
    beta = np.zeros(features.shape[1])
    for step in range(num_steps):  
        y_pred = logistic(features @ beta)

        gradient = features.T @ (target - y_pred) - 2*lambda_0*beta
        hessian = features.T * y_pred * (y_pred-1) @ features - 2*lambda_0*np.eye(beta.size)

        update = np.linalg.solve(-hessian, gradient)
        beta += update
        # Check convergence
        if np.linalg.norm(update) < tolerance:
            break
    return beta

if __name__ == "__main__":
    df = pd.read_csv(r'./mnist.csv')
    #make dataframe into numpy array
    df.to_numpy()
    y_labels_data=df['label'].to_numpy()
    df_xdata=df.drop(columns='label')
    x_features_data=df_xdata.to_numpy()
    x_features_data.shape

    # we will only use the zeros and ones in this empirical study
    y_labels_01 = y_labels_data[np.where(y_labels_data <=1)[0]]
    x_features_01 = x_features_data[np.where(y_labels_data <=1)[0]]

    # create test and training set
    n_train=100
    y_train=y_labels_01[0:n_train]
    x_train=x_features_01[0:n_train]
    n_total=y_labels_01.size
    y_test=y_labels_01 [n_train:n_total]
    x_test=x_features_01[n_train:n_total]

    print(f"Rank of x_train: {np.linalg.matrix_rank(x_train.T)}")
    print(f"Dimensions of x_train: {x_train.shape}")
    #In logistic_regression_NR we try to solve an equation system that lacks solutions, when we add a penalty term this problem disappears.

    #See weight map for some penalty parameter
    penalty_weight = 1e7
    fig,ax=plt.subplots(1)
    beta_ml = logistic_regression_NR_penalized(x_train, y_train, 100, 1e-9, penalty_weight)
    matrix = np.reshape(beta_ml, (28, 28))
    im = ax.matshow(matrix, cmap='gray')
    fig.colorbar(im)
    ax.set_title(f'Weight Map for lambda_0={penalty_weight}')
    y_hat = logistic_forecast(x_test, beta_ml)
    accuracy = prediction_accuracy(y_hat, y_test)
    print(accuracy)
    plt.show()


