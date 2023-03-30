#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:22:28 2023

@author: Elias Stenhede Johansson
"""

import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    """The logistic function"""
    return 1 / (1 + np.exp(-x))

def logistic_simulation(features,beta,n):
    """Simulate the labels""" 
    signal = np.dot(features, beta)
    p=logistic(signal)
    y= np.array([np.random.binomial(1, p[i] ) for i in range(n)])
    return y
 
#
def logistic_regression_NR(features, target, num_steps, tolerance):
    """Newton-Raphson for logisic regression"""
    beta = np.zeros(features.shape[1])
    for step in range(num_steps):     
        # Compute the predicted values
        y_pred = logistic(features @ beta)
        
        # Compute the gradient and the Hessian
        gradient = features.T @ (target - y_pred)
        hessian = features.T * y_pred * (1 - y_pred) @ features
        
        update = np.linalg.solve(hessian, gradient)
        beta += update
        # Check convergence
        if np.linalg.norm(update) < tolerance:
            break
    return beta

def simulate_data(beta_star, n):
    """Simulate data from multivariate normal distribution"""
    half_sample=int(n/2)
    x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
    x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
    simulated_features = np.vstack((x1, x2)).astype(np.float64)
    return simulated_features

def monte_carlo_histogram(S, n, beta_star):
    """Create a histogram of S trials of n samples"""
    data = []
    simulated_features = simulate_data(beta_star, n)
    for _ in range(S):
        simulated_labels = logistic_simulation(simulated_features, beta_star, n)
        data.append(logistic_regression_NR(simulated_features, simulated_labels, 1000, 1e-6))
    data = np.array(data)

    fig, ax = plt.subplots()
    bins=np.linspace(min(data.flatten()), max(data.flatten()), 30)
    ax.hist(data, bins=bins, alpha=0.5, edgecolor='k', width=bins[0]-bins[1], label=["beta[0]", "beta[1]"])
    ax.axvline(x=beta_star[0], color='blue', linestyle='--')
    ax.axvline(x=beta_star[1], color='orange', linestyle='--')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.title(f"Maximum likelihood estimates of beta with S={S}, n={n}.")
    plt.savefig(f"./figures/S{S}_n{n}.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    beta_star=np.array([0.2,-0.8])
    n = 1000
    #Create our two histograms with 1000 and 100 datapoints
    monte_carlo_histogram(1000, n, beta_star)
    monte_carlo_histogram(1000, n//10, beta_star)

    #
    simulated_features = simulate_data(beta_star, n)
    simulated_labels = logistic_simulation(simulated_features, beta_star, n)
    logistic_regression_NR(simulated_features, simulated_labels, 10, 0.01)