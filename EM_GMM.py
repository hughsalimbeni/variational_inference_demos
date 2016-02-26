# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:15:53 2016

@author: hrs13
"""

import numpy as np
from scipy.stats import  multivariate_normal as mvn
from scipy.misc import logsumexp

def E_step(X, parameters):
    pis, mus, Sigmas = parameters
    N = X.shape[0]
    K = len(pis)
    gammas_flat = np.asarray(
                [pis[k] * mvn.pdf(X[n, :], mus[k, :], Sigmas[k, :, :]) 
                 for k in range(K) for n in range(N)])
    gammas = np.reshape(gammas_flat, (K, N)).T
    return (gammas.T/np.sum(gammas, axis=1)).T

def M_step(X, gammas):
    N, K = gammas.shape
    N_k = np.sum(gammas, axis=0)
    mus = ((gammas.T.dot(X)).T/N_k).T
    X_c = np.reshape([X[n, :] - mus[k, :] for n in range(N) for k in range(K)], (N, K, 2))
    Sigmas = np.einsum('nk,nkd,nke, k->kde', gammas, X_c, X_c, 1/N_k)
    pis = N_k/N    
    return pis, mus, Sigmas
    
def loglikelihood(X, parameters):
    pis, mus, Sigmas = parameters
    N = X.shape[0]
    K = len(pis)
    log_probs_flat = np.asarray(
                [np.log(pis[k]) + mvn.logpdf(X[n, :], mus[k, :], Sigmas[k, :, :]) 
                 for k in range(K) for n in range(N)])
    log_probs = np.reshape(log_probs_flat, (K, N)).T
    L = np.sum(logsumexp(log_probs, axis=1))
    return L
    
    
from utils import generate_parameters, generate_data 
from plotting import double_panel_demo

if __name__ == '__main__':
    K = 3
    N = 100
    num_its = 16
    
    X = generate_data(N, generate_parameters(K))[0]
    plt = double_panel_demo(K)
    
    while True:
        X = generate_data(N, generate_parameters(K))[0]
        plt.set_new_lims(X, num_its)
        params = generate_parameters(K)
        # these initial parameters are an independent draw from the prior  
        
        objective = []    
        
        plt.cla('ax1')
        plt.cla('ax2')
        plt.plot_points_black(X)
        plt.draw()
        plt.pause(2.)
        
        plt.cla('ax1')
        plt.plot_points_black(X)
        plt.plot_parameters(params)
        plt.draw()
        plt.pause(1.)


        for i in range(num_its):           
            gamma = E_step(X, params)
            objective.append(loglikelihood(X, params))
            plt.cla('ax1')
            plt.plot_parameters(params)
            plt.plot_GMM_objective(objective)
            plt.plot_data_coloured(X, gamma)
            plt.pause(0.5)       
            
            params =  M_step(X, gamma)
            plt.cla('ax1')
            plt.plot_parameters(params)
            plt.plot_data_coloured(X, gamma)
            plt.pause(0.5)            


         
