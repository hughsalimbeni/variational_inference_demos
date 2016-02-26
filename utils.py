# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:16:26 2016

@author: hrs13
"""

import numpy as np
import scipy.stats as stats
from scipy.special import gamma

d = lambda i, j, X, mus: np.sum((X[i, :] - mus[j, :])**2)

def squared_distances(X, mus):
    N, D = X.shape
    K, D = mus.shape
    return np.reshape([d(i, j, X, mus) for i in range(N) for j in range(K)], (N, K))
    
    
def perp_bisector(a, b):
    mid_point = 0.5*(a + b)
    grad = -(a[0] - b[0])/(a[1] - b[1])
    return grad,  (mid_point[1] - grad*mid_point[0])
    

def generate_parameters(K, priors=None):
    if priors is None:
        a_0 = 10. # larger makes the mixing coefficients more similar 
        b_0 = 1. # larger makes the clusters closer to the origin
        W_0 = np.eye(2) # 2D, but would work just as well for higher dimensions
        v_0 = 3. # this number being higher (min is 2) makes the clusters more irregular in shape
        m_0 = np.zeros(2) # clusters to be around the origin
    else:
        a_0, b_0, m_0, W_0, v_0 = priors
    
    Sigmas = stats.wishart.rvs(v_0, W_0, K)
    mus = np.empty((K, 2))
    for i in range(K):
        mus[i, :] = stats.multivariate_normal.rvs(m_0, np.linalg.inv(b_0 * Sigmas[i]))
    pis = stats.dirichlet.rvs(np.ones(K) * a_0).flatten()
    return pis, mus, Sigmas

def log_B(W, v):
    ret = (-v/2.)*np.log(np.linalg.det(W))
    ret -= 2*np.log(2) + 0.5*np.log(np.pi) # 2D only
    ret -= gamma(v/2) + gamma((v+1)/2) #2D only
    return ret
    


def generate_data(N, parameters):
    pis, mus, Sigmas = parameters
    X = np.empty((N, 2))
    Z = np.random.multinomial(1, pis.flatten(), N) # so we have a 1-of-N encoding for the class
    for i in range(N):
        z = int(np.where(Z[i, :] == 1)[0]) # the class number 
        X[i, :] = stats.multivariate_normal.rvs(mus[z, :], Sigmas[z, :, :])
    return X, Z
    
def generate_rand_samples(a_k, b_k, m_k, W_k, v_k):
    K = len(a_k)
    Sigmas = np.empty((K, 2, 2))
    mus = np.empty((K, 2))
    for k in range(K):
        Ls = stats.wishart.rvs(v_k[k], W_k[k])
        Sigmas[k, :, :] = np.linalg.inv(Ls)
        mus[k, :] = stats.multivariate_normal.rvs(m_k[k], np.linalg.inv(b_k[k] * Ls))
    pis = stats.dirichlet.rvs(a_k).flatten()
    return pis, mus, Sigmas
    
def interpolate(i, start, end, total):
    p = float(i)/float(total)
    return p * end + (1 - p) * start

def interpolate_wishart(i, L, start_z, end_z, total):
    z = interpolate(i, start_z, end_z, total)
    x = z.dot(L)
    return x.T.dot(x)

def generate_samples_correlated_new(num_samples, a_k, b_k, m_k, W_k, v_k, init):
    K = len(a_k)
    Sigmas = np.empty((num_samples, K, 2, 2))
    mus = np.empty((num_samples, K, 2))
    pis = np.empty((num_samples, K))
    
    df_max = np.floor(max(v_k))
    dfs = np.floor(v_k)
    Z_current = np.reshape(np.random.randn((df_max+1)*2*K), (K, df_max+1, 2))
    pi_current = stats.dirichlet.rvs(a_k).flatten()
    if init is not None:
        Z_old, pi_current = init
        old_df_max = Z_old.shape[1]-1
        if df_max > old_df_max:
            Z_current[:, :(old_df_max+1), :] = Z_old
            Z_current[:, -1, :] = Z_old[:, -1, :]
        else:
            Z_current = Z_old
            df_max = old_df_max
    
    
    c = 0.995
    s = (1-c**2)**0.5
    
    L = np.empty((K, 2, 2))
    for k in range(K):
        L[k, :, :] = np.linalg.cholesky(W_k[k, :, :])
        
    for i in range(num_samples):
        Z_new = np.reshape(np.random.randn((df_max+1)*2*K), (K, df_max+1, 2))
        pi_new = stats.dirichlet.rvs(a_k).flatten()  
        Z_current = c*Z_current + s*Z_new
#        pi_current = abs(c*pi_current + s*(pi_new - a_k/sum(a_k)))
        for k in range(K):
            z = np.reshape(Z_current[k, :dfs[k], :], (dfs[k], 2))
            x = z.dot(L[k, :, :])
            W = x.T.dot(x)
            Sigmas[i, k, :, :] = np.linalg.inv(W)
            WL = np.linalg.cholesky(np.linalg.inv(b_k[k]*W))
            mus[i, k, :] = m_k[k] + np.reshape(WL.dot(np.reshape(Z_current[k, df_max, :], (2, 1))), (2, ))
        pis[i, :] = pi_new
    out = Z_current, pi_current
    return pis, mus, Sigmas, out
         
    
def generate_samples_correlated(num_samples, num_steps, a_k, b_k, m_k, W_k, v_k):
    K = len(a_k)
    Sigmas = np.empty((num_samples, num_steps, K, 2, 2))
    mus = np.empty((num_samples, num_steps, K, 2))
    pis = np.empty((num_samples, num_steps, K))
    
    df_max = np.floor(max(v_k))
    dfs = np.floor(v_k)
    Z_current = np.reshape(np.random.randn((df_max+1)*2*K), (K, df_max+1, 2))
    pi_current = stats.dirichlet.rvs(a_k).flatten()  

    L = np.empty((K, 2, 2))
    for k in range(K):
        L[k, :, :] = np.linalg.cholesky(W_k[k, :, :])
        
    for i in range(num_samples):
        Z_next = np.reshape(np.random.randn((df_max+1)*2*K), (K, df_max+1, 2))
        pi_next = stats.dirichlet.rvs(a_k).flatten() 
        for j in range(num_steps):
            for k in range(K):
                z_c = Z_current[k, :dfs[k], :]
                z_n = Z_next[k, :dfs[k], :]
                W = interpolate_wishart(j, L[k], z_c, z_n, num_steps)
                Sigmas[i, j, k, :, :] = np.linalg.inv(W)
        
        for k in range(K):
            start_L = np.linalg.cholesky(np.linalg.inv(b_k[k] * np.linalg.inv(Sigmas[i, 0, k, :, :])))
            start_mu = m_k[k] + np.reshape(start_L.dot(Z_current[k, df_max, :]), (2, ))
            end_L = np.linalg.cholesky(np.linalg.inv(b_k[k] * np.linalg.inv(Sigmas[i, num_steps-1, k, :, :])))
            end_mu = m_k[k] + np.reshape(end_L.dot(Z_next[k, df_max, :]), (2, ))
            for j in range(num_steps):        
                mus[i, j, k, :] = interpolate(j, start_mu, end_mu, num_steps)

        for j in range(num_steps):
            pis[i, j, :] = interpolate(i, pi_current, pi_next, num_steps) 
        Z_current = Z_next
        pi_current = pi_next
    return pis, mus, Sigmas
        
#    for k in range(K):
#        Ws = np.empty((num_steps, 2, 2))
#        
#        degs_freedom = np.floor(v_k[k])
#    
#        def Ws_unif_interpolated(i):
#            return interpolate(i, wishart_uniforms_start, wishart_uniforms_end, num_steps)
#
#        
#            
#        L = np.linalg.cholesky(W_k[k])
#        
#        x_start = Ws_unif_interpolated(0).dot(L.T)
#        W_start = x_start.T.dot(x_start)
#            
#        x_end = Ws_unif_interpolated(num_steps).dot(L)
#        W_end = x_start.T.dot(x_end)
#        
#        mu_start = stats.multivariate_normal.rvs(m_k[k], np.linalg.inv(b_k[k] * W_start))
#        mu_end = stats.multivariate_normal.rvs(m_k[k], np.linalg.inv(b_k[k] * W_end))
#    
#        def mu_interpolated(i):
#            return interpolate(i, mu_start, mu_end, num_steps)
#    
#        def pi_interpolated(i):
#            return interpolate(i, pi_start, pi_end, num_steps)
#
#            
#        for i in range(num_steps):
#            x = Ws_unif_interpolated(i).dot(L)
#            Ws[i, :, :] = x.T.dot(x)
#            Sigmas[i, k, :, :] = np.linalg.inv(x.T.dot(x))
#            mus[i, k, :] = mu_interpolated(i)
#        
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    