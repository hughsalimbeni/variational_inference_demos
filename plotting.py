# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:20:53 2016

@author: hrs13
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import squared_distances, perp_bisector
from scipy.stats import dirichlet, chi2
from matplotlib.patches import Ellipse

class double_panel_demo:
    def __init__(self, K):
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax_1 = fig.add_subplot(121, frameon=False)
        ax_2 = fig.add_subplot(122, frameon=False)
        plt.show(block=False)
        self.ax_1 = ax_1
        self.ax_2 = ax_2
        self.fig = fig
       
        self.xlim = None
        self.ylim = None
        self.num_its = 10
        
        if K==3:
            colours = np.eye(3) # if there are there colours may as well use rgb
        else:
            colours = dirichlet.rvs(0.1*np.ones(3), K) \
                      + dirichlet.rvs(0.1*np.ones(3), K) # else try to take some contrasting ones
            colours = colours/sum(colours)
        self.colours = colours
        self.z_to_colour = lambda z: colours.T.dot(np.reshape(z, (K, 1)))
   
    def set_new_lims(self, X, n):
        l = 1.5
        self.xlim = (min(X[:, 0])*l, max(X[:, 0])*l)
        self.ylim = (min(X[:, 1])*l, max(X[:, 1])*l)
        self.num_its = n
        
    def draw(self):
        self.ax_1.set_xlim(self.xlim)
        self.ax_1.set_ylim(self.ylim)
        self.ax_2.set_xlabel('number of iterations')
        self.ax_2.set_ylabel('objective function')
        self.ax_2.set_xlim((0, self.num_its))
        plt.draw()
    
    def cla(self, axis):
        if axis is 'ax1':
            self.ax_1.cla()
        if axis is 'ax2':
            self.ax_2.cla()
            
    def pause(self, time):
        plt.pause(time)
        
    def plot_points_black(self, X):
        self.ax_1.scatter(X[:, 0], X[:, 1], color='k')
    
        
    ##################### K-means specfic 
    def plot_data_coloured(self, X, Z):
        N, K = Z.shape
        for i in range(N):
            self.ax_1.scatter(X[i, 0], X[i, 1], marker='o', alpha=0.8, color=self.z_to_colour(Z[i, :]))
            
    def plot_regions(self, Z, mus):
        N, K = Z.shape
        eps = 0.00000001
        for j in range(K):
            A = [(0., 1.), (0., 1.)] 
            C = [50., -50.] 
            for k in range(K):
                if k != j:
                    m, c = perp_bisector(mus[j, :], mus[k, :])
                    A.append((-m, 1))
                    C.append(c)
            A = np.asarray(A)
            C = np.asarray(C)
            points = []
            for i in range(K+1):
                for k in range(i+1, K+1):
                    AA= np.vstack((A[i, :], A[k, :]))
                    if abs(np.linalg.det(AA))>eps:
                        CC = np.vstack((C[i], C[k]))
                        points.append(np.linalg.solve(AA, CC))
            
            retained_points = []
            for point in points:
                dist_to_mu = squared_distances(point.T, mus).flatten()
                sorted_args = np.argsort(dist_to_mu)
                diffs = dist_to_mu[sorted_args[0]] - dist_to_mu[sorted_args[1]]
                if abs(diffs)<eps:
                    if j in sorted_args[0:3]:
                        retained_points.append(point)
    
            retained_points = np.reshape(np.asarray(retained_points), (len(retained_points), 2))
            diff_x = retained_points[:, 0] - mus[j, 0]
            diff_y = retained_points[:, 1] - mus[j, 1]
            angles = np.arctan2(diff_x, diff_y)
            ordering = np.argsort(angles)
            orderd_points = retained_points[ordering]
    
            self.ax_1.fill(orderd_points[:, 0], orderd_points[:, 1], color=self.colours[j], alpha=0.1)
    
    def plot_means_as_crosses(self, mus):
        K, D = mus.shape
        for j in range(K):
            col = self.colours[j, :]
            self.ax_1.scatter(mus[j, 0], mus[j, 1], marker='x', color=col, s=300)
    
    def plot_K_means_objective(self, vals):
        for i in range(len(vals)):
            if i%2 == 0:
                self.ax_2.scatter(i/2., vals[i], marker='x')
            else:
                self.ax_2.scatter(i/2., vals[i], marker='o')
        plt.plot(np.arange(len(vals))/2., vals, color='k')
    ############# GMM specific 

    def plot_ellipse(self, mean, cov, mag, colour):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        for volume in np.linspace(0, 0.9, 9):
            width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
            ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                            alpha=0.3*mag, color=colour)
            self.ax_1.add_artist(ellip)
    
    def plot_parameters(self, parameters):
        pis, mus, Sigmas = parameters
        K = len(pis)
        for i in range(K):
            self.plot_ellipse(mus[i, :], Sigmas[i, :, :], pis[i], self.colours[i, :])
      
    def plot_GMM_objective(self, vals, num_its=1):
        if num_its>1:
            n = len(vals)
            m = n/num_its
            p = n%num_its
            print n, m, p
            for j in range(m):
                self.ax_2.plot(np.arange(num_its), vals[j*num_its:(j+1)*num_its], color='k')
                self.ax_2.scatter(np.arange(num_its), vals[j*num_its:(j+1)*num_its], color='r')
            for j in range(p):
                self.ax_2.plot(np.arange(j), vals[m*num_its:(m*num_its+j)], color='k')
                self.ax_2.scatter(np.arange(j), vals[m*num_its:(m*num_its+j)], color='r')
        else:
            self.ax_2.plot(np.arange(len(vals)), vals, color='k')
            self.ax_2.scatter(np.arange(len(vals)), vals, color='r')
        
    ############# GMM specific 
    



class single_panel_demo(double_panel_demo):
    def __init__(self, K):
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax_1 = fig.add_subplot(111, frameon=False)
        plt.show(block=False)
        self.ax_1 = ax_1
        self.fig = fig
       
        self.xlim = [-6, 6.]
        self.ylim = [-6, 6.]
        self.num_its = 10
        
        if K==3:
            colours = np.eye(3) # if there are there colours may as well use rgb
        else:
            colours = dirichlet.rvs(0.1*np.ones(3), K) \
                      + dirichlet.rvs(0.1*np.ones(3), K) # else try to take some contrasting ones
            colours = colours/sum(colours)
        self.colours = colours
        self.z_to_colour = lambda z: colours.T.dot(np.reshape(z, (K, 1)))
   
    def draw(self):
        self.ax_1.set_xlim(self.xlim)
        self.ax_1.set_ylim(self.ylim)
        plt.draw()
    
    def cla(self):
        self.ax_1.cla()
            
    def pause(self, time):
        plt.pause(time)
        
    def plot_points_black(self, X):
        self.ax_1.scatter(X[:, 0], X[:, 1], color='k')
    
    def plot_ellipse(self, mean, cov, mag, colour):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        for volume in np.linspace(0, 0.9, 9):
            width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
            ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                            alpha=0.3*mag, color=colour)
            self.ax_1.add_artist(ellip)
    
    def plot_parameters(self, parameters):
        pis, mus, Sigmas = parameters
        K = len(pis)
        for i in range(K):
            self.plot_ellipse(mus[i, :], Sigmas[i, :, :], pis[i], self.colours[i, :])
            
        
    ##################### K-means specfic 
    def plot_data_coloured(self, X, Z):
        N, K = Z.shape
        for i in range(N):
            self.ax_1.scatter(X[i, 0], X[i, 1], marker='o', alpha=0.8, color=self.z_to_colour(Z[i, :]))
            
    def plot_regions(self, Z, mus):
        N, K = Z.shape
        eps = 0.00000001
        for j in range(K):
            A = [(0., 1.), (0., 1.)] 
            C = [50., -50.] 
            for k in range(K):
                if k != j:
                    m, c = perp_bisector(mus[j, :], mus[k, :])
                    A.append((-m, 1))
                    C.append(c)
            A = np.asarray(A)
            C = np.asarray(C)
            points = []
            for i in range(K+1):
                for k in range(i+1, K+1):
                    AA= np.vstack((A[i, :], A[k, :]))
                    if abs(np.linalg.det(AA))>eps:
                        CC = np.vstack((C[i], C[k]))
                        points.append(np.linalg.solve(AA, CC))
            
            retained_points = []
            for point in points:
                dist_to_mu = squared_distances(point.T, mus).flatten()
                sorted_args = np.argsort(dist_to_mu)
                diffs = dist_to_mu[sorted_args[0]] - dist_to_mu[sorted_args[1]]
                if abs(diffs)<eps:
                    if j in sorted_args[0:3]:
                        retained_points.append(point)
    
            retained_points = np.reshape(np.asarray(retained_points), (len(retained_points), 2))
            diff_x = retained_points[:, 0] - mus[j, 0]
            diff_y = retained_points[:, 1] - mus[j, 1]
            angles = np.arctan2(diff_x, diff_y)
            ordering = np.argsort(angles)
            orderd_points = retained_points[ordering]
    
            self.ax_1.fill(orderd_points[:, 0], orderd_points[:, 1], color=self.colours[j], alpha=0.1)
    
    def plot_means_as_crosses(self, mus):
        K, D = mus.shape
        for j in range(K):
            col = self.colours[j, :]
            self.ax_1.scatter(mus[j, 0], mus[j, 1], marker='x', color=col, s=300)
    
      