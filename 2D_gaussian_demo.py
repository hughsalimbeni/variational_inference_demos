# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:26:19 2016

@author: hughsalimbeni
"""

import numpy as np
from utils import generate_parameters 
from plotting import single_panel_demo
from matplotlib import animation

plt = single_panel_demo(3)

def anim(i):
    print i
    plt.cla()
    pis, mus, Sigmas = generate_parameters(2)
    mu = mus[0, :]
    Sigma = Sigmas[0, :, :]
    Lambda = np.linalg.inv(Sigma)
    plt.plot_ellipse(mu, Sigma, 0.7, plt.colours[:, 1])   
    var_Sigma = np.reshape(np.array((Lambda[0, 0]**(-1.), 0., 0., Lambda[1, 1]**(-1.))), (2, 2))
    plt.plot_ellipse(mu, var_Sigma, 0.5, plt.colours[:, 0])
    plt.draw()

FFMpegWriter = animation.writers['ffmpeg']

frames = 50 
fps = 1

writer = FFMpegWriter(fps=fps, bitrate=100*fps)
anim = animation.FuncAnimation(plt.fig, anim, frames=frames)
anim.save("2d_gaussian.mp4", writer=writer)













#
#def make_demo(num_samples):    
#    K = 1
#    Sigmas = np.empty((num_samples, K, 2, 2))
#    mus = np.empty((num_samples, K, 2))
#    pis = np.empty((num_samples, K))
#    out = None
#    
#    a_k = 10.*np.ones(K) 
#    b_k = 1.*np.ones(K)
#    v_k = 2.5*np.ones(K)
#    m_k = np.empty((K, 2))
#    W_k = np.empty((K, 2, 2))
#    for k in range(K):
#        m_k[k, :] = np.zeros(2) 
#        W_k[k, :, :] = np.eye(2)
#        
#    pis, mus, Sigmas, out = generate_samples_correlated_new(num_samples, 
#                                                   a_k, b_k, m_k, W_k, v_k, out)
#
##    def animate(n):
#    for i in range(5):
#        n = np.random.random_integers(0, num_samples, 1)
#        plt.cla()
#        mu = mus[n, 0, :]
#        Sigma = Sigmas[n, 0, :, :]
#        Lambda = np.linalg.inv(Sigma)
#        plt.plot_ellipse(mu, Sigma, 0.7, plt.colours[:, 1])   
#        var_Sigma = np.reshape(np.array((Lambda[0, 0]**(-1.), 0., 0., Lambda[1, 1]**(-1.))), (2, 2))
#        plt.plot_ellipse(mu, var_Sigma, 0.5, plt.colours[:, 0])
#        plt.fig.plt.fig.savefig('2D_gaussian'+ str(i)+'.png', format='png')        
#        plt.draw()
#    return animate
    

#FFMpegWriter = animation.writers['ffmpeg']
#
#frames = 500
#fps=15
#
#writer = FFMpegWriter(fps=fps, bitrate=100*fps)
#animate = make_demo(frames)
#anim = animation.FuncAnimation(plt.fig, animate, frames=frames)
#anim.save("2d_gaussian.mp4", writer=writer)

#a_k = 10.*np.ones(K) 
#b_k = 1.*np.ones(K)
#v_k = 2.5*np.ones(K)
#m_k = np.empty((K, 2))
#W_k = np.empty((K, 2, 2))
#for k in range(K):
#    m_k[k, :] = np.zeros(2) 
#    W_k[k, :, :] = np.eye(2)
#
#pis, mus, Sigmas, out = generate_samples_correlated_new(num_samples, 
#                                                   a_k, b_k, m_k, W_k, v_k, out)
#
#
#
#n = np.random.random_integers(0, 499, 1)[0]
#plt.cla()
#mu = mus[n, 0, :]
#Sigma = Sigmas[n, 0, :, :]
#Lambda = np.linalg.inv(Sigma)
#plt.plot_ellipse(mu, Sigma, 0.7, plt.colours[:, 1])   
#var_Sigma = np.reshape(np.array((Lambda[0, 0]**(-1.), 0., 0., Lambda[1, 1]**(-1.))), (2, 2))
#plt.plot_ellipse(mu, var_Sigma, 0.5, plt.colours[:, 0])
#plt.fig.savefig('2D_gaussian'+ str(i)+'.png', format='png')        
#









