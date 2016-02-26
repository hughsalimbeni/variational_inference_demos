# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:16:52 2016

@author: hrs13
"""

import numpy as np
import pickle
import matplotlib.animation as manimation
from utils import generate_parameters, generate_samples_correlated, generate_samples_correlated_new

from plotting import double_panel_demo, single_panel_demo
from K_means import update_K_means_Z, K_means_objective, update_K_means_mus
from EM_GMM import loglikelihood, E_step, M_step
from variational_GMM import variational_E_step, variational_M_step, variational_LB

def K_means_demo(X, K, num_its):
    plt.set_new_lims(X, num_its)
    mus = generate_parameters(K)[1] 
    # these initial means are an independent draw from the prior  
    
    objective = []    
    
    plt.cla('ax1')
    plt.cla('ax2')
    plt.plot_points_black(X)
    plt.draw()
    writer.grab_frame()    
    
    plt.cla('ax1')
    plt.plot_points_black(X)
    plt.plot_means_as_crosses(mus)
    plt.draw()
    writer.grab_frame()
    
    for i in range(num_its):
        # update Z        
        
        Z = update_K_means_Z(X, mus)
        objective.append(K_means_objective(X, Z, mus))
       
        plt.cla('ax1')
        plt.plot_means_as_crosses(mus)
        plt.plot_points_black(X)
        plt.plot_K_means_objective(objective)
        plt.plot_regions(Z, mus)
        plt.draw()
        writer.grab_frame()
        
        #show colours
        plt.cla('ax1')
        plt.plot_regions(Z, mus)
        plt.plot_means_as_crosses(mus)
        plt.plot_data_coloured(X, Z)
        plt.draw()
        writer.grab_frame()
        
        # update means 
        new_mus = update_K_means_mus(X, Z)
        objective.append(K_means_objective(X, Z, new_mus))
        
        plt.cla('ax1')
        plt.plot_means_as_crosses(new_mus)
        plt.plot_data_coloured(X, Z)
        plt.plot_K_means_objective(objective)
        plt.draw()
        writer.grab_frame()
        
        mus = new_mus
        
        #plot with black points 
        plt.cla('ax1')
        plt.plot_points_black(X)
        plt.plot_means_as_crosses(mus)
        plt.draw()
        writer.grab_frame()
        
           
        #move regions
        plt.cla('ax1')
        plt.plot_regions(Z, mus)
        plt.plot_means_as_crosses(mus)
        plt.plot_points_black(X)
        plt.draw()
        writer.grab_frame()
        
        
def EM_make_demo(X, K, num_its, num_trails):
    N = X.shape[0]
    objective = []    
    
    Sigmas = np.empty((num_iterations*num_trails, K, 2, 2))
    mus = np.empty((num_iterations*num_trails, K, 2))
    pis = np.empty((num_iterations*num_trails, K))
    rs = np.empty((num_iterations*num_trails, N, 3))
    
    for j in range(num_trails):
        params = generate_parameters(K)
        for i in range(num_its):           
            n = j*num_its + i
            print n
            r = E_step(X, params)  
            rs[n, :, :] = r 
            pis[n, :] = params[0]
            mus[n, :, :] = params[1]
            Sigmas[n, :, :, :] = params[2]
            params = M_step(X, r)
            objective.append(loglikelihood(X, params))
#        objectives.append(objective)

    def EM_animate(i):
        n = i
        plt.cla('ax1')
        plt.cla('ax2')
        params = pis[n, :], mus[n, :, :], Sigmas[n, :, :, :]
        plt.plot_parameters(params)
        plt.plot_data_coloured(X, rs[n, :, :])
        plt.plot_GMM_objective(objective[:n], num_iterations)
        plt.draw()  
    return EM_animate
                
                                


def variational_make_demo_new(X, K, num_iterations, num_samples):
    N = X.shape[0]
    objective = []

    rand = np.reshape(np.random.rand(N*K), (N, K))
    r = (rand.T/np.sum(rand, axis=1)).T
    
    Sigmas = np.empty((num_iterations, num_samples, K, 2, 2))
    mus = np.empty((num_iterations, num_samples, K, 2))
    pis = np.empty((num_iterations, num_samples, K))
    rs = np.empty((num_iterations, N, K))
    out = None
    for i in range(num_iterations):     
        a_k, b_k, m_k, W_k, v_k = variational_M_step(X, r)
        r = variational_E_step(X, a_k, b_k, m_k, W_k, v_k)          
        rs[i, :, :] = r 
        objective.append(variational_LB(X, r, a_k, b_k, m_k, W_k, v_k))
        pis_, mus_, Sigmas_, out = generate_samples_correlated_new(num_samples, 
                                                       a_k, b_k, m_k, W_k, v_k, out)
        pis[i, :, :] = pis_
        mus[i, :, :, :] = mus_
        Sigmas[i, :, :, :, :] = Sigmas_
    def variational_animate(n):
        i = n/num_samples
        j = n%num_samples
        print n, i, j
        plt.cla('ax1')
        params = pis[i, j, :], mus[i, j, :, :], Sigmas[i, j, :, :, :]
        print pis[i, j, :]
        plt.plot_parameters(params)
        plt.plot_data_coloured(X, rs[i, :, :])
        if j==0:
            plt.cla('ax2')
            plt.plot_GMM_objective(objective[:i])
        plt.draw()
    return variational_animate
    
    
    
#def variational_make_demo(X, K, num_iterations, num_samples, num_interpolation_steps):
#    N = X.shape[0]
#    objective = []
#
#    rand = np.reshape(np.random.rand(N*K), (N, K))
#    r = (rand.T/np.sum(rand, axis=1)).T
#    
#    Sigmas = np.empty((num_iterations, num_samples, num_interpolation_steps, K, 2, 2))
#    mus = np.empty((num_iterations, num_samples, num_interpolation_steps, K, 2))
#    pis = np.empty((num_iterations, num_samples, num_interpolation_steps, K))
#    rs = np.empty((num_iterations, N, K))
#
#    for i in range(num_iterations):     
#        a_k, b_k, m_k, W_k, v_k = variational_M_step(X, r)
#        r = variational_E_step(X, a_k, b_k, m_k, W_k, v_k)          
#        rs[i, :, :] = r 
#        objective.append(variational_LB(X, r, a_k, b_k, m_k, W_k, v_k))
#        pis_, mus_, Sigmas_ = generate_samples_correlated(num_samples, 
#                                                           num_interpolation_steps, 
#                                                       a_k, b_k, m_k, W_k, v_k)
#        pis[i, :, :, :] = pis_
#        mus[i, :, :, :, :] = mus_
#        Sigmas[i, :, :, :, :] = Sigmas_
##    return pis, mus, Sigmas, rs, objective
#    def variational_animate(n):
#        i = n/(num_samples*num_interpolation_steps)
#        j = (n%(num_samples*num_interpolation_steps))/num_interpolation_steps
#        k = (n%(num_samples*num_interpolation_steps))%num_interpolation_steps
#        print n, i, j, k
#        plt.cla('ax1')
#        params = np.abs(pis[i, j, k, :]), mus[i, j, k, :, :], Sigmas[i, j, k, :, :, :]
#        print pis[i, j , k :]
#        plt.plot_parameters(params)
#        plt.plot_data_coloured(X, rs[i])
#        if k==0:
#            plt.cla('ax2')
#            plt.plot_GMM_objective(objective[:i])
#        plt.draw()
#    return variational_animate
#    


FFMpegWriter = manimation.writers['ffmpeg']

#metadata = dict(title='K means 1', artist='hughsalimbeni',
#        comment='Shows K means working well')
#writer = FFMpegWriter(fps=15, metadata=metadata)

#data_1, data_2 = pickle.load( open( "data.p", "rb" ))
#
#plt = double_panel_demo(3)
#
#num_iterations = 30
#num_trials = 8
#X = data_1
#
#fps = 5
#writer = FFMpegWriter(fps=fps, bitrate=fps*100)
#
#plt.set_new_lims(X, num_iterations)
#frames = num_iterations*num_trials
#EM_animate = EM_make_demo(X, 3, num_iterations, num_trials)
#anim = manimation.FuncAnimation(plt.fig, EM_animate, frames=frames)
#
#anim.save("EM_low.mp4", writer=writer)

#    


#
#data_1, data_2 = pickle.load( open( "data.p", "rb" ))
#plt = double_panel_demo(3)
#X = data_1
#
#num_iterations = 30
#num_samples = 20
#var_animate = variational_make_demo_new(X, 3, 
#                                  num_iterations, 
#                                  num_samples)
#fps=30
#plt.set_new_lims(X, num_iterations)
#
#writer = FFMpegWriter(fps=fps, bitrate=100*fps)
#frames = num_iterations * num_samples
#anim = manimation.FuncAnimation(plt.fig, var_animate, frames=frames)
#anim.save("var_high.mp4", writer=writer)


#    
#    
    

def prior_make_demo(K, num_samples):

    Sigmas = np.empty((num_samples, K, 2, 2))
    mus = np.empty((num_samples, K, 2))
    pis = np.empty((num_samples, K))
    out = None
    
    a_k = 1000.*np.ones(K) 
    b_k = 1.*np.ones(K)
    v_k = 25*np.ones(K)
    m_k = np.empty((K, 2))
    W_k = np.empty((K, 2, 2))
    for k in range(K):
        m_k[k, :] = np.zeros(2) 
        W_k[k, :, :] = np.eye(2)/20
        
    pis, mus, Sigmas, out = generate_samples_correlated_new(num_samples, 
                                                   a_k, b_k, m_k, W_k, v_k, out)

    def prior_animate(n):
        print n
        plt.cla()
        params = pis[n, :], mus[n, :, :], Sigmas[n, :, :, :]
        plt.plot_parameters(params)
        plt.draw()
    return prior_animate


plt = single_panel_demo(3)
num_samples = 200
fps = 15
writer = FFMpegWriter(fps=fps, bitrate=100*fps)
frames = num_samples
prior_animate = prior_make_demo(3, num_samples)

anim = manimation.FuncAnimation(plt.fig, prior_animate, frames=frames)
anim.save("prior2.mp4", writer=writer)


        
#writer = FFMpegWriter(fps=1)
#num_its = 10
#frames = (2 + num_its*5)
#with writer.saving(plt.fig, "K_means_low_1.mp4", frames):
#    K_means_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "K_means_low_2.mp4", frames):
#    K_means_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "K_means_low_3.mp4", frames):
#    K_means_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "K_means_low_1.mp4", frames):
#    K_means_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "K_means_low_2.mp4", frames):
#    K_means_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "K_means_low_3.mp4", frames):
#    K_means_demo(plt, data_2, 3, num_its)
    
#    
#writer = FFMpegWriter(fps=5)
#num_its = 30
#frames = 2 + num_its*2
#with writer.saving(plt.fig, "EM_low_1.mp4", frames):
#    EM_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "EM_low_2.mp4", frames):
#    EM_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "EM_low_3.mp4", frames):
#    EM_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "EM_high_1.mp4", frames):
#    EM_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "EM_high_2.mp4", frames):
#    EM_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "EM_high_3.mp4", frames):
#    EM_demo(plt, data_2, 3, num_its)
#
#
#
#









#with writer.saving(plt.fig, "var_low_2.mp4", frames):
#    variational_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "var_low_3.mp4", frames):
#    variational_demo(plt, data_1, 3, num_its)
#with writer.saving(plt.fig, "var_high_1.mp4", frames):
#    variational_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "var_high_2.mp4", frames):
#    variational_demo(plt, data_2, 3, num_its)
#with writer.saving(plt.fig, "var_high_3.mp4", frames):
#    variational_demo(plt, data_2, 3, num_its)
#
#


