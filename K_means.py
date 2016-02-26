# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:15:18 2016

@author: hrs13
"""
import numpy as np
from utils import squared_distances

def update_K_means_Z(X, mus):
    d2 = squared_distances(X, mus)
    return (abs((d2.T-np.min(d2, axis=1)).T)==0).astype(int)

def update_K_means_mus(X, Z):
    if sum(np.sum(Z, axis=0)==0)!=0:
        print 'singluar !!!'
    else:
        return np.einsum('nk,nd->kd', Z/(np.sum(Z, axis=0).astype(float)), X)

def K_means_objective(X, Z, mus):
    d2 = squared_distances(X, mus)
    return np.einsum('nk,nk',d2, Z)

from utils import generate_parameters, generate_data 
from plotting import double_panel_demo, single_panel_demo

if __name__ == '__main__':
    
    K = 4
    N = 100
    num_its = 5
    
    X = generate_data(N, generate_parameters(K))[0]
    plt = single_panel_demo(K)
    
    while True:
        X = generate_data(N, generate_parameters(K))[0]
        plt.set_new_lims(X, num_its)
        mus = generate_parameters(K)[1] 
        # these initial means are an independent draw from the prior  
        
        objective = []    
        
        plt.cla()
        plt.plot_points_black(X)
        plt.draw()
        plt.fig.savefig('K_means_1.png', format='png')
        plt.pause(2.)
        
        
        plt.cla()
        plt.plot_points_black(X)
        plt.plot_means_as_crosses(mus)
        plt.draw()
        plt.fig.savefig('K_means_2.png', format='png')
        plt.pause(1.)


                
        
        for i in range(num_its):
            # update Z        
            
            Z = update_K_means_Z(X, mus)
            objective.append(K_means_objective(X, Z, mus))
           
            plt.cla()
            plt.plot_means_as_crosses(mus)
            plt.plot_points_black(X)
            plt.plot_regions(Z, mus)
            plt.draw()
            plt.fig.savefig('K_means_'+ str(i)+'_1.png', format='png')
            plt.pause(0.5)
            
            #show colours
            plt.cla()
            plt.plot_regions(Z, mus)
            plt.plot_means_as_crosses(mus)
            plt.plot_data_coloured(X, Z)
            plt.draw()
            plt.fig.savefig('K_means_'+ str(i)+'_2.png', format='png')
            plt.pause(0.5)
            
            # update means 
            new_mus = update_K_means_mus(X, Z)
            objective.append(K_means_objective(X, Z, new_mus))
            
            plt.cla()
            plt.plot_means_as_crosses(new_mus)
            plt.plot_data_coloured(X, Z)
            plt.draw()
            plt.fig.savefig('K_means_'+ str(i)+'_3.png', format='png')
            plt.pause(0.5)
            
            mus = new_mus
            
            #plot with black points 
            plt.cla()
            plt.plot_points_black(X)
            plt.plot_means_as_crosses(mus)
            plt.draw()
            plt.fig.savefig('K_means_'+ str(i)+'_4.png', format='png')
            plt.pause(0.5)
            
            #move regions
            plt.cla()
            plt.plot_regions(Z, mus)
            plt.plot_means_as_crosses(mus)
            plt.plot_points_black(X)
            plt.draw()
            plt.fig.savefig('K_means_'+ str(i)+'_5.png', format='png')
            plt.pause(0.5)

#
#if __name__ == '__main__':
#    
#    K = 4
#    N = 100
#    num_its = 5
#    
#    X = generate_data(N, generate_parameters(K))[0]
#    plt = double_panel_demo(K)
#    
#    while True:
#        X = generate_data(N, generate_parameters(K))[0]
#        plt.set_new_lims(X, num_its)
#        mus = generate_parameters(K)[1] 
#        # these initial means are an independent draw from the prior  
#        
#        objective = []    
#        
#        plt.cla('ax1')
#        plt.cla('ax2')
#        plt.plot_points_black(X)
#        plt.draw()
#        plt.fig.savefig('K_means_1.png', format='png')
#        plt.pause(2.)
#        
#        
#        plt.cla('ax1')
#        plt.plot_points_black(X)
#        plt.plot_means_as_crosses(mus)
#        plt.draw()
#        plt.fig.savefig('K_means_2.png', format='png')
#        plt.pause(1.)
#
#
#                
#        
#        for i in range(num_its):
#            # update Z        
#            
#            Z = update_K_means_Z(X, mus)
#            objective.append(K_means_objective(X, Z, mus))
#           
#            plt.cla('ax1')
#            plt.plot_means_as_crosses(mus)
#            plt.plot_points_black(X)
#            plt.plot_K_means_objective(objective)
#            plt.plot_regions(Z, mus)
#            plt.draw()
#            plt.fig.savefig('K_means_'+ str(i)+'_1.png', format='png')
#            plt.pause(0.5)
#            
#            #show colours
#            plt.cla('ax1')
#            plt.plot_regions(Z, mus)
#            plt.plot_means_as_crosses(mus)
#            plt.plot_data_coloured(X, Z)
#            plt.draw()
#            plt.fig.savefig('K_means_'+ str(i)+'_2.png', format='png')
#            plt.pause(0.5)
#            
#            # update means 
#            new_mus = update_K_means_mus(X, Z)
#            objective.append(K_means_objective(X, Z, new_mus))
#            
#            plt.cla('ax1')
#            plt.plot_means_as_crosses(new_mus)
#            plt.plot_data_coloured(X, Z)
#            plt.plot_K_means_objective(objective)
#            plt.draw()
#            plt.fig.savefig('K_means_'+ str(i)+'_3.png', format='png')
#            plt.pause(0.5)
#            
#            mus = new_mus
#            
#            #plot with black points 
#            plt.cla('ax1')
#            plt.plot_points_black(X)
#            plt.plot_means_as_crosses(mus)
#            plt.draw()
#            plt.fig.savefig('K_means_'+ str(i)+'_4.png', format='png')
#            plt.pause(0.5)
#            
#            #move regions
#            plt.cla('ax1')
#            plt.plot_regions(Z, mus)
#            plt.plot_means_as_crosses(mus)
#            plt.plot_points_black(X)
#            plt.draw()
#            plt.fig.savefig('K_means_'+ str(i)+'_5.png', format='png')
#            plt.pause(0.5)
#
