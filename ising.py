# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:14:32 2016

@author: hughsalimbeni
"""


import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import threshold_adaptive

IMAGE = 'penguin.jpg'

block_size = 40
image = np.array(Image.open(IMAGE).convert('L'))
image = threshold_adaptive(image, block_size, offset=10)
image_scaled = (image * 2) - 1

#plt.gray()
#plt.imshow(image_scaled)
#plt.colorbar()
#plt.show()


noise = 1*np.random.randn(image.shape[0]*image.shape[1]).reshape(image.shape)
noise = 2*np.random.binomial(1, 0.7, image.shape[0]*image.shape[1]).reshape(image.shape)-1.
noisey_image = (image_scaled*noise)

#noisey_image = (image_scaled + noise)
#plt.gray()
#plt.imshow(noisey_image)
#plt.colorbar()
#plt.show()
#plt.bar

# Helper functions for the updates

def fast_convolve(matrix, weights):
    h, w = matrix.shape
    conv = np.zeros((h, w))
    matrix = np.pad(matrix,((1, 1), (1, 1)),mode='constant')
    for i in range(h):
        for j in range(w):
            conv[i, j] = (matrix[i:i + 3,j:j + 3] * weights).sum()
    return conv
    
def fast_convolve_double(matrix, weights):
    h, w = matrix.shape
    conv = np.zeros((h, w))
    weights2 = weights**2
    matrix = np.pad(matrix,((1, 1), (1, 1)),mode='constant')
    for i in range(h):
        for j in range(w):
            conv[i, j] = (matrix[i:i + 3,j:j + 3] * weights2).sum()
    return conv    
        
def prob(img, mu, sigma):
#    # this is an optimisation
#    res = np.exp(-((img - mu)**2 / (2 * sigma**2)))
#    res = res / (sigma * np.sqrt(2 * np.pi))
#    return res
    return sigma*img*mu

def logprob(img, sigma):
    log_plus = prob(img, 1, sigma)
    log_minus = prob(img, -1, sigma)
    return log_plus - log_minus

#def LB(img, mu, sigma):
#    L_plus = prob(img, 1, sigma)
#    L_minus = prob(img, -1, sigma)
#    a = fast_convolve(mu, weights) 
#    b = fast_convolve_double(mu, weights)
#    lb = np.sum(b) + np.sum(a) + np.sum(0.5*(L_plus - L_minus))
#    return np.log(lb)
                       
def update(mu, weights, L):
    weighted_mu = fast_convolve(mu, weights)
    new_mu = np.tanh(weighted_mu + 0.5 * L)
    return new_mu


# Set model paramters and run updates

steps = np.arange(10)
mu = noisey_image

weights = np.ones((3, 3))
weights[1,1] = 0
decay = 1

L_1 = logprob(noisey_image, 0.0001)
L_2 = logprob(noisey_image, 0.1)
L_3 = logprob(noisey_image, 1.0)
L_4 = logprob(noisey_image, 3.)


#fig = plt.figure(figsize=(12, 8), facecolor='white')
fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(2, 3)
#plt.show(block=False)


def draw(mu_1, mu_2, mu_3, mu_4):
    for ax in (ax_1, ax_2, ax_3, ax_4, ax_5, ax_6):
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.gray()
    ax_1.set_title('Original')
    ax_1.imshow(image_scaled)
    
    ax_4.set_title('Corrupted')
    ax_4.imshow(noisey_image)
    
    ax_2.set_title('Sigma = 0.0001')
    ax_2.imshow(mu_1)
    
    ax_3.set_title('Sigma = 0.1')
    ax_3.imshow(mu_2)
    
        
    ax_5.set_title('Sigma = 1.0')
    ax_5.imshow(mu_3)
    
    ax_6.set_title('Sigma = 3.0')
    ax_6.imshow(mu_4)
    
    
mu_1 = np.zeros_like(noisey_image.copy())
mu_2 = np.zeros_like(noisey_image.copy())
mu_3 = np.zeros_like(noisey_image.copy())
mu_4 = np.zeros_like(noisey_image.copy())


for i in steps:
    print i
    mu_1 = (1 - decay)*mu_1 + (decay * update(mu_1, weights, L_1))
    mu_2 = (1 - decay)*mu_2 + (decay * update(mu_2, weights, L_2))
    mu_3 = (1 - decay)*mu_3 + (decay * update(mu_3, weights, L_3))
    mu_4 = (1 - decay)*mu_4 + (decay * update(mu_4, weights, L_4))
draw(mu_1, mu_2, mu_3, mu_4)
plt.savefig('ising.pdf')
plt.show()
    
#def draw(mu, objective):
#    for ax in (ax_1, ax_2, ax_3):
#        ax.axes.get_xaxis().set_visible(False)
#        ax.axes.get_yaxis().set_visible(False)
#    plt.gray()
#    ax_1.set_title('Original')
#    ax_1.imshow(image_scaled)
#    ax_2.set_title('Corrupted')
#    ax_2.imshow(noisey_image)
#    ax_3.set_title('Posterior')
#
#    ax_3.imshow(mu)
#    
#    ax_4.set_title('L')
#    ax_4.scatter(steps[:len(objective)], objective, color='b')
#    ax_4.set_xlim([0, len(steps)])
#    ax_4.set_xlabel('Number iterations')
#    ax_4.set_ylabel('Lower bound log ML')
#    plt.draw()
#    plt.pause(0.1)

#objective = []
#for i in steps:
#    mu = (1 - decay)*mu + (decay * update(mu, weights, L))
#    lb = LB(noisey_image, mu, sigma)
#    objective.append(lb)
#    draw(mu, objective)

# plt.gray()
# plt.imshow(image_scaled)
# plt.colorbar()
# plt.show()

