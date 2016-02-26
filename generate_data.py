# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:21:14 2016

@author: hughsalimbeni
"""

from utils import generate_data
import pickle
import numpy as np
import matplotlib.pyplot as plt

mus = np.reshape((3., 3., -3, 3, 0, -3), (3, 2))
Sigmas = np.reshape((1., 0., 0., 1., 1., 0., 0., 1., 2., 0., 0., 0.5), (3, 2, 2))

pis_ = np.array((2., 1., 1.,))
pis = pis_/np.sum(pis_)

params = (pis, mus, Sigmas)

data_1 = generate_data(50, params)[0]
plt.scatter(data_1[:, 0], data_1[:, 1])

data_2 = generate_data(500, params)[0]
plt.scatter(data_2[:, 0], data_2[:, 1])

pickle.dump((data_1, data_2), open( "data.p", "wb" ))