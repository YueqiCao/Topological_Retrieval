# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:01:26 2021

@author: Administrator
"""

import numpy as np
import main_algorithm as mydist
import gudhi as gd
import DI_distance as NS

np.random.seed(1) 
num_points = 32 # change this parameter to 64, 128, ... to get different number of points
low_end = 0.01 # change this parameter to 0.1, 1, ... to get different scale 
high_end = 10 # change this parameter to 100, 1000, ... to get different scale

Xx = np.random.uniform(low = low_end, high = high_end, size = num_points)
Xy = Xx + np.random.uniform(low = low_end, high = high_end, size= num_points)
X = np.transpose(np.vstack((Xx, Xy)))

Yx = np.random.uniform(low = low_end, high = high_end, size = num_points)
Yy = Yx + np.random.uniform(low = low_end, high = high_end, size= num_points)
Y = np.transpose(np.vstack((Yx, Yy)))

ori = gd.bottleneck_distance(X, Y)

# DI distance

shiftInv, bottleneckDistance, minBottleneckDistance = NS.myDistance(X, Y, 100)

# Don distance

X = [tuple(x) for x in X.tolist()]
Y = [tuple(x) for x in Y.tolist()]
DShift = mydist.shifted_bottleneck_distance(X,Y, analysis = False)

print('Don distance is ', DShift)

