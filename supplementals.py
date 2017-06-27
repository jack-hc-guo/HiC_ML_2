# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:14:59 2017

@author: Jack_2
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy
import numpy as np

def plotHeatMap(features):
#    print features
#    features = [math.log10(p) for p in features]
#    square_size = int(math.sqrt(len(features)))
#    tmp = []
#    for i in range(0, len(features), square_size):
#        if len(features[i:i+square_size]) != 1:
#            tmp.append(features[i:i+square_size])
#    print tmp, features[-1]
#    features = [abs(feat) for feat in features]
    win_size = int(math.sqrt(len(features)))
    tmp = np.array(features[:-1]).reshape(win_size, win_size)
    df = pd.DataFrame(data = tmp)
    ax = plt.axes()
    sns.heatmap(df, ax=ax, cmap="gist_heat_r")
    ax.set_title("Feature Importance of Size-"+str(np.shape(tmp)[0])+" Window")
#    plt.cla()
    
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2, intercept, slope

''' We use only the distance as features '''
def distanceOnlyFeature(train_feat, bp = True):
    if bp:
        distanceOnly = [[feat[-2]] for feat in train_feat]
    else:
        distanceOnly = [[feat[-1]] for feat in train_feat]
    return distanceOnly

'''Generate window vs rc MSE  as we vary window size'''
def windowVsRC(X, window, rc):
    l1 = plt.scatter(X, window, color="red")
    l2 = plt.scatter(X, rc, color="blue")
    plt.xlabel("Window Size")
    plt.ylabel("MSE")
    plt.title("MSE vs Window Size for window or row+column")
    plt.legend([l1, l2], ["Full window", "Row + column"]) 