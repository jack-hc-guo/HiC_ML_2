# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:45:32 2017

@author: Jack_2
"""
import numpy as np
import pickle
import json
import supplementals as sup
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
''' Reverse propagation '''
def rprop(inp, weights, c):
    init_rel, X_mat = fprop(inp, weights)
    X_mat = list(reversed(X_mat))
    weights = list(reversed(weights))
    relevance = []
    relevance.append(init_rel)
    for ind, w in enumerate(weights):
        neur_out = X_mat[ind]
        Z_tot = np.dot(neur_out, w)
        tmpRel = calrel(w, neur_out, relevance[ind], Z_tot)
        relevance.append(tmpRel)
    
    inp_rel = relevance[-1]
    inp_rel = [i/sum(inp_rel) for i in inp_rel]
#    print sum(inp_rel)
    
#    inp_rel = [math.log10(abs(i)) for i in inp_rel]
#    print inp_rel
#    sup.plotHeatMap(inp_rel)
    if (c%10000) == 0:
        print str(sum(relevance[-1]))[:10], init_rel[0]
    return inp_rel
#    print relevance

''' Calculates relevance for certain layer '''
def calrel(weights, neuron_out, rele, total):
    R_tmp = []
    for i in zip(weights, neuron_out):
        R_i = 0.0
        for j in zip(i[0], total, rele):
            R_i = R_i + (float(i[1])*j[0]/j[1])*j[2]
        R_tmp.append(R_i)
    return R_tmp
    
''' Forward propagation, calculates final output and neuron output at each layer '''
def fprop(inp, weights):
    X_mat = []
    X_mat.append(inp)
    for w in weights:
        inp = np.dot(inp, w)
        X_mat.append(inp)
    return X_mat[-1], X_mat[:-1]

def addMatrices(mat1, mat2):
#    print mat1, mat2
    return [i[0]+i[1] for i in zip(mat1,mat2)]

def plotHeatMap(inp_features, rel_features):
    fig,ax = plt.subplots()
    win_size = int(math.sqrt(len(inp_features)))
    tmp = np.array(inp_features[:-1]).reshape(win_size, win_size)
    df = pd.DataFrame(data = tmp)
    ax = plt.axes()
    sns.heatmap(df, ax=ax, cmap="gist_heat_r")
    ax.set_title("Features of Size-"+str(np.shape(tmp)[0])+" Window")
    
    fig2,ax2 = plt.subplots()
    win_size = int(math.sqrt(len(rel_features)))
    tmp = np.array(rel_features[:-1]).reshape(win_size, win_size)
    df = pd.DataFrame(data = tmp)
    ax2 = plt.axes()
    sns.heatmap(df, ax=ax2, cmap="gist_heat_r")
    ax2.set_title("Feature Relevance of Size-"+str(np.shape(tmp)[0])+" Window")
    
if __name__=='__main__':
    weights = pickle.load(open("NN_weights_test_0.json"))
    inp = json.load(open("Fullfeatures_RF_BP_DISTANCE/train_features.txt"))
    print len(inp)
#    input1 = inp[2][:-2]+[inp[2][-1]]

    avgPlot = False
    flag = False
    if avgPlot:
        init = []
        for ind, i in enumerate(inp):
            i = i[:-2]+[i[-1]]
            if ind == 0:
                init = rprop(i, weights, ind)
            else:
                init = addMatrices(init, rprop(i, weights, ind))
        init = [float(weight)/len(inp) for weight in init]
        init = [abs(w) for w in init]
        json.dump(list(init), open("feature_relevance.json", "w"))
        sup.plotHeatMap(init)
    elif flag:
        init = []
        for ind, i in enumerate(inp):
            i = i[:-2]+[i[-1]]
            if ind == 0:
                init = i
            else:
                init = addMatrices(init, i)
        init = [float(weight) for weight in init]
        sup.plotHeatMap(init)
    else:
#        input1 = [[1 for i in range(7)] for j in range(7)]
        input1 = [1 for i in range(50)]
        print input1
#        ii = 17
#        input1 = inp[ii][:-2]+[inp[ii][-1]]
        rel = rprop(input1, weights, ind)
        rel = [abs(w) for w in rel]
        print rel
        plotHeatMap(input1, rel)
          