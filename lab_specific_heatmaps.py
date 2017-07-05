# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 13:39:02 2017

@author: Jack_2
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def splitByLabels(feat, lab, prefix, write=False):
    if prefix == "Full":
        root = "Fullfeatures_RF_BP_DISTANCE/"
    else:
        root = "Subfeatures_RF_BP_distance/"
        
    tmpDict = {}
    for instance in zip(feat, lab):
        f, label = instance[0], instance[1]
        if label in tmpDict:
            tmpDict[label].append(f)
        else:
            tmpDict[label] = [f]
    if write:
        outfile = root+"/label_heatmaps/log.txt"
        with open(outfile, "w") as wf:
            wf.write("Counts"+'\n')
            wf.write("========================"+'\n')
            for k in tmpDict.keys():
                count = len(tmpDict[k])
                wf.write(str(k)+'\t'+str(count)+'\n')
    
    return tmpDict
    
def plotHeatMap(inp_features, k, mode, prefix):
    if prefix == "Full":
        root = "Fullfeatures_RF_BP_DISTANCE/"
    else:
        root = "Subfeatures_RF_BP_distance/"
        
    fig,ax = plt.subplots()
    win_size = int(math.sqrt(len(inp_features)))
    tmp = np.array(inp_features).reshape(win_size, win_size)
    df = pd.DataFrame(data = tmp)
    ax = plt.axes()
    sns.heatmap(df, ax=ax, cmap="gist_heat_r")
    ax.set_title("Features "+mode+" for label "+str(k))
    plt.savefig(root+"/label_heatmaps/Features_"+mode+"_for_label "+str(k)+".jpg",format="jpg",dpi=300.0)
    
def avgHeatMaps(dic, prefix):
#    print dic[1][:2]
    for k in dic.keys():
#        print np.shape(np.array(dic[k]))
        avgHeatMap = [float(c)/len(dic[k]) for c in np.sum(np.array(dic[k]), axis=0)]
        print len(avgHeatMap)
        plotHeatMap(avgHeatMap, k, "avg", prefix)

def varHeatMaps(dic, prefix):
#    print dic[1][:2]
    for k in dic.keys():
#        print np.shape(np.array(dic[k]))
        avgHeatMap = np.var(np.array(dic[k]), axis=0)
        print len(avgHeatMap)
        plotHeatMap(avgHeatMap, k, "var", prefix)