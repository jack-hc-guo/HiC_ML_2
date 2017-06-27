# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:23:26 2017

@author: Jack_2
"""
import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
#import gridspec
def read():
    chr17_labels = "GM12878.HindIII_1_2.chr17.test.RF.tsv"
    chr18_labels = "GM12878.HindIII_1_2.chr18.test.RF.tsv"
    f = open(chr17_labels, "r")
    f2 = open(chr18_labels, "r")
    chr17_lines = f.readlines()
    chr18_lines = f2.readlines()
    chr17_lab = []
    chr18_lab = []
    for l in chr17_lines:
        l = l.strip().split('\t')
        chr17_lab.append(int(l[0]))
        
    for l in chr18_lines:
        l = l.strip().split('\t')
        chr18_lab.append(int(l[0]))
        
    f.close()
    f2.close()
    return chr17_lab, chr18_lab

def getViolinplots(c17_l, c18_l):
    df = pd.DataFrame(data = c17_l, columns = ['chr 17 Labels'] )
    df2 = pd.DataFrame(c18_l, columns=['chr 18 labels'])
#    gs = gridspec.GridSpec(1, 2)
    ax1 = sns.violinplot(data = df)
#    ax2 = sns.violinplot(data = df2)

def getaxHist(c17, c18, ctmp=[], l1="Chr17", l2="Chr18", l3=""):
    print len(c17), len(c18)
    bins = [i for i in range(0, 50, 1)]
    plt.hist((c17,c18, ctmp), bins, rwidth=0.95, label=(l1,l2,l3))
    #plt.hist(c18_labels, bins, alpha=0.5, label='Chr18 (Test labels)')
    plt.yscale('symlog')    
    plt.legend(loc='upper right')
    #plt.ylim([0, 1000])
    plt.xlim([0, 40])
    plt.xlabel("Interaction Frequency")
    plt.ylabel("Counts")
    plt.title("Count frequency histogram for all data")
    plt.show()
    
def conv2dict(lab):
    tmp = {}
    for i in lab:
        if i in tmp:
            tmp[i] = tmp[i] + 1
        else:
            tmp[i] = 1
#    s = sum([tmp[i] for i in tmp])
#    print s
    return tmp

if __name__=='__main__':
    c17 = "Fullfeatures_RF_BP_DISTANCE/train_labels.txt"
    c18 = "Fullfeatures_RF_BP_DISTANCE/test_labels.txt"
    c17_labels, c18_labels = json.load(open(c17)), json.load(open(c18))
    #c17_labels, c18_labels = read()
    #c17_dict = conv2dict(c17_labels)
    #c18_dict = conv2dict(c18_labels)
    #plt.hist(c17_dict.items(), c18_dict.items(), alpha=0.7, label=['x', 'y'])
    #
    #print c17_dict
    #print max(c18_labels)
    #
    #getViolinplots(c17_labels, c18_labels)
    getaxHist(c17_labels, c18_labels)
    #print len(c17_labels), len(c18_labels)