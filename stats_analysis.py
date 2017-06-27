# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:30:20 2017

@author: Jack_2
"""
import json
import matplotlib.pyplot as plt

def labelDistribution(labels, tag):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim([0,40])
    ax1.hist(labels,bins=100)
    ax1.set_title("Label distribution "+tag+" dataset")
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Counts") 
    fig1.savefig(tag+"_label.jpg")

def distanceDistribution(distances, tag):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(distances, bins=100)
    ax2.set_title("Distance distribution of "+tag+" dataset")
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Counts") 
    fig2.savefig(tag+"_distance.jpg")
    
def distanceOnlyFeature(feats, flag):
    if flag == "rf":
        distanceOnly = [feat[-1] for feat in feats]
    elif flag == "bp":
        distanceOnly = [feat[-2] for feat in feats]
    return distanceOnly

def split(distances, labs, tag):
    print len(distances), len(labs)
    zero_dist = []
    non_zero_dist = []
#    non_zero_labels = []
    for inst in zip(distances, labs):
        label = inst[1]
        dist = inst[0]
#        print label, dist
        if label == 0:
            zero_dist.append(dist)
        else:
            non_zero_dist.append(dist)
#            non_zero_labels.append(label)
#    print max(non_zero_labels), min(non_zero_labels)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.hist(zero_dist, bins=100)
    ax3.set_title("Distance distribution of zero-label for "+tag+" dataset")
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Counts")
#    fig3.savefig(tag+"_zero_label_distri.jpg")
    
    fig4 = plt.figure()
    ax3 = fig4.add_subplot(111)
    ax3.hist(non_zero_dist, bins=100)
    ax3.set_title("Distance distribution of nonzero-label for "+tag+" dataset")
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Counts")
#    fig4.savefig(tag+"_non_zero_label_distri.jpg")
            
def returnData(flag = "Partial"):
    if flag == "Full":
        root = "Fullfeatures_RF_BP_DISTANCE/"
    else:
        root = "Subfeatures_RF_BP_distance/"
        
    train_features = root+"train_features.txt"
    train_labels = root+"train_labels.txt"
    train_test_features = root+"test_features.txt"
    train_test_labels = root+"test_labels.txt"
    train_feat = json.load(open(train_features))
    train_lab = json.load(open(train_labels))
    
    test_feat = json.load(open(train_test_features))
    test_lab = json.load(open(train_test_labels))
    return train_feat, train_lab, test_feat, test_lab

def scatterDistanceVSLabel(distance, labels):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(distance, labels)
    ax2.set_title("IF vs Distance")
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("IF") 
#    fig2.savefig(tag+"_distance.jpg")
    
if __name__=='__main__':
    train_feat, train_lab, test_feat, test_lab = returnData("Full")
    train_distance_bp = distanceOnlyFeature(train_feat, flag="bp")
    train_distance_rf = distanceOnlyFeature(train_feat, flag="rf")
    test_distance_bp = distanceOnlyFeature(test_feat, flag="bp")
    test_distance_rf = distanceOnlyFeature(test_feat, flag="rf")
    train_feat = [feat[:-2] for feat in train_feat]
    test_feat = [feat[:-2] for feat in test_feat]
#    print len(train_feat), len(train_test_feat)
#    print len(distanceOnly), len(train_lab)
#    scatterDistanceVSLabel(train_distance_rf, train_lab)
    split(train_distance_rf, train_lab, "training")
#    split(train_test_feat, train_test_lab, "testing")
#    labelDistribution(train_lab, "training")
#    distanceDistribution(train_distance_rf, "training")
#    labelDistribution(train_test_lab)
#    distanceDistribution(train_test_distanceOnly)
    
    