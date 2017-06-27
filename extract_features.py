# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:39:59 2017

@author: Jack_2
"""
import json
import time
import random
import math
import sys
''' Parses counts and store it in double indexed dictionary '''
def parseData(fname, flag, chrnum):
    with open(fname, "r") as f:
        lines = f.readlines()
        if flag == "features":
            [rowsize, colsize] = [int(lines[2].split('\t')[2]), int(lines[2].split('\t')[4].strip())]
#            print rowsize*colsize            
            sparse_matrix = {}
            for l in lines[3:]:
                l = l.strip().split('\t')
                if chrnum == 17:
                    count, i, j = int(l[0]), int(l[2]), int(l[4])
                else:
                    count, i, j = int(l[0]), int(l[2])+1, int(l[4])+1
                sparse_matrix[i, j] = count
		sparse_matrix[j, i] = count
#            print len(sparse_matrix)
#            print "================"
        elif flag == "labels":
            [rowsize, colsize] = [0, 0]
            sparse_matrix = {}
            for l in lines:
                l = l.strip().split('\t')
                if chrnum == 17:
                    count, i, j = int(l[0]), int(l[2]), int(l[4])
                else:
                    count, i, j = int(l[0]), int(l[2])+1, int(l[4])+1
                sparse_matrix[i, j] = count
		sparse_matrix[j, i] = count
    return sparse_matrix, [rowsize, colsize]
    
''' Parses restriction fragment file and stores the end position to RF# in dictionary ''' 
def parseRFSizes(fname, flag):
    with open(fname, "r") as f:
        lines = f.readlines()
        if flag == "train":
            mydict = {}
            for line in lines:
                line = line.split()
                if line[0] == "chr17":
                    mydict[line[3]] = line[2]
        elif flag == "test":
            mydict = {}
            for line in lines:
                line = line.split()
                if line[0] == "chr18":
                    mydict[line[3]] = line[2]
    return mydict

''' Returns the feature and labels for training dataset '''             
def getFeatandLabels(sm, ps, size, sm_size, digestFrags, sm_test):
    totalLabels, totalFeatures = [], []
    [colsize, rowsize] = sm_size
    
    keys = {}
    ''' Enforces that we have size # of features '''
    while len(totalFeatures) < size:
        sizegap = 1000000000
        ''' Enforces that the distance between RF is smaller than 1 million bps '''
        ''' Possiblility of duplicating keys --> might want to fix later '''
        while sizegap > 1000000:
            rf1 = random.randint(1, rowsize-1000)
            if rf1+1000 >= colsize:
                if rf1 +1 != colsize:
                    rf2 = random.randint(rf1+1, colsize)
                else: 
                    rf2 = colsize
            else:
                rf2 = random.randint(rf1+1, rf1+1000)
            
            if (rf1, rf2) not in keys:
                rkey = (rf1, rf2)
                sizegap = abs(int(digestFrags[str(rf1)])-int(digestFrags[str(rf2)]))
                keys[rf1, rf2] = sizegap
            else:
                sizegap = 1000000000
        
        ''' Append the label '''
        if rkey in sm_test:
            totalLabels.append(sm_test[rkey])
#            print rkey, sm_test[rkey]
        else:
            totalLabels.append(0)
#            print rkey, 0
        
        tmp = []
#        print rf1, rf2, sizegap
#        print len(totalFeatures)
        ''' Takes all points within window size 2*ps+1 * 2ps+1 '''
        for i in range(rf1-ps, rf1+ps+1):
            for j in range(rf2-ps, rf2+ps+1):
                if i <= 0 or i >= rowsize: 
                    tmp.append(-1)
                elif j <= 0 or j >= colsize:
                    tmp.append(-1)
                else:
                    tmpkey = (i, j)
                    if tmpkey in sm:
                        tmp.append(sm[tmpkey])
                    else:
                        tmp.append(0)
        tmp.append(sizegap)
        tmp.append(abs(rf2-rf1))
        totalFeatures.append(tmp)
    return totalLabels, totalFeatures

def getReflectionFeatAndLabels(sm, ps, size, sm_size, digestFrags, sm_test):
    upperLabels, upperFeatures = [], []
    lowerLabels, lowerFeatures = [], []
    [colsize, rowsize] = sm_size
    
    keys = {}
    ''' Enforces that we have size # of features '''
    while len(upperFeatures) < size:
        sizegap = 1000000000
        ''' Enforces that the distance between RF is smaller than 1 million bps '''
        ''' Possiblility of duplicating keys --> might want to fix later '''
        while sizegap > 1000000:
            rf1 = random.randint(1, rowsize-1000)
            if rf1+1000 >= colsize:
                if rf1 +1 != colsize:
                    rf2 = random.randint(rf1+1, colsize)
                else: 
                    rf2 = colsize
            else:
                rf2 = random.randint(rf1+1, rf1+1000)
            
            if (rf1, rf2) not in keys:
                rkey = (rf1, rf2)
                sizegap = abs(int(digestFrags[str(rf1)])-int(digestFrags[str(rf2)]))
                keys[rf1, rf2] = sizegap
            else:
                sizegap = 1000000000
        
        ''' Append the label '''
        if rkey in sm_test:
            upperLabels.append(sm_test[rkey])
            lowerLabels.append(sm_test[rkey])
#            print rkey, sm_test[rkey]
        else:
            upperLabels.append(0)
            lowerLabels.append(sm_test[rkey])
#            print rkey, 0
        
        upTmp, lowTmp = [], []
#        print rf1, rf2, sizegap
#        print len(totalFeatures)
        ''' Takes all points within window size 2*ps+1 * 2ps+1 '''
        for i in range(rf1-ps, rf1+ps+1):
            for j in range(rf2-ps, rf2+ps+1):
                if i <= 0 or i >= rowsize: 
                    upTmp.append(-1)
                    lowTmp.append(-1)
                elif j <= 0 or j >= colsize:
                    upTmp.append(-1)
                    lowTmp.append(-1)
                else:
                    tmpkey = (i, j)
                    if tmpkey in sm:
                        upTmp.append(sm[tmpkey])
                    else:
                        upTmp.append(0)
                        
                    tmpkey_2 = (j, i)
                    if tmpkey_2 in sm:
                        lowTmp.append(sm[tmpkey_2])
                    else:
                        lowTmp.append(0)
        
        upTmp.append(sizegap)
        upTmp.append(abs(rf2-rf1))
        upperFeatures.append(upTmp)
        
        lowTmp.append(sizegap)
        lowTmp.append(abs(rf2-rf1))
        lowerFeatures.append(lowTmp)
    return upperLabels, upperFeatures, lowerLabels, lowerFeatures

    
if __name__=='__main__':
    ''' Output files '''
    train_features = "train_features.txt"
    train_labels = "train_labels.txt"
    train_test_features = "test_features.txt"
    train_test_labels = "test_labels.txt"
    
    ''' Digestion file '''
    fdigest = "Digest_hg19_HindIII.bed"
    
    ''' Training files '''
    train_feat = "GM12878.HindIII_1_2.chr17.train.100.RF.tsv"
    train_label = "GM12878.HindIII_1_2.chr17.test.RF.tsv"
    
    test_feat = "GM12878.HindIII_1_2.chr18.train.100.RF.tsv"
    test_label = "GM12878.HindIII_1_2.chr18.test.RF.tsv"
    
    flag = "Reflect"
    start_time = time.time()    
    print >> sys.stderr, "Parsing fragments...",
    digestFrags_train = parseRFSizes(fdigest, "train") 
    digestFrags_test = parseRFSizes(fdigest, "test")
    print >> sys.stderr, "Done"
    
    print >> sys.stderr, "Reading counts...",
    sm, sm_size = parseData(train_feat, "features", 17) 
    sm_labels, sm_labels_size = parseData(train_label,"labels", 17)
    
    sm_test_feat, sm_test_size = parseData(test_feat, "features", 18)
    sm_test_labels, sm_test_labels_size = parseData(test_label, "labels", 18)
    print >> sys.stderr, "Done"
    
    if flag != "Reflect":
        print >> sys.stderr, "Extracting normal features...",
        train_lab, train_feat = getFeatandLabels(sm, 3, 200000, sm_size, digestFrags_train, sm_labels)
        test_lab, test_feat = getFeatandLabels(sm_test_feat, 3, 100000, sm_test_size, digestFrags_test, sm_test_labels)
        print >> sys.stderr, "Done"
        print len(train_lab), len(train_feat)
        print len(test_lab), len(test_feat)
        with open(train_features, "w") as f:
            json.dump(list(train_feat), f)
        with open(train_labels, "w") as f:
            json.dump(list(train_lab), f)
            
        with open(train_test_features, "w") as f:
            json.dump(list(test_feat), f)
        with open(train_test_labels, "w") as f:
            json.dump(list(test_lab), f)
    
    if flag == "Reflect":
        print >> sys.stderr, "Extracting normal and reflected feaures...",
        upper_train_lab, upper_train_feat, lower_train_lab, lower_train_feat = getReflectionFeatAndLabels(sm, 1, 1, sm_size, digestFrags_train, sm_labels)
        upper_test_lab, upper_test_feat, lower_test_lab, lower_test_feat = getReflectionFeatAndLabels(sm, 1, 1, sm_size, digestFrags_train, sm_labels)
        print >> sys.stderr, "Done"
        print len(upper_train_lab), len(upper_train_feat), len(lower_train_lab), len(lower_train_feat)
        print len(upper_test_lab), len(upper_test_feat), len(lower_test_lab), len(lower_test_feat)
        print upper_train_lab, upper_train_feat, lower_train_lab, lower_train_feat
        print upper_train_lab, upper_train_feat, lower_train_lab, lower_train_feat
        
        with open("upper_train_features.txt", "w") as f:
            json.dump(list(upper_train_feat), f)
        with open("upper_train_labels.txt", "w") as f:
            json.dump(list(upper_train_lab), f)
        with open("lower_train_features.txt", "w") as f:
            json.dump(list(lower_train_feat), f)
        with open("lower_train_labels.txt", "w") as f:
            json.dump(list(lower_train_lab), f)
            
        with open("upper_test_features.txt", "w") as f:
            json.dump(list(upper_test_feat), f)
        with open("upper_test_labels.txt", "w") as f:
            json.dump(list(upper_test_lab), f)
        with open("lower_test_features.txt", "w") as f:
            json.dump(list(upper_test_feat), f)
        with open("lower_test_labels.txt", "w") as f:
            json.dump(list(upper_test_lab), f)
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    
    print "Finished"
