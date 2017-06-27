# -*- coding: utf-8 -*-
"""
Created on Tue May 02 11:30:39 2017

@author: Jack_2
"""
import begin
import extract_features
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor as MLPR
import time
def getFeatures(sm_feats, key, size):
        [rowsize, colsize] = size
        [rf1, rf2] = key        
        tmp = []
        ps = 3
        ''' Takes all points within window size 2*ps+1 * 2ps+1 '''
        for i in range(rf1-ps, rf1+ps+1):
            for j in range(rf2-ps, rf2+ps+1):
                if i <= 0 or i >= rowsize: 
                    tmp.append(-1)
                elif j <= 0 or j >= colsize:
                    tmp.append(-1)
                else:
                    tmpkey = (i, j)
                    if tmpkey in sm_feats:
                        tmp.append(sm_feats[tmpkey])
                    else:
                        tmp.append(0)
#        tmp.append(sizegap)
        tmp.append(abs(rf2-rf1))
        return tmp

def train_and_predict(train_features, train_labels, size, sm_test_feats, logfile classifier="randforest", num=[50]):
#    [rowsze, colsze] = size
    rowsze, colsze = 5, 5
    if classifier == "randforest":
        ''' Generate weights for trees and number of trees '''
        ''' Change to random forest regressor ''' 
        clf = RandomForestRegressor(n_estimators=(num))
    elif classifier == "MLPR":
        layers = [len(train_features[0]), len(train_features[0])-1]
        logfile.write("ARCHITECTURE:" + layers)
        clf = MLPR(hidden_layer_sizes=(layers), activation='relu', solver='adam')
        
    print "training..."
    clf.fit(train_features, train_labels)
    wf = open("chr18_ud_NN.predict.RF.tsv", "a")
    print "predicting..."
    for i in range(1, rowsze+1):
        for j in range(1, colsze+1):
		if j > i:
#            print i,j
#            if (i, j) in sm_test_feats:
#                print i,j
#                print sm_test_feats[(i,j)]
            		feat_vector = getFeatures(sm_test_feats, [i,j], [rowsze, colsze])
           		res = clf.predict([feat_vector])
			if res[0] > 0.001:
	    			s = str(res[0])+'\t'+"18"+'\t'+str(i)+'\t'+"18"+'\t'+str(j)+'\n'
	    			wf.write(s)
    wf.close()
            #if res[0] != 0:
            #    print res, "18", i, "18", j

def returnData(flag = "Partial", distance = "bp"):
    if flag == "Full":
        root = "Fullfeatures_RF_BP_DISTANCE/"
    else:
        root = "Subfeatures_newindex/"
        
    train_features = root+"train_features.txt"
    train_labels = root+"train_labels.txt"
    
    train_feat = json.load(open(train_features))
    train_lab = json.load(open(train_labels))
    if distance == "bp":
        train_feat = [feat[:-1] for feat in train_feat]
    elif distance == "rf":
        train_feat = [feat[:-2]+[feat[-1]] for feat in train_feat]
    else:
        train_feat = [feat[:-2] for feat in train_feat]
    print "training instances: ", len(train_feat)
    return train_feat, train_lab

def calTrueMat(labels, fname):
    contacts = labels.values()
    total_contacts = sum(contacts)
    with open(fname, "r") as f:
        lines = f.readlines()
        wf = open("normalized_chr18.test.RF.tsv", "a")
	print "Writing..."
        for l in lines:
            l = l.split('\t')
	    #print l
            l[0] = str(float(l[0])*1000000/total_contacts)
            wf.write('\t'.join(l))
        wf.close()

@begin.start     
def run(distance = "rf", classifier = "MLPR", log = "log_"+time.strftime("%d%m%Y")+".txt"):
    test_feat = "GM12878.HindIII_1_2.chr18.train.100.RF.tsv"
    test_lab = "GM12878.HindIII_1_2.chr18.test.RF.tsv"
    log_out = open(log, "w")
    log_out.write("CONFIGURATION"+'\n')
    log_out.write("DISTANCE:"+distance+'\n')
    log_out.write("CLASSIFIER:"+classifier+'\n')
    train_feat, train_lab = returnData(flag = "Full", distance = distance)
    log_out.write("TRAINING INSTANCES:"+str(len(train_feat))+'\n')
    start_time = time.time()
    log_out.write("START TIME:"+str(start_time)+'\n')
    sm_test_feats, sm_test_size = extract_features.parseData(test_feat, "features", 18)
    sm_test_labs, sm_lab_size = extract_features.parseData(test_lab, "labels", 18)
    
    train_and_predict(classifier = classifier, train_feat, train_lab, sm_test_size, sm_test_feats, log_out)
#    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    log_out.write("END TIME:"+str(time.time() - start_time)/60)+'\n')
    log_out.close()
#    calTrueMat(sm_test_labs, test_lab)   
    
