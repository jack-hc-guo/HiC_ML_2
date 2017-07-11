# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:37:19 2017

@author: Jack_2
"""

''' Preprocessing '''
from sklearn.model_selection import cross_val_score, train_test_split as tts
from sklearn.utils import shuffle

''' Classifiers '''
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
import ae
from theano import tensor as T

''' Analysis '''
from sklearn.metrics import precision_score as ps, recall_score as rs, mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn import preprocessing as preproc

''' Supplementals '''
import math
#import matplotlib.pyplot as plt
#import violinPlots as vp
import sys
#import iterators as it
#import supplementals as sup
import json
import numpy as np
import time
#import lab_specific_heatmaps as lsh
import argparse

allCV = []
allMSE = []
window = []
rc = []
''' Cross Validation (5-Fold default) '''
def CV(clf, train, train_label, test, test_label, flag = "window", k=5, 
       getDistr = False, verbose = False):  
     trainscores, valscores, testscores = [], [], []
     train=np.asarray(train)
     train_label=np.asarray(train_label)
     if verbose:
         print "Cross Validation"
         print "================"
         print '     ', "Train MSE   ", "Val MSE   ", "Test MSE   "
     kf=KFold(n_splits=5,shuffle=True)
     for i,(train_index,val_index) in enumerate(kf.split(train)):
         clf.fit(train[train_index],train_label[train_index])
         train_mse=mse(train_label[train_index],clf.predict(train[train_index]))
         val_mse=mse(train_label[val_index],clf.predict(train[val_index]))
         test_mse=mse(test_label,clf.predict(test))
         if getDistr:
             vp.getaxHist(train_label[train_index], train_label[val_index], test_label, l1="Train MSE Labels", l2="Val MSE labels", l3="Test MSE labels")
         if verbose:
             print "fold",i
             print "mse:",train_mse,val_mse,test_mse
         trainscores.append(train_mse)
         valscores.append(val_mse)
         testscores.append(test_mse)
     if flag == "window":
         window.append(np.mean(testscores))
     elif flag == "rc":
         rc.append(np.mean(testscores))
     return np.mean(trainscores), np.mean(valscores), np.mean(testscores)
     sys.exit()    

def count_fun(arr, threshold):
    c = 0
    for i in arr:
        if i == threshold:
            c += 1
    return c
    
''' Train and prediction, also returns cross validation score with k = 5 '''
def trainAndPredict(train, train_label, test, test_label, classifier = "randforest", num=[10], flag="window",
                    calCV = True, fullPred = False, getRegression = False, getFeatImportance = False,
                    verbose = False, initLrnRate = 1e-3, plot=False):
    print classifier+' architecture'
    print '------------------------'
    if classifier == "randforest":
        ''' Generate weights for trees and number of trees '''
        ''' Change to random forest regressor '''
        depth = None
        print "num of trees:", num[0]
        print "max depth:",depth
        clf = RandomForestRegressor(n_estimators=num[0], verbose = True, max_depth = depth)
    if classifier == "RBM":
        clf = BernoulliRBM(n_components=100, learning_rate = 1e-7, n_iter = 3, verbose = True)
        train = clf.fit_transform(train, train_label)
        print len(train)
#        print clf.
        sys.exit()
        classifier = "MLPR"
    if classifier == "ae":
        act_fn = T.nnet.sigmoid
        out_fn = act_fn
        a = ae.AutoEncoder(np.array(train), 50, act_fn, out_fn)
        a.train(2)
        sys.exit()
    if classifier == "MLPR":
        print "lul"
        a_fns = ["identity", "logistic", "tanh", "relu"]
        solvers = ["adam", "sgd", "lbfgs"]
        lrn_mode = ["constant", "adaptive", "invscaling"]
        if num[0] == 0:
            num[0] = 1
        layers = num
        act_fn = a_fns[3]
        solv = solvers[0]
        lrn_rate = lrn_mode[1]
        max_iteration = 1000
        print "MLPR Layers:", layers
        print "Activation function:",act_fn
        print "Solver:",solv
        print "Learning rate:", lrn_rate
        print "Max iter:",max_iteration
        print '\n'
        clf = MLPR(hidden_layer_sizes=(layers), activation=act_fn, solver=solv, learning_rate=lrn_rate,
                   max_iter=max_iteration, verbose=False, shuffle = False, learning_rate_init = initLrnRate,
                   tol=1e-4, momentum = 0.9, nesterovs_momentum=True, alpha = 1e1, beta_1 = 0.2, 
                   beta_2 = 0.8)
    if classifier == "linreg":
        clf = LinearRegression()
    if classifier == "SVR":
        clf = SVR()
    if classifier == "nb":
        clf = BayesianRidge(alpha_1=1e-6, alpha_2=1e-06, compute_score=True, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=1000,
       normalize=False, tol=0.001, verbose=True)
    
    ''' CV score '''  
    if calCV:
#        print >> sys.stderr, "Running Cross Validation..."
        train_mse, val_mse, test_mse = CV(clf, train, train_label, test, test_label, 
                                          flag, verbose = verbose)
        return train_mse, val_mse, test_mse
        
    ''' Train test prediction score --> i.e. on the test file of the same chromosome '''
    if fullPred:
#        print >> sys.stderr, "Training..."
        clf.fit(train, train_label)
        if verbose and classifier == "MLPR":
            print "Architecture properties after training"
            print "-------------------------------------"
            print "loss:", clf.loss_          
            print "coefs:",np.shape(clf.coefs_[0]), len(clf.coefs_)
#            pickle.dump((clf.coefs_), open("NN_weights_test_0.json", "w"))
            print "params:",clf.get_params
            if plot:
                ''' Sum by row '''
                coefs = [sum(weights) for weights in clf.coefs_[0]]
                coefs = [w/sum(coefs) for w in coefs]  
                print sum(coefs)
                coefs = [math.log10(abs(w)) for w in coefs]           
                ''' Sum by column '''
    #            coefs = np.sum(clf.coefs_[0], axis=1)
                ''' Just taking the column (50 pictures), as seen in MNIST example '''
    #            coefs_2 = [weights[0] for weights in clf.coefs_[0]]
                sup.plotHeatMap(coefs)
    #            sup.plotHeatMap(coefs_2)
            print "num_of_iterations:",clf.n_iter_
            print "num_of_outputs:",clf.n_outputs_
#            f = "loss_curve_"+solv+"_"+lrn_rate+"_relu.json"
#            json.dump(list(clf.loss_curve_), open(f, "w"))
#            print "mlpr loss_curve:", clf.loss_curve_
            print '\n'
        
        if verbose and classifier == "randforest":
            print "Architecture properties after training"
            print "-------------------------------------"
            print "estimators:", len(clf.estimators_)
            print "features:", clf.n_features_
            print "outputs:", clf.n_outputs_
            if plot:
                print sum(clf.feature_importances_)
                sup.plotHeatMap([math.log10(f) for f in clf.feature_importances_])
        
        if verbose and classifier == "nb":
            print "Architecture properties after training"
            print "-------------------------------------"
            print "class_prior:", clf.class_prior_
            print "class_count:", clf.class_count_
#        print >> sys.stderr, "Predicting..."
        pred = clf.predict(test)
        thresh = 1.0
        countf = count_fun(pred, thresh)
        print "Less than or equal to",thresh, "count in prediction:",countf,"out of",len(pred)
        if countf != 0:
            print "Average of these counts:",sum([f for f in pred if f < thresh])/float(countf)
        print min(pred), max(pred)
        full_test_score = mse(test_label, pred)
        if getRegression:
            r_2, intercept, slope = sup.rsquared(pred, test_label)
            line = intercept + slope * pred
            plt.scatter(pred, test_label)
            plt.plot(pred, line, color='red')
            plt.xlabel("Predicted labels")
            plt.ylabel("True Labels")
            plt.title("True labels vs predicted labels")
            print "R^2:",str(r_2)
            plt.legend(('data', 'line-regression r={}'.format(np.round(r_2, 3))), 'best')
            plt.legend('R^2 = '.format(r_2), 'best')
            r_2 = r2_score(pred, test_label)
        
        if getFeatImportance:
            features = list(clf.feature_importances_)
            sup.plotHeatMap(features)
        return full_test_score

def classify(classifier, train_feat, train_lab, test_feat, test_lab, flag, 
             CV = True, fullPred = False, v = False):
    if classifier == "randforest":
        layers = [40]
    
    if classifier == "MLPR" or classifier == "ae" or classifier == "RBM":
        layers = [len(train_feat[0]), len(train_feat[0])-1]
#        layers = [1]
        
    if classifier == "SVR" or classifier == "nb":
        layers = None
    
    if CV:
        if flag == "window":
            train_MSE, val_MSE, test_MSE = trainAndPredict(train_feat, train_lab, test_feat, test_lab, 
                                                           classifier, num = layers, flag = flag, verbose = v)
        else:
            train_MSE, val_MSE, test_MSE = trainAndPredict(train_feat, train_lab, test_feat, test_lab, 
                                                           classifier, num = layers, flag = flag, verbose = v)
    if fullPred:
        if flag == "window":
            fullTestScore = trainAndPredict(train_feat, train_lab, test_feat, test_lab, 
                                            classifier, num = layers, 
                                            flag = flag, verbose = v, calCV = False, fullPred = True)
        else:
            fullTestScore = trainAndPredict(train_feat, train_lab, test_feat, test_lab, 
                                            classifier, num = layers, 
                                            flag = flag, verbose = v, calCV = False, fullPred = True)
        print "Test Error:", fullTestScore


def classifierIterator(predictor, train_feat, train_lab, test_feat, test_lab, win_size, mode,
                       flag="window", distance = "bp", v = False):
    print "Parameters:"
    print "--------------"
    print "Distance: ", distance
    print "Feature mode: ", flag
    predFull, predCV, iterateForest, iterateNN, iterateConv = False, False, False, False, False
    
    if mode == "predFull":
        predFull = True
    elif mode == "predCV":
        predCV = True
    elif mode == "iterateForest":
        iterateForest = True
    elif mode == "iterateNN":
        iterateNN = True
    elif mode == "iterateConv":
        iterateConv = True
        
    if flag == "window":    
        ''' Defines indices of window '''
        tmp = []
        low, up = [24-win_size/2, 24+win_size/2]
        for i in range(low, up+1):
            tmp.append(i)
            for t in range(1, win_size/2+1):
                tmp.append(i+t*int(math.sqrt(len(train_feat[0])-1)))
                tmp.append(i-t*int(math.sqrt(len(train_feat[0])-1)))
        tmp = sorted(tmp)
    elif flag == "rc":
        ''' Use row and column only '''
        tmp = []
        t = win_size/2+1
        tmp.append(24)
        for i in range(1, t):
            tmp.append(24-i)
            tmp.append(24+i)
            tmp.append(24-i*7)
            tmp.append(24+i*7)
        tmp = sorted(tmp)
        print tmp
    else:
        print "Invalid mode"
        sys.exit()
        
    new_train_feat, new_test_feat = [], []
    for i, instance in enumerate(train_feat):
        temp = []
        for ind in tmp:
            temp.append(instance[ind])
        if distance == "bp":
            temp.append(instance[-2])
        elif distance == "rf":
            temp.append(instance[-1])
        new_train_feat.append(temp)
    print "Feature size: ",len(new_train_feat[0])            
    for i, instance in enumerate(test_feat):
        temp = []
        for ind in tmp:
            temp.append(instance[ind])
            
        if distance == "bp":
            temp.append(instance[-2])
        elif distance == "rf":
            temp.append(instance[-1])
        new_test_feat.append(temp)
    
    if iterateForest:
        modes = ["Trees", "Depth"]
        mode = modes[0]
        print "Iterating random forests on hyperparameter ", mode, '\n'
        it.iterate_forest(new_train_feat, train_lab, new_test_feat, test_lab, flag, iterateMode = mode,
                          v = v)
    
    if iterateNN:
        modes = ["Layers", "Neurons", "initLearningRate"]
        mode = modes[2]
        print "Iterating MLPR on hyperparameter ", mode, '\n'
        it.iterate_NN(new_train_feat, train_lab, new_test_feat, test_lab, flag, iterateMode = mode, 
                      v = v)
    
    if iterateConv:
        modes = ["Layers", "momentum"]
        mode = modes[1]
        print "Generating images for weights on ", mode, '\n'
        it.iterateConvergence(new_train_feat, train_lab, new_test_feat, test_lab, flag, iterateMode = mode, 
                      v = v)
                      
    if predCV:
        print "Cross-Validation Mode..."
        print "Predictor:", predictor, '\n'
        classify(predictor, new_train_feat, train_lab, new_test_feat, test_lab, flag, v = v)
    elif predFull:
        print "Full prediction Mode..."
        print "Predictor:", predictor, '\n'
        classify(predictor, new_train_feat, train_lab, new_test_feat, test_lab, flag, 
                 CV = False, fullPred = predFull, v=v) 
 
''' Input files '''
def featureStrip(lab_dic, tag = "train"):
    X, y = [], []
    if tag == "train":
        for k in lab_dic.keys():
            if k <= 6:
    #        if len(lab_dic[k]) >= 0:
                c = 0
                for f in lab_dic[k]:
                    if c < 500:
#                    if np.count_nonzero(f[:-2]) >= 0:
                        X.append(f)
                        y.append(k)
                        c += 1
    if tag == "test":
        for k in lab_dic.keys():
            if k == 1:
                c = 0
                for f in lab_dic[k]:
                    if c < 500:
                        X.append(f)
                        y.append(k)
                        c += 1
    return X, y
    
def returnData(flag = "Partial"):
    if flag == "Full":
        root = "Fullfeatures_RF_BP_DISTANCE/"
    elif flag == "Partial":
        root = "Subfeatures_RF_BP_distance/"
    elif flag == "same_chr":
        root = "Same_chr_features/"
        
    train_features = root+"train_features.txt"
    train_labels = root+"train_labels.txt"
    train_test_features = root+"test_features.txt"
    train_test_labels = root+"test_labels.txt"
    
    if flag == "Reflected":
        root= "Fullfeatures_reflected/"
        train_features = root+"upper_train_features.txt"
        train_labels = root+"upper_train_labels.txt"
        train_test_features = root+"lower_train_features.txt"
        train_test_labels = root+"lower_train_labels.txt"
        
    train_feat = json.load(open(train_features))
    train_lab = json.load(open(train_labels))
    
    test_feat = json.load(open(train_test_features))
    test_lab = json.load(open(train_test_labels))
    
    strip = False
    if strip:
        train_dict = lsh.splitByLabels(train_feat, train_lab)
        test_dict = lsh.splitByLabels(test_feat, test_lab)
        train_feat, train_lab = featureStrip(train_dict, "train")
        train_feat, train_lab = shuffle(train_feat, train_lab)
#        print train_feat[0][:-2]
        test_feat, test_lab = featureStrip(test_dict, "test")
        test_feat, test_lab = shuffle(test_feat, test_lab)
        print len(train_feat), sum(test_lab), len(test_lab)
        
    combine = True
    if flag == "Reflected" and combine:
        tmp_train_feat, test_feat, tmp_train_lab, test_lab = tts(test_feat, test_lab, test_size=0.2)
        endind = int(0*len(tmp_train_feat))
        train_feat = train_feat+tmp_train_feat[:endind]
        train_lab = train_lab+tmp_train_lab[:endind]
        train_feat, train_lab = shuffle(train_feat, train_lab)
    print "Training instances:", len(train_feat), "Testing instances:", len(test_feat)
    
    scale = False
    if scale:
         min_max_scaler = preproc.MinMaxScaler()
         train_feat = min_max_scaler.fit_transform(train_feat)
         test_feat = min_max_scaler.fit_transform(test_feat)
#    print train_feat[200000:200010], train_lab[200000:200010]
    return train_feat, train_lab, test_feat, test_lab
    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prefix", required = True,
    	help = "prefix of file path: Full, Partial, Reflected")
    ap.add_argument("-s", "--size", required = False, default=7,
    	help = "window size to be considered: 1, 3, 5, 7. Default = 7")
    ap.add_argument("-c", "--classifier", required = False, default = "MLPR",
      help = "classifier to use: randforest, MLPR, RBM, ae. Default = MLPR")
    ap.add_argument("-m", "--mode", required = False, default = "predFull",
      help = "mode of prediction: predCV, predFull, iterateForest, iterateNN, iterateConv. Default = predFull")
    ap.add_argument("-d", "--distance", required = False, default = "rf", 
      help = "distance to be used: rf, bp. Default = rf")
    ap.add_argument("-f", "--flag", required = False, default = "window",
      help = "input feature type: window, row+column. Default = window")
    ap.add_argument("-v", "--verbose", required = False, default = False,
      help = "Default = False")
    ap.add_argument("-o", "--out", required = False, default = "log.txt",
                    help = "log file to be written to")

    args = vars(ap.parse_args())

    print >> sys.stderr, "Reading inputs..."
    train_feat, train_lab, test_feat, test_lab = returnData(args["prefix"])
#    print max(train_lab), min(train_lab)
#    print max(test_lab), min(test_lab)
    
    start_time = time.time()
    X = []
    for i in range(args["size"], args["size"]+1, 2):
        X.append(i)
        print "------ Window size ", i," -----"
        classifierIterator(args["classifier"], train_feat, train_lab, 
                           test_feat, test_lab, i, mode = args["mode"], distance= args["distance"], 
                           flag = args["flag"], v = args["verbose"])
#        classifierIterator(train_feat, train_lab, test_feat, test_lab, i, flag="rc")

    print("--- %s minutes ---" % ((time.time() - start_time)/60))
   