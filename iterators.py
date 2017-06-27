# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:12:10 2017

@author: Jack_2
"""
import predictors as pr
import matplotlib.pyplot as plt
import sys
import json
import numpy as np
import supplementals as sup
import math
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor as MLPR
''' Iterate some hyperparameter of randomforest depending on iterateMode '''
def iterate_forest(new_train_feat, train_lab, new_test_feat, test_lab, flag, 
                   iterateMode = "Trees", saveImg = False, v = False):
    train_errors = []
    validation_errors = []
    test_errors = []
    X = []
    if iterateMode == "Trees":
        for k in range(10, 160, 10):
            X.append(k)
            train_mse, val_mse, test_mse = pr.trainAndPredict(new_train_feat, train_lab, new_test_feat, test_lab, 
                                                              flag = flag, num=[k], verbose = v)
    #        print k, test_score, CV_score
            train_errors.append(train_mse)
            validation_errors.append(val_mse)
            test_errors.append(test_mse)
    if iterateMode == "Depth":
        for k in range(10, 160, 10):
            X.append(k)
            train_mse, val_mse, test_mse = pr.trainAndPredict(new_train_feat, train_lab, new_test_feat, test_lab, 
                                                              flag = flag, num=[k], verbose = v)
    #        print k, test_score, CV_score
            train_errors.append(train_mse)
            validation_errors.append(val_mse)
            test_errors.append(test_mse)
    if v:
        fig,ax = plt.subplots()
        #ax1 = fig1.add_subplot(111)
        #    ax1.set_xlim([0, 160])
        ax.plot(X, train_errors, color='red',label="Train MSE")
        ax.plot(X, validation_errors, color='blue',label="Validation MSE")
        ax.plot(X, test_errors, color='green',label="Test MSE")
        ax.set_title("MSE vs "+iterateMode,fontweight="bold")
        ax.set_xlabel(iterateMode)
        ax.set_ylabel("Mean Squared Error (MSE)") 
        lgd=ax.legend(frameon=False,loc="center",bbox_to_anchor=(0.5,-0.2),ncol=3)
        #ax1.legend((l1, l2, l3), ("Train MSE", "Validation MSE", "Testing MSE"))
    if saveImg:
        plt.savefig("MSE_vs_"+iterateMode+"_1.jpg",format="jpg",dpi=300.0,bbox_extra_artists=(lgd,),bbox_inches="tight")       

def iterate_NN(new_train_feat, train_lab, new_test_feat, test_lab, flag,
               iterateMode = "Layers", saveImg = False, v = False, write = True):
    train_errors = []
    validation_errors = []
    test_errors = []
    X = []
    print "Length of feature: ", len(new_train_feat[0])
    if iterateMode == "Layers":
        for k in range(1, 10):
            X.append(k)
            layers = k*[len(new_train_feat[0])]
            train_mse, val_mse, test_mse = pr.trainAndPredict(new_train_feat, train_lab, new_test_feat, test_lab, classifier="MLPR", 
                                                              flag=flag, num=layers, verbose = v)
    #        print k, test_score, CV_score
            train_errors.append(train_mse)
            validation_errors.append(val_mse)
            test_errors.append(test_mse)
            
    if iterateMode == "Neurons":
        for k in range(1, 2*len(new_train_feat[0]), 5):
            X.append(k)
            layers = [k]
            train_mse, val_mse, test_mse = pr.trainAndPredict(new_train_feat, train_lab, new_test_feat, test_lab, classifier="MLPR", 
                                                              flag=flag, num=layers, verbose = v)
    #        print k, test_score, CV_score
            train_errors.append(train_mse)
            validation_errors.append(val_mse)
            test_errors.append(test_mse)
    if iterateMode == "initLearningRate":
        rates = [0.0001, 0.00005, 0.00001, 0.0005, 0.001, 0.005]
        for r in rates:
            X.append(r)
            layers = [len(new_train_feat[0]), len(new_train_feat[0])-1]
            train_mse, val_mse, test_mse = pr.trainAndPredict(new_train_feat, train_lab, new_test_feat, test_lab, classifier="MLPR", 
                                                              flag=flag, num=layers, verbose = v, initLrnRate = r)
    #        print k, test_score, CV_score
            train_errors.append(train_mse)
            validation_errors.append(val_mse)
            test_errors.append(test_mse)
    
    if write:
        tre = open(iterateMode+"_train_errors.json", "w")
        ve = open(iterateMode+"_val_errors.json", "w")
        te = open(iterateMode+"_test_errors.json", "w")
        xax = open(iterateMode+"_X.json", "w")
        json.dump(list(X), xax)
        json.dump(list(train_errors), tre)
        json.dump(list(validation_errors), ve)
        json.dump(list(test_errors), te)
        tre.close()
        ve.close()
        te.close()
        xax.close()
    v = True
    if v:
        fig,ax = plt.subplots()
        #ax1 = fig1.add_subplot(111)
        #    ax1.set_xlim([0, 160])
        ax.plot(X, train_errors, color='red',label="Train MSE")
        ax.plot(X, validation_errors, color='blue',label="Validation MSE")
        ax.plot(X, test_errors, color='green',label="Test MSE")
        ax.set_title("MSE vs "+iterateMode,fontweight="bold")
        ax.set_xlabel(iterateMode)
        ax.set_ylabel("Mean Squared Error (MSE)") 
        lgd=ax.legend(frameon=False,loc="center",bbox_to_anchor=(0.5,-0.2),ncol=3)
        #ax1.legend((l1, l2, l3), ("Train MSE", "Validation MSE", "Testing MSE"))
    if saveImg:
        plt.savefig("MSE_vs_"+iterateMode+"_1.jpg",format="jpg",dpi=300.0,bbox_extra_artists=(lgd,),bbox_inches="tight")
        
def iterateConvergence(new_train_feat, train_lab, new_test_feat, test_lab, flag,
               iterateMode = "Layers", saveImg = False, v = False, write = True):
    if iterateMode == "momentum":
        momentums = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        a_fns = ["identity", "logistic", "tanh", "relu"]
        solvers = ["adam", "sgd", "lbfgs"]
        lrn_mode = ["constant", "adaptive", "invscaling"]
        layers = [len(new_train_feat[0])]
        act_fn = a_fns[1]
        solv = solvers[1]
        lrn_rate = lrn_mode[1]
        max_iteration = 1000
        if v:
            print "MLPR Layers:", layers
            print "Activation function:",act_fn
            print "Solver:",solv
            print "Learning rate:", lrn_rate
            print "Max iter:",max_iteration
            print '\n'
        for i in momentums:
            for k in range(3):            
                clf = MLPR(hidden_layer_sizes=(layers), activation=act_fn, solver=solv, learning_rate=lrn_rate,
                           max_iter=max_iteration, verbose=False, shuffle = False, learning_rate_init = 0.001,
                           tol=1e-4, momentum = i)
                clf.fit(new_train_feat, train_lab)
                print "Architecture properties after training"
                print "-------------------------------------"      
                print "coefs:",np.shape(clf.coefs_[0]), len(clf.coefs_)
                print "params:",clf.get_params
                ''' Sum by row '''
                coefs = [sum(weights) for weights in clf.coefs_[0]]
                coefs = [w/sum(coefs) for w in coefs]  
                print sum(coefs)
                coefs = [math.log10(abs(w)) for w in coefs]           
                win_size = int(math.sqrt(len(coefs)))
                tmp = np.array(coefs[:-1]).reshape(win_size, win_size)
                df = pd.DataFrame(data = tmp)
                ax = plt.axes()
                sns.heatmap(df, ax=ax, cmap="gist_heat_r")
                ax.set_title("Feature Importance of Size-"+str(np.shape(tmp)[0])+" Window")
                plt.savefig("FtImp_"+solv+"_"+lrn_rate+"_"+str(i)+"momen_"+str(win_size)+"winsize_"+str(k)+".png")
                plt.clf()
                print "num_of_iterations:",clf.n_iter_
                print "num_of_outputs:",clf.n_outputs_
                print '\n'