# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 13:59:01 2017

@author: Jack_2
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import time
import argparse
import predictors

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required = False, default = "MLPR",
	help = "pipeline to be tested")
ap.add_argument("-o", "--out", required = False, default = "MLPR_log.txt",
                help = "log file to be written to")
args = vars(ap.parse_args())

print "IMPORTING DATA"
X_train, y_train, X_test, y_test = predictors.returnData("Full")

#args["pipeline"] == "MLPR"
#args["out"] == "MLPR_log.txt"
if args["pipeline"] == "MLPR":
    print "lol"
    layers = [50, 49]
    MLPR = MLPRegressor(hidden_layer_sizes=(layers))
    classifier = Pipeline([("mlp", MLPR)])
    params = {"mlp__activation": ['identity', 'logistic', 'tanh', 'relu'],
              "mlp__solver": ['adam', 'sgd', 'lbfgs'], 
              "mlp__learning_rate": ['constant', 'adaptive', 'invscaling'], 
              "mlp__alpha": [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
#    params = {"mlp__activation": ['logistic', 'relu'],
#              "mlp__learning_rate": ['constant', 'adaptive']}
elif args["pipeline"] == "RBM":
    rbm = BernoulliRBM()
    classifier = Pipeline([("rbm", rbm)])
    params = {
	"rbm__learning_rate": [0.1, 0.01, 0.001],
	"rbm__n_iter": [20, 40, 80],
      "rbm__n_components": [50, 100, 200]}
elif args["pipeline"] == "RBM+MLPR":
    rbm = BernoulliRBM()
    layers = [50, 49]
    MLPR = MLPRegressor(hidden_layer_sizes=(layers))
    classifier = Pipeline([("rbm", rbm), ("mlp", MLPR)])
    params = {
	"rbm__learning_rate": [0.1, 0.01, 0.001],
	"rbm__n_iter": [20, 40, 80],
      "rbm__n_components": [50, 100, 200],
	"mlp__learning_rate_init": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
      "mlp__activation": ['identity', 'logistic', 'tanh', 'relu'],
      "mlp__solver": ['adam', 'sgd', 'lbfgs'], 
      "mlp__learning_rate": ['constant', 'adaptive', 'invscaling'], 
      "mlp__alpha": [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
      "mlp__momentum": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}

print "SEARCHING", args["pipeline"]

start = time.time()
gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
gs.fit(X_train, y_train)
 
# print diagnostic information to the user and grab the
# best model
with open(args["out"], "w") as wf:
    wf.write(str(gs.best_score_)+'\n')
    print "\ndone in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print args["pipeline"], "PARAMETERS"
    bestParams = gs.best_estimator_.get_params()
 
    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print p, bestParams[p]
        wf.write(str(p)+'\t'+str(bestParams[p])+'\n')