# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:06:57 2021

@author: BoesAn
"""

import numpy as np
import time
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size = .000001
max_iters = 10000000    

def main():

  logging.info("Running logistic regression w/ step size: {}, max iters: {}".format(step_size, max_iters))  
  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()
 
  X_train_bias = dummyAugment(X_train)
  logging.info("\n---------------------------------------------------------------------------\n")

# =============================================================================
#   # Fit a logistic regression model on train and plot its losses
#   logging.info("Training logistic regression model (No Bias Term)")
#   t0 = time.time()
#   w, losses = trainLogistic(X_train,y_train)
#   logging.info("exe time {}".format(time.time()-t0))
#   y_pred_train = X_train @ w >= 0
#   
#   logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
#   logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train) * 100))
#   
#   logging.info("\n---------------------------------------------------------------------------\n")
# 
#  
#   # Fit a logistic regression model on train and plot its losses
#   logging.info("Training logistic regression model (Added Bias Term)")
# 
#   w, bias_losses = trainLogistic(X_train_bias,y_train)
#   y_pred_train = X_train_bias @ w >= 0
#   
#   logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
#   logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train) * 100))
# 
# 
#   plt.figure(figsize=(16,9))
#   plt.plot(range(len(losses)), losses, label="No Bias Term Added")
#   plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
#   plt.title("Logistic Regression Training Curve")
#   plt.xlabel("Epoch")
#   plt.ylabel("Negative Log Likelihood")
#   plt.legend()
#   plt.show()
# 
#   logging.info("\n---------------------------------------------------------------------------\n")
# 
# =============================================================================
  logging.info("Running cross-fold vclidation for bias case:")

  # Perform k-fold cross
  for k in [2,3,4, 5, 10]:
     #w = [0.43463937, 0.09842774, 0.3646923, 0.37318118, 0, 0.33230682, 0.2072864, 0]
     w = [-9.2681, 0.5374, -0.0413, 0.5004, 0, 0.0219, 0.4403, 0.2525, 0]
     cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k,w)
     logging.info("{}-fold Cross Val Accuracy (s={}, m={}) -- Mean (stdev): {:.4}% ({:.4}%)".format(k,1,1,cv_acc * 100, cv_std * 100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  #raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')

  kaggleOutput(X_test)
  



######################################################################
# Q3.1 logistic
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input:
#   z -- a n-by-1 vector
#
# Output:
#   logit_z -- a n-by-1 vector where logit_z[i] is the result of
#               applying the logistic function to z[i]
######################################################################
def logistic(z):
    ##sigma(z) = 1/(1+e^(w*X))
    return (1 / (1 + np.exp(-z))).T


######################################################################
# Q3.2 calculateNegativeLogLikelihood
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input:
#   X -- a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y -- a n-by-1 vector representing the labels of the examples in X
#
#   w -- a d-by-1 weight bector
#
# Output:
#   nll -- the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):    
    logLikelihood = 0
    for i,x in enumerate(X):
        logisiticOfx = logistic(w.T @ x)
        logLikelihood += y[i] * np.log(logisiticOfx) + (1 - y[i]) * np.log(1 - logisiticOfx)
    return -logLikelihood


######################################################################
# Q4 trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of
# iterations max_iters, and step size step_size -- run max_iters of
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input:
#   X -- a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y -- a n-by-1 vector representing the labels of the examples in X
#
#   max_iters -- the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w -- the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):
    # Initialize our weights with zeros
    w = np.zeros((X.shape[1],1))
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
        # Todo: Compute the gradient over the dataset and store in w_grad
        # .
        # .  Implement equation 9.
        # .
        gradientW = np.zeros((X.shape[1],1))
        
        for i, x in enumerate(X):
            gw = (logistic(w.T @ x) - y[i]) * x
            gradientW = gradientW + gw[:, np.newaxis]

        # This is here to make sure your gradient is the right shape
        assert(gradientW.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size * gradientW
        
        # Calculate the negative log-likelihood with the
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))
    
        
    return w, losses


######################################################################
# Q5 dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input:
#   X -- a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X -- a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
    return np.hstack((np.ones((len(X),1)),X))
  


def kaggleOutput(X_test):
    X_test = dummyAugment(X_test)
# =============================================================================
#     w = [-9.2681, 0.5374, -0.0413, 0.5004, 0.585, 0.0219, 0.4403, 0.2525, 0.5046]
# =============================================================================
# =============================================================================
#     w = [-9.2681, 0.5374, -0.0413, 0.5004, 0, 0.0219, 0.4403, 0.2525, 0]
# =============================================================================
# =============================================================================
#     w = [0.43463937, 0.09842774, 0.3646923, 0.37318118, 0, 0.33230682, 0.2072864, 0, 0.5046]
# =============================================================================
    w = [-10.5847, 0.6384, -0.111, 0.562, 0.6698, 0.0487, 0.5016, 0.2672, 0.6799]
    predictedY = X_test @ w >= 0
    test_out = np.column_stack((np.expand_dims(np.array(range(233),dtype=np.int), axis=1), predictedY.T))
    header = np.array([["id", "type"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')



##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k, w):
  fold_size = int(np.ceil(len(X) / k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size * j)
    end = min(len(X),fold_size * (j + 1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate([np.arange(0,start), np.arange(end, len(X))])
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
  

    #w, losses = trainLogistic(X_fold_train, y_fold_train,m,s)

    acc.append(np.mean((X_fold_test @ w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for
# test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.

main()
