import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# LEARNING RATE
step_size=0.0001
max_iters=2000

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()
  #X_train, y_train, X_test, y_test = adminLoadData()


  print(X_train.shape)
  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  #return
  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  for k in [2,3,4, 5, 10, 20, 50]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  X_test_bias = dummyAugment(X_test)
  y_pred_test = X_test_bias@w >= 0
  
  out = np.zeros( (len(X_test_bias), 2))
  out[:,0] = np.arange(len(X_test_bias))
  out[:,1] = y_pred_test.astype(np.int)[:,0]

  np.savetxt("sample_sub.csv", out, delimiter=",", fmt="%d")

def dummyAugment(X):
  return np.hstack( [np.ones( (len(X), 1)), X])

def basisAugment(X):
  X = (X-np.mean(X,axis=0))
  X = X / np.max(X, axis=0)
  mats = []
  for i in range(2):
    mats.append(np.power(X,i))
  X = np.concatenate(mats,axis=1)
  print(X.shape)
  return X

# the below definitions are for numerical stability.
# they use identities of the logistic function to keep
# things in a reasonable range of values even for large z
# lets us still see output for huge learning rates

def sig(z):
  return 1. / (1.+np.exp(-z))

def logSig(z):
  return -np.logaddexp(0,-z)

def log1MinSig(z):
  return logSig(-z)


def calculateNegativeLogLikelihood(X,y,w):
  l = -y.T@logSig(X@w)-(1-y).T@log1MinSig(X@w)
  return l[0]

# This is the direct-but-unstable implementation that require an epsillon
# constant in the log to avoid overflow and infs from the log
def calculateNegativeLogLikelihoodUnsafe(X,y,w):
  sig = 1/(1+np.exp(-(X@w)))
  eps = 0.00000000000000000001
  l = -y.T@np.log(sig+eps) - (1-y).T@np.log(1-sig+eps)
  return l[0]

def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )
    

    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):

        # Make a variable to store our gradient
        w_grad = np.zeros( (X.shape[1],1) )

        # Compute the gradient over the dataset and store in w_grad
        # .
        # . Implement equation 5.
        # .
       
        #raise Exception('Student error: You haven\'t implemented the gradient calculation for trainLogistic yet.')
        w_grad = X.T@(sig(X@w)-y)

        
        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad 
        
        # Calculate the negative log-likelihood with the 
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))
        
    return w, losses


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test

  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.

# Loads the train and test splits, passes back x/y for train and just x for test
def adminLoadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test[:, 0:-1]
  y_test = test[:, -1]

  return X_train, y_train[:, np.newaxis], X_test, y_test[:,np.newaxis] # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
