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


 ######################################################################
 # Q6 Learning rates/step sizes
 ######################################################################
 for k in [2,3,4, 5, 10]:
     for s in [.1, .01, .001, .0001]:
         for m in [100, 1000, 5000, 10000, 20000]:
             cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k, s, m)
             logging.info("{}-fold Cross Val Accuracy (s={}, m={}) -- Mean (stdev): {:.4}% ({:.4}%)".format(k,s,m,cv_acc * 100, cv_std * 100))
             
# =============================================================================
# As you can see in the output below, as the step size (s) decreases the accuracy increases. 
# As the max_iters increases the accuracy also increases
# But there is a point when the accuracy stops increasing. My guess is that the weight vector is not getting better or the max_iters needs to increase even faster than step size decreases.
# Plots discussed in accompanying PDF
#
# OUTPUT
# 2021-10-19 10:39:27 INFO     2-fold Cross Val Accuracy (s=0.1, m=100) -- Mean (stdev): 92.7% (0.4292%)
# 2021-10-19 10:39:35 INFO     2-fold Cross Val Accuracy (s=0.1, m=1000) -- Mean (stdev): 95.28% (0.8584%)
# 2021-10-19 10:40:12 INFO     2-fold Cross Val Accuracy (s=0.1, m=5000) -- Mean (stdev): 95.71% (0.8584%)
# 2021-10-19 10:41:26 INFO     2-fold Cross Val Accuracy (s=0.1, m=10000) -- Mean (stdev): 93.56% (3.004%)
# 2021-10-19 10:43:55 INFO     2-fold Cross Val Accuracy (s=0.1, m=20000) -- Mean (stdev): 95.49% (1.502%)
# 2021-10-19 10:43:56 INFO     2-fold Cross Val Accuracy (s=0.01, m=100) -- Mean (stdev): 93.56% (0.4292%)
# 2021-10-19 10:44:03 INFO     2-fold Cross Val Accuracy (s=0.01, m=1000) -- Mean (stdev): 93.78% (0.2146%)
# 2021-10-19 10:44:36 INFO     2-fold Cross Val Accuracy (s=0.01, m=5000) -- Mean (stdev): 96.57% (0.0%)
# 2021-10-19 10:45:49 INFO     2-fold Cross Val Accuracy (s=0.01, m=10000) -- Mean (stdev): 94.42% (0.8584%)
# 2021-10-19 10:48:27 INFO     2-fold Cross Val Accuracy (s=0.01, m=20000) -- Mean (stdev): 94.64% (3.648%)
# 2021-10-19 10:48:28 INFO     2-fold Cross Val Accuracy (s=0.001, m=100) -- Mean (stdev): 93.35% (2.79%)
# 2021-10-19 10:48:34 INFO     2-fold Cross Val Accuracy (s=0.001, m=1000) -- Mean (stdev): 95.71% (1.288%)
# 2021-10-19 10:49:09 INFO     2-fold Cross Val Accuracy (s=0.001, m=5000) -- Mean (stdev): 95.71% (0.8584%)
# 2021-10-19 10:50:15 INFO     2-fold Cross Val Accuracy (s=0.001, m=10000) -- Mean (stdev): 95.49% (0.2146%)
# 2021-10-19 10:52:27 INFO     2-fold Cross Val Accuracy (s=0.001, m=20000) -- Mean (stdev): 94.21% (1.073%)
# 2021-10-19 10:52:28 INFO     2-fold Cross Val Accuracy (s=0.0001, m=100) -- Mean (stdev): 88.41% (0.0%)
# 2021-10-19 10:52:35 INFO     2-fold Cross Val Accuracy (s=0.0001, m=1000) -- Mean (stdev): 93.99% (0.8584%)
# 2021-10-19 10:53:08 INFO     2-fold Cross Val Accuracy (s=0.0001, m=5000) -- Mean (stdev): 95.49% (1.073%)
# 2021-10-19 10:54:14 INFO     2-fold Cross Val Accuracy (s=0.0001, m=10000) -- Mean (stdev): 95.49% (1.073%)
# 2021-10-19 10:56:37 INFO     2-fold Cross Val Accuracy (s=0.0001, m=20000) -- Mean (stdev): 96.78% (0.6438%)
# 2021-10-19 10:56:38 INFO     3-fold Cross Val Accuracy (s=0.1, m=100) -- Mean (stdev): 72.79% (27.64%)
# 2021-10-19 10:56:54 INFO     3-fold Cross Val Accuracy (s=0.1, m=1000) -- Mean (stdev): 96.13% (0.9379%)
# 2021-10-19 10:58:10 INFO     3-fold Cross Val Accuracy (s=0.1, m=5000) -- Mean (stdev): 93.56% (1.374%)
# 2021-10-19 11:00:38 INFO     3-fold Cross Val Accuracy (s=0.1, m=10000) -- Mean (stdev): 94.0% (1.295%)
# 2021-10-19 11:05:32 INFO     3-fold Cross Val Accuracy (s=0.1, m=20000) -- Mean (stdev): 94.42% (0.3213%)
# 2021-10-19 11:05:33 INFO     3-fold Cross Val Accuracy (s=0.01, m=100) -- Mean (stdev): 93.35% (1.67%)
# 2021-10-19 11:05:47 INFO     3-fold Cross Val Accuracy (s=0.01, m=1000) -- Mean (stdev): 94.86% (2.387%)
# 2021-10-19 11:06:57 INFO     3-fold Cross Val Accuracy (s=0.01, m=5000) -- Mean (stdev): 95.92% (0.3297%)
# 2021-10-19 11:09:17 INFO     3-fold Cross Val Accuracy (s=0.01, m=10000) -- Mean (stdev): 95.29% (1.584%)
# 2021-10-19 11:13:55 INFO     3-fold Cross Val Accuracy (s=0.01, m=20000) -- Mean (stdev): 95.06% (1.349%)
# 2021-10-19 11:13:56 INFO     3-fold Cross Val Accuracy (s=0.001, m=100) -- Mean (stdev): 94.43% (0.7736%)
# 2021-10-19 11:14:11 INFO     3-fold Cross Val Accuracy (s=0.001, m=1000) -- Mean (stdev): 96.36% (1.079%)
# 2021-10-19 11:15:24 INFO     3-fold Cross Val Accuracy (s=0.001, m=5000) -- Mean (stdev): 96.14% (0.02355%)
# 2021-10-19 11:17:51 INFO     3-fold Cross Val Accuracy (s=0.001, m=10000) -- Mean (stdev): 95.49% (0.5241%)
# 2021-10-19 11:22:37 INFO     3-fold Cross Val Accuracy (s=0.001, m=20000) -- Mean (stdev): 95.92% (1.35%)
# 2021-10-19 11:22:38 INFO     3-fold Cross Val Accuracy (s=0.0001, m=100) -- Mean (stdev): 89.7% (0.06279%)
# 2021-10-19 11:22:53 INFO     3-fold Cross Val Accuracy (s=0.0001, m=1000) -- Mean (stdev): 95.49% (1.57%)
# 2021-10-19 11:24:02 INFO     3-fold Cross Val Accuracy (s=0.0001, m=5000) -- Mean (stdev): 96.14% (1.378%)
# 2021-10-19 11:26:21 INFO     3-fold Cross Val Accuracy (s=0.0001, m=10000) -- Mean (stdev): 95.92% (0.6186%)
# 2021-10-19 11:31:05 INFO     3-fold Cross Val Accuracy (s=0.0001, m=20000) -- Mean (stdev): 96.35% (0.8205%)
# 2021-10-19 11:31:08 INFO     4-fold Cross Val Accuracy (s=0.1, m=100) -- Mean (stdev): 67.96% (26.56%)
# 2021-10-19 11:31:31 INFO     4-fold Cross Val Accuracy (s=0.1, m=1000) -- Mean (stdev): 93.98% (2.548%)
# 2021-10-19 11:33:25 INFO     4-fold Cross Val Accuracy (s=0.1, m=5000) -- Mean (stdev): 96.35% (1.964%)
# 2021-10-19 11:37:05 INFO     4-fold Cross Val Accuracy (s=0.1, m=10000) -- Mean (stdev): 95.29% (3.219%)
# 2021-10-19 11:44:11 INFO     4-fold Cross Val Accuracy (s=0.1, m=20000) -- Mean (stdev): 95.49% (2.727%)
# 2021-10-19 11:44:13 INFO     4-fold Cross Val Accuracy (s=0.01, m=100) -- Mean (stdev): 93.36% (2.906%)
# 2021-10-19 11:44:33 INFO     4-fold Cross Val Accuracy (s=0.01, m=1000) -- Mean (stdev): 96.36% (2.365%)
# 2021-10-19 11:46:15 INFO     4-fold Cross Val Accuracy (s=0.01, m=5000) -- Mean (stdev): 96.35% (1.262%)
# 2021-10-19 11:49:38 INFO     4-fold Cross Val Accuracy (s=0.01, m=10000) -- Mean (stdev): 85.25% (16.5%)
# 2021-10-19 11:56:21 INFO     4-fold Cross Val Accuracy (s=0.01, m=20000) -- Mean (stdev): 95.06% (2.932%)
# 2021-10-19 11:56:23 INFO     4-fold Cross Val Accuracy (s=0.001, m=100) -- Mean (stdev): 94.21% (3.095%)
# 2021-10-19 11:56:43 INFO     4-fold Cross Val Accuracy (s=0.001, m=1000) -- Mean (stdev): 96.37% (2.446%)
# 2021-10-19 11:58:22 INFO     4-fold Cross Val Accuracy (s=0.001, m=5000) -- Mean (stdev): 95.92% (1.864%)
# 2021-10-19 12:01:43 INFO     4-fold Cross Val Accuracy (s=0.001, m=10000) -- Mean (stdev): 95.91% (1.796%)
# 2021-10-19 12:50:29 INFO     4-fold Cross Val Accuracy (s=0.001, m=20000) -- Mean (stdev): 96.36% (2.033%)
# 2021-10-19 12:50:31 INFO     4-fold Cross Val Accuracy (s=0.0001, m=100) -- Mean (stdev): 89.07% (2.171%)
# 2021-10-19 12:50:53 INFO     4-fold Cross Val Accuracy (s=0.0001, m=1000) -- Mean (stdev): 94.62% (2.551%)
# 2021-10-19 12:52:40 INFO     4-fold Cross Val Accuracy (s=0.0001, m=5000) -- Mean (stdev): 96.14% (0.4467%)
# 2021-10-19 12:56:19 INFO     4-fold Cross Val Accuracy (s=0.0001, m=10000) -- Mean (stdev): 95.49% (0.7038%)
# 2021-10-19 13:03:23 INFO     4-fold Cross Val Accuracy (s=0.0001, m=20000) -- Mean (stdev): 96.78% (1.267%)
# 2021-10-19 13:03:26 INFO     5-fold Cross Val Accuracy (s=0.1, m=100) -- Mean (stdev): 91.37% (4.614%)
# 2021-10-19 13:03:56 INFO     5-fold Cross Val Accuracy (s=0.1, m=1000) -- Mean (stdev): 95.49% (2.06%)
# 2021-10-19 13:06:24 INFO     5-fold Cross Val Accuracy (s=0.1, m=5000) -- Mean (stdev): 95.27% (1.116%)
# 2021-10-19 13:11:18 INFO     5-fold Cross Val Accuracy (s=0.1, m=10000) -- Mean (stdev): 95.06% (1.107%)
# 2021-10-19 13:21:01 INFO     5-fold Cross Val Accuracy (s=0.1, m=20000) -- Mean (stdev): 95.03% (2.413%)
# 2021-10-19 13:21:04 INFO     5-fold Cross Val Accuracy (s=0.01, m=100) -- Mean (stdev): 69.24% (27.75%)
# 2021-10-19 13:21:32 INFO     5-fold Cross Val Accuracy (s=0.01, m=1000) -- Mean (stdev): 94.46% (3.099%)
# 2021-10-19 13:23:46 INFO     5-fold Cross Val Accuracy (s=0.01, m=5000) -- Mean (stdev): 95.1% (2.174%)
# 2021-10-19 13:28:18 INFO     5-fold Cross Val Accuracy (s=0.01, m=10000) -- Mean (stdev): 94.42% (0.4122%)
# 2021-10-19 13:37:49 INFO     5-fold Cross Val Accuracy (s=0.01, m=20000) -- Mean (stdev): 96.78% (1.781%)
# 2021-10-19 13:37:52 INFO     5-fold Cross Val Accuracy (s=0.001, m=100) -- Mean (stdev): 93.78% (1.211%)
# 2021-10-19 13:38:19 INFO     5-fold Cross Val Accuracy (s=0.001, m=1000) -- Mean (stdev): 95.92% (1.249%)
# 2021-10-19 13:40:35 INFO     5-fold Cross Val Accuracy (s=0.001, m=5000) -- Mean (stdev): 96.14% (1.264%)
# 2021-10-19 13:45:10 INFO     5-fold Cross Val Accuracy (s=0.001, m=10000) -- Mean (stdev): 96.36% (1.424%)
# 2021-10-19 13:54:35 INFO     5-fold Cross Val Accuracy (s=0.001, m=20000) -- Mean (stdev): 95.94% (1.008%)
# 2021-10-19 13:54:38 INFO     5-fold Cross Val Accuracy (s=0.0001, m=100) -- Mean (stdev): 89.7% (1.048%)
# 2021-10-19 13:55:07 INFO     5-fold Cross Val Accuracy (s=0.0001, m=1000) -- Mean (stdev): 94.84% (1.873%)
# 2021-10-19 13:57:28 INFO     5-fold Cross Val Accuracy (s=0.0001, m=5000) -- Mean (stdev): 96.58% (1.817%)
# 2021-10-19 14:02:02 INFO     5-fold Cross Val Accuracy (s=0.0001, m=10000) -- Mean (stdev): 95.5% (2.046%)
# 2021-10-19 14:11:25 INFO     5-fold Cross Val Accuracy (s=0.0001, m=20000) -- Mean (stdev): 95.95% (2.157%)
# 2021-10-19 14:11:32 INFO     10-fold Cross Val Accuracy (s=0.1, m=100) -- Mean (stdev): 87.23% (16.78%)
# 2021-10-19 14:12:42 INFO     10-fold Cross Val Accuracy (s=0.1, m=1000) -- Mean (stdev): 94.6% (2.494%)
# 2021-10-19 14:18:22 INFO     10-fold Cross Val Accuracy (s=0.1, m=5000) -- Mean (stdev): 95.92% (3.744%)
# 2021-10-19 14:29:35 INFO     10-fold Cross Val Accuracy (s=0.1, m=10000) -- Mean (stdev): 95.28% (2.654%)
# 2021-10-19 14:52:39 INFO     10-fold Cross Val Accuracy (s=0.1, m=20000) -- Mean (stdev): 96.6% (2.17%)
# 2021-10-19 14:52:46 INFO     10-fold Cross Val Accuracy (s=0.01, m=100) -- Mean (stdev): 94.18% (3.097%)
# 2021-10-19 14:53:51 INFO     10-fold Cross Val Accuracy (s=0.01, m=1000) -- Mean (stdev): 95.53% (2.417%)
# 2021-10-19 14:59:19 INFO     10-fold Cross Val Accuracy (s=0.01, m=5000) -- Mean (stdev): 87.0% (18.28%)
# 2021-10-19 15:10:03 INFO     10-fold Cross Val Accuracy (s=0.01, m=10000) -- Mean (stdev): 95.94% (1.465%)
# 2021-10-19 15:31:49 INFO     10-fold Cross Val Accuracy (s=0.01, m=20000) -- Mean (stdev): 94.45% (4.154%)
# 2021-10-19 15:31:56 INFO     10-fold Cross Val Accuracy (s=0.001, m=100) -- Mean (stdev): 93.36% (2.381%)
# 2021-10-19 15:32:58 INFO     10-fold Cross Val Accuracy (s=0.001, m=1000) -- Mean (stdev): 95.96% (2.417%)
# 2021-10-19 15:38:16 INFO     10-fold Cross Val Accuracy (s=0.001, m=5000) -- Mean (stdev): 95.53% (2.597%)
# 2021-10-19 15:49:03 INFO     10-fold Cross Val Accuracy (s=0.001, m=10000) -- Mean (stdev): 96.15% (2.645%)
# 2021-10-19 16:10:30 INFO     10-fold Cross Val Accuracy (s=0.001, m=20000) -- Mean (stdev): 96.58% (1.688%)
# 2021-10-19 16:10:36 INFO     10-fold Cross Val Accuracy (s=0.0001, m=100) -- Mean (stdev): 90.52% (4.28%)
# 2021-10-19 16:11:40 INFO     10-fold Cross Val Accuracy (s=0.0001, m=1000) -- Mean (stdev): 95.72% (2.676%)
# 2021-10-19 16:16:57 INFO     10-fold Cross Val Accuracy (s=0.0001, m=5000) -- Mean (stdev): 96.15% (2.468%)
# 2021-10-19 16:27:29 INFO     10-fold Cross Val Accuracy (s=0.0001, m=10000) -- Mean (stdev): 96.75% (2.464%)
# 2021-10-19 16:48:47 INFO     10-fold Cross Val Accuracy (s=0.0001, m=20000) -- Mean (stdev): 96.56% (1.44%)
# =============================================================================


 ######################################################################
 # Q7 Evaluating Cross Validation
 ######################################################################
# =============================================================================
#  My score on the leader board was 4% lower than my best performance using cross validation. 
#  There were some crazy variance (16%, 27%) in some of the test output from question 6
#  Also, the value K for the same step size and max_iters didn't  have the same accuracy. This is may be due to not 
#  enough training data.
# =============================================================================


 ######################################################################
 # Q8 Kaggle
 ######################################################################
# =============================================================================
#  
#  For my submission (PnwDrew) I looked at the output from question 6 and chose step size .0001 and max_iters 10000 as that seemed to have the best overall perf with low variance. 
#  My weight vector was: w = [-9.2681, 0.5374, -0.0413, 0.5004, 0.585, 0.0219, 0.4403, 0.2525, 0.5046]
#  
#  Code:
# =============================================================================

def kaggleOutput(X_test):
    X_test = dummyAugment(X_test)
    w = [-9.2681, 0.5374, -0.0413, 0.5004, 0.585, 0.0219, 0.4403, 0.2525, 0.5046]
    predictedY = X_test @ w >= 0
    test_out = np.column_stack((np.expand_dims(np.array(range(233),dtype=np.int), axis=1), predictedY.T))
    header = np.array([["id", "type"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')
