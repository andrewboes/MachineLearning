import numpy as np
import time


def main():

    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    # I made up some random 3-dimensional data and some labels for us
    example_train_x = np.array([[1, 0, 2], [3, -2, 4], [5, -2, 4],
                                [4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])

    #########
    # Sanity Check 1: If I query with examples from the training set
    # and k=1, each point should be its own nearest neighbor

    for i in range(len(example_train_x)):
        v = get_nearest_neighbors(example_train_x, example_train_x[i], 1)
        assert([i] == v)
    print("Sanity Check 1...Complete")
    #########
    # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, 4, 2]), 2)
    assert(set(nn_idx).difference(set([4, 3])) == set())

    rankAndDist = get_nearest_neighbors_with_dist(example_train_x, np.array([1, 4, 2]), 2)

    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, -4, 2]), 3)
    assert(set(nn_idx).difference(set([1, 0, 2])) == set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array([10, 40, 20]), 5)
    assert(set(nn_idx).difference(set([4, 3, 0, 2, 1])) == set())
    print("Sanity Check 2...Complete")
    #########
    # Sanity Check 3: Neighbors for increasing k should be subsets
    query = np.array([10, 40, 20])
    p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    for k in range(2, 7):
        nn_idx = get_nearest_neighbors(example_train_x, query, k)
        assert(set(p_nn_idx).issubset(nn_idx))
        p_nn_idx = nn_idx

    print("Sanity Check 3...Complete")
    #########
    # Test out our prediction code
    queries = np.array([[10, 40, 20], [-2, 0, 5], [0, 0, 0]])
    pred = predict(example_train_x, example_train_y, queries, 3)
    assert(np.all(pred == np.array([[0], [1], [0]])))

    print("Testing Prediction code...Complete")
    #########
    # Test our our accuracy code
    true_y = np.array([[0], [1], [2], [1], [1], [0]])
    pred_y = np.array([[5], [1], [0], [0], [1], [0]])
    assert(compute_accuracy(true_y, pred_y) == 3/6)

    pred_y = np.array([[5], [1], [2], [0], [1], [0]])
    assert(compute_accuracy(true_y, pred_y) == 4/6)

    print("Testing code accuracy...Complete")

    queries = np.array([[10, 40, 20], [-2, 0, 5], [0, 0, 0]])
    pred = predict_with_weights(example_train_x, example_train_y, queries, 3)
    assert(np.all(pred == np.array([[0], [1], [0]])))

    print("Testing weighted knn...Complete")

    #######################################
    # Now on to the real data!
    #######################################

    # Load training and test data as numpy matrices
    train_X, train_y, test_X = load_data()
# =============================================================================
#     testCols = [[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]]
#     testSubMatrix = train_X[:, np.concatenate(testCols)]
# =============================================================================

    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    # Search over possible settings of k
    print("Performing 4-fold cross validation")
    # Breaking up all cols to see if some groups aren't helpful
    basicCols = list(range(4))
    workClassCols = list(range(4, 11))
    maritalCols = list(range(11, 18))
    occupationCols = list(range(18, 32))
    relashonshipCols = list(range(32, 38))
    raceCols = list(range(38, 43))
    countryCols = list(range(43, 83))
    sexCols = list(range(83, 84))
    allTrainingCols = [basicCols, workClassCols, maritalCols,
                       occupationCols, relashonshipCols, raceCols, countryCols, sexCols]
    allSubsetsOfAllTrainigCols = [[allTrainingCols[j] for j in range(
        len(allTrainingCols)) if i >> j & 1] for i in range(2 ** len(allTrainingCols))]
    allSubsets = [x for x in allSubsetsOfAllTrainigCols if x != []]
    for k in [99]:
        t0 = time.time()

        #######################################
        # TODO Compute train accuracy using whole set
        #######################################
        train_acc = 0
        #######################################
        # TODO Compute 4-fold cross validation accuracy
        ######################################
        val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k, 50)
        t1 = time.time()
        print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(
            k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))

        for subset in allSubsets:
              t0 = time.time()
              if(len(subset) != 0):
                  print("Training cols: ", subset)
                  trainingSubset = train_X[:, np.concatenate(subset)]
                  val_acc, val_acc_var = cross_validation(trainingSubset, train_y, 4, k)
                  t1 = time.time()
                  print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
# =============================================================================
#       for sigma in [.01,.1,2,5,10,20,50,100]:
#           val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k,sigma)
#           t1 = time.time()
#           print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}] sigma = {:.2f}".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0,sigma))
# =============================================================================

    #######################################

    #######################################
    # Q10 Kaggle Submission
    #######################################

    # TODO set your best k value and then run on the test set
    best_k = 99

    columnSubset = [0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17], [
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 37], [38, 39, 40, 41, 42]
    trainingSubset = train_X[:, np.concatenate(columnSubset)]
    testSubset = test_X[:, np.concatenate(columnSubset)]
    # Make predictions on test set
    pred_test_y = predict(trainingSubset, train_y, testSubset, best_k)

    # add index and header then save to file
    test_out = np.concatenate(
        (np.expand_dims(np.array(range(2000), dtype=np.int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')

######################################################################
# Q7 get_nearest_neighbors
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy.
#
# Input:
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################


def get_nearest_neighbors(X, y, k):
    lengths = np.linalg.norm(X-y, axis=1)
    return np.argsort(lengths)[0:k]


def get_nearest_neighbors_with_dist(X, y, k):
    lengths = np.linalg.norm(X-y, axis=1)
    indexesAndDistance = np.vstack((np.argsort(lengths)[0:k], lengths[0:k]))
    return indexesAndDistance

######################################################################
# Q7 knn_classify_point
######################################################################
# Runs a kNN classifier on the query point
#
# Input:
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################


def knn_classify_point(examples_X, examples_y, query, k):
    neighbors = get_nearest_neighbors(examples_X, query, k)  # indexes of knn
    neighborClasses = []
    for neighbor in neighbors:
        neighborClasses.append(examples_y[neighbor][0])
    return np.bincount(neighborClasses).argmax()


def knn_weighted_classification(examples_X, examples_y, query, k):
    neighbors = get_nearest_neighbors_with_dist(examples_X, query, k)  # indexes of knn
    zeroScore = 0  # weighted sum of group 0
    oneScore = 0  # weighted sum of group 1

    # for n,i in neighbors[0]:
    for i, neighbor in enumerate(neighbors[0]):
        neighborClass = examples_y[int(neighbor)][0]
        neighborDist = neighbors[1][i]
        if neighborDist == 0:
            return neighborClass
        if neighborClass == 0:
            zeroScore += (1/neighborDist)
        elif neighborClass == 1:
            oneScore += (1/neighborDist)

    return 0 if zeroScore > oneScore else 1


def knn_gauss_weighted_classification(examples_X, examples_y, query, k, sigma):
    neighbors = get_nearest_neighbors_with_dist(examples_X, query, k)  # indexes of knn
    zeroScore = 0  # weighted sum of group 0
    oneScore = 0  # weighted sum of group 1

    # for n,i in neighbors[0]:
    for i, neighbor in enumerate(neighbors[0]):
        neighborClass = examples_y[int(neighbor)][0]
        neighborDist = neighbors[1][i]
        weight = np.exp(-(neighborDist**2)/sigma)
        if neighborClass == 0:
            zeroScore += weight
        elif neighborClass == 1:
            oneScore += weight

    return 0 if zeroScore > oneScore else 1

# =============================================================================
#     neighborClasses = []
#     for neighbor in neighbors[0]:
#         neighborClasses.append(examples_y[int(neighbor)][0])
# =============================================================================
    # return np.bincount(neighborClasses).argmax()


######################################################################
# Q8 cross_validation
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input:
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1, sigma=5):
    splits = np.split(train_X, num_folds)  # creates n matrices each with numrows = rows/n
    classSplits = np.split(train_y, num_folds)
    results = []
    for i in range(num_folds):
        test = splits[i]
        testClasses = classSplits[i]
        train = np.array([])
        classes = np.array([])
        for j in range(num_folds):
            if j != i:
                if train.size == 0:
                    train = splits[j]
                    classes = classSplits[j]
                else:
                    train = np.vstack((train, splits[j]))
                    classes = np.vstack((classes, classSplits[j]))
        knnClasses = []
        for x in test:
            knnClasses.append(knn_classify_point(train, classes, x, k))
        results.append(compute_accuracy(testClasses.T[0], knnClasses))
    return sum(results)/len(results), np.var(results)
# =============================================================================
#     return avg_val_acc, varr_val_acc
# =============================================================================


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy
######################################################################
# Runs a kNN classifier on the query point
#
# Input:
#   true_y --  a n-by-1 vector where each value corresponds to
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input:
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################


def predict(examples_X, examples_y, queries_X, k):
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k)
                   for query in queries_X]

    return np.array(predicted_y, dtype=np.int)[:, np.newaxis]


def predict_with_weights(examples_X, examples_y, queries_X, k):
    # For each query, run a knn classifier
    predicted_y = [knn_weighted_classification(
        examples_X, examples_y, query, k) for query in queries_X]
    return np.array(predicted_y, dtype=np.int)[:, np.newaxis]

# Load data


def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:, np.newaxis]

    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


if __name__ == "__main__":
    main()
