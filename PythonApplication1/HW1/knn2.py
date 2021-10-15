import numpy as np
import time

def main():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:, np.newaxis]
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]
    #testNN(train_X, train_y)
    #findGoodK(train_X, train_y)
    #findGoodSigma(train_X, train_y)
    #findColumnWeights(train_X, train_y)
    kaggleOutput(train_X, train_y, test_X)
    
def kaggleOutput(train_X, train_y, test_X, k=9):
    bestColWeights = [2, 1, 20000, 1000, 0.00005, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.1, 1, 1, 1, 2, 1, 1, 10, 1, 1, 200, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0.5, 2, 0, 0, 0, 1, 0.5, 1, 2, 0, 10, 0, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.001, 1, 1, 2, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
    train_X = train_X * bestColWeights
    test_X = test_X * bestColWeights
    pred_test_y = predict(train_X, train_y, test_X, k)    
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=np.int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')
    

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k, weightFunction=0) for query in queries_X]
    return np.array(predicted_y,dtype=np.int)[:,np.newaxis]

def findColumnWeights(train_X, train_y):
    #bestColWeights = [1, 1, 20000, 1000, 0.00005, 1, 1, 1, 1, 1, 1, 1, 1, 0.001, 0.001, 0.001, 0.1, 1, 1, 1, 2, 1, 1, 10, 1, 1, 200, 0.001, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0.5, 2, 0.001, 0.001, 0.0000001, 1, 0.5, 1, 2, 0.001, 10, 0.001, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.001, 1, 1, 2, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
    print("Performing n-fold cross validation for tuning column weights")
    for k in [9,50,99]:
        for d in [0,1,2]:
            bestColWeights = [1, 1, 10000, 1000, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.1, 1, 1, 1, 2, 1, 1, 10, 1, 1, 200, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0.5, 2, 0, 0, 0, 1, 0.5, 1, 2, 0, 10, 0, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.001, 1, 1, 2, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
            train_X = train_X * bestColWeights
            t0 = time.time()
            val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k,d=d)
            t1 = time.time()
            print("k = {:5d} -- val acc = {:.2f}% ({:.4f}) d={:.2f}\t\t[exe_time = {:.2f}]".format(k, val_acc*100, val_acc_var*100,d, t1-t0))


def findGoodK(train_X, train_y, n=4):
    #99
    print("Performing n-fold cross validation for tuning k")
    for k in [1,3,5,7,9,99,999]:
        for d in [0,1,2]:
            t0 = time.time()
            train_acc = 0
            val_acc, val_acc_var = cross_validation(train_X, train_y, n, k,d)
            t1 = time.time()
            print("k = {:5d} -- val acc = {:.2f}% ({:.4f}) d={:.2f}\t\t[exe_time = {:.2f}]".format(k, val_acc*100, val_acc_var*100,d, t1-t0))

def findGoodSigma(train_X, train_y, n=4):
    #10
    print("Performing n-fold cross validation for tuning sigma")
    for k in [9,99,999]:
        for sigma in [.1,1,10,50]:
            t0 = time.time()
            train_acc = 0
            val_acc, val_acc_var = cross_validation(train_X, train_y, n, k,2,sigma=sigma)
            t1 = time.time()
            print("k = {:5d} -- val acc = {:.2f}% ({:.4f}) sigma={:.2f}\t\t[exe_time = {:.2f}]".format(k, val_acc*100, val_acc_var*100,sigma, t1-t0))
    
    
def cross_validation(train_X, train_y, num_folds=4, k=1, d=0, sigma=10):
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
            knnClasses.append(knn_classify_point(train, classes, x, k, weightFunction = d, sigma=sigma))
        results.append(compute_accuracy(testClasses.T[0], knnClasses))
    return sum(results)/len(results), np.var(results)

def get_nearest_neighbors(X, y, k):
    lengths = np.linalg.norm(X-y, axis=1)
    indexesAndDistance = np.vstack((np.sort(np.argpartition(lengths, k)[:k]), np.sort(lengths)[0:k]))
    return indexesAndDistance
    
def knn_classify_point(examples_X, examples_y, query, k, weightFunction = 0, sigma = 10):
    neighbors = get_nearest_neighbors(examples_X, query, k)
    if weightFunction == 0:
        neighborClasses = []
        for neighbor in neighbors[0]:
            neighborClasses.append(examples_y[int(neighbor)][0])
        return np.bincount(neighborClasses).argmax()
    elif weightFunction == 1:
        zeroScore = 0  # weighted sum of group 0
        oneScore = 0  # weighted sum of group 1
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
    else:
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

def testNN(train_X, train_y, k=99):
    print("Testing...")
    index = 1
    nn = get_nearest_neighbors(train_X, train_X[index], k)
    assert(nn[0][0] == index)
    assert(nn[1][0] <= nn[1][1])
    c = knn_classify_point(train_X, train_y, train_X[index], k)
    assert(c == 0)
    c = knn_classify_point(train_X, train_y, train_X[9], k)
    assert(c == 1)
    c = knn_classify_point(train_X, train_y, train_X[index], k, 1)
    assert(c == 0)
    c = knn_classify_point(train_X, train_y, train_X[9], k, 1)
    assert(c == 1)
    c = knn_classify_point(train_X, train_y, train_X[index], k, 2)
    assert(c == 0)
    c = knn_classify_point(train_X, train_y, train_X[9], k, 2)
    assert(c == 1)
    c = knn_classify_point(train_X, train_y, train_X[index], k, 2, .1)
    assert(c == 0)
    c = knn_classify_point(train_X, train_y, train_X[9], k, 2, .1)
    assert(c == 1)
    print("...Complete")

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy
if __name__ == "__main__":
    main()