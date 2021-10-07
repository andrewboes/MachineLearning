import numpy as np

def main():
    msg = "Hello world"
    print(msg)

    #v = np.array([[1,2],[3,4]])
    #print(v, '\n')
    #print("no axis:", np.sum(v))
    #print("no axis=0:", np.sum(v, axis=0))
    #print("no axis=1:", np.sum(v, axis=1))

    example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],[ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0,1,1]])

    print(example_train_x, '\n')
    print(example_train_y, '\n')

    ans = knn(example_train_x, example_train_y,6)
    print(ans)

def knn(xTrain, xTest, k):
    """
    Finds the k nearest neighbors of xTest in xTrain.
    Input:
    xTrain = n x d matrix. n=rows and d=features
    xTest = m x d matrix. m=rows and d=features (same amount of features as xTrain)
    k = number of nearest neighbors to be found
    Output:
    dists = distances between xTrain/xTest points. Size of n x m
    indices = kxm matrix with indices of yTrain labels
    """
    #the following formula calculates the Euclidean distances.
    distances = -2 * xTrain@xTest.T + np.sum(xTest**2,axis=1) + np.sum(xTrain**2,axis=1)[:, np.newaxis]
    #because of numpy precision, some really small numbers might 
    #become negatives. So, the following is required.
    distances[distances < 0] = 0
    #for speed you can avoid the square root since it won't affect
    #the result, but apply it for exact distances.
    distances = distances**.5
    indices = np.argsort(distances, 0) #get indices of sorted items
    distances = np.sort(distances,0) #distances sorted in axis 0
    #returning the top-k closest distances.
    return indices[0:k, : ], distances[0:k, : ]


if __name__ == "__main__":
    main()
