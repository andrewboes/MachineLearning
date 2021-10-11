import numpy as np

def main():
    numfolds = 4
    randArray = np.random.randint(10, size=[8,8])
    splits = np.split(randArray, numfolds) 
    rows = int(len(randArray)/numfolds)
    cols = int(len(randArray[0]))
    for i in range(numfolds):
        test = splits[i]
        train = np.array([])
        for j in range(numfolds):
            if j != i:
                if train.size == 0:
                    train = splits[j]
                else:
                    train = np.vstack((train, splits[j]))
        print("i", i)
        print("test", test)
        print("train", train)
                
        
        
        
        
        
      
# =============================================================================
#     example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],
#                                  [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
#     example_train_y = np.array([[0], [1], [1], [1], [0], [1]])
#   
#     #########
#     # Sanity Check 1: If I query with examples from the training set 
#     # and k=1, each point should be its own nearest neighbor
#     
#     for i in range(len(example_train_x)):
#         print(myKnn(example_train_x, example_train_x[i], 1))
# =============================================================================


    #v = np.array([[1,2],[3,4]])
    #print(v, '\n')
    #print("no axis:", np.sum(v))
    #print("no axis=0:", np.sum(v, axis=0))
    #print("no axis=1:", np.sum(v, axis=1))

    #example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],[ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    #example_train_y = np.array([3,-2,4])

    #print(example_train_x, '\n')
    #print(example_train_y, '\n')

    #ans = knn(example_train_x, example_train_y,1)
    #print(ans)
    #print(ans)
    #normOfVectorAndMatrix()
 
def myKnn(X,x,k) :
    #dist = []
    #for i in range(len(X)):
    lengths = np.linalg.norm(X-x,axis=1)
    d3 = np.argpartition(lengths,0)
    return d3[0:k]


def distanceBetweenMatrixAndVector():
    t = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],[ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    s = np.array([[0,1,1]])
    d = (t-s)**2
    print(d)
    d = np.sum(d, axis=0)
    d = np.sqrt(d)
    print(d)

def normOfVectorAndMatrix():
    A = np.array([])
    t = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],[ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    s = np.array([[1,0,2]])
    for i in range(len(t)):
        print(np.linalg.norm(s-i))



def test():
    single_point = [3, 4]
    points = np.arange(20).reshape((10,2))
    print(points)
    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    print(dist)

def anothertest(x,y,k):
    dist = [] 
    #Computing Euclidean distance
    dist_ind = np.sqrt(np.sum((x-y)**2, axis=1)) 
    #Concatinating the label with the distance
    #main_arr = np.column_stack((train_label,dist_ind))
    #Sorting the distance in ascending order
    #main = main_arr[main_arr[:,1].argsort()] 
    #Calculating the frequency of the labels based on value of K
    count = Counter(main[0:k,0])
    keys, vals = list(count.keys()), list(count.values())
    if len(vals)>1:
        if vals[0]>vals[1]:
            return int(keys[0])
        else:
            return int(keys[1])
    else:
        return int(keys[0])

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
    print(distances)
    #because of numpy precision, some really small numbers might 
    #become negatives. So, the following is required.
    distances[distances < 0] = 0
    #for speed you can avoid the square root since it won't affect
    #the result, but apply it for exact distances.
    #distances = distances**.5
    indices = np.argsort(distances, 0) #get indices of sorted items
    distances = np.sort(distances,0) #distances sorted in axis 0
    #returning the top-k closest distances.
    return indices[0:k, : ], distances[0:k, : ]


if __name__ == "__main__":
    main()
