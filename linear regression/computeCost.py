
import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    y.shape = (97,1)  #reshape 97 to 97X1
    predictions = np.dot(X, theta)
    err = np.array(predictions - y)
    sqrErr = np.square(err)

    J = np.sum(sqrErr)/(2*m)

    return J
