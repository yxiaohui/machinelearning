import numpy as np
def computeCost(X, y, theta):
    m = len(y)

    predictions = np.dot(X, theta)
    sqrErr = (predictions-y)**2

    J = 1/(2*m)*