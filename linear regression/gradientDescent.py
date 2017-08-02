import numpy as np
import computeCost

def gradient_descent(X, y, theta, alpha, num_inters):

    J_history = np.zeros([num_inters, 1])
    m = len(y)
    for i in range(0, num_inters):
        predictions = np.dot(X, theta)
        tmp0 = theta[0] - alpha*np.sum(predictions-y)/m
        tmp1 = theta[1] - alpha*np.sum((predictions-y)*X[:, 1:])/m
        theta[0] = tmp0
        theta[1] = tmp1

        J_history[i] = computeCost.computeCost(X, y, theta)
        if i == (num_inters-1):
            print("Cost J = ")
            print(J_history[i])

    return theta, J_history