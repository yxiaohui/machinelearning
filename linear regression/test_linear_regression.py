import warmUpExercise
import msvcrt
import numpy as np
import matplotlib.pylab as plt
import computeCost
import unittest
#import matplotlib.dates as mdates

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
warmUpExercise.warmUpExercise()

print('Program pasued. ress enter to continue.\n')
ord(msvcrt.getch())

print('Plotting Data ...\n')
X, y = np.loadtxt('ex1data1.txt', delimiter=',', unpack=True)
m = len(X)

plt.plot(X, y, 'r*')
plt.show()

print('Program pasued. ress enter to continue.\n')
ord(msvcrt.getch())

print('Running Gradient descent')

X = np.c_[np.ones(m), X]  # m*1 --> m*2
theta = np.zeros([2, 1])  # 2*1

iterations = 1500
alpha = 0.01

computeCost.computeCost(X, y, theta)







