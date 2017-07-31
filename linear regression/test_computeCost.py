import computeCost
import unittest
import numpy as np

class test_compute_cost(unittest.TestCase):

    def setUp(self):
        self.X, self.y = np.loadtxt('ex1data1.txt', delimiter=',', unpack=True)
        m = len(self.y)
        self.X = np.c_[np.ones(m), self.X]  # m*1 --> m*2
        theta = np.zeros([2, 1])  # 2*1

    def test_theat0(self):

        theta = np.zeros([2, 1])  # 2*1
        self.assertAlmostEquals(32.07, computeCost.computeCost(self.X, self.y, theta), delta=0.004)

if __name__ == '__main__':
    unittest.main()