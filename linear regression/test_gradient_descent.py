import gradientDescent
import unittest
import numpy as np

class test_gradient_descent(unittest.TestCase):

    def setUp(self):
        self.X, self.y = np.loadtxt('ex1data1.txt', delimiter=',', unpack=True)
        m = len(self.y)
        self.X = np.c_[np.ones(m), self.X]  # m*1 --> m*2

    def test_theta_init0(self):
        theta = np.zeros([2, 1])  # 2*1
        iterations = 1500
        alpha = 0.01
        theta, J_h = gradientDescent.gradient_descent(self.X, self.y, theta, alpha, iterations)
        print('Theta found by gradent descent:++++++')
        print(theta[0], theta[1])
        self.assertTrue(np.allclose(theta, np.array([[-3.56218887], [1.15952071]])))

if __name__ == '__main__':
    unittest.main()