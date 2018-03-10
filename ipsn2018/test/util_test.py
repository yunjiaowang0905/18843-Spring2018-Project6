__author__ = 'Nanshu Wang'
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from util import getA_c, get_par2, cvx_solve_u, get_next_state, limit_range

class getA_c_Test(unittest.TestCase):
    def test(self):
        a = getA_c(1, 1, np.zeros((2, 2)), np.zeros((2, 2)))
        b = np.array([[-4, 1, 1, 0],[1, -4, 0, 1], [1, 0, -4, 1], [0, 1, 1, -4]])
        self.assertEqual(np.array_equal(a, b), True, "getA_c() test fail")
        a = getA_c(3, 10, np.zeros((2, 2)), np.zeros((2, 2)))
        b = np.array([[-0.12,0.03,0.03,0],[0.03,-0.12,0,0.03],
                    [0.03,0,-0.12,0.03],[0,0.03,0.03,-0.12]])
        self.assertEqual(np.array_equal(a, b), True, "getA_c() test fail")

class limit_range_Test(unittest.TestCase):
    def test(self):
        a = limit_range(np.array([-1,0,1,200,3,4,5,6]), 0, 5)
        b = np.array([0,0,1,5,3,4,5,5])
        self.assertEqual(np.array_equal(a, b), True, "limit_range() test fail")


if __name__ == '__main__':
    unittest.main()