import math
import unittest

import torch

from irl.kernels import GaussianKernel, DotKernel, KernelVector

class GaussianKernel_Tests(unittest.TestCase):

    def test_singleton_eval(self): 
        kernel = GaussianKernel()

        s1 = [[1]]
        s2 = [[0],[1],[2]]

        actual   = kernel(s1, s2)
        expected = [[math.exp(-1), math.exp(0), math.exp(-1)]]

        self.assertEqual(actual,expected)

class KernelVector_Tests(unittest.TestCase):
    
    def test_make_vector1(self):
        items = [[1],[2]]
        coefs = [1,2]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual([[1],[2]], v.items)
        self.assertEqual([1,2], v.coefs)

    def test_make_vector2(self):
        items = [[1],[2],[1],[2]]
        coefs = [1,2,1,2]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual([[1],[2]], v.items)
        self.assertEqual([2,4], v.coefs)

    def test_make_vector3(self):
        items = [[1]]
        coefs = [1]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual([[1]], v.items)
        self.assertEqual([1], v.coefs)
    
    def test_make_vector6(self):
        items = [[1,2,3],[1,2,3]]
        coefs = [1,2]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual([[1,2,3]], v.items)
        self.assertEqual([3], v.coefs)

    def test_add1(self):
        v1 = KernelVector(DotKernel(), [1], [[1]])
        v2 = KernelVector(DotKernel(), [1], [[1]])

        v3 = v1+v2

        self.assertEqual([[1]], v3.items)
        self.assertEqual([2], v3.coefs)

    def test_add2(self):
        v1 = KernelVector(DotKernel(), [1   ], [[1]])
        v2 = KernelVector(DotKernel(), [1, 3], [[1],[2]])

        v3 = v1+v2

        self.assertEqual([[1],[2]], v3.items)
        self.assertEqual([2,3], v3.coefs)

    def test_add3(self):
        v1 = KernelVector(DotKernel(), [1   ], [[1]])
        v2 = KernelVector(DotKernel(), [3, 1], [[2],[1]])

        v3 = v1+v2

        self.assertEqual([[1],[2]], v3.items)
        self.assertEqual([2,3], v3.coefs)

    def test_sub1(self):
        v1 = KernelVector(DotKernel(), [1], [[1]])
        v2 = KernelVector(DotKernel(), [1], [[1]])

        v3 = v1-v2

        self.assertEqual(0, len(v3.items))
        self.assertEqual(0, len(v3.coefs))

    def test_sub2(self):
        v1 = KernelVector(DotKernel(), [1   ], [[1]])
        v2 = KernelVector(DotKernel(), [3, 1], [[2],[1]])

        v3 = v1-v2

        self.assertEqual([[2]], v3.items)
        self.assertEqual([-3 ], v3.coefs)

    def test_mul(self):
        v1 = KernelVector(DotKernel(), [3,1], [[1],[2]])
        v2 = 2*v1
        v3 = v1*2

        self.assertEqual([[1],[2]], v1.items)
        self.assertEqual([3,1]    , v1.coefs)

        self.assertEqual([[1],[2]], v2.items)
        self.assertEqual([6,2]    , v2.coefs)

        self.assertEqual([[1],[2]], v3.items)
        self.assertEqual([6,2]    , v3.coefs)

    def test_div(self):
        v1 = KernelVector(DotKernel(), [6,2], [[1],[2]])
        v2 = v1/2

        self.assertEqual([[1],[2]], v1.items)
        self.assertEqual([6,2]    , v1.coefs)

        self.assertEqual([[1],[2]], v2.items)
        self.assertEqual([3,1]    , v2.coefs)

    def test_len(self):
        self.assertEqual(2, len(KernelVector(DotKernel(), [2,1], [[3,1],[1,1]])))

    def test_assert(self):
        with self.assertRaises(AssertionError):
            KernelVector(DotKernel(), [2], [[3,1],[1,1]])

    def test_matmul1(self):
        v1 = KernelVector(DotKernel(), [2,1], [[3,1],[1,1]])
        v2 = KernelVector(DotKernel(), [1  ], [[3,1]])

        #this direct solution method is only valid for the dot kernel
        expected = (2*1)*(3*3+1*1) + (1*1)*(3*1+1*1)
        actual   = v1@v2

        self.assertEqual(actual, expected)

    def test_matmul2(self):
        v1 = KernelVector(DotKernel(), [2,1], [[3,1],[1,1]])
        v2 = [[3,1]]

        #this direct solution method is only valid for the dot kernel
        expected = [(2*1)*(3*3+1*1) + (1*1)*(3*1+1*1)]
        actual   = v1 @ v2

        self.assertEqual(actual, expected)

    def test_matmul3(self):
        v1 = KernelVector(DotKernel(), [2,1], [[3,1],[1,1]])
        v2 = [[3,1], [4,2]]

        #this direct solution method is only valid for the dot kernel
        expected = [2*(3*3+1*1) + (3*1+1*1), 2*(4*3+2*1) + (4*1+2*1)]
        actual   = v1 @ v2

        self.assertEqual(expected, actual)
