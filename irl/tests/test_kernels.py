import unittest

import torch

from combat.representations import EqualityKernel, GaussianKernel, DotKernel, KernelVector

class EqualityKernel_Tests(unittest.TestCase):

    def test_singleton_eval(self): 
        kernel = EqualityKernel()

        s1 = torch.tensor([[1]]).float()
        s2 = torch.tensor([[0],[1],[2]]).float()

        actual = kernel.eval(s1, s2)
        expected = torch.tensor([[0,1,0]]).float()

        self.assertTrue(torch.equal(actual,expected))

class GaussianKernel_Tests(unittest.TestCase):

    def test_singleton_eval(self): 
        kernel = GaussianKernel()

        s1 = torch.tensor([[1]])
        s2 = torch.tensor([[0],[1],[2]])

        actual   = kernel.eval(s1, s2)
        expected = torch.exp(-torch.tensor([[1.,0.,1.]]))

        self.assertTrue(torch.equal(actual,expected))

class KernelVector_Tests(unittest.TestCase):
    
    def test_make_vector1(self):
        items = torch.tensor([[1,2,3],[1,2,3]])
        coefs = torch.tensor([[1],[2]]).squeeze().tolist()

        v = KernelVector(EqualityKernel(), coefs, items)
        
        self.assertEqual(1, len(v.items))
        self.assertTrue(torch.equal(v.items[0],torch.tensor([1,2,3])))
        self.assertEqual(v.coefs,[3])

    def test_make_vector2(self):
        items = torch.tensor([[1],[2],[1],[2]])
        coefs = torch.tensor([[1],[2],[1],[2]]).squeeze().tolist()

        v = KernelVector(EqualityKernel(), coefs, items)

        self.assertEqual(2, len(v.items))

        self.assertTrue(torch.equal(v.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v.items[1],torch.tensor([2])))
        self.assertEqual(v.coefs,[2,4])

    def test_make_vector3(self):
        items = [torch.tensor([1])]
        coefs = [1]

        v = KernelVector(EqualityKernel(), coefs, items)

        self.assertTrue(torch.equal(v.items[0], torch.tensor([1])))
        self.assertEqual(v.coefs, [1])

    def test_make_vector4(self):
        items = [torch.tensor([1,2])]
        coefs = [1]

        v = KernelVector(EqualityKernel(), coefs, items)

        self.assertTrue(torch.equal(v.items[0], torch.tensor([1, 2])))
        self.assertEqual(v.coefs,[1])

    def test_make_vector5(self):
        items = torch.tensor([[1,2]])
        coefs = [1]

        v = KernelVector(EqualityKernel(), coefs, items)

        self.assertTrue(torch.equal(v.items[0], torch.tensor([1,2])))
        self.assertEqual(v.coefs, [1])
    
    def test_make_vector6(self):
        items = torch.tensor([[1,2,3],[1,2,3]])
        coefs = [1,2]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual(1, len(v.items))
        self.assertTrue(torch.equal(v.items[0],torch.tensor([1,2,3])))
        self.assertEqual(v.coefs,[3])

    def test_make_vector7(self):
        items = torch.tensor([[1],[2],[1],[2]])
        coefs = [1,2,1,2]

        v = KernelVector(DotKernel(), coefs, items)

        self.assertEqual(2, len(v.items))
        self.assertTrue(torch.equal(v.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v.items[1],torch.tensor([2])))
        self.assertEqual(v.coefs,[2,4])

    def test_add1(self):
        v1 = KernelVector(EqualityKernel(), [1], torch.tensor([[1]]))
        v2 = KernelVector(EqualityKernel(), [1], torch.tensor([[1]]))

        v3 = v1+v2

        self.assertEqual(1, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertEqual(v3.coefs,[2])

    def test_add2(self):
        v1 = KernelVector(EqualityKernel(), [1   ], torch.tensor([[1]]))
        v2 = KernelVector(EqualityKernel(), [1, 3], torch.tensor([[1],[2]]))

        v3 = v1+v2

        self.assertEqual(2, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v3.items[1],torch.tensor([2])))
        self.assertEqual(v3.coefs, [2,3])

    def test_add3(self):
        v1 = KernelVector(EqualityKernel(), [1  ], torch.tensor([[1]]))
        v2 = KernelVector(EqualityKernel(), [3,1], torch.tensor([[2],[1]]))

        v3 = v1+v2

        self.assertEqual(2, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v3.items[1],torch.tensor([2])))
        self.assertEqual(v3.coefs, [2,3])

    def test_sub1(self):
        v1 = KernelVector(EqualityKernel(), [1], torch.tensor([[1]]))
        v2 = KernelVector(EqualityKernel(), [1], torch.tensor([[1]]))

        v3 = v1-v2

        self.assertEqual(1, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertEqual(v3.coefs, [0])

    def test_sub2(self):
        v1 = KernelVector(EqualityKernel(), [1   ], torch.tensor([[1]]))
        v2 = KernelVector(EqualityKernel(), [3, 1], torch.tensor([[2],[1]]))

        v3 = v1-v2

        self.assertEqual(2, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v3.items[1],torch.tensor([2])))
        self.assertEqual(v3.coefs, [0,-3])

    def test_rmul(self):
        v1 = KernelVector(EqualityKernel(), [3,1], torch.tensor([[2],[1]]))
        v2 = 2*v1

        self.assertEqual(2, len(v2.items))
        self.assertTrue(torch.equal(v2.items[0],torch.tensor([2])))
        self.assertTrue(torch.equal(v2.items[1],torch.tensor([1])))
        self.assertEqual(v2.coefs, [6,2])

    def test_mul1(self):
        v1 = KernelVector(EqualityKernel(), [3,1], torch.tensor([[2],[1]]))
        v2 = v1*2

        self.assertEqual(2, len(v2.items))
        self.assertTrue(torch.equal(v2.items[0],torch.tensor([2])))
        self.assertTrue(torch.equal(v2.items[1],torch.tensor([1])))
        self.assertEqual(v2.coefs, [6,2])

    def test_mul2(self):
        v1 = KernelVector(EqualityKernel(), [3,1], torch.tensor([[2],[1]]))
        v2 = KernelVector(EqualityKernel(), [2,2], torch.tensor([[2],[1]]))
        v3 = v1*v2

        self.assertEqual(2, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([2])))
        self.assertTrue(torch.equal(v3.items[1],torch.tensor([1])))
        self.assertEqual(v3.coefs, [6,2])

    def test_mul3(self):
        v1 = KernelVector(EqualityKernel(), [1,3], torch.tensor([[1],[2]]))
        v2 = KernelVector(EqualityKernel(), [2,2], torch.tensor([[2],[1]]))
        v3 = v1*v2

        self.assertEqual(2, len(v3.items))
        self.assertTrue(torch.equal(v3.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v3.items[1],torch.tensor([2])))
        self.assertEqual(v3.coefs, [2,6])

    def test_mul4(self):
        v1 = KernelVector(EqualityKernel(), [1,3,4], torch.tensor([[1],[2],[3]]))
        v2 = KernelVector(EqualityKernel(), [2,2  ], torch.tensor([[2],[1]    ]))

        with self.assertRaises(AssertionError):
            v3 = v1*v2

    def test_div(self):
        v1 = KernelVector(EqualityKernel(), [1,3,9], torch.tensor([[1],[2],[3]]))
        v2 = 9/v1

        self.assertEqual(3, len(v2.items))
        self.assertTrue(torch.equal(v2.items[0],torch.tensor([1])))
        self.assertTrue(torch.equal(v2.items[1],torch.tensor([2])))
        self.assertTrue(torch.equal(v2.items[2],torch.tensor([3])))
        self.assertEqual(v2.coefs, [9,3,1])

    def test_len(self):
        v1 = KernelVector(EqualityKernel(), [2,1], torch.tensor([[3,1],[1,1]]))

        self.assertEqual(2, len(v1))

    def test_assert(self):
        with self.assertRaises(AssertionError):
            KernelVector(EqualityKernel(), torch.tensor([[2]]), torch.tensor([[3,1],[1,1]]))

    def test_matmul1(self):
        v1 = KernelVector(DotKernel(), [2,1], torch.tensor([[3,1],[1,1]]))
        v2 = KernelVector(DotKernel(), [1  ], torch.tensor([[3,1]]))

        #this direct solution method is only valid for the dot kernel
        expected = (2*1)*(3*3+1*1) + (1*1)*(3*1+1*1)
        actual   = v1@v2

        self.assertEqual(actual, expected)

    def test_matmul2(self):
        v1 = KernelVector(DotKernel(), [2,1], torch.tensor([[3,1],[1,1]]))
        v2 = torch.tensor([[3,1]]).float()

        #this direct solution method is only valid for the dot kernel
        expected = (2*1)*(3*3+1*1) + (1*1)*(3*1+1*1)
        actual   = v1 @ v2

        self.assertEqual(actual, expected)

    def test_matmul3(self):
        coefs = torch.tensor([[2],[1]]).float()
        items = torch.tensor([[3,1],[1,1]]).float()

        v1 = KernelVector(DotKernel(), coefs.squeeze().tolist(), items)
        v2 = torch.tensor([[3,1], [4,1], [10,1]]).float()

        #this direct solution method is only valid for the dot kernel
        expected = v2 @ items.T @ coefs
        actual   = v1 @ v2

        self.assertTrue(torch.equal(expected, actual))

    def test_matmul4(self):
        coefs = torch.tensor([[2],[1]]).float()
        items = torch.tensor([[3,1],[1,1]]).float()

        v1 = KernelVector(DotKernel(), coefs.squeeze().tolist(), items)
        v2 = list(torch.tensor([[3,1], [4,1], [10,1]]).float())

        #this direct solution method is only valid for the dot kernel
        expected = torch.stack(v2) @ items.T @ coefs
        actual   = v1 @ v2

        self.assertTrue(torch.equal(expected, actual))

    def test_matmul5(self):
        coefs = torch.tensor([[2],[1]]).float()
        items = torch.tensor([[3,1],[1,1]]).float()

        v1 = KernelVector(DotKernel(), coefs.squeeze().tolist(), items)
        v2 = list(torch.rand(75000,2).float())

        #this direct solution method is only valid for the dot kernel
        expected = torch.stack(v2) @ items.T @ coefs
        actual   = v1 @ v2

        self.assertTrue(torch.equal(expected, actual))

    def test_dist(self):
        coefs = torch.tensor([[2],[1]]).float()
        items = torch.tensor([[3,1],[1,1]]).float()

        v1 = KernelVector(DotKernel(), coefs.squeeze().tolist(), items)

        expected = ((coefs.T @ items) @ (coefs.T @ items).T).sqrt().item()

        self.assertAlmostEqual(expected, v1.norm(), 5)