import math

from abc import ABC, abstractmethod
from operator import mul
from typing import Any, Sequence, Tuple, Iterator, List

from irl.models import Reward

class Kernel(ABC):

    @abstractmethod
    def __call__(self, items1: Sequence[Any], items2: Sequence[Any]) -> Sequence[Sequence[float]]:
        ...

class DotKernel(Kernel):

    def __call__(self, items1: Sequence[Sequence[float]], items2: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:

        row_size = len(items1[0])
        assert all([ len(item1) == row_size for item1 in items1])
        assert all([ len(item2) == row_size for item2 in items2])

        gram: List[List[float]] = [ [0]*len(items2) for _ in range(len(items1)) ]
        
        for r in range(len(items1)):
            for c in range(len(items2)):
                gram[r][c] = sum(map(mul,items1[r],items2[c]))

        return gram

class GaussianKernel(Kernel):

    def __init__(self, gamma: float = 1) -> None:
        self._gamma = gamma

    def __call__(self, items1: Sequence[Sequence[float]], items2: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:

        dots1 = [ sum(map(mul,item1,item1)) for item1 in items1 ]
        dots2 = [ sum(map(mul,item2,item2)) for item2 in items2 ]
        dist2 = lambda r,c: dots1[r]+dots2[c]-2*sum(map(mul,items1[r],items2[c]))
        gram = [ [0]*len(items2) for _ in range(len(items1)) ]

        for r in range(len(items1)):
            for c in range(len(items2)):                
                gram[r][c] = math.exp(-dist2(r,c)/self._gamma)

        return gram 

class KernelVector:

    def __init__(self, kernel: Kernel, coefs: Sequence[float], items: Sequence[Any]) -> None:
        
        assert len(coefs) == len(items), "Invalid kernel vector"

        self.kernel = kernel
        self.items  = items
        self.coefs  = coefs

    def __iter__(self) -> Iterator[Tuple[float,Any]]:
        return zip(self.coefs, self.items)

    def __len__(self) -> int:
        return len(self.coefs)

    def __add__(self, other: 'KernelVector'):
        assert other.kernel == self.kernel
        return KernelVector(self.kernel, list(self.coefs)+list(other.coefs), list(self.items)+list(other.items))

    def __sub__(self, other: 'KernelVector'):
        assert other.kernel == self.kernel
        return self + (-1 * other)

    def __mul__(self, other:float):
        return KernelVector(self.kernel, [other*c for c in self.coefs], self.items)

    def __truediv__(self, other:float):
        return self * (1/other)

    def __rmul__(self, other:float):
        return self * other

    def __rtruediv__(self, other:float):
        return other * KernelVector(self.kernel, (1/self._coefs).squeeze(1).tolist(), self.items)

    def __matmul__(self, other: 'KernelVector'):
        assert other.kernel == self.kernel
        K = self.kernel.eval(self.items, other.items)
        return sum([ self.coefs[i]*other.coefs[j]*K[i,j] for i in range(len(self)) for j in range(len(other))])

class KernelReward(Reward):

    def __init__(self, alpha: 'KernelVector') -> None:
        self._alpha  = alpha
        self._reward = lambda s: sum([ c*self._alpha.kernel(i,s) for c,i in self._alpha ])

    def observe(self, state = None, states = None):       
        if state:
            return self._reward(state)
        else:
            return list(map(self._reward,states))
