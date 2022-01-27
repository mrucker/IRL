import collections

from math import sqrt
from numbers import Real
from itertools import compress
from abc import ABC, abstractmethod
from typing import Callable, Any, Sequence, Tuple, cast, overload, Union

import torch
import numpy as np

Observation = Any
Action      = Any

class Episode:

    def __init__(self, states: Sequence[Observation], actions: Sequence[Action], rewards:Sequence[float] = None) -> None:

        assert len(states)-1 == len(actions), "We assume that there is a trailing state"

        self.states   = states 
        self.actions  = actions
        self.rewards  = rewards

    @property
    def start(self) -> Observation:
        return self.states[0]

    def __len__(self):
        return len(self.states)

class Policy(ABC):
    @abstractmethod
    def act(self, observation) -> Action:
        ...

class Reward(ABC):
    @overload
    def observe(self,*,state: Observation) -> float: ...

    @overload
    def observe(self,*,states: Sequence[Observation]) -> Sequence[float]: ...

    @abstractmethod
    def observe(self, state: Observation = None, states:Sequence[Observation] = None) -> Union[float, Sequence[float]]: ...

class KernelVectorReward(Reward):

    def __init__(self, alpha: 'KernelVector') -> None:
        self._alpha = alpha

    def observe(self, state = None, states = None):
        if state:
            return (self._alpha @ [state]).squeeze().item()
        else:
            return (self._alpha @ states).squeeze(1).tolist()

class LambdaReward(Reward):
    def __init__(self, reward: Callable[[Any], float]) -> None:
        self._reward = reward

    def observe(self, state: Observation = None, states:Sequence[Observation] = None) -> Union[Sequence[float], float]:
        return self._reward(state) if state is not None else [ self._reward(state) for state in states ]

class Regressor(ABC):

    @abstractmethod
    def predict(self, X) -> torch.Tensor:
        ...

class Featurizer(ABC):
    @abstractmethod
    def to_features(self, items: Sequence[Any]) -> torch.Tensor:
        ...

class IdentityFeaturizer(Featurizer):
    def to_features(self, items: torch.Tensor) -> torch.Tensor:
        return items

class LambdaFeaturizer(Featurizer):
    def __init__(self, lambda_featurizer:Callable[[Sequence[Any]], torch.Tensor]) -> None:
        self._lambda_featurizer = lambda_featurizer
    
    def to_features(self, items: Sequence[Any]) -> torch.Tensor:
        return self._lambda_featurizer(items)

class SafeFeaturizer(Featurizer):
    def to_features(self, items: Sequence[Any]) -> torch.Tensor:

        if isinstance(items, torch.Tensor):
            assert items.dim() in [1,2]
            return items if items.dim() == 2 else items.unsqueeze(1)

        if isinstance(items[0], torch.Tensor):
            return torch.vstack(items)

        def safe_list(i):
            return list(i) if isinstance(i, (collections.Sequence, np.ndarray)) else [i]

        return torch.tensor([safe_list(item) for item in items])

class SA_Featurizer(Featurizer):

    def __init__(self, S_featurizer: Featurizer = None, A_featurizer: Featurizer = None):

        self._S_featurizer = S_featurizer or SafeFeaturizer()
        self._A_featurizer = A_featurizer or SafeFeaturizer()

    def to_features(self, items: Sequence[Tuple[Any,Any]]):
    
        S, A = zip(*items)
        return torch.hstack([self._S_featurizer.to_features(S), self._A_featurizer.to_features(A)])

class CachedFeaturizer(Featurizer):
    def __init__(self, featurizer: Featurizer):
        self._cache = {}
        self._featurizer = featurizer

    def _states_in_cache(self, states: Sequence[Observation]) -> bool:

        if not self._cache: return False

        for state in states:

            if state not in self._cache:
                return False

        return True

    def to_features(self, states: Sequence[Observation]) -> torch.Tensor:

        if self._states_in_cache(states):
            return torch.vstack(list(map(self._cache.__getitem__, states)))

        features = self._featurizer.to_features(states)
        
        self._cache.update(zip(states,features))

        return features

class RBF_Featurizer(Featurizer):

    def __init__(self, featurizer: Featurizer) -> None:

        from sklearn.kernel_approximation import RBFSampler

        self._featurizer = featurizer
        self._rbf_featurizer = RBFSampler(n_components=500, gamma=1)

    def to_features(self, items: Sequence[Any]) -> torch.Tensor:

        A = torch.exp(-torch.cdist(self._featurizer.to_features(items),self._featurizer.to_features(items))/0.1)
        B = torch.from_numpy(self._rbf_featurizer.fit_transform(self._featurizer.to_features(items)))
        C = B@B.T

        #Fit only cares about how many columns are in items. So we can call fit each time.
        return torch.from_numpy(self._rbf_featurizer.fit_transform(self._featurizer.to_features(items)))

class Poly_Featurizer(Featurizer):

    def __init__(self, featurizer: Featurizer, degree:int=2, interaction_only:bool=False) -> None:

        from sklearn.preprocessing import PolynomialFeatures

        self._featurizer      = featurizer
        self._poly_featurizer = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    def to_features(self, items: Sequence[Any]) -> torch.Tensor:
        return torch.from_numpy(self._poly_featurizer.fit_transform(self._featurizer.to_features(items)))

class Kernel(ABC):

    @abstractmethod
    def eval(self, items1: Sequence[Any], items2: Sequence[Any]) -> torch.Tensor:
        ...

    def equal(self, items1: Sequence[Any], items2: Sequence[Any]) -> torch.BoolTensor:
        #Warning KPIRL is incredibly sensitive to equality issues. So, be careful
        #when implementing custom kernels and strongly consider overriding this
        
        items1_dot_items1_diag = torch.tensor([ [self.eval([i],[i])] for i in items1 ])
        items2_dot_items2_diag = torch.tensor([ [self.eval([i],[i])] for i in items2 ])
        items1_dot_items2      = self.eval(items1, items2)

        #the first two items are row and column vectors so they broadcast over the last item
        squared_distance_matrix = items1_dot_items1_diag + items2_dot_items2_diag - 2*items1_dot_items2

        #pytorch with float32 precision isn't the most 
        #accurate so we use < .00001 rather than strict == 0 
        return squared_distance_matrix < 10**(-5)

class DotKernel(Kernel):

    def eval(self, items1: Sequence[torch.Tensor], items2: Sequence[torch.Tensor]) -> torch.Tensor:

        s1_stacked = torch.vstack(list(items1)) if not isinstance(items1, torch.Tensor) else items1
        s2_stacked = torch.vstack(list(items2)) if not isinstance(items2, torch.Tensor) else items2
        
        return s1_stacked.float() @ s2_stacked.T.float()

    def equal(self, items1: Sequence[torch.Tensor], items2: Sequence[torch.Tensor]) -> torch.BoolTensor:
        return EqualityKernel().equal(items1,items2)

class EqualityKernel(Kernel):

    def eval(self, items1: Sequence[Any], items2: Sequence[Any]) -> torch.Tensor:
        return self.equal(items1,items2).float()

    def equal(self, items1: Sequence[Any], items2: Sequence[Any]) -> torch.BoolTensor:
        #this needs to stay all true for the tensor logic to work
        equality_matrix = torch.full((len(items1),len(items2)), True)
        
        s1_is_tensor_list = not isinstance(items1, torch.Tensor) and isinstance(items1[0], torch.Tensor)
        s2_is_tensor_list = not isinstance(items2, torch.Tensor) and isinstance(items2[0], torch.Tensor)

        if  s1_is_tensor_list and s2_is_tensor_list:
            items1 = torch.vstack(items1) #type:ignore
            items2 = torch.vstack(items2) #type:ignore

        if isinstance(items1, torch.Tensor) and isinstance(items2, torch.Tensor):
            if items1.ndim == 1: items1 = items1.unsqueeze(0)
            if items2.ndim == 1: items2 = items2.unsqueeze(0)

            equality_matrix = (items1.unsqueeze(1) == items2.unsqueeze(0)).all(dim=2)
        
        else:
            equality_matrix = torch.full((len(items1),len(items2)), True)

            for i in range(len(items1)):
                for j in range(len(items2)):
                    is_equal = (items1[i] == items2[j])
                    equality_matrix[i,j] = is_equal
        
        return cast(torch.BoolTensor, equality_matrix)

class GaussianKernel(Kernel):

    def __init__(self, bandwidth: float = 1) -> None:
        self._bandwidth = bandwidth

    def eval(self, items1: Sequence[torch.Tensor], items2: Sequence[torch.Tensor]) -> torch.Tensor:
        
        s1_stacked = items1 if isinstance(items1, torch.Tensor) else torch.vstack(list(items1))
        s2_stacked = items2 if isinstance(items2, torch.Tensor) else torch.vstack(list(items2))

        #if we don't manually set the mode then the cdist matrix is actually wrong when dealing with large matrices
        return torch.exp(-torch.cdist(s1_stacked.float(),s2_stacked.float(), compute_mode='donot_use_mm_for_euclid_dist')/self._bandwidth)

    def equal(self, items1: Sequence[torch.Tensor], items2: Sequence[torch.Tensor]) -> torch.BoolTensor:
        return EqualityKernel().equal(items1,items2)

class KernelVector:

    def __init__(self, kernel: Kernel, coefs: Sequence[float], items: Sequence[Any]) -> None:
        
        assert len(coefs) == len(items), "Invalid tensor params"

        if len(items) > 1:
            pequal = kernel.equal(items,items).float()

            coefs  = (pequal @ torch.tensor(coefs).float().unsqueeze(1)).squeeze().tolist()
            unique = pequal.argmax(0).unique()

            coefs = [coefs[i] for i in unique]
            items = [items[i] for i in unique]

        self.kernel = kernel
        self.items  = items
        self._coefs = torch.tensor(coefs).unsqueeze(1).float()

    @property
    def coefs(self) -> Sequence[float]:
        return self._coefs.squeeze(1).tolist()

    def norm(self) -> float:
        return sqrt(self @ self)

    def __len__(self) -> int:
        return len(self.coefs)

    def __add__(self, other):
        if isinstance(other, KernelVector):
            pequal      = self.kernel.equal(self.items, other.items)
            not_in_self = (~pequal).all(0)

            #this calculation is only correct if self and other are already reduced
            new_coefs = self._coefs + pequal.float() @ other._coefs

            coefs = torch.vstack([new_coefs , other._coefs[not_in_self]]).squeeze(1).tolist()
            items = list(self.items) + list(compress(other.items, not_in_self))

            return KernelVector(self.kernel, coefs, items)
        
        raise Exception("Invalid KernelVector addition type")

    def __sub__(self, other):
        
        if isinstance(other, KernelVector):
            return self + (-1 * other)
        
        raise Exception("Invalid KernelVector subtraction type")

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, Real):
            return KernelVector(self.kernel, (other*self._coefs).squeeze(1).tolist(), self.items)

        if isinstance(other, KernelVector):
            pequal = self.kernel.equal(self.items, other.items).float()

            assert len(self.items) == len(other.items) and all(pequal.sum(1) == 1)

            return KernelVector(self.kernel, (self._coefs*(pequal@other._coefs)).squeeze(1).tolist(), self.items)

        raise Exception("Invalid KernelVector scalar multiplication type")

    def __truediv__(self, other):
        return self * (1/other)

    def __rtruediv__(self, other):
        return other * KernelVector(self.kernel, (1/self._coefs).squeeze(1).tolist(), self.items)

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            other = torch.from_numpy(other)

        if isinstance(other, KernelVector):
            return (self._coefs.T @ self.kernel.eval(self.items, other.items) @ other._coefs).squeeze().item()

        # is_multi_tensor = isinstance(other,torch.Tensor) and other.dim() > 1

        # if isinstance(other, type(self.items[0])) and not is_multi_tensor:
        #     other = [other]

        if isinstance(other[0], type(self.items[0])):
            result      = torch.tensor([[0]]*len(other)).float()
            batch_size  = 20000
            upper_index = 0
            lower_index = 0

            for i in range(int(len(other)/batch_size)):
                lower_index =  i   *batch_size
                upper_index = (i+1)*batch_size
                result[lower_index:upper_index] = self.kernel.eval(other[lower_index:upper_index], self.items) @ self._coefs

            if upper_index != len(other):
                result[upper_index:] = self.kernel.eval(other[upper_index:], self.items) @ self._coefs

            return result
        
        raise Exception("Invalid KernelVector vector multiplication type")