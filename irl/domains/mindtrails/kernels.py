from irl.kernels import Kernel, GaussianKernel
from irl.domains.mindtrails.models import MindtrailModel

class MindtrailsKernel(Kernel):

    def __init__(self, model: MindtrailModel) -> None:
        self._model = model

    def __call__(self, items1, items2):

        items1 = list(map(self._model.post_state, *zip(*items1)))
        items2 = list(map(self._model.post_state, *zip(*items2)))        

        return GaussianKernel()(items1, items2)
