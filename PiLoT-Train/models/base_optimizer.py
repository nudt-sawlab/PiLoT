import logging
from typing import Tuple, Dict, Optional
import torch
from torch import Tensor

from .base_model import BaseModel
from ..utils.geometry.wrappers import Pose


class BaseOptimizer(BaseModel):
    default_conf = dict(
        jacobi_scaling=False,
        lambda_=0,
        grad_stop_criteria=1e-4,
        dt_stop_criteria=5e-3,  # in meters
        dR_stop_criteria=5e-2,  # in degrees
    )
    logging = None

    def _init(self, conf):
        assert conf.lambda_ >= 0

    def log(self, **args):
        if self.logging_fn is not None:
            self.logging_fn(**args)

    def _forward(self, data: Dict):
        return self._run(data['pose'], data['B'], data['A'])

    # @torchify
    # def run(self, *args, **kwargs):
    #     return self._run(*args, **kwargs)

    def _run(self, pose: Pose, B: torch.Tensor, A: torch.Tensor):
        raise NotImplementedError

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
