import logging
from typing import Tuple, Optional
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..utils.geometry.wrappers import Pose
from packaging import version

logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
    solve = torch.linalg.solve
else:
    cholesky = torch.cholesky
    solve = torch.solve

class DampingNet(nn.Module):
    def __init__(self, conf, num_params=6):
        super().__init__()
        self.conf = conf
        if conf.type == 'constant':
            const = torch.zeros(num_params)
            self.register_parameter('const', torch.nn.Parameter(const))
        else:
            raise ValueError(f'Unsupported type of damping: {conf.type}.')

    def forward(self):
        min_, max_ = self.conf.log_range
        # lambda_ = 5. * (10.**(min_ + self.const.sigmoid()*(max_ - min_)))
        lambda_ = (10.**(min_ + self.const.sigmoid()*(max_ - min_)))
        return lambda_

class LearnedOptimizer(BaseOptimizer):
    default_conf = dict(
        tikhonov_matrix=[5000, 5000, 5000, 500000, 500000, 500000],
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        
        # deprecated entries
        lambda_=0.,
    )

    eps = 1e-6

    def _init(self, conf):
        self.dampingnet = DampingNet(conf.damping)
        # assert conf.trainable
        super()._init(conf)
        self.tikhonov_matrix = torch.diag(torch.tensor(self.conf.tikhonov_matrix, dtype=torch.float32)).cuda()
        
    def optimizer_step(self, B: torch.Tensor, A: torch.Tensor):
        lambda_ = self.dampingnet()
        if lambda_ is 0:
            diag = torch.zeros_like(B)
        else:
            diag = A.diagonal(dim1=-2, dim2=-1) * lambda_.unsqueeze(0)

        if self.conf.trainable:
            self.tikhonov_matrix = diag.clamp(min=self.eps).diag_embed()
        # else:
        #     tikhonov_matrix = torch.diag(torch.tensor([5000, 5000, 5000, 500000, 500000, 500000], dtype=torch.float32)).cuda()
        # self.tikhonov_matrix = self.tikhonov_matrix.cuda()
        self.tikhonov_matrix = self.tikhonov_matrix.to(A)
        A = A + self.tikhonov_matrix

        A_ = A.cpu()
        B_ = B.cpu()
        # try:
        #     U = cholesky(A_)
        #     delta = torch.cholesky_solve(B_, U)[..., 0]
        # except RuntimeError as e:
        #     if "not positive-definite" in str(e) or "singular U" in str(e):
        #         logger.debug("Cholesky failed (likely not PD), fallback to LU.")
        #         try:
        #             delta = torch.linalg.solve(A_, B_)[..., 0]
        #         except RuntimeError as e2:
        #             logger.debug(f"A is not invertible: {e2}")
        #             delta = torch.zeros_like(B_)[..., 0]
        #     else:
        #         raise e
        try:
            U = cholesky(A_)
            
        except RuntimeError as e:
            import ipdb;
            ipdb.set_trace();
            if 'singular U' in str(e):
                logger.debug(
                    'Cholesky decomposition failed, fallback to LU.')
                try:
                    delta = solve(A_, B_)[..., 0]
                except RuntimeError:
                    delta = torch.zeros_like(B_)[..., 0]
                    logger.debug('A is not invertible')
            else:
                raise
        else:
            delta = torch.cholesky_solve(B_, U)[..., 0]        

        return delta.to(A.device)

    def _run(self, pose: Pose, B: torch.Tensor, A: torch.Tensor):
        delta = self.optimizer_step(B, A)
        aa, t = delta.split([3, 3], dim=-1)
        delta_pose = Pose.from_aa(aa, t)
        deformed_pose = pose @ delta_pose

        return deformed_pose