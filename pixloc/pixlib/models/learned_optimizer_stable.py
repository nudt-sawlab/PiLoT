import logging
from typing import Tuple, Optional
import torch
from torch import nn, Tensor
from ..geometry import DirectAbsoluteCost
from ..geometry import direct_abs_cost_cuda
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Agg')
from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa
import numpy as np
logger = logging.getLogger(__name__)

def transform_p3d(body2view_pose_data, p3d):
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)
def project_p3d(camera_data, p3d):
    eps=1e-4

    z = p3d[..., -1]
    valid1 = z > eps
    z = z.clamp(min=eps)
    p2d = p3d[..., :-1] / z.unsqueeze(-1)

    f = camera_data[..., 2:4]
    c = camera_data[..., 4:6]
    p2d = p2d * f.unsqueeze(-2) + c.unsqueeze(-2)

    size = camera_data[..., :2]
    size = size.unsqueeze(-2)
    # valid2 = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
    valid2 = torch.logical_and(p2d >= 0, p2d <= (size - 1))
    valid2 = torch.logical_and(valid2[..., 0], valid2[..., 1])
    valid = torch.logical_and(valid1, valid2)

    return p2d, valid
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
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
        return lambda_


class LearnedOptimizer(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        feature_dim=None,

        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.dampingnet = DampingNet(conf.damping)
        assert conf.learned_damping
        self.ratio = conf.ratio
        self.fn1 = DirectAbsoluteCost.residual_jacobian_batch_quat
        
        self.fn = direct_abs_cost_cuda.residual_jacobian_batch_quat_cuda
        self.optimizer_cuda = direct_abs_cost_cuda.optimizer_step_cuda
        super()._init(conf)
    def compute_overall_loss(self, res, valid, weights, loss_fn=None):
        """
        Compute the overall loss after each iteration of pose optimization.

        Args:
            res (Tensor): Residuals tensor of shape (..., N), where N is the number of residuals.
            valid (Tensor): Boolean mask indicating valid residuals of shape (..., N).
            weights (Tensor): Weights tensor of shape (..., N).
            loss_fn (callable, optional): Robust loss function applied to squared residuals.

        Returns:
            Tensor: Overall weighted loss for the current iteration.
        """
        # Compute squared residuals
        squared_residuals = (res**2).sum(-1)
        # Apply robust loss function if provided
        if loss_fn is not None:
            loss = loss_fn(squared_residuals)
        else:
            loss = squared_residuals

        # Apply valid mask and weights
        loss = loss * valid.float()
        loss = loss * weights

        # Sum the loss over all residuals
        overall_loss = loss.sum(dim=-1)  # Sum over residual dimension

        return overall_loss
    def plot_loss_curve(self, overall_losses, save_path=None):
        """
        Plot the loss curve for the optimization process.

        Args:
            overall_losses (list or array): List of overall loss values for each iteration.
            save_path (str, optional): Path to save the plot image. If None, the plot will not be saved.
        """
        # Generate the iterations corresponding to the losses
        iterations = list(range(len(overall_losses)))

        # # Plot the curve
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, overall_losses, marker='o', linestyle='-', label='Overall Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Overall Loss vs Iteration')
        plt.grid(True)
        plt.legend()

        
        # # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Show the plot
        # plt.show()
    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init, camera: Camera, T_render, ref_camera: Camera,mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None, T_gt = None, batch = True):

        T = T_init
        T_flat = T_init.to_flat()

        num_init_pose = T_init.shape[0]
        T_render = T_render.to_flat().expand(1, -1)
        qcamera = camera.unsqueeze(0).expand(num_init_pose, -1)
        p3D = p3D.unsqueeze(0).expand(1, -1, -1)
        ref_camera = ref_camera.unsqueeze(0).expand(1, -1)
        c_ref, c_query = W_ref_query
        c_ref = c_ref.unsqueeze(0)
        c_query = c_query.unsqueeze(0)
        F_ref = F_ref.unsqueeze(0)
        F_query = F_query.unsqueeze(0)
        args = (F_ref, 
                T_render.unsqueeze(0),
                ref_camera.unsqueeze(0),
                F_query, 
                qcamera.unsqueeze(0), 
                p3D.unsqueeze(0), 
                c_ref,
                c_query
                )
        # args = tuple(x.double() if torch.is_tensor(x) else x for x in args)
        # T_flat = T_flat.double()
        # T = T.double()
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()
        # lambda_ = 0
        overall_losses = []
        ratio = self.ratio
        if ref_camera[0][0] == 480/ratio:
            num_iters = 2
        elif ref_camera[0][0] == 960/ratio:
            num_iters = 3
        elif ref_camera[0][0] == 1920/ratio:
            num_iters = 4
        # if T_gt is not None:
        #     p3d_q_gt = transform_p3d(T_gt, p3D)
        #     p2d_q_gt, _ = project_p3d(qcamera, p3d_q_gt)
        #     p2d_q_gt = p2d_q_gt[0]
        # print('优化前：',T_flat[0])
        # err_list= []
        
        for i in range(num_iters): #self.conf.num_iters
            # g1, H1, w_loss1, valid1, p2d_q1, cost1 = self.fn1(T_flat.unsqueeze(0), *args)
            start_time = time.time()
            # inputs = {
            # "pose_data_q": T_flat.unsqueeze(0),
            # "f_r": F_ref,
            # "pose_data_r": T_render.unsqueeze(0),
            # "cam_data_r": ref_camera.unsqueeze(0),
            # "f_q": F_query,
            # "cam_data_q": qcamera.unsqueeze(0),
            # "p3D": p3D.unsqueeze(0),
            # "c_ref": c_ref,
            # "c_query": c_query
            # }
            # torch.save(inputs, "/home/ubuntu/Documents/code/FPV-Test/DirectAbsoluteCostCuda/sample_inputs.pt")
            g, H, w_loss, valid, p2d_q, cost = direct_abs_cost_cuda.residual_jacobian_batch_quat_cuda(T_flat.unsqueeze(0), *args)
            # g, H, w_loss, valid, p2d_q, cost = self.fn(T_flat.unsqueeze(0), *args)
            # g2, H2, w_loss2, valid2, p2d_q2, cost2 = self.cost_fn.residual_jacobian_batch_quat(T_flat.unsqueeze(0), *args)
            # import ipdb; ipdb.set_trace()
            g, H, w_loss, valid, p2d_q, cost = [
                x.squeeze(0) if x.shape[0] == 1 else x for x in [g, H, w_loss, valid, p2d_q, cost]
            ]
            g = g.unsqueeze(-1)
            failed = failed | (valid.long().sum(-1) < 10)  # too few points
            
            # compute the cost and aggregate the weights
            # delta = optimizer_step(g, H, lambda_, mask=~failed)
            # delta = self.optimizer_cuda(g, H,0.1, ~failed)
            delta = direct_abs_cost_cuda.optimizer_step_cuda(g, H,0.1, ~failed)
            end_time = time.time()
            # print('LM time: ', end_time - start_time)
            # print(f"Step 3 - Optimizer step: {time.time() - st*art_time:.6f}s")
            # compute the pose update
            dw, dt = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T @ T_delta
            T_flat = T.to_flat()
            # p2d_q_gt1 = p2d_q_gt.repeat(T_flat.shape[0], 1, 1)
            # error = p2d_q_gt1 - p2d_q  

            # # 计算每个点的欧氏误差，再对有效点求平均
            # error_norm = torch.norm(error, dim=-1)  # [B, N]
            # error_mean = (error_norm * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)  # [B]
            # err_list.append(error_mean)
            overall_losses.append(cost.sum(dim=1))
            # if ref_camera[0][0] == 480/ratio and len(overall_losses) == 2 and len(T_flat) > 1:
            #     # descents = overall_losses[0] - overall_losses[-1]  # 计算两组的下降量
            #     # fastest_index = int(torch.argmax(descents))  # 取最大下降值对应的索引
            #     # _, topk_indices = torch.topk(descents, 4, dim=-1, largest=True, sorted=True)
            #     _, topk_indices = torch.topk(-overall_losses[-1], 4, dim=-1, largest=True, sorted=True)
            #     # topk_indices = topk_indices[0]
            #     T_flat = T_flat[topk_indices]
            #     T = T[topk_indices]
            #     failed = failed[topk_indices]
            #     qcamera = qcamera[:len(topk_indices)]
            #     cost = cost[topk_indices]
            #     args = (F_ref, 
            #     T_render.unsqueeze(0),
            #     ref_camera.unsqueeze(0),
            #     F_query, 
            #     qcamera.unsqueeze(0), 
            #     p3D.unsqueeze(0), 
            #     c_ref,
            #     c_query
            #     )
            #     # print("下降最快的是第 {} 组".format(topk_indices))
            # if ref_camera[0][0] == 960/ratio and len(overall_losses) == 1 and len(T_flat) > 1:
            #     _, topk_indices = torch.topk(-overall_losses[0], 2, dim=-1, largest=True, sorted=True)
            #     # topk_indices = topk_indices[0]
            #     T_flat = T_flat[topk_indices]
            #     T = T[topk_indices]
            #     failed = failed[topk_indices]
            #     qcamera = qcamera[:len(topk_indices)]
            #     cost = cost[topk_indices]
            #     args = (F_ref, 
            #     T_render.unsqueeze(0),
            #     ref_camera.unsqueeze(0),
            #     F_query, 
            #     qcamera.unsqueeze(0), 
            #     p3D.unsqueeze(0), 
            #     c_ref,
            #     c_query
            #     )
            # if ref_camera[0][0] == 1920/ratio and len(overall_losses) == 1 and len(T_flat) > 1:
            #     _, topk_indices = torch.topk(-overall_losses[0], 1, dim=-1, largest=True, sorted=True)
            #     # topk_indices = topk_indices[0]
            #     T_flat = T_flat[topk_indices]
            #     T = T[topk_indices]
            #     failed = failed[topk_indices]
            #     qcamera = qcamera[:len(topk_indices)]
            #     cost = cost[topk_indices]
            #     args = (F_ref, 
            #     T_render.unsqueeze(0),
            #     ref_camera.unsqueeze(0),
            #     F_query, 
            #     qcamera.unsqueeze(0), 
            #     p3D.unsqueeze(0), 
            #     c_ref,
            #     c_query
            #     )
        # errors_all = torch.stack(err_list, dim=0).T  # [b, 10]
        # 展示每个样本的误差变化 
        # for i in range(len(errors_all)):
        #     print(f"样本 {i} 的误差变化: {errors_all[i]}")

        # print('优化后：',T_flat[0])
        total_cost_loss = cost.sum(dim=1)
        # best_idx = torch.argmin(total_loss)  # 最佳收敛位姿索引
        # print('argmax weight loss: ', torch.argmax(total_loss))
        # print('argmin cost: ', torch.argmin(total_cost_loss))
        
        
        return T, failed, total_cost_loss # T[initial pose]
