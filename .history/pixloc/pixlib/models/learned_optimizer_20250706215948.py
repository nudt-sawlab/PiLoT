import logging
from typing import Tuple, Optional
import torch
import copy
from torch import nn, Tensor
from ..geometry import DirectAbsoluteCost
import direct_abs_cost_cuda
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
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None, T_gt = None, batch = True, last_F_query=None, last_c_query=None, prior = False, T_kf =None):
        # torch.cuda.synchronize()
        # t0 = time.perf_counter()
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

        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)
        overall_losses = []
        ratio = self.ratio
        if ref_camera[0][0] == 480/ratio:
            num_iters = 2
        elif ref_camera[0][0] == 960/ratio:
            num_iters = 3
        elif ref_camera[0][0] == 1920/ratio:
            num_iters = 4
        if last_F_query is not None:
            last_c_query = last_c_query.unsqueeze(0)
            last_F_query = last_F_query.unsqueeze(0)
            inputs_last = {
            "pose_data_q": T_flat.unsqueeze(0).clone(),
            "f_r": last_F_query.clone(),
            "pose_data_r": T_render.unsqueeze(0).clone(),
            "cam_data_r": ref_camera.unsqueeze(0).clone(),
            "f_q": F_query.clone(),
            "cam_data_q": qcamera.unsqueeze(0).clone(),
            "p3D": p3D.unsqueeze(0).contiguous().clone(),
            "c_ref": last_c_query.clone(),
            "c_query": c_query.clone()
            }
        inputs = {
            "pose_data_q": T_flat.unsqueeze(0).clone(),
            "f_r": F_ref.clone(),
            "pose_data_r": T_render.unsqueeze(0).clone(),
            "cam_data_r": ref_camera.unsqueeze(0).clone(),
            "f_q": F_query.clone(),
            "cam_data_q": qcamera.unsqueeze(0).clone(),
            "p3D": p3D.unsqueeze(0).contiguous().clone(),
            "c_ref": c_ref.clone(),
            "c_query": c_query.clone()
            }
        
        T_kf_pred = torch.eye(4, 4, device=T_render.device)
        T_kf_pred[:3, :3] = T_kf.R
        T_kf_pred[:3, 3] = T_kf.t
        T_kf_batch = T_kf_pred.unsqueeze(0).repeat(T.shape[0], 1, 1)  # [B,4,4]
        T_kf_batch = Pose.from_Rt(T_kf_batch[:, :3, :3], T_kf_batch[:, :3, 3])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(num_iters): #self.conf.num_iters
            start_time = time.time()
            # torch.save(inputs, "/home/ubuntu/Documents/code/FPV-Test/DirectAbsoluteCostCuda/sample_inputs.pt")
            g, H, w_loss, cost = self.fn(
            inputs['pose_data_q'],
            inputs['f_r'],
            inputs['pose_data_r'],
            inputs['cam_data_r'],
            inputs['f_q'],
            inputs['cam_data_q'],
            inputs['p3D'],
            inputs['c_ref'],
            inputs['c_query']
            )
            g, H, w_loss, cost = [
                x.squeeze(0) if x.shape[0] == 1 else x for x in [g, H, w_loss, cost]
            ]
            if prior and num_iters == 4:
                lambda_prior = 1e-3  # 可调参数

                # 相对变换 T_kf^-1 * T
                T_rel = T_kf_batch.inv() @ T  # [N, 4, 4]
                aa_rel = T_rel.to_aa()  # [N, 6]

                # 添加一阶近似的正则梯度项和单位Hessian
                g_prior = lambda_prior * aa_rel
                H_prior = lambda_prior * torch.eye(6, device=g.device).unsqueeze(0).repeat(T.shape[0], 1, 1)

                g += g_prior
                H += H_prior
            if last_F_query is not None:
                g1, H1, w_loss1, cost1 = self.fn(
                inputs_last['pose_data_q'],
                inputs_last['f_r'],
                inputs_last['pose_data_r'],
                inputs_last['cam_data_r'],
                inputs_last['f_q'],
                inputs_last['cam_data_q'],
                inputs_last['p3D'],
                inputs_last['c_ref'],
                inputs_last['c_query']
            )
                g1, H1, w_loss1, cost1 = [
                    x.squeeze(0) if x.shape[0] == 1 else x for x in [g1, H1, w_loss1,cost1]
                ]
                ratio_last = 0.5  # 控制上一帧在LM中的权重（0~1之间）
                g += ratio_last * g1
                H += ratio_last * H1
                w_loss += ratio_last * w_loss1
                cost += ratio_last * cost1
            g = g.unsqueeze(-1)
            failed = failed 
            
            # compute the cost and aggregate the weights
            delta = self.optimizer_cuda(g, H,0.1, ~failed)
            
            # print(f"Step 3 - Optimizer step: {time.time() - start_time:.6f}s")
            # compute the pose update
            dw, dt = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T @ T_delta
            T_flat = T.to_flat()
            inputs["pose_data_q"]= T_flat.unsqueeze(0).contiguous().clone()
           
            overall_losses.append(w_loss)
            
            if ref_camera[0][0] == 480/ratio and len(overall_losses) == 2 and len(T_flat) > 1:
                # num = torch.count_nonzero(cost, dim=1)
                # row_sum = cost.sum(dim=1)                 # shape = [B]
                # nonzero_count = torch.count_nonzero(cost, dim=1)  # PyTorch ≥ 1.7
                # safe_count = nonzero_count.clone()
                # safe_count[safe_count == 0] = 1         # 把 0 改成 1，暂时防止除 0
                # row_avg = row_sum / safe_count 
                # _, topk_indices = torch.topk(overall_losses[-1], 32, dim=-1, largest=True, sorted=True)
                
                _, topk_indices = torch.topk(-overall_losses[-1], 32, dim=-1, largest=True, sorted=True)
                T_flat = T_flat[topk_indices]
                T = T[topk_indices]
                failed = failed[topk_indices]
                qcamera = qcamera[:len(topk_indices)]
                cost = cost[topk_indices]
                inputs["pose_data_q"] =  T_flat.unsqueeze(0).contiguous().clone()
                inputs["cam_data_q"] = qcamera.unsqueeze(0).clone()
                # print('topk_indices: ',topk_indices)
                if last_F_query is not None:
                    inputs_last["pose_data_q"] =  T_flat.unsqueeze(0).contiguous().clone()
                    inputs_last["cam_data_q"] = qcamera.unsqueeze(0).clone()

                # print("下降最快的是第 {} 组".format(topk_indices))
            # if ref_camera[0][0] == 960/ratio and len(overall_losses) == 1 and len(T_flat) > 1:
            #     _, topk_indices = torch.topk(-overall_losses[0], 32, dim=-1, largest=True, sorted=True)
            #     T_flat = T_flat[topk_indices]
            #     T = T[topk_indices]
            #     failed = failed[topk_indices]
            #     qcamera = qcamera[:len(topk_indices)]
            #     cost = cost[topk_indices]
            #     inputs["pose_data_q"] =  T_flat.unsqueeze(0).contiguous().clone()
            #     inputs["cam_data_q"] = qcamera.unsqueeze(0).contiguous().clone()
            #     # print("下降最快的是第 {} 组".format(topk_indices))
            # if ref_camera[0][0] == 1920/ratio and len(overall_losses) == 1 and len(T_flat) > 1:
            #     _, topk_indices = torch.topk(-overall_losses[0], 32, dim=-1, largest=True, sorted=True)
            #     # topk_indices = topk_indices[0]
            #     T_flat = T_flat[topk_indices]
            #     T = T[topk_indices]
            #     failed = failed[topk_indices]
            #     qcamera = qcamera[:len(topk_indices)]
            #     cost = cost[topk_indices]
            #     inputs["pose_data_q"] =  T_flat.unsqueeze(0).contiguous().clone()
            #     inputs["cam_data_q"] = qcamera.unsqueeze(0).clone()
                # print("下降最快的是第 {} 组".format(topk_indices))
            
        # errors_all = torch.stack(err_list, dim=0).T  # [b, 10]
        # 展示每个样本的误差变化 
        # for i in range(len(errors_all)):
        #     print(f"样本 {i} 的误差变化: {errors_all[i]}")

        # print('优化后：',T_flat[0])
        total_cost_loss = w_loss
        # best_idx = torch.argmin(total_loss)  # 最佳收敛位姿索引
        # print('argmax weight loss: ', torch.argmax(total_loss))
        # print('argmin cost: ', torch.argmin(total_cost_loss))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        time_lm = (t1 - t0)*1000
        # print(f"latency: {time_lm} ms")
        
        return T, failed, total_cost_loss # T[initial pose]
