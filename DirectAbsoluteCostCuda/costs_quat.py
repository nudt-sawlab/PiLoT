import torch
from torch import Tensor
from typing import Optional, Tuple
# from mmcv.ops.point_sample import bilinear_grid_sample
import torch.nn.functional as F
import copy

@torch.jit.script
def J_project( p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=1e-3)
        J = torch.stack([
            1/z, zero, -x / z**2,
            zero, 1/z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1]+(2, 3))
        return J  # N x 2 x 3
@torch.jit.script
# def transform_p3d(body2view_pose_data, p3d):
#     R = body2view_pose_data[..., :9].view(-1, 3, 3)
#     t = body2view_pose_data[..., 9:]
#     return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2), R, t
@torch.jit.script
def transform_p3d(body2view_pose_data, p3d):
    # 假设 body2view_pose_data: [B, V, 12]
    B, V = body2view_pose_data.shape[:2]
    R = body2view_pose_data[..., :9].reshape(B, V, 3, 3)            # [B, V, 3, 3]
    t = body2view_pose_data[..., 9:].reshape(B, V, 1, 3)            # [B, V, 1, 3]
    # 显式扩展 R 为 [B, V, 1, 3, 3]，p3d: [B, V, N, 3]
    R = R.unsqueeze(2)                                              # [B, V, 1, 3, 3]

    # torch.matmul: [B, V, N, 3] @ [B, V, 1, 3, 3] -> [B, V, N, 3]
    out = torch.matmul(p3d.unsqueeze(-2), R.transpose(-1, -2)).squeeze(-2) + t
    return out
# @torch.jit.script
def project_p3d(camera_data, p3d):
    eps=1e-4

    z = p3d[..., -1]
    valid1 = z > eps
    z = z.clamp(min=eps)
    p2d = p3d[..., :-1] / z.unsqueeze(-1)

    f = camera_data[..., 2:4].unsqueeze(-2)
    c = camera_data[..., 4:6].unsqueeze(-2)
    f = f.repeat(1, 1, p2d.size(2), 1) 
    c = c.repeat(1, 1, p2d.size(2), 1)  
    p2d = p2d * f + c

    size = camera_data[..., :2]
    size = size.unsqueeze(-2)
    # valid2 = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
    valid2 = torch.logical_and(p2d >= 0, p2d <= (size - 1))
    valid2 = torch.logical_and(valid2[..., 0], valid2[..., 1])
    valid = torch.logical_and(valid1, valid2)

    return p2d, valid
@torch.jit.script
def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M
class DirectCostModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def bilinear_grid_sample(self, im, grid, align_corners=False):
        n, c, h, w = im.shape
        gn, gh, gw, _ = grid.shape
        assert n == gn
    
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]
    
        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2
    
        x = x.view(n, -1)
        y = y.view(n, -1)
    
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1
    
        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)
    
        # Apply default for grid_sample function zero padding
        im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0.0)
        padded_h = h + 2
        padded_w = w + 2
    
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1
    
        # Clip coordinates to padded image size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
        x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
        x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
        x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
        y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
        y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
        y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
        y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)
    
        im_padded = im_padded.view(n, c, -1)
    
        x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    
        Ia = torch.gather(im_padded, 2, x0_y0)
        Ib = torch.gather(im_padded, 2, x0_y1)
        Ic = torch.gather(im_padded, 2, x1_y0)
        Id = torch.gather(im_padded, 2, x1_y1)
    
        return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)
    def interpolate_feature_map(
                            self,
                            feature: torch.Tensor,
                            p2d: torch.Tensor,
                            return_gradients: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interpolation_pad = 4
        b, c, h, w = feature.shape
        scale = torch.tensor([w-1, h-1]).to(p2d)
        pts = (p2d / scale) * 2  - 1
        pts = pts.clamp(min=-2.0, max=2.0)
        # fp = torch.nn.functional.grid_sample(feature, pts[:, None], mode='bilinear', align_corners=True)
        fp = self.bilinear_grid_sample(feature, pts[:, None], align_corners=True)
        fp = fp.reshape(b, c, -1).transpose(-1, -2)
        image_size_ = torch.tensor([w-interpolation_pad-1, h-interpolation_pad-1]).to(pts)
        valid0 = torch.logical_and(p2d >= interpolation_pad, p2d <= image_size_)
        valid = torch.logical_and(valid0[..., 0], valid0[..., 1])
        # valid = torch.all((p2d >= interpolation_pad) & (p2d <= image_size_), -1)
        if return_gradients:
            dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
            dx, dy = dxdy.chunk(2, dim=0)
            pts_d = torch.cat([pts-dx, pts+dx, pts-dy, pts+dy], 1)
            # tensor_d = torch.nn.functional.grid_sample(
            #         feature, pts_d[:, None], mode='bilinear', align_corners=True)
            tensor_d = self.bilinear_grid_sample(
                    feature, pts_d[:, None], align_corners=True)
            tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
            tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
            gradients = torch.stack([
                (tensor_x1 - tensor_x0)/2, (tensor_y1 - tensor_y0)/2], dim=-1)
        else:
            gradients = torch.zeros(b, pts.shape[1], c, 2).to(feature)

        return fp, valid, gradients
    def loss_fn1(
            self,
            cost: Tensor,
            alpha: float = 0.0,
            truncate: float = 0.1,
            eps: float = 1e-7,
            derivatives: bool = True
        ) -> Tuple[Tensor, Tensor, Tensor]:  
        # x = cost / (truncate**2)
        # alpha = cost.new_tensor(alpha)[None]
        x = cost / (truncate ** 2)
        alpha = torch.full((1,), alpha, dtype=cost.dtype, device=cost.device)

        loss_two = x
        # loss_zero = 2 * torch.log1p(torch.clamp(0.5*x, max=33e37))
        loss_zero = 2 * torch.log(torch.clamp(0.5*x, max=33e37)+1)

        # The loss when not in one of the above special cases.
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.abs(alpha - 2.).clamp(min=eps)

        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        alpha_safe = alpha_safe * torch.abs(alpha).clamp(min=eps)

        loss_otherwise = 2 * (beta_safe / alpha_safe) * (
            torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == 0, loss_zero,
            torch.where(alpha == 2, loss_two, loss_otherwise))
        dummy = torch.zeros_like(x)

        loss_two_d1 = torch.ones_like(x)
        loss_zero_d1 = 2 / (x + 2)
        loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)
        loss_d1 = torch.where(
            alpha == 0, loss_zero_d1,
            torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

        loss, loss_d1, loss_d2 = loss, loss_d1, dummy

        
        return loss*(truncate**2), loss_d1, loss_d2/(truncate**2)
    def forward(
        self, pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D: Tensor,
        c_ref,c_query):
        num_init_pose = pose_data_q.shape[1]
        # === Step 1: 参考帧中投影、采样、可见性判断 ===
        p3d_r = transform_p3d(pose_data_r, p3D)
        p2d_r, visible_r = project_p3d(cam_data_r, p3d_r)
        p2d_r = p2d_r[0]
        
        visible_r = visible_r[0]
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)

        valid_ref_mask = (valid_r & visible_r).to(torch.float32)  # shape: [1, N]
        mask = valid_ref_mask.transpose(1, 0)  # shape: [N, 1]，避免 unsqueeze
        
        fp_r = fp_r[0] * mask.repeat(1, fp_r[0].shape[-1])  # [N, C]
        p3D = p3D[0] * mask.repeat(1, p3D[0].shape[-1])    # [N, 3]
        fp_r = fp_r.repeat(1, num_init_pose, 1, 1)
        p3D = p3D.repeat(1, num_init_pose, 1, 1)
        
        fp_r = torch.nn.functional.normalize(fp_r, dim=-1)
        

        # # === Step 2: 查询帧中投影 ===
        p3d_q = transform_p3d(pose_data_q, p3D)
        p2d_q, visible_q = project_p3d(cam_data_q, p3d_q)

        p2d_q = p2d_q * mask.T.view(1, 1, -1, 1)  # [B, N, 2] * [1, N, 1]  # [B, N, 2]，广播避免 unsqueeze
        
        # # === Step 3: 查询帧特征采样 ===
        fp_q, valid_q, J_f = self.interpolate_feature_map(f_q, p2d_q.reshape(1, -1, 2), return_gradients=True)
        fp_q = fp_q.view(1, num_init_pose, -1, fp_q.shape[-1])
        valid_q = valid_q.view(1, num_init_pose, -1)
        J_f = J_f.view(1, num_init_pose, -1, J_f.shape[-2], J_f.shape[-1])
        visible_q = visible_q.view(1, num_init_pose, -1)
        
        res = fp_q - fp_r

        # # === Step 4: Jacobians ===
        R = pose_data_q[..., :9].view(1, -1, 3, 3)
        J_p3d_rot = R[:, :, None].matmul(-skew_symmetric(p3D))
        J_p3d_tran = R[:, :, None].expand(-1, -1, J_p3d_rot.shape[2], -1, -1)
        J_p3d_pose = torch.cat((J_p3d_rot, J_p3d_tran), dim=-1)

        fx, fy = cam_data_q[:, :, 2], cam_data_q[:, :, 3]
        zero = torch.zeros_like(fx)
        f_diag_embed = torch.stack([fx, zero, zero, fy], dim=-1).reshape(1, -1, 1, 2, 2)

        J_p2d_p3d = f_diag_embed @ J_project(p3d_q)  # [B, N, 2, 3]
        J = J_f @ J_p2d_p3d @ J_p3d_pose             # [B, N, 2, 6]

        # # === Step 5: loss 和加权 ===
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn1(cost)

        valid_query = (valid_q & visible_q).float()
        weight_loss = w_loss * valid_query

        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r *  mask.T.view(1, -1, 1)  # [N, C]
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [B, N]
        weights = weight_loss * w_unc

        grad = (J * res[..., :, :, None]).sum(dim=-2)
        grad = weights[:, :, :, None] * grad
        grad = grad.sum(-2)

        Hess = J.permute(0, 1, 2, 4, 3) @ J
        Hess = weights[:, :, :, None, None] * Hess
        Hess = Hess.sum(-3)
        import ipdb; ipdb.set_trace()
        return -grad, Hess, w_loss, valid_query, p2d_q, cost   # torch.Size([64, 6]) , torch.Size([64, 6, 6]), torch.Size([64, 447]), torch.Size([64, 447])
    



# =====================转torchscript
# model = DirectCostModule()
# model.eval()

# # 编译为 TorchScript
# scripted_model = torch.jit.script(model)

# # 保存
# scripted_model.save("direct_cost_module_scripted.pt")

# ====================测试tensorscript速度
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat/pixloc/pixlib/geometry/sample_inputs.pt")
# # 2. 加载 TorchScript 模型
# scripted_model = torch.jit.load("direct_cost_module_scripted.pt")
# # 1. 加载模型（已加载并 to(device)）
# # scripted_model = torch.jit.load("direct_cost_module.pt").to(device)
# scripted_model.eval()

# # 2. 将所有输入张量移到 GPU 上
# inputs = {k: v.to("cuda") for k, v in inputs.items()}

# # 3. 预热 GPU（避免冷启动影响时间）
# for _ in range(50):
#     _ = scripted_model(
#         inputs["pose_data_q"],
#         inputs["f_r"],
#         inputs["pose_data_r"],
#         inputs["cam_data_r"],
#         inputs["f_q"],
#         inputs["cam_data_q"],
#         inputs["p3D"],
#         inputs["c_ref"],
#         inputs["c_query"]
#     )

# 
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat/pixloc/pixlib/geometry/sample_inputs.pt")
# # 4. 测量推理时间
# torch.cuda.synchronize()
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# start_event.record()
# for _ in range(100):  # 跑100次取平均
#     _ = scripted_model(
#         inputs["pose_data_q"],
#         inputs["f_r"],
#         inputs["pose_data_r"],
#         inputs["cam_data_r"],
#         inputs["f_q"],
#         inputs["cam_data_q"],
#         inputs["p3D"],
#         inputs["c_ref"],
#         inputs["c_query"]
#     )
# end_event.record()

# # 等待 GPU 完成
# torch.cuda.synchronize()

# # 5. 输出平均推理时间（ms）
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(f"平均单次推理时间：{elapsed_time_ms / 100:.3f} ms")


# 加载示例输入
# ====================导出ONNX模型
model = DirectCostModule()
model.eval().cuda()
inputs = torch.load("/home/ubuntu/Documents/code/FPV-Test/sample_inputs_512.pt")
save_onnx = "/home/ubuntu/Documents/code/FPV-Test/direct_cost_module_512.onnx"
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_128.pt")
# save_onnx = "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/direct_cost_module_128.onnx"
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_64.pt")
# save_onnx = "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/direct_cost_module_64.onnx"
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_256.pt")
# save_onnx = "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/direct_cost_module_256.onnx"
inputs = {k: v.cuda() for k, v in inputs.items()}

# 准备输入元组（按 forward 参数顺序）
input_tuple = (
    inputs["pose_data_q"],
    inputs["f_r"],
    inputs["pose_data_r"],
    inputs["cam_data_r"],
    inputs["f_q"],
    inputs["cam_data_q"],
    inputs["p3D"],
    inputs["c_ref"],
    inputs["c_query"]
)

# 导出 ONNX 模型
torch.onnx.export(
    model,
    input_tuple,
    save_onnx,
    export_params=True,
    opset_version=17,
    do_constant_folding=False,
    verbose = True,
    input_names=[
        "pose_data_q", "f_r", "pose_data_r", "cam_data_r",
        "f_q", "cam_data_q", "p3D", "c_ref", "c_query"
    ],
    output_names=["output"],
    # dynamic_axes={
    #     "p3D": {0: "batch_size", 1: "num_points"},
    #     "output": {0: "batch_size"}  # 根据输出修改
    # }
)
print("ONNX export completed.", save_onnx)
#========================测试onnx模型
import onnxruntime as ort
import onnx
import torch
import numpy as np
# 加载原始 FP32 模型
from onnxconverter_common import float16
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# model_fp32 = onnx.load(save_onnx)
# # 转换为 FP16
# model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
# save_onnx = "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/direct_cost_module_128_fp16.onnx"
# # 保存
# onnx.save(model_fp16, save_onnx)
# 加载输入数据
# inputs = torch.load("/home/ubuntu/Documents/code/github/FPVLoc_dev_1/sample_inputs_240.pt")
# inputs = torch.load("/home/ubuntu/Documents/code/github/FPVLoc_dev_1/sample_inputs_480.pt")
# inputs = torch.load("/home/ubuntu/Documents/code/github/FPVLoc_dev_1/sample_inputs_960.pt")

inputs = {k: v.cuda() for k, v in inputs.items()}
# inputs["pose_data_q"] = inputs["pose_data_q"].unsqueeze(0)
# inputs["pose_data_r"] = inputs["pose_data_r"].unsqueeze(0)
# inputs["cam_data_r"] = inputs["cam_data_r"].unsqueeze(0)
# inputs["cam_data_q"] = inputs["cam_data_q"].unsqueeze(0)
# inputs["p3D"] = inputs["p3D"].unsqueeze(0)
# 准备输入元组（按 forward 参数顺序）

# 1. **冷启动**（首次加载模型，进行推理）：
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 创建 ONNX 推理 session（冷启动）
start_event.record()
# save_onnx = "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/direct_cost_module_128_sim.onnx"
# 创建 ONNX 推理 session
session = ort.InferenceSession(save_onnx, providers=["CUDAExecutionProvider"])

# 准备输入数据，转换为 numpy 格式
onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

# 进行一次推理（加载模型及准备操作）
for _ in range(50):
    output = session.run(None, onnx_inputs)

# 结束冷启动记录
end_event.record()

# 等待 GPU 操作完成
torch.cuda.synchronize()

# 计算冷启动时间（单位：ms）
cold_start_time_ms = start_event.elapsed_time(end_event)
print(f"冷启动时间：{cold_start_time_ms:.3f} ms")

# 2. **温热启动**（不再有模型加载过程，只进行推理）

# 使用 torch.cuda.Event 来记录推理时间
start_event.record()
# 进行多次推理（例如 100 次，取平均）
for _ in range(100):
    ort_outputs = session.run(None, onnx_inputs)

# 结束时间记录
end_event.record()

# 等待 GPU 操作完成
torch.cuda.synchronize()

# 计算温热启动时间（单位：ms）
warm_start_time_ms = start_event.elapsed_time(end_event)
print(f"平均单次推理时间（温热启动）：{warm_start_time_ms / 100:.3f} ms")
print("是否使用 CUDAExecutionProvider:", session.get_providers()[0] == "CUDAExecutionProvider")


# 1. 将模型移到 GPU
model = model.to('cuda')
model.eval()  # 推荐同时调用 eval() 模式（特别是 inference 时）

# 2. 将所有输入张量移到 GPU（如果是一个 tuple）
input_tuple = tuple(t.to('cuda') for t in input_tuple)

# 3. 执行前向推理
with torch.no_grad():
    torch_outputs = model(*input_tuple)

for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outputs)):
    np_out = to_numpy(torch_out)
    if not np.allclose(np_out, ort_out, rtol=1e-05, atol=1e-08):
        print(f"❌ Output {i} mismatch!")
        diff = np.abs(np_out - ort_out)
        print("Max abs diff:", diff.max())
    else:
        print(f"✅ Output {i} matches.")
# =======================转tensorrt
# from torch2trt import torch2trt
# inputs = torch.load("/home/liuxy24/code/FPV_dev_quat/pixloc/pixlib/geometry/sample_inputs.pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}
# # 准备输入元组（按 forward 参数顺序）
# input_tuple = (
#     inputs["pose_data_q"],
#     inputs["f_r"],
#     inputs["pose_data_r"],
#     inputs["cam_data_r"],
#     inputs["f_q"],
#     inputs["cam_data_q"],
#     inputs["p3D"],
#     inputs["c_ref"],
#     inputs["c_query"]
# )
# model_trt = torch2trt(model, input_tuple, fp16_mode=True)
# torch.save(model_trt.state_dict(), "direct_cost_module_trt.pth")