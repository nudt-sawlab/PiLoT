import torch
import multiprocessing
from ..geometry import Pose, Camera
def main():
    # —— 1. 显式设置多进程启动方式为 spawn —— 
    #    这样每个子进程都会单独启动一个 Python 解释器，不会继承父进程的 CUDA 上下文
    multiprocessing.set_start_method('spawn', force=True)

    # —— 2. 准备所有要送给子进程的输入张量 —— 
    #    这里以随机值举例，你要把真实的 F_ref、p3D 等替换进来。
    B, C, H, W, P = 1, 32, 512, 512, 500  # 例如 B=4 组待优化姿态
    device = torch.device('cuda:0')

    # T_init_flat: [B,6]
    T_init_flat = torch.randn(B, 6, device=device, dtype=torch.float32)

    # reference-query 特征图
    F_ref   = torch.randn(1, C, H, W, device=device)
    F_query = torch.randn(1, C, H, W, device=device)

    # cam 内参，假设维度是 6
    cam_ref = torch.randn(1, 6, device=device, dtype=torch.float32)
    cam_q   = torch.randn(1, 6, device=device, dtype=torch.float32)

    # p3D：假设形状先是 [P,3]，扩成 [B,P,3]
    p3D = torch.randn(P, 3, device=device, dtype=torch.float32)
    p3D_b = p3D.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,P,3]

    # visibility mask：假设先是 [H,W]，扩成 [B,1,H,W]
    c_ref   = torch.randint(0, 2, (H, W), device=device).float()
    c_query = torch.randint(0, 2, (H, W), device=device).float()
    # 把它们补成 [1,1,H,W] 再 expand
    c_ref_b   = c_ref.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).contiguous()
    c_query_b = c_query.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).contiguous()

    # 把 F_ref、F_query 补成 [B,C,H,W]
    F_ref_b   = F_ref.expand(B, C, H, W).contiguous()
    F_query_b = F_query.expand(B, C, H, W).contiguous()

    # T_render_flat: 单个 Pose to_flat() 得到 [6]，补成 [B,6]
    # 这里直接随机初始化
    T_render_flat = torch.randn(6, device=device, dtype=torch.float32)
    T_render_b    = T_render_flat.unsqueeze(0).expand(B, -1).contiguous()

    # qcamera_b: [B,6]
    qcam_b = cam_q.expand(B, -1).contiguous()

    # ref_cam_b: [B,6]
    ref_cam_b = cam_ref.expand(B, -1).contiguous()

    # dampingnet 返回的 λ 
    lambda_val = float(0.1)

    # —— 3. 把所有输入打包成一个 dict，发给子进程 —— 
    all_inputs = {
        "T_init": 
        "T_flat":     T_init_flat,   # [B,6]
        "F_ref_b":    F_ref_b,       # [B,C,H,W]
        "T_render_b": T_render_b,    # [B,6]
        "ref_cam_b":  ref_cam_b,     # [B,6]
        "F_query_b":  F_query_b,     # [B,C,H,W]
        "qcam_b":     qcam_b,        # [B,6]
        "p3D_b":      p3D_b,         # [B,P,3]
        "c_ref_b":    c_ref_b,       # [B,1,H,W]
        "c_query_b":  c_query_b,     # [B,1,H,W]
        "lambda_val": lambda_val     # float
    }

    # —— 4. 创建两个 Queue：一个给子进程拿输入，一个用于子进程把结果放回 —— 
    in_queue  = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()

    # —— 5. 把输入发给子进程，然后启动子进程 —— 
    in_queue.put(all_inputs)
    p = multiprocessing.Process(
        target=worker_process,
        args=(in_queue, out_queue),
    )
    p.start()

    # —— 6. 等子进程跑完，拿回结果 —— 
    p.join()
    result = out_queue.get()

    # result 是一个 dict，包含 g_b、H_b、w_loss_b、valid_b、p2d_q_b、cost_b、delta_b
    g_b      = result["g_b"]      # Tensor [B,6]
    H_b      = result["H_b"]      # Tensor [B,6,6]
    w_loss_b = result["w_loss_b"] # Tensor [B]
    valid_b  = result["valid_b"]  # Tensor [B,P?]
    p2d_q_b  = result["p2d_q_b"]  # Tensor [B,P,2]
    cost_b   = result["cost_b"]   # Tensor [B]
    delta_b  = result["delta_b"]  # Tensor [B,6]

    # —— 7. 验证：确保你拿到的每一组 g_b[i], H_b[i] 都不是“只第 0 维度有效”—— 
    print("g_b shape:", g_b.shape)      # 应该是 [B,6]
    print("H_b shape:", H_b.shape)      # 应该是 [B,6,6]
    print("w_loss_b shape:", w_loss_b.shape)  # [B]
    print("delta_b shape:", delta_b.shape)    # [B,6]

    # 例如：
    for i in range(B):
        print(f"第 {i} 组 g:", g_b[i])
        print(f"第 {i} 组 H:", H_b[i])
        print(f"第 {i} 组 delta:", delta_b[i])
def worker_process(
    in_queue: "multiprocessing.Queue",
    out_queue: "multiprocessing.Queue"
):
    import torch
    from ..geometry import direct_abs_cost_cuda
    torch.cuda.set_device(0)

    # 拿到所有输入
    data = in_queue.get()
    
    T_flat     = data["T_flat"].clone()      # [B,6]
    F_ref_b    = data["F_ref_b"]
    T_render_b = data["T_render_b"]
    ref_cam_b  = data["ref_cam_b"]
    F_query_b  = data["F_query_b"]
    qcam_b     = data["qcam_b"]
    p3D_b      = data["p3D_b"]
    c_ref_b    = data["c_ref_b"]
    c_query_b  = data["c_query_b"]
    lambda_val = data["lambda_val"]
    num_iters  = data.get("num_iters", 4)    # 比如跑 4 轮

    for it in range(num_iters):
        # 每轮都从 T_flat 得到 g_b, H_b, 等
        g_b, H_b, w_loss_b, valid_b, p2d_q_b, cost_b = \
            direct_abs_cost_cuda.residual_jacobian_batch_quat_cuda(
                T_flat, F_ref_b, T_render_b, ref_cam_b,
                F_query_b, qcam_b, p3D_b, c_ref_b, c_query_b
            )

        # 计算 valid mask
        if valid_b.dim() == 2:
            valid_mask = (valid_b.long().sum(dim=1) >= 10)
        elif valid_b.dim() == 1:
            valid_mask = (valid_b.long() >= 10)
        else:
            valid_mask = torch.ones((T_flat.shape[0],), dtype=torch.bool, device=T_flat.device)

        # 计算 delta
        delta_b = direct_abs_cost_cuda.optimizer_step_cuda(g_b, H_b, lambda_val, valid_mask)

        # 更新 T_flat
        dw, dt = delta_b.split([3, 3], dim=-1)
        T_delta = Pose.from_aa(dw, dt).to_flat()  # [B,6]
        # 假设 Pose.from_aa(...).to_flat() 能把 dw,dt 转为 [B,6] 再叠加到 T_flat 上
        T = T @ T_delta
        T_flat = T.to_flat()
        # 如果需要中途把每轮的结果回传给主进程，可以 out_queue.put(...)。
        # 这里只等到最后一轮再发送：
        if it == num_iters - 1:
            out_queue.put({
                "g_b":      g_b.cpu(),
                "H_b":      H_b.cpu(),
                "w_loss_b": w_loss_b.cpu(),
                "valid_b":  valid_b.cpu(),
                "p2d_q_b":  p2d_q_b.cpu(),
                "cost_b":   cost_b.cpu(),
                "delta_b":  delta_b.cpu(),
                "T_flat":   T_flat.cpu()    # 最终更新后的扁平化 T
            })
if __name__ == "__main__":
    main()
