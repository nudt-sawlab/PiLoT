import torch
import direct_abs_cost_cuda as ext
import time
def benchmark(device="cuda", warmup=5, repeat=20):
    # 1. 加载输入并搬到 device
    inputs = torch.load("/home/ubuntu/Documents/code/FPV-Test/sample_inputs_512_64.pt", map_location="cpu")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
        print(k, v.shape)
    # 2. 取方法
    fn = ext.residual_jacobian_batch_quat_cuda
    optimizer_cuda = ext.optimizer_step_cuda
    # 3. 预热若干次（保证 JIT / CUDA kernel 都加载完）
    
    for _ in range(5):
        grad, Hess, w_loss, valid_qm, p2d_q, loss = fn(
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
        Hess = Hess[0]
        valid = valid_qm[0].long().sum(-1)
        grad = grad.squeeze(0).unsqueeze(-1)
        delta1 = optimizer_cuda(grad, Hess,0.1, ~valid)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        

    # 4. 测量多次，累加总时间
    times = []
    for _ in range(20):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        grad, Hess, w_loss, valid_qm, p2d_q, loss = fn(
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
        Hess = Hess[0]
        valid = valid_qm[0].long().sum(-1)
        grad = grad.squeeze(0).unsqueeze(-1)
        delta1 = optimizer_cuda(grad, Hess,0.1, ~valid)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0)*1000)  # ms

    # 5. 输出结果
    print(f"Device: {device}")
    print(f"Warmup runs: {warmup}, Benchmark runs: {repeat}")
    print(f"Average latency: {sum(times)/len(times):.2f} ms")
    print(f"Min latency: {min(times):.2f} ms")
    print(f"Max latency: {max(times):.2f} ms")
    print(times)

if __name__ == "__main__":
    # CPU 测试
    # benchmark(device="cpu")
    # GPU 测试（如果可用）
    if torch.cuda.is_available():
        benchmark(device="cuda")
        