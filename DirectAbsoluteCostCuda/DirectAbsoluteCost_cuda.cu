// DirectAbsoluteCost_cuda.cu
// 完整 GPU 加速版 DirectAbsoluteCost CUDA 扩展（Float32 版）

#include <torch/extension.h>
#include <unordered_map>
#include <mutex>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#define CUDA_CHECK_ERRORS()                   \
    do                                        \
    {                                         \
        cudaError_t err = cudaGetLastError(); \
        C10_CUDA_CHECK(err);                  \
    } while (0)

// 在文件顶部添加
static void print_tensor(const torch::Tensor &t, const char *name)
{
    // 打印形状
    std::cout << name << " shape=[";
    for (int i = 0; i < t.dim(); ++i)
    {
        std::cout << t.size(i) << (i + 1 < t.dim() ? "," : "");
    }
    std::cout << "] values=[";
    // 展平并打印前 10 个元素
    auto flat = t.flatten();
    int64_t count = std::min<int64_t>(10, flat.numel());
    for (int64_t i = 0; i < count; ++i)
    {
        // 这里假设都是 float32
        std::cout << flat[i].item<float>() << (i + 1 < count ? "," : "");
    }
    if (flat.numel() > count)
        std::cout << ",...";
    std::cout << "]\n";
}

// ------------------------------------------------------------------------------------------------
// 1) 内联的双线性插值函数：给定单通道指针 im_ptr（大小 H×W），在 (x, y) 浮点坐标处做 bilinear sampling，
//    越界时返回 0。此处 x,y 均为 [0, W-1]、[0, H-1] 空间坐标（非归一化）。
// ------------------------------------------------------------------------------------------------
__device__ inline float get_pixel_bilinear(
    const float *__restrict__ im_ptr, // 指向单通道图像数据的行主序指针
    float x,                          // 浮点横坐标（已反归一化到 [0, W-1]）
    float y,                          // 浮点纵坐标（已反归一化到 [0, H-1]）
    int H,                            // 图像高度
    int W                             // 图像宽度
)
{
    // —— 1. “未经 clamp 的位置” 判定，完全拷贝自 kernel 的写法 ——
    //    只有当 (x >= 4 && x <= W-5 && y >= 4 && y <= H-5) 时，才算 in_bounds。
    //    否则都当作越界，直接返回 0.0f。
    // if (!(x >= 4.0f && x <= (float)(W - 5) && y >= 4.0f && y <= (float)(H - 5)))
    // {
    //     return 0.0f;
    // }

    // —— 2. 计算插值所需的四邻域整数坐标（floor + 1） ——
    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // —— 3. clamp 到合法像素范围 [0, W-1], [0, H-1] ——
    x0 = min(max(x0, 0), W - 1);
    x1 = min(max(x1, 0), W - 1);
    y0 = min(max(y0, 0), H - 1);
    y1 = min(max(y1, 0), H - 1);

    // —— 4. 计算插值权重（与 kernel 中一模一样） ——
    float wa = ((float)x1 - x) * ((float)y1 - y);
    float wb = ((float)x1 - x) * (y - (float)y0);
    float wc = (x - (float)x0) * ((float)y1 - y);
    float wd = (x - (float)x0) * (y - (float)y0);

    // —— 5. 从 im_ptr 中读取四个像素值 ——
    //    im_ptr 按 “行主序” 存储，也就是 im_ptr[row * W + col]
    float Ia = im_ptr[y0 * W + x0];
    float Ib = im_ptr[y1 * W + x0];
    float Ic = im_ptr[y0 * W + x1];
    float Id = im_ptr[y1 * W + x1];

    // —— 6. 返回加权和 ——
    return wa * Ia + wb * Ib + wc * Ic + wd * Id;
}
__device__ inline float sample_by_norm(
    const float *__restrict__ im_ptr,
    float xnorm, // 归一化横坐标 ∈ [-1,1]
    float ynorm, // 归一化纵坐标 ∈ [-1,1]
    int H,       // 高度
    int W)       // 宽度
{
    // 1) 先从归一化坐标算回像素空间： x ∈ [0, W-1], y ∈ [0, H-1]
    float x = (xnorm + 1.0f) * ((W - 1) / 2.0f);
    float y = (ynorm + 1.0f) * ((H - 1) / 2.0f);

    // 2) 调用原来的 get_pixel_bilinear，保证与“直接用像素坐标”结果一致
    return get_pixel_bilinear(im_ptr, x, y, H, W);
}
// ------------------------------------------------------------------------------------------------
// 2) fused_residual_hess_grad_kernel：
//    – 每个线程对应 (b, n, p) 中的一个 3D 点，直接在 kernel 内部做：
//      * Reference 侧：P_world -> 相机坐标 -> 像素 (ur, vr) -> bilinear 采样 fr_feat[c]
//      * Query    侧：P_world -> 相机坐标 -> 像素 (uq, vq) -> 四次 bilinear 采样 (中心差分) 得到 Jf_x/Jf_y[c]
//      * 计算 Jp2d、Jp3d
//      * 计算残差 res_c[c] = f_q(uq,vq)[c] - fr_feat[c]
//      * 计算 loss 的一阶导数 g_d1，appearance 权重 unc
//      * 在 6 维度上做 Σ_c (Jfp2d[c,k] * res_c[c]) 并乘 weight，得到 grad_i[k]
//      * 在 6×6 维度上做 Σ_c (Jfp2d[c,r] * Jfp2d[c,c2]) 并乘 weight，得到 Hess_i[r,c2]
//      * 最后用 atomicAdd 将 grad_i[k] 累加到 “grad_out[b,n,k]”，将 Hess_i 累加到 “Hess_out[b,n,:,:]”
//    完整合并了原本需要多次 kernel launch、reshape、matmul 等开销。
// ------------------------------------------------------------------------------------------------
__global__ void fused_debug_all_kernel(
    const float *__restrict__ pose_data_q, // [B, N, 12]
    const float *__restrict__ f_r,         // [B, C, H, W]
    const float *__restrict__ pose_data_r, // [B, 1, 12]
    const float *__restrict__ cam_data_r,  // [B, 1, 6]
    const float *__restrict__ f_q,         // [B, C, H, W]
    const float *__restrict__ cam_data_q,  // [B, N, 6]
    const float *__restrict__ p3D,         // [B, 1, P, 3]
    const float *__restrict__ c_ref,       // [B, 1, H, W]
    const float *__restrict__ c_query,     // [B, 1, H, W]
    int B, int N, int P, int C, int H, int W,
    float truncate, float alpha, float eps,
    // Outputs：
    float *__restrict__ grad_out, // [B, N, 6]
    float *__restrict__ Hess_out, // [B, N, 6, 6]
    float *__restrict__ loss_out  // [B * N * P]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N * P)
        return;

    // 1. 解出 (b, n, p)
    int pn = tid % (N * P);
    int n = pn / P;
    int p = pn % P;
    int b = tid / (N * P);

    // 少量点打印条件：仅 (b=0, n=0, p<2)：
    bool is_debug = (b == 0 && n == 0 && p < 7);

    // ---- 2. 参考帧：计算 p3d_r_world -> p3d_r_cam -> p2d_r -> 采样 fr，以及 fp_r（略） ----
    // 这一段如果你只关注 Query 侧 Jp2d,Jp3d,Jf，可以先跳过；否则可复制你已有的计算逻辑在此。
    // 本示例重点演示 Query 侧，因此略去参考帧部分，假设 fr_feat[c] 已在上面算好并存在一个数组 fr_feat[C]。
    // —— 步骤 A：读参考 pose_r -> Rr, tr ——
    const float *Pr = pose_data_r + (b * 12);
    float Rr[9], tr[3];
#pragma unroll
    for (int i = 0; i < 9; i++)
        Rr[i] = Pr[i];
#pragma unroll
    for (int i = 0; i < 3; i++)
        tr[i] = Pr[9 + i];
    const float *P_world = p3D + ((b * P + p) * 3);
    float Xw = P_world[0], Yw = P_world[1], Zw = P_world[2];

    // —— 步骤 C：参考侧投影 P_cam_r = Rr * P_world + tr ——
    float Xr = Rr[0] * Xw + Rr[1] * Yw + Rr[2] * Zw + tr[0];
    float Yr = Rr[3] * Xw + Rr[4] * Yw + Rr[5] * Zw + tr[1];
    float Zr = Rr[6] * Xw + Rr[7] * Yw + Rr[8] * Zw + tr[2];

    if (Zr <= 1e-4f)
    {
        // 如果参考深度无效，直接写零 loss 并返回
        loss_out[(b * N + n) * P + p] = 0.0f;
        return;
    }

    // 读取 cam_data_r[b,0,:]
    const float *Cr = cam_data_r + (b * 6);
    float wr = Cr[0], hr = Cr[1];
    float fxr = Cr[2], fyr = Cr[3];
    float cxr = Cr[4], cyr = Cr[5];

    float ur = Xr / Zr * fxr + cxr;
    float vr = Yr / Zr * fyr + cyr;
    bool inbr = (ur >= 0.0f && ur <= wr - 1.0f && vr >= 0.0f && vr <= hr - 1.0f);

    if (!inbr)
    {
        loss_out[(b * N + n) * P + p] = 0.0f;
        return;
    }
    // *此处省略参考帧的具体实现，仅留一个占位*
    float inv_Wm1 = 2.0f / float(W - 1);
    float inv_Hm1 = 2.0f / float(H - 1);
    const float eps_x = 2.0f * 4.0f / float(W - 1); // = 8/(W-1)
    const float eps_y = 2.0f * 4.0f / float(H - 1); // = 8/(H-1)
    float xnorm_r = ur * inv_Wm1 - 1.0f;            // ur/(W-1)*2 - 1
    float ynorm_r = vr * inv_Hm1 - 1.0f;            // vr/(H-1)*2 - 1

    // 判断参考帧可见性（是否在 [-1,1] 范围内，sample_by_norm 会再判是否在距离边界 ≥4 区域）
    bool visible_r = (xnorm_r >= -1.0f + eps_x &&
                      xnorm_r <= 1.0f - eps_x &&
                      ynorm_r >= -1.0f + eps_y &&
                      ynorm_r <= 1.0f - eps_y);
    float fr_feat[32];
    for (int c = 0; c < C; ++c)
    {
        const float *imr_ptr = f_r + ((b * C + c) * H) * W;
        if (visible_r)
        {
            fr_feat[c] = sample_by_norm(imr_ptr, xnorm_r, ynorm_r, H, W);
        }
        else
        {
            fr_feat[c] = 0.0f;
        }
    }

    // ---- 3. 查询帧：先把 pose_data_q[b,n,:] 拆出为 Rq(3×3)、tq(3) ----
    const float *Pq = pose_data_q + ((b * N + n) * 12);
    float Rq[9], tq[3];
#pragma unroll
    for (int i = 0; i < 9; i++)
        Rq[i] = Pq[i];
#pragma unroll
    for (int i = 0; i < 3; i++)
        tq[i] = Pq[9 + i];
    // 3D -> 相机坐标
    float Xq = Rq[0] * Xw + Rq[1] * Yw + Rq[2] * Zw + tq[0];
    float Yq = Rq[3] * Xw + Rq[4] * Yw + Rq[5] * Zw + tq[1];
    float Zq = Rq[6] * Xw + Rq[7] * Yw + Rq[8] * Zw + tq[2];
    if (Zq <= 1e-4f)
    {
        loss_out[(b * N + n) * P + p] = 0.0f;
        return;
    }

    // 读取 cam_data_q[b,n,:] -> (wq, hq, fxq, fyq, cxq, cyq)
    const float *Cq = cam_data_q + ((b * N + n) * 6);
    float wq = Cq[0], hq = Cq[1];
    float fxq = Cq[2], fyq = Cq[3];
    float cxq = Cq[4], cyq = Cq[5];

    // 投影到像素 (uq, vq)
    float uq = Xq / Zq * fxq + cxq;
    float vq = Yq / Zq * fyq + cyq;
    bool inbq = (uq >= 0.0f && uq <= wq - 1.0f && vq >= 0.0f && vq <= hq - 1.0f);

    if (!inbq)
    {
        loss_out[(b * N + n) * P + p] = 0.0f;
        return;
    }
    // 查询帧归一化坐标
    float xnorm_c = uq * inv_Wm1 - 1.0f;
    float ynorm_c = vq * inv_Hm1 - 1.0f;

    // 查询帧可见性：确保在 [-1+dx, 1-dx] 范围
    float dx_norm = inv_Wm1; // 2/(W-1)
    float dy_norm = inv_Hm1; // 2/(H-1)
    bool visible_q = (xnorm_c >= -1.0f + eps_x &&
                      xnorm_c <= 1.0f - eps_x &&
                      ynorm_c >= -1.0f + eps_y &&
                      ynorm_c <= 1.0f - eps_y);
    // ---- 4. 计算 Jf_x[c], Jf_y[c]（中心差分四次采样） ----
    const float delta_x_norm = dx_norm; // 等价 1 像素
    const float delta_y_norm = dy_norm;

    float Jf_x_arr[32], Jf_y_arr[32];
    float fq_center_arr[32];
    float fr_norm2 = 0.0f, fq_norm2 = 0.0f;

    int base_img_q = (b * C) * H * W; // f_q 基地址

    for (int c = 0; c < C; ++c)
    {
        const float *imq_ptr = f_q + base_img_q + c * H * W;

        float fq_m = sample_by_norm(imq_ptr, xnorm_c - delta_x_norm, ynorm_c, H, W);
        float fq_p = sample_by_norm(imq_ptr, xnorm_c + delta_x_norm, ynorm_c, H, W);
        float fq_v = sample_by_norm(imq_ptr, xnorm_c, ynorm_c - delta_y_norm, H, W);
        float fq_d = sample_by_norm(imq_ptr, xnorm_c, ynorm_c + delta_y_norm, H, W);

        float Jfx = 0.5f * (fq_p - fq_m);
        float Jfy = 0.5f * (fq_d - fq_v);
        Jf_x_arr[c] = Jfx;
        Jf_y_arr[c] = Jfy;

        float fq_center = sample_by_norm(imq_ptr, xnorm_c, ynorm_c, H, W);
        fq_center_arr[c] = fq_center;

        fr_norm2 += fr_feat[c] * fr_feat[c];
        fq_norm2 += fq_center * fq_center;
    }
    const float eps_norm = 1e-6f;
    float inv_sqrt_fr = rsqrtf(fr_norm2 + eps_norm); // 1/||fr||  (CUDA 自带的快速反开方)
    float inv_sqrt_fq = rsqrtf(fq_norm2 + eps_norm); // 1/||fq||

    // ---- 3. 归一化后计算 residual ----
    float res_c[32];
    for (int c = 0; c < C; c++)
    {
        float fr_hat = fr_feat[c] * inv_sqrt_fr;
        float fq_hat = fq_center_arr[c] * inv_sqrt_fq;
        res_c[c] = fq_hat - fr_hat;
    }
    // ---- 5. 计算 Jp2d（2×3） ----
    float invZq = 1.0f / Zq;
    float Zq2 = Zq * Zq;
    float Jp2d_arr[6];
    Jp2d_arr[0] = fxq * invZq;     // ∂u/∂Xq
    Jp2d_arr[1] = 0.0f;            // ∂u/∂Yq
    Jp2d_arr[2] = -fxq * Xq / Zq2; // ∂u/∂Zq
    Jp2d_arr[3] = 0.0f;            // ∂v/∂Xq
    Jp2d_arr[4] = fyq * invZq;     // ∂v/∂Yq
    Jp2d_arr[5] = -fyq * Yq / Zq2; // ∂v/∂Zq
    // ---- 6. 计算 Jp3d（3×6） ----
    // 用相机坐标构造 skew，而不是世界坐标
    float xw = Xw, yw = Yw, zw = Zw;
    float skew_world[9] = {
        0.0f, -zw, yw,
        zw, 0.0f, -xw,
        -yw, xw, 0.0f};
    float Jp3d_arr[18];
    // -R_q * skew_world  （旋转部分）
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; k++)
            {
                sum += Rq[3 * i + k] * skew_world[3 * k + j];
            }
            Jp3d_arr[i * 6 + j] = -sum;
        }
    }

    // 2) 计算平移部分： Jtran = Rq，本身放到“每行的后 3 列”
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // Jtran[i,j] 就是 Rq[i,j]，放到 Jp3d_arr[i*6 + (3 + j)]
            Jp3d_arr[i * 6 + 3 + j] = Rq[3 * i + j];
        }
    }

    // ---- 7. 计算 M = Jp2d(2×3) @ Jp3d(3×6) → 得到 2×6 ----
    float M_row0[6], M_row1[6];
// M_row0[k] 对应 M[0, k]，M_row1[k] 对应 M[1, k]
#pragma unroll
    for (int k = 0; k < 6; k++)
    {
        float Jp3d_0k = Jp3d_arr[0 * 6 + k];
        float Jp3d_1k = Jp3d_arr[1 * 6 + k];
        float Jp3d_2k = Jp3d_arr[2 * 6 + k];

        // M_row0[k] = Jp2d(0,0)*Jp3d(0,k) + Jp2d(0,1)*Jp3d(1,k) + Jp2d(0,2)*Jp3d(2,k)
        M_row0[k] = Jp2d_arr[0] * Jp3d_0k + Jp2d_arr[1] * Jp3d_1k + Jp2d_arr[2] * Jp3d_2k;

        // M_row1[k] = Jp2d(1,0)*Jp3d(0,k) + Jp2d(1,1)*Jp3d(1,k) + Jp2d(1,2)*Jp3d(2,k)
        M_row1[k] = Jp2d_arr[3] * Jp3d_0k + Jp2d_arr[4] * Jp3d_1k + Jp2d_arr[5] * Jp3d_2k;
    }
    // ---- 8. 计算 Jfp2d[c * 6 + k] = Jf_x[c] * M_row0[k] + Jf_y[c] * M_row1[k] ----
    // **注意：必须在这里先声明 Jfp2d_arr，然后再使用它！**
    float Jfp2d_arr[32 * 6]; // 假设 C==32，如果你的 C 有变化，请相应调整数组大小
    for (int c = 0; c < C; c++)
    {
        for (int k = 0; k < 6; k++)
        {
            Jfp2d_arr[c * 6 + k] = Jf_x_arr[c] * M_row0[k] + Jf_y_arr[c] * M_row1[k];
        }
    }
    // ---- 9. 计算 cost = Σ_c res_c[c]^2 ----
    float cost = 0.0f;
    for (int c = 0; c < C; c++)
    {
        cost += res_c[c] * res_c[c];
    }

    // ---- 10. 计算 w_loss = d1 (Cauchy) ----
    float x_val = cost / (truncate * truncate);
    float loss_val;
    // 当 alpha=0 时，loss = 2 * log(0.5*x + 1) * (truncate^2)
    loss_val = 2.0f * logf(0.5f * x_val + 1.0f) * (truncate * truncate);
    loss_out[(b * N + n) * P + p] = loss_val;
    float beta = fabsf(alpha - 2.0f);
    if (beta < eps)
        beta = eps;
    float as_alpha = fabsf(alpha);
    if (as_alpha < eps)
        as_alpha = eps;
    float g_d1;
    if (alpha == 0.0f)
    {
        g_d1 = 2.0f / (x_val + 2.0f);
    }
    else if (alpha == 2.0f)
    {
        g_d1 = 1.0f;
    }
    else
    {
        g_d1 = powf(x_val / beta + 1.0f, 0.5f * alpha - 1.0f);
    }

    // ---- 11. appearance 权重 ----
    // 读取 c_query[b,:,:], c_ref[b,:,:] 单通道图
    // 同样用归一化插值 c_query, c_ref
    const float *cref_ptr = c_ref + (b * H * W); // [H×W] 单通道
    const float *cqry_ptr = c_query + (b * H * W);

    float wq_ap = sample_by_norm(cqry_ptr, xnorm_c, ynorm_c, H, W);
    float wr_ap = sample_by_norm(cref_ptr, xnorm_r, ynorm_r, H, W);
    float unc = wq_ap * wr_ap;

    float valid_qm = (Zq > 1e-4f && inbq) ? 1.0f : 0.0f;
    // visible_q: (xnorm_c ∈ [-1+dx_norm,1-dx_norm]) ∧ (ynorm_c ∈ [-1+dy_norm,1-dy_norm])
    float visible_q_f = visible_q ? 1.0f : 0.0f;
    // visible_r: 已在插值阶段保证 fr_feat 仅在 visible_r 时非零
    float visible_r_f = visible_r ? 1.0f : 0.0f;
    float weight = g_d1 * unc * valid_qm * visible_q_f * visible_r_f;
    // ---- 12. 计算 grad_i[6] 和 Hess_i[6×6]，累加到 grad_out/Hess_out ----
    float grad_i[6];
    for (int k = 0; k < 6; k++)
    {

        float sumtemp = 0.0f;
        for (int c = 0; c < C; c++)
        {
            sumtemp += Jfp2d_arr[c * 6 + k] * res_c[c];
        }
        grad_i[k] = weight * sumtemp;
    }
    float Hess_i[36];
    for (int r = 0; r < 6; r++)
    {
        for (int c2 = 0; c2 < 6; c2++)
        {
            float tsum = 0.0f;
            for (int c = 0; c < C; c++)
            {
                tsum += Jfp2d_arr[c * 6 + r] * Jfp2d_arr[c * 6 + c2];
            }
            Hess_i[r * 6 + c2] = weight * tsum;
        }
    }
    // atomicAdd 到 grad_out[b,n,:]
    int base_g = (b * N + n) * 6;
    for (int k = 0; k < 6; k++)
    {
        atomicAdd(grad_out + base_g + k, grad_i[k]);
    }
    // atomicAdd 到 Hess_out[b,n,:,:]
    int base_H = ((b * N + n) * 6) * 6;
    for (int i = 0; i < 36; i++)
    {
        atomicAdd(Hess_out + base_H + i, Hess_i[i]);
    }
}

// -------------------------------------------------------------
// Host 包装：创建所有 Debug Tensor 并返回给 Python
// -------------------------------------------------------------
std::tuple<
    torch::Tensor, // -grad        [B, N, 6]
    torch::Tensor, //  Hess        [B, N, 6, 6]
    torch::Tensor, //  w_loss      [B, N, P]       （本例中留空占位）
    torch::Tensor  //  loss        [B, N, P]       （本例中留空占位）
    >
residual_jacobian_batch_quat_cuda(
    torch::Tensor pose_data_q, // [B, N, 12]
    torch::Tensor f_r,         // [B, C, H, W]
    torch::Tensor pose_data_r, // [B, 1, 12]
    torch::Tensor cam_data_r,  // [B, 1, 6]
    torch::Tensor f_q,         // [B, C, H, W]
    torch::Tensor cam_data_q,  // [B, N, 6]
    torch::Tensor p3D,         // [B, 1, P, 3]
    torch::Tensor c_ref,       // [B, 1, H, W]
    torch::Tensor c_query      // [B, 1, H, W]
)
{
    TORCH_CHECK(pose_data_q.is_cuda() && f_r.is_cuda() &&
                    pose_data_r.is_cuda() && cam_data_r.is_cuda() &&
                    f_q.is_cuda() && cam_data_q.is_cuda() &&
                    p3D.is_cuda() && c_ref.is_cuda() && c_query.is_cuda(),
                "All inputs must be CUDA Tensors");

    TORCH_CHECK(pose_data_q.scalar_type() == torch::kFloat32 &&
                    f_r.scalar_type() == torch::kFloat32 &&
                    pose_data_r.scalar_type() == torch::kFloat32 &&
                    cam_data_r.scalar_type() == torch::kFloat32 &&
                    f_q.scalar_type() == torch::kFloat32 &&
                    cam_data_q.scalar_type() == torch::kFloat32 &&
                    p3D.scalar_type() == torch::kFloat32 &&
                    c_ref.scalar_type() == torch::kFloat32 &&
                    c_query.scalar_type() == torch::kFloat32,
                "All inputs must be float32");

    int64_t B = pose_data_q.size(0);
    int64_t N = pose_data_q.size(1);
    int64_t C = f_q.size(1);
    int64_t H = f_q.size(2);
    int64_t W = f_q.size(3);
    int64_t P = p3D.size(2);

    auto opts = pose_data_q.options().dtype(torch::kFloat32);

    // 1) grad, Hess
    torch::Tensor grad = torch::zeros({B, N, 6}, opts);
    torch::Tensor Hess = torch::zeros({B, N, 6, 6}, opts);
    torch::Tensor loss = torch::zeros({B, N, P}, opts);

    int64_t total = B * N * P;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    float truncate = 0.1f, alpha = 0.0f, eps = 1e-7f;
    fused_debug_all_kernel<<<blocks, threads>>>(
        pose_data_q.data_ptr<float>(),
        f_r.data_ptr<float>(),
        pose_data_r.data_ptr<float>(),
        cam_data_r.data_ptr<float>(),
        f_q.data_ptr<float>(),
        cam_data_q.data_ptr<float>(),
        p3D.data_ptr<float>(),
        c_ref.data_ptr<float>(),
        c_query.data_ptr<float>(),
        (int)B, (int)N, (int)P, (int)C, (int)H, (int)W,
        truncate, alpha, eps,
        grad.data_ptr<float>(),
        Hess.data_ptr<float>(),
        loss.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    auto nz_mask = loss.ne(0.0f);    // 布尔张量，[B, N, P]，true 表示对应位置 loss ≠ 0
    auto nz_count = nz_mask.sum(-1); // 在 P 维度上求和，得到 [B, N]，表示每个 (b,n) 下非零元素的个数
    // 为了避免除以零的情况，如果某些 (b,n) 一行里 loss 全为 0，就让计数至少为 1
    nz_count = nz_count.clamp_min(1);                // [B, N]，把所有值 < 1 的先设为 1
    auto loss_sum = loss.sum(-1);                    // [B, N]
    auto nz_count_f = nz_count.to(loss_sum.dtype()); // [B, N], float32
    auto loss_avg = loss_sum / nz_count_f;           // [B, N]
    return std::make_tuple(
        -grad, Hess, loss_avg, loss);
}

// CUDA 版 optimizer_step：解 (A + λI) δ = B，并支持 mask
torch::Tensor optimizer_step_cuda(
    torch::Tensor grad, // [B,6,1], float32
    torch::Tensor Hess, // [B,6,6], float32
    double lambda_d,    // λ
    torch::Tensor mask  // [B], bool or uint8; 可传空 Tensor
)
{
    TORCH_CHECK(grad.is_cuda() && Hess.is_cuda(),
                "optimizer_step_cuda: grad and Hess must be CUDA");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32 &&
                    Hess.scalar_type() == torch::kFloat32,
                "optimizer_step_cuda: dtype must be float32");
    TORCH_CHECK(grad.dim() == 3 && Hess.dim() == 3 &&
                    grad.size(0) == Hess.size(0) &&
                    grad.size(1) == 6 && grad.size(2) == 1 &&
                    Hess.size(1) == 6 && Hess.size(2) == 6,
                "optimizer_step_cuda: expected shapes [B,6,1] & [B,6,6]");

    int64_t B = grad.size(0);
    float lambda = static_cast<float>(lambda_d);
    auto opts = Hess.options().dtype(torch::kFloat32);

    // 1) 构造 λI：[6,6] -> [1,6,6] 用于广播
    auto eye6 = torch::eye(6, opts); // [6,6]
    auto tikh = eye6 * lambda;       // [6,6]
    auto tikh_b = tikh.unsqueeze(0); // [1,6,6]

    // 2) Hess + λI
    auto Hreg = Hess + tikh_b; // [B,6,6]
    // 3) 可选 mask
    if (mask.defined() && mask.numel() == B)
    {
        auto m = mask.to(torch::kBool).view({B, 1, 1}); // [B,1,1]
        // 保持 Hreg 在 m=1 的位置，m=0 的位置设为单位阵(1s)
        Hreg = Hreg.where(m, torch::ones_like(Hreg));
        // grad 在 m=0 的位置设为 0
        auto m2 = mask.to(torch::kBool).view({B, 1, 1}); // [B,1,1]
        grad = grad.masked_fill(~m2, 0.0f);
    }

    // 4) 求解 Hreg * delta = grad
    torch::Tensor delta;
    try
    {
        // 尝试 Cholesky
        auto L = at::cholesky(Hreg, /*upper=*/false);         // [B,6,6]
        delta = at::cholesky_solve(grad, L, /*upper=*/false); // [B,6,1]
    }
    catch (const c10::Error &e)
    {
        // 回退到通用求解
        delta = at::linalg_solve(Hreg, grad); // [B,6,1]
    }

    // 5) squeeze -> [B,6]
    return delta.squeeze(-1);
}
// ---------- 1. 设备常量 eye(6) 缓存 ---------- //
static torch::Tensor get_eye6(const at::TensorOptions &opts)
{
    // ---------- 修正处 ---------- //
    const auto scalar_type = c10::typeMetaToScalarType(opts.dtype());
    const int64_t key = (static_cast<int64_t>(opts.device().index()) << 8) |
                        static_cast<int64_t>(scalar_type);
    // --------------------------------

    static std::mutex mtx;
    static std::unordered_map<int64_t, torch::Tensor> cache;

    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(key);
        if (it != cache.end())
            return it->second;

        auto eye6 = torch::eye(6, opts);
        cache.emplace(key, eye6);
        return eye6;
    }
}

// ---------- 2. 优化后的求解函数 ---------- //
torch::Tensor optimizer_step_cuda_v2(
    torch::Tensor grad,      // [B,6,1] (float32, CUDA)
    torch::Tensor Hess,      // [B,6,6] (float32, CUDA)
    double lambda_d,         // Levenberg λ
    torch::Tensor mask = {}) // [B]  (bool / uint8, 可选)
{
    TORCH_CHECK(grad.is_cuda() && Hess.is_cuda(),
                "optimizer_step_cuda: grad & Hess must be CUDA");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32 &&
                    Hess.scalar_type() == torch::kFloat32,
                "optimizer_step_cuda: dtype must be float32");
    TORCH_CHECK(grad.dim() == 3 && Hess.dim() == 3 &&
                    grad.size(0) == Hess.size(0) &&
                    grad.size(1) == 6 && grad.size(2) == 1 &&
                    Hess.size(1) == 6 && Hess.size(2) == 6,
                "optimizer_step_cuda: expect grad[B,6,1], Hess[B,6,6]");

    const int64_t B = grad.size(0);
    const float lambda = static_cast<float>(lambda_d);
    const float eps = 1e-4f;          // 数值稳定扰动
    const auto opts = Hess.options(); // 保存 device/dtype

    // ------------- 2.1 复制 / 就地加 λI+epsI ------------- //
    // 若上游还要用到 Hess，需克隆一份；否则可直接就地修改：
    auto Hreg = Hess.clone(); // [B,6,6]

    // (λ+eps) 加到所有样本的对角线
    // 利用 view 把对角元素当作一维张量，批量加
    Hreg.diagonal(0, /*dim1=*/1, /*dim2=*/2)
        .add_(lambda + eps); // in-place

    // ------------- 2.2 处理可选掩码 ------------- //
    if (mask.defined() && mask.numel() == B)
    {
        auto m = mask.to(torch::kBool).view({B, 1, 1}); // [B,1,1] bool
        // grad: m==0 的样本直接置零
        grad.masked_fill_(~m, 0.0f);
        // Hreg: m==0 的对角线设 1，其余保持 Hreg
        // 这里不需要全置单位阵，只需改对角线
        Hreg.diagonal(0, 1, 2).masked_fill_(~m.squeeze(-1), 1.0f);
    }

    // ------------- 2.3 批量 Cholesky & 解方程 ------------- //
    // 现在 Hreg 一定是 SPD，直接求解
    auto L = at::cholesky(Hreg, /*upper=*/false);              // [B,6,6]
    auto delta = at::cholesky_solve(grad, L, /*upper=*/false); // [B,6,1]

    return delta.squeeze(-1); // [B,6]
}
// ---------------------------
// 2. 主函数：优化后的 optimizer_step_cuda
//    输入：grad [B,6,1]，Hess [B,6,6]，lambda_d，mask [B]（可选）
//    输出：delta [B,6] (float32, CUDA)
// ---------------------------
torch::Tensor optimizer_step_cuda_v3(
    torch::Tensor grad, // [B,6,1]，float32，CUDA
    torch::Tensor Hess, // [B,6,6]，float32，CUDA
    double lambda_d,    // Tikhonov 正则化 λ
    torch::Tensor mask  // [B]，bool or uint8，可传空 Tensor
)
{
    // 1. 检查输入合法性
    TORCH_CHECK(grad.is_cuda() && Hess.is_cuda(),
                "optimizer_step_cuda: grad & Hess must be CUDA tensors.");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32 &&
                    Hess.scalar_type() == torch::kFloat32,
                "optimizer_step_cuda: grad & Hess must be float32.");
    TORCH_CHECK(grad.dim() == 3 && Hess.dim() == 3 &&
                    grad.size(0) == Hess.size(0) &&
                    grad.size(1) == 6 && grad.size(2) == 1 &&
                    Hess.size(1) == 6 && Hess.size(2) == 6,
                "optimizer_step_cuda: expected shapes grad[ B,6,1 ] and Hess[ B,6,6 ].");

    const int64_t B = grad.size(0);
    const float lambda = static_cast<float>(lambda_d);
    auto opts = grad.options(); // 保留 grad 的 device 和 dtype 信息 (CUDA, float32)

    // 2. 为所有样本初始化输出 delta_out = zeros([B,6])
    //    对应 mask=false 的样本，会保持为 0。
    torch::Tensor delta_out = torch::zeros({B, 6}, opts);

    // 3. 处理 mask —— 生成一个 [B] 的 bool Tensor 在 CUDA 上
    //    若 mask 未定义（undefined）或 numel()==0，则视作全 true。
    torch::Tensor mask_bool;
    if (!mask.defined() || mask.numel() == 0)
    {
        mask_bool = torch::ones({B}, torch::dtype(torch::kBool).device(opts.device()));
    }
    else
    {
        // 先转换为 bool，再搬到 CUDA，最后展平成 [B]
        mask_bool = mask.to(torch::kBool);
        if (!mask_bool.is_cuda())
        {
            mask_bool = mask_bool.to(opts.device());
        }
        mask_bool = mask_bool.view({B});
    }

    // 4. 在 mask_bool 中找出所有为 true 的下标 idx_active —— [K]（CUDA 端 int64）
    //    注意：C++ 版 nonzero 只接受一个输入参数。
    auto idx_active = torch::nonzero(mask_bool).squeeze(-1); // 去掉第二维，得到 [K]
    int64_t K = idx_active.numel();
    if (K == 0)
    {
        // 全部样本都被 mask 掉，直接返回全 0 输出
        return delta_out;
    }

    // 5. 从 grad, Hess 中挑出活跃子批次 grad_k:[K,6,1], Hess_k:[K,6,6]
    torch::Tensor grad_k = grad.index_select(0, idx_active); // [K,6,1]
    torch::Tensor Hess_k = Hess.index_select(0, idx_active); // [K,6,6]

    // 6. 构造正则化矩阵 λI 于子批次
    torch::Tensor eye6 = torch::eye(6, opts); // [6,6] 单位阵
    torch::Tensor tikh = eye6.mul(lambda);    // [6,6] = λ * I
    torch::Tensor tikh_b = tikh.unsqueeze(0); // [1,6,6]，以便广播到 [K,6,6]
    torch::Tensor Hreg_k = Hess_k + tikh_b;   // [K,6,6]

    // 7. 对每个子批次样本做 Cholesky 或通用求解
    torch::Tensor delta_k; // 最终大小 [K,6,1]
    try
    {
        // 7.1 尝试 Cholesky 分解
        auto L = at::cholesky(Hreg_k, /*upper=*/false);           // [K,6,6]
        delta_k = at::cholesky_solve(grad_k, L, /*upper=*/false); // [K,6,1]
    }
    catch (const c10::Error &e)
    {
        // 7.2 如果奇异或 Cholesky 失败，则回退到通用 linalg_solve
        delta_k = at::linalg_solve(Hreg_k, grad_k); // [K,6,1]
    }

    // 8. 将 [K,6,1] 的 delta_k squeeze(-1) → [K,6]，copy 回 delta_out 对应行
    torch::Tensor delta_k_2d = delta_k.squeeze(-1); // [K,6]
    delta_out.index_copy_(0, idx_active, delta_k_2d);

    // 9. 返回最终结果，mask=false 的样本行依然全 0
    return delta_out;
}

// -------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("residual_jacobian_batch_quat_cuda", &residual_jacobian_batch_quat_cuda);
    m.def("optimizer_step_cuda", &optimizer_step_cuda);
    m.def("optimizer_step_cuda_v2", &optimizer_step_cuda_v2);
    m.def("optimizer_step_cuda_v3", &optimizer_step_cuda_v3);
}
