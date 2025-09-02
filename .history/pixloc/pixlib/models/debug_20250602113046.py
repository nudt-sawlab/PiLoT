import torch
import direct_abs_cost_cuda
def _run():
    device = 'cuda'
    inputs = torch.load("/home/ubuntu/Documents/code/FPV-Test/DirectAbsoluteCostCuda/sample_inputs.pt", map_location="cpu")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    grad, Hess, w_loss, valid_qm, p2d_q, loss = direct_abs_cost_cuda.residual_jacobian_batch_quat_cuda(
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
    print(grad, Hess, w_loss, valid_qm, p2d_q, loss)

if __name__ == "__main__":
    _run()