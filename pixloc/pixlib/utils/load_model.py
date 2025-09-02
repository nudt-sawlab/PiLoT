
import torch
ckpt = "/home/ubuntu/Documents/code/github/FPVLoc_dev/outputs/training/pixloc_megadepth/checkpoint_best.tar"
ckpt1 = "/home/ubuntu/Documents/code/github/src_open/workspace/train_Aero_seq_deepac_plus_fusion/logs-2025-04-14-22-47-20/model_last.ckpt"

model = torch.load(str(ckpt), map_location='cpu')

model1 = torch.load(str(ckpt1), map_location='cpu')
print("finish")
