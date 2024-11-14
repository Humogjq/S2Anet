import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args
from utils import mkdir, build_dataset, build_model, Visualizer
from losses import build_loss
from train import train_first_stage, val_first_stage
from test import predict_first_stage
import numpy as np
import os


# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

database = build_dataset("zeiss", "/home/imed/Desktop/GJQworkspace/230315_MPVit_OCTA/dataset", 1, isTraining=False, crop_size=(args.crop_size, args.crop_size), scale_size=(512,512))
sub_dir = args.dataset + "/" + args.model + "/" + args.loss  # ## _thin _thick


test_dataloader = DataLoader(database, batch_size=1)
print(device)
#net = torch.load(args.models_dir + "/" + sub_dir + "/exp5/front_model-regular-" + args.model_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
net = torch.load("/home/imed/Desktop/GJQworkspace/230315_MPVit_OCTA/front_model-regular-460-0.8818836601630807.pth")
net.eval()

# start testing
print("Start testing...")

predict_first_stage(test_dataloader, net, device,"/home/imed/Desktop/GJQworkspace/230315_MPVit_OCTA/compare_models/test", isSave=True, crop_size=512)


print("Testing finished.")