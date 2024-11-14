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

net = torch.load('/home/imed/Desktop/GJQworkspace/220810_FAZ/models/FAZtest/cenet_221109_changeDAC/focal/exp5/front_model-regular-190-0.9233911565409618.pth')
net = net.cuda().module
torch.save(net.state_dict(),'test.pth')