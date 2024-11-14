# -*- coding: utf-8 -*-

import os
import torch
import cv2
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args
from utils import mkdir, build_dataset, build_model, Visualizer
from losses import build_loss
from train import train_first_stage, val_first_stage
from test import test_first_stage
from mpvit import mpvit_gjq
from mpvit_test import mpvit_gjq_test
from DS_TransUNet import UNet_DS
from model_JSN import base

from mpvit_230315_6layer import mpvit_gjq_6layer
from torchinfo import summary

from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("gpu using")
print(torch.cuda.is_available())

if args.mode == "train":
    isTraining = True
else:
    isTraining = False


database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
                         crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
sub_dir = args.dataset + "/" + args.model + "/" + args.loss  # ## _thin _thick

if isTraining:  # train
    NAME = args.dataset + "_" + args.model + "_" + args.loss  # ## _thin _thick
    viz = Visualizer(env=NAME)
    writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
    mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹

    # 加载数据集
    train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
                                 crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
    val_dataloader = DataLoader(val_database, batch_size=1)





    # 构建模型

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    #net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    #net = base(1)
    #net = mpvit_gjq()
    #net = mpvit_gjq_test()
    #net = UNet_DS(128,1)
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    net = build_model(args.model, device, channel=args.input_nc)
    net = torch.nn.DataParallel(net, device_ids=[0]).to(device)
    #net = net.to(device)
    #optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)



    #net = torch.nn.DataParallel(net)
    #optimizer = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr , betas=(0.9, 0.999), eps=1e-8)
    criterion = build_loss(args.loss)
    criterion2 = build_loss(args.loss2)
    best_metric = {"epoch": 0, "dice": 0}
    # start training
    print("Start training...")
    #epoch = 0
    #net = val_first_stage(best_metric,viz, writer, val_dataloader, net, device, args.save_epoch_freq, args.models_dir + "/" + sub_dir, args.results_dir + "/" + sub_dir, epoch, args.first_epochs)
    for epoch in range(args.first_epochs):
        print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
        print('-'*10)
        net = train_first_stage(viz, writer, train_dataloader, net, optimizer, args.init_lr, criterion,criterion2, device, args.power, epoch, args.first_epochs)
        if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1 or epoch ==1 :
            net = val_first_stage(best_metric,viz, writer, val_dataloader, net, device, args.save_epoch_freq,
                  args.models_dir + "/" + sub_dir, args.results_dir + "/" + sub_dir, epoch, args.first_epochs)
    print("Training finished.")

else:  # test
    # 加载数据集和模型
    test_dataloader = DataLoader(database, batch_size=1)
    net = torch.load(args.models_dir + "/" + sub_dir + "/exp5/front_model-regular-" + args.model_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
    net.eval()
    
    # start testing
    print("Start testing...")
    test_first_stage(test_dataloader, net, device, args.results_dir + "/" + sub_dir, criterion=None, isSave=True)
    print("Testing finished.")
