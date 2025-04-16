# -*- coding: utf-8 -*-

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=str, default="0,1,4,6", help="device")
parser.add_argument("--dataset1", type=str, default="refuge", choices=["rose", "cria", "drive", "origa", "refuge", "refuge1","rimone1"], help="dataset")  # choices可扩展
parser.add_argument("--dataset", type=str, default="FAZ500", choices=["rose", "FAZ500","rose1","zeiss_1", "cria", "drive", "origa", "refuge", "origa2", "mix", "iostar", "rimone", "rimone1"], help="dataset")  # choices可扩展
parser.add_argument("--model", type=str, default="VES500net", choices=["unet","FAZ500net","VES500net", "VESpial500net","cenet", "resunet", "csnet", "srfunet", "r2unet"], help="model")  # choices可扩展
parser.add_argument("--loss", type=str, default="dice", choices=["mse", "l1", "smoothl1", "bce", "focal", "dice", "smooth", "dice2"], help="loss")  # choices可扩展
parser.add_argument("--loss2", type=str, default="dice", choices=["mse", "l1", "smoothl1", "bce", "focal", "dice", "smooth", "dice2"], help="loss")  # choices可扩展
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train or test")

# data settings
parser.add_argument("--data_dir", type=str, default="./dataset", help="path to folder for getting dataset")
parser.add_argument("--input_nc", type=int, default=3, choices=[1, 3], help="gray or rgb")

parser.add_argument("--img_size", type=int, default=304, help="image size")      ####################################
parser.add_argument('--vit_patches_size', type=int,default=16, help='vit_patches_size, default is 16')


parser.add_argument("--crop_size", type=int, default=512, help="crop size")
parser.add_argument("--scale_size", type=int, default=512, help="scale size (applied in drive and cria)")
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')


# training
parser.add_argument("--batch_size", type=int, default=4 ,help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
parser.add_argument("--val_epoch_freq", type=int, default=10, help="frequency of validation at the end of epochs")
parser.add_argument("--save_epoch_freq", type=int, default=10, help="frequency of saving models at the end of epochs")


parser.add_argument("--init_lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument('--base_lr', type  =float,  default=0.01,help='segmentation network learning rate')


parser.add_argument("--power", type=float, default=0.9, help="power")
parser.add_argument("--weight_decay", type=float, default=0.00001, help="weight decay")
# first stage
parser.add_argument("--first_epochs", type=int, default=500, help="train epochs of first stage")
# second stage (if necessary)
parser.add_argument("--second_epochs", type=int, default=100, help="train epochs of second stage")
parser.add_argument("--pn_size", type=int, default=7, help="size of propagation neighbors")
parser.add_argument("--base_channels", type=int, default=256, help="basic channels")

parser.add_argument('--n_skip', type=int,default=3, help='using number of skip-connect, default is num')
parser.add_argument('--num_classes', type=int,default=1, help='output classes num')

# results
parser.add_argument("--logs_dir", type=str, default="logs", help="path to folder for saving logs")
parser.add_argument("--models_dir", type=str, default="models", help="path to folder for saving models")
parser.add_argument("--results_dir", type=str, default="results", help="path to folder for saving results")
parser.add_argument("--model_suffix", type=str, default="100-0.9734680437771199.pth", help="front_model-regular-[model_suffix].pth will be loaded in models_dir")

args = parser.parse_args()
