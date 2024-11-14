import os
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil


# 获得mask质心坐标
gt_dir = "/media/imed/data/qh/disc_cup_seg/data/refuge/test/labels"
img_dir = "/media/imed/data/qh/disc_cup_seg/data/refuge/train/images"
savepath = "/media/imed/data/qh/disc_cup_seg/data/refuge/test/disc_small/labels/"

gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

name_lst = []
x_lst = []
y_lst = []
for gtPath in gt_lst:
    gt_name = gtPath.split("/")[-1]
    gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
    gt = np.array(gt)
    x, y = np.shape(gt)
    gt1 = np.array(gt, np.float32) / 255.0
    gt_ = np.zeros((x, y), np.float32)
    gt_[gt1 > 0.6] = 0
    gt_[gt1 < 0.6] = 1
    print(np.unique(gt_))
    labels = label(gt_, connectivity=2)  # 8连通区域标记
    regions = regionprops(labels)
    C_x = int(regions[0].centroid[0])
    C_y = int(regions[0].centroid[1])
    disc_region = np.ones((512, 512), dtype=gt.dtype) * 255
    crop_coord = np.array([C_x - 256, C_x + 256, C_y - 256, C_y + 256], dtype=int)
    err_coord = [0, 512, 0, 512]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > gt.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - gt.shape[0])
        crop_coord[1] = gt.shape[0]

    if crop_coord[3] > gt.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - gt.shape[1])
        crop_coord[3] = gt.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = gt[crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3], ]
    cv2.imwrite(savepath + gt_name[:-4] + ".png", disc_region)

    # x_lst.append(C_x)
    # y_lst.append(C_y)
    # name_lst.append(gt_name)

# dataframe = pd.DataFrame({"img_name":name_lst,'x':x_lst,'y':y_lst})
# dataframe.to_csv(r"/media/imed/data/qh/disc_cup_seg/data/ORIGA_REFUGE_mix/test/label.csv",sep=',', index=None)



# gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
#
# name_lst = []
# x_lst = []
# y_lst = []
# for gtPath in gt_lst:
#     gt_name = gtPath.split("/")[-1]
#     gt = cv2.imread(gtPath, cv2.IMREAD_COLOR)
#     gt = np.array(gt, np.float32)
#     print()
#     x,y,_ = gt.shape
#     disc_region = np.ones((x, y), dtype=gt.dtype) * 255
#     disc_region[gt[:, :, 1] >= 128] = 128
#     disc_region[gt[:, :, 2] >= 128] = 0
#
#     cv2.imwrite(savepath + gt_name[:-4] + ".png", disc_region)
    # labels = label(gt, connectivity=2)  # 8连通区域标记
    # regions = regionprops(labels)
    # C_x = int(regions[0].centroid[0])
    # C_y = int(regions[0].centroid[1])
    # x_1 = regions[0].bbox[0]
    # y_1 = regions[0].bbox[1]
    # x_2 = regions[0].bbox[2]
    # y_2 = regions[0].bbox[3]
    # print("x_1",x_1, "y_1",y_1,"x_2",x_2, "y_2",y_2)
    # print("C_x",C_x, "C_y",C_y)

#
# # dataframe = pd.DataFrame({"img_name":name_lst,'x':x_lst,'y':y_lst})
# # dataframe.to_csv(r"/media/imed/data/qh/disc_cup_seg/data//label.csv",sep=',', index=None)
#
# train_path = "/media/imed/data/qh/disc_cup_seg/data/ORIGA_650/train/labels"
# test_path = "/media/imed/data/qh/disc_cup_seg/data/ORIGA_650/test/images"
# train_save = "/media/imed/data/qh/disc_cup_seg/data/ORIGA_650/train_labels"
# test_save = "/media/imed/data/qh/disc_cup_seg/data/ORIGA_650/test_labels"
# mask_root = "/media/imed/data/qh/disc_cup_seg/data/ORIGA_650/labels"
# test_list = os.listdir(test_path)
# for file in test_list:
#     file_name = file[:-4]
#     src = os.path.join(mask_root, file_name + ".png")
#     dst = os.path.join(test_save, file_name + ".png")
#     print('src:', src)
#     print('dst:', dst)
#     shutil.move(src, dst)



