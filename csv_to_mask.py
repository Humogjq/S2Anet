import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import cv2

# csv_file = "/home/imed/data_disk/data/qh/train_Result"
# csv_list = sorted(list(map(lambda x: os.path.join(csv_file, x), os.listdir(csv_file))))
# mask_root = "/home/imed/data_disk/data/qh/train_GT"
# for csv_name in csv_list:
#     if csv_name.split('.')[1] == "bmp-layer-1-lesion":
#         img_name = csv_name.split(".")[0][-5:] + ".bmp"
#         # print(img_name)
#         file_path = '/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/images/'
#         image_path = file_path + img_name
#         print(image_path)
#         image = cv2.imread(image_path)
#         image = np.asarray(image)
#         print(image.shape)
#         x, y, _ = image.shape
#
#         csv_data = pd.read_csv(csv_name)
#         pts = np.array(csv_data).astype("int")
#         mask = np.zeros((x,y), dtype='uint8')
#
#         cv2.polylines(mask, [pts], 1, (255,255,255))
#         cv2.fillPoly(mask, [pts], (255,255,255))  # 关键的一步，用这个函数就可以将轮廓内的部分标记出来。
#
#         # csv_name2 = csv_name.split("-")[0] + "-layer-4-lesion.csv"
#         # csv_data2 = pd.read_csv(csv_name2)
#         # pts2 = np.array(csv_data2).astype("int")
#
#         save_path = os.path.join(mask_root, img_name[:-4] + ".bmp")
#         # cv2.polylines(mask, [pts2], 1, (0, 0, 0))
#         # cv2.fillPoly(mask, [pts2], (0, 0, 0))  # 关键的一步，用这个函数就可以将轮廓内的部分标记出来。
#         cv2.imwrite(save_path, mask) # 最后就是存储了



csv_file = "/media/imed/data/qh/mx_Result"
csv_list = sorted(list(map(lambda x: os.path.join(csv_file, x), os.listdir(csv_file))))
img_root = "/media/imed/data/qh/mingxing"
img_save = "/media/imed/data/qh/mx_crop"
for csv_name in csv_list:
    if csv_name.split('.')[1] == "jpg-layer-1-mark":
        img_name = csv_name.split("-")[0].split("/")[-1]
        # print(img_name)
        image_path = os.path.join(img_root, img_name)
        # print(image_path)
        image = cv2.imread(image_path)
        img = np.asarray(image)
        print(image.shape)
        disc_region = np.zeros((512, 512, 3), dtype=img.dtype)
        csv_data = pd.read_csv(csv_name)
        C_x, C_y = csv_data
        C_x = int(C_x)
        C_y = int(C_y)
        crop_coord = np.array([C_y - 256, C_y + 256, C_x - 256, C_x + 256], dtype=int)
        err_coord = [0, 512, 0, 512]

        if crop_coord[0] < 0:
            err_coord[0] = abs(crop_coord[0])
            crop_coord[0] = 0

        if crop_coord[2] < 0:
            err_coord[2] = abs(crop_coord[2])
            crop_coord[2] = 0

        if crop_coord[1] > img.shape[0]:
            err_coord[1] = err_coord[1] - (crop_coord[1] - img.shape[0])
            crop_coord[1] = img.shape[0]

        if crop_coord[3] > img.shape[1]:
            err_coord[3] = err_coord[3] - (crop_coord[3] - img.shape[1])
            crop_coord[3] = img.shape[1]

        disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = img[crop_coord[0]:crop_coord[1],
                                                                              crop_coord[2]:crop_coord[3], ]
        cv2.imwrite(os.path.join(img_save, img_name), disc_region)
