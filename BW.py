import os
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



gt_dir = "/media/imed/data/qh/disc_cup_seg/data/labels_trans"
gt_dir2 = "/media/imed/data/qh/disc_cup_seg/data/label"

gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

for gtPath in gt_lst:
    if gtPath.endswith(".bmp"):
        gt_name = gtPath.split("/")[-1]
        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
        gt = np.array(gt, np.float32)
        x, y = gt.shape
        gt_ = np.zeros((x,y), np.float32)
        gt_[gt >130] = 0
        gt_[gt<130] = 255
        gt_path = os.path.join(gt_dir2, gt_name[:-4] + ".jpg")
        cv2.imwrite(gt_path, gt_)