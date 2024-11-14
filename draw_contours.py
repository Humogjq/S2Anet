import os
import cv2
import numpy as np
from skimage.measure import label, regionprops

img_root = "/home/imed/data_disk/data/qh/line_train"
gt_root = "/home/imed/data_disk/data/qh/train_label_or"
save_root = "/home/imed/data_disk/data/qh/line_train/"

for img in os.listdir(img_root):
    img_name = img[:-4]
    img_path = os.path.join(img_root, img)
    gt_path = os.path.join(gt_root,"Img" +img_name[2:5] + "-exp-or.bmp")
    print(gt_path)
    save_name = img_name + ".bmp"

    image = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.COLOR_BGR2GRAY)
    gt = np.array(gt)
    print(gt.shape)
    gt_ = np.ones((gt.shape[0], gt.shape[1]), dtype='uint8')
    gt_[gt[:, :] == 255] = 0
    im2, contours = cv2.findContours(gt_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, im2, -1, (255, 0, 0), 1)
    # gt2 = np.ones((gt.shape[0], gt.shape[1]), dtype='uint8')
    # gt2[gt[:, :] == 255] = 0
    # gt2[gt[:, :] == 128] = 0
    # im2_, contours2 = cv2.findContours(gt2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, im2_, -1, (255, 0, 0), 3)

    # image[:, :, 0][(mask[:, :, 0] == 128)] = 255
    # image[:, :, 1][(mask[:, :, 1] == 128)] = 0
    # image[:, :, 2][(mask[:, :, 2] == 128)] = 0
    cv2.imwrite(save_root + save_name, image)

