# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve
from fundus_dataset import recompone_overlap
from utils import mkdir, Visualizer
from evaluation import *

from skimage import morphology
import scipy
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from skimage import measure,draw
import matplotlib.pyplot as plt
import os


def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def disc_crop(org_img, DiscROI_size, C_x, C_y, fill_value=0):
    tmp_size = int(DiscROI_size / 2)
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    if fill_value != 0:
        disc_region = disc_region + fill_value
    crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
    err_coord = [0, DiscROI_size, 0, DiscROI_size]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3], ]

    return disc_region, err_coord, crop_coord
def get_best_thresh(gt_arr, pred_arr):
    gt_arr = gt_arr // 255
    # num_pos = np.sum(gt_arr)
    # num_neg = gt_arr.reshape(-1)[0] - num_pos
    pred_arr = pred_arr / 255.0

    fpr, tpr, thresholds = roc_curve(gt_arr.reshape(-1), pred_arr.reshape(-1), pos_label=1)
    """
    RightIndex = 0.5 * ((tpr - 1) ** 2 + fpr ** 2) / (tpr - fpr) ** 2
    thresh_idx = np.argmin(RightIndex)
    thresh_value = thresholds[thresh_idx]
    thresh_arr[thresh_arr >= thresh_value] = 255
    thresh_arr[thresh_arr < thresh_value] = 0
    """

    best_dice = 0
    thresh_value = 0
    for i in range(2, thresholds.shape[0] - 1):
        thresh_arr = pred_arr.copy()
        thresh_arr[thresh_arr >= thresholds[i]] = 1
        thresh_arr[thresh_arr < thresholds[i]] = 0
        current_dice = 0.3 * calc_iou(thresh_arr, gt_arr) + calc_acc(thresh_arr, gt_arr)
        if current_dice >= best_dice:
            best_dice = current_dice
            thresh_value = thresholds[i]

    thresh_arr = pred_arr.copy()
    thresh_arr[thresh_arr >= thresh_value] = 255
    thresh_arr[thresh_arr < thresh_value] = 0

    return thresh_value, thresh_arr

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(pred, gt):

    prediction = np.asarray(pred)
    mask = np.asarray(gt)
    mask_binary = np.zeros((mask.shape[0], mask.shape[1],1))
    prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],1))
    mask_binary[mask < 128] = 0
    mask_binary[mask > 128] = 1
    prediction_binary[prediction < 128] = 0
    prediction_binary[prediction > 128] = 1
    disc_dice = dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0])
    return disc_dice

def test_first_stage(viz, dataloader, net, device, results_dir, criterion=None, isSave=True):
    dice_lst = []  # Loss
    #dice_lst_faz = []
    dice_lst_ves = []
    disc_lst = []
    cup_lst = []

    i = 1
    with torch.no_grad():
        for sample in dataloader:
            #if len(sample) != 5 and len(sample) != 4 and len(sample) != 2:
            #    print("Error occured in sample %03d, skip" % i)
            #    continue

            print("Evaluate %03d..." % i)
            i += 1

            img = sample[0].to(device)
            gt = sample[1].to(device)
            #faz = sample[2].to(device)
            #faz = faz.unsqueeze(1)
            # print(gt.shape)
            #pred,pred_faz = net(img)
            pred= net(img)




            viz.img(name="prediction_test_ves"+str(i), img_=pred[:, :, :, :])
            #viz.img(name="prediction_test_faz"+str(i), img_=pred_faz[:, :, :, :])


            gt = gt.cpu().squeeze(0)
            #faz = faz.cpu().squeeze(0)
            #gt_ = np.array(gt)
            #disc_gt = gt_[0, :, :].squeeze()

            gt = transforms.ToPILImage()(gt).convert('L')   #似乎有点重复
            gt = np.array(gt)

            #faz = transforms.ToPILImage()(faz).convert('L')   #似乎有点重复
            #faz = np.array(faz)

            pred = pred.cpu().squeeze(0)
            #pred_faz = pred_faz.cpu().squeeze(0)

            #pred_ = np.array(pred)
            #disc_pred = pred_[0, :, :].squeeze()
            # print("disc_pred:", disc_pred.shape)
            # print("disc_max:", disc_pred.max())

            pred = transforms.ToPILImage()(pred).convert('L')  #似乎有点重复
            pred = np.array(pred)

            #pred_faz = transforms.ToPILImage()(pred_faz).convert('L')  #似乎有点重复
            #pred_faz = np.array(pred_faz)

            print(pred.shape)
            ves_dice = calculate_dice(pred, gt)
            #faz_dice = calculate_dice(pred_faz, faz)
            #dice_rate = (ves_dice + faz_dice)/2
            dice_rate = ves_dice
            #dice_rate = faz_dice
            print('dice_rate:', dice_rate)

            # thresh_value_disc, thresh_pred_disc = get_best_thresh(disc_gt, disc_pred)
            # thresh_value_cup, thresh_pred_cup = get_best_thresh(cup_gt, cup_pred)
            thresh_pred_disc = (pred > 128)
            thresh_pred_disc = thresh_pred_disc.astype(np.uint8)

            #thresh_pred_disc_faz = (pred_faz > 128)
            #thresh_pred_disc_faz = thresh_pred_disc_faz.astype(np.uint8)
            #thresh_pred_disc = morphology.binary_erosion(thresh_pred_disc, morphology.diamond(7)).astype(np.uint8)  # return 0,1
            #thresh_pred_disc = get_largest_fillhole(thresh_pred_disc)
            #thresh_pred_disc = morphology.binary_dilation(thresh_pred_disc, morphology.diamond(7)).astype(np.uint8)  # return 0,1
            #thresh_pred_disc = get_largest_fillhole(thresh_pred_disc).astype(np.uint8)  # return 0,1

            #thresh_pred_disc = (thresh_pred_disc > 0.5)

            #thresh_pred_disc = thresh_pred_disc.astype(np.uint8)

            thresh_pred_img = thresh_pred_disc
            #thresh_pred_img_faz = thresh_pred_disc_faz
            # print(thresh_pred_img.max())
            # print(thresh_pred_img.min())

            thresh_pred_img[thresh_pred_img == 0] = 0
            thresh_pred_img[thresh_pred_img == 1] = 255           #thresh

            #thresh_pred_img_faz[thresh_pred_img_faz == 0] = 0
            #thresh_pred_img_faz[thresh_pred_img_faz == 1] = 255           #thresh

            thresh_pred_img = np.array(thresh_pred_img, np.uint8)
            #thresh_pred_img_faz = np.array(thresh_pred_img_faz, np.uint8)
            # kernel = np.ones((3, 3))
            # thresh_pred_img = cv2.erode(thresh_pred_img,kernel=kernel)
            # thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("shape of prediction", thresh_pred_disc.shape)
            print("shape of groundtruth", gt.shape)

            dice_lst.append(dice_rate)
            #dice_lst_faz.append(faz_dice)
            dice_lst_ves.append(ves_dice)

            # Save Results
            imgName = dataloader.dataset.getFileName()
            if isSave:
                mkdir(results_dir + "/first_stage/Thresh")
                mkdir(results_dir + "/first_stage/noThresh")
                #mkdir(results_dir + "/first_stage/Thresh_faz")
                #mkdir(results_dir + "/first_stage/noThresh_faz")

                cv2.imwrite(results_dir + "/first_stage/noThresh/" + imgName, pred)
                cv2.imwrite(results_dir + "/first_stage/Thresh/" + imgName, thresh_pred_img)
                #cv2.imwrite(results_dir + "/first_stage/noThresh_faz/" + imgName, pred_faz)
                #cv2.imwrite(results_dir + "/first_stage/Thresh_faz/" + imgName, thresh_pred_img_faz)

    dice_arr = np.array(dice_lst)
    #dice_arr_faz = np.array(dice_lst_faz)
    dice_arr_ves = np.array(dice_lst_ves)

    print("Dice - mean: " + str(dice_arr.mean()) + "\tstd: " + str(dice_arr.std()))

    return dice_arr,dice_arr_ves

def predict_first_stage(dataloader, net, device, results_dir, isSave=True, crop_size=512):
    i = 1
    with torch.no_grad():

        ##############mask_220310####################################
        '''
        mask_erosion = np.zeros([512,512])
        for i in range(512):
            for j in range(512):
                if ((i-255.5)*(i-255.5)+(j-255.5)*(j-255.5)) < 35000:
                    mask_erosion[i,j] = 1
        mask_erosion.astype(np.uint8)

        sidecut = np.zeros([1984,450])
        sidecut.astype(np.uint8)
        sidecut = sidecut+255
        '''
        ##############mask_220310####################################


        for sample in dataloader:
            # print(sample.shape)

            print("Evaluate %03d..." % i)
            # with open("/home/imed/qh/disc_cup_seg/2.csv", 'r') as file_to_read:
            #     lines = file_to_read.readlines()
            #     name,err_coord0,err_coord1,err_coord2,err_coord3,crop_coord0,crop_coord1,crop_coord2,crop_coord3 = lines[i-1].split()
            i += 1
            #print(sample)

            #240527更改 解决AttributeError: 'list' object has no attribute 'to'，使用torch.stack更改sample
            shapes = [tensor.shape for tensor in sample]
            print(shapes)
            #sample_tensor = torch.stack(sample, dim=0)
            shape = sample[0]
            # print(err_coord2,err_coord3)
            img = shape.to(device)
            # img = img.cpu().unsqueeze(0)




            print(img.shape)
            pred, pred_faz= net(img)

            pred = pred.cpu().squeeze(0).squeeze(0)
            pred = np.array(pred)*255
            print(pred.shape)
            pred_faz = pred_faz.cpu().squeeze(0).squeeze(0)
            pred_faz = np.array(pred_faz)*255
            pred[pred < 70] = 0
            pred[pred > 69] = 255
            pred_faz[pred_faz < 70] = 0
            pred_faz[pred_faz > 69] = 255
            imgName = dataloader.dataset.getFileName()
            print(results_dir + "/ves/" + imgName)
            cv2.imwrite(results_dir + "/ves/" + imgName, pred)
            cv2.imwrite(results_dir + "/faz/" + imgName, pred_faz)

            '''         
            #220309_pass
            file_path = '/media/imed/data/qh/disc_cup_seg/data/RIM-ONE/valid/images/'
            t = i +83
            name = "Im%03d.bmp" % t
            print(name)
            image_path = file_path + name
            print(image_path)
            image = cv2.imread(image_path)
            image = np.asarray(image)
            print(image.shape)
            x, y, _ = image.shape
            '''


            #######################
            ###24.5.28修改，以下部分为适用视杯视盘分割部分

            '''




            pred_ = np.array(pred)
            disc_pred = pred_[0, :, :].squeeze()
            #cv2.imshow("IMG_2", disc_pred)
            # print("disc_pred:", disc_pred.shape)
            # print("disc_max:", disc_pred.max())
            # cup_pred = pred_[1, :, :].squeeze()
            # print("cup_max:", cup_pred.max())

            pred = transforms.ToPILImage()(pred).convert('L')
            #pred.show()
            pred = np.array(pred)
            print(pred.shape)

            # thresh_value_disc, thresh_pred_disc = get_best_thresh(disc_gt, disc_pred)
            # thresh_value_cup, thresh_pred_cup = get_best_thresh(cup_gt, cup_pred)
            disc_pred = disc_pred * mask_erosion           ## erosion round
            disc_pred = disc_pred * (1/disc_pred.max())
            print(disc_pred.max())
            thresh_pred_disc = (disc_pred > 0.35)
            thresh_pred_cup = (disc_pred > 0.95)
            thresh_pred_disc = thresh_pred_disc.astype(np.uint8)
            thresh_pred_cup = thresh_pred_cup.astype(np.uint8)
            #cv2.imshow("IMG_5", thresh_pred_disc)
            # thresh_pred_cup = thresh_pred_cup.astype(np.uint8)

            thresh_pred_disc = morphology.binary_erosion(thresh_pred_disc, morphology.diamond(5)).astype(
                np.uint8)  # return 0,1
            #cv2.imshow("IMG_6", thresh_pred_disc)
            thresh_pred_cup = morphology.binary_erosion(thresh_pred_cup, morphology.diamond(2)).astype(
                np.uint8)  # return 0,1
            thresh_pred_disc = get_largest_fillhole(thresh_pred_disc)
            #cv2.imshow("IMG_7", thresh_pred_disc)
            thresh_pred_cup = get_largest_fillhole(thresh_pred_cup)

            thresh_pred_disc = morphology.binary_dilation(thresh_pred_disc, morphology.diamond(5)).astype(
                np.uint8)  # return 0,1
            #cv2.imshow("IMG_8", thresh_pred_disc)
            thresh_pred_cup = morphology.binary_dilation(thresh_pred_cup, morphology.diamond(2)).astype(
                np.uint8)  # return 0,1

            thresh_pred_disc = get_largest_fillhole(thresh_pred_disc).astype(np.uint8)  # return 0,1
            thresh_pred_cup = get_largest_fillhole(thresh_pred_cup).astype(np.uint8)
            #cv2.imshow("IMG_9", thresh_pred_disc)
            #thresh_pred_disc = (thresh_pred_disc > 0.5)
            # thresh_pred_cup = (thresh_pred_cup > 0.5)
            #thresh_pred_disc = thresh_pred_disc.astype(np.uint8)
            #cv2.imshow("IMG_11", thresh_pred_disc)
            # thresh_pred_cup = thresh_pred_cup.astype(np.uint8)
            thresh_pred_cup = thresh_pred_cup*thresh_pred_disc
            thresh_pred_img = thresh_pred_disc + thresh_pred_cup

            #thresh_pred_img = thresh_pred_disc
            print(thresh_pred_img.max())
            print(thresh_pred_img.min())

            #thresh_pred_img[thresh_pred_img == 0] = 0
            #thresh_pred_img[thresh_pred_img == 1] = 255
            thresh_pred_img[thresh_pred_img == 0] = 255
            thresh_pred_img[thresh_pred_img == 1] = 128
            thresh_pred_img[thresh_pred_img == 2] = 0

            thresh_pred_img = np.array(thresh_pred_img, np.uint8)
            #cv2.imshow("IMG_12", thresh_pred_img)
            # Img_result = np.zeros((x, y, 3), dtype=int) + 255
            # Img_result[int(crop_coord0):int(crop_coord1), int(crop_coord2):int(crop_coord3), 0] = thresh_pred_img[
            #                                                                           int(err_coord0):int(err_coord1),
            #                                                                           int(err_coord2):int(err_coord3)]

            # Save Results
            imgName = dataloader.dataset.getFileName()

            ##################220311 resize

            thresh_pred_img = cv2.resize(thresh_pred_img, (2076, 1984))
            thresh_pred_img = np.concatenate((sidecut, thresh_pred_img),axis = 1)
            thresh_pred_img = np.concatenate((thresh_pred_img, sidecut), axis=1)



            if isSave:
                mkdir(results_dir + "/exp-5")
                mkdir(results_dir + "/exp-5_nothresh")
                # mkdir(results_dir + "/first_stage/BW_imgs")
                cv2.imwrite(results_dir + "/exp-5/" + imgName, thresh_pred_img)
                cv2.imwrite(results_dir + "/exp-5_nothresh/" + imgName, pred)

            '''


