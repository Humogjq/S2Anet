import skimage.io as io
import cv2
import numpy as np


str1 = '/home/imed/data_disk/data/qh/train_label_and/*.bmp'
label1 = io.ImageCollection(str1)
print(len(label1))


str2 = '/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/RIM-ONE-exp5/*.bmp'
label2 = io.ImageCollection(str2)
print(len(label2))

str3 = '/home/imed/data_disk/data/qh/train_label_and/'

for i in range(0,len(label2)):
    img = cv2.bitwise_and(label1[i],label2[i])
    io.imsave(str3 + 'Img' + np.str(i+1).zfill(3) + '-exp-and.bmp',img)

# import os
#
# path = '/home/imed/RIM-ONE/label_fusion_and'
# num_list = [12,13,14,15,23,24,25,34,35,45,123,124,125,134,135,145,234,235,245,345,1234,1235,1245,1345,2345,12345]
# print(len(num_list))
#
# for i in range(len(num_list)):
#     file_name = os.path.join(path,'RIM-ONE-exp' + str(num_list[i]))
#     os.makedirs(file_name)


