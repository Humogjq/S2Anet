import cv2
import numpy as np
from PIL import Image
import os
file_path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/seg_templete/data/OCTA/ROSE/train/thin_labels'
file_path1 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/seg_templete/data/OCTA/ROSE/train/thin_labels_1'
file_list = os.listdir(file_path)
file_list.sort()
for item in file_list:
    image = Image.open(os.path.join(file_path,item))
    image = image.convert('L')
    image = np.array(image)
    kernel = np.ones((2, 2))
    # image1 = cv2.erode(image, kernel, iterations=1)
    image2 = cv2.dilate(image, kernel)
    image2 = cv2.erode(image2, kernel, iterations=1)
    # image1 = Image.open('_WS BIguan_Lin__4359_Angio Retina_OS_2018-02-27_14-51-02_M_1948-07-22_Enface-304x3041.tif')
    # image1 = image1.convert('L')
    # image1 = np.array(image1)
    # image2 = image-image1
    cv2.imwrite(os.path.join(file_path1,item), image2)

