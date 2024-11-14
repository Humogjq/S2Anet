from PIL import Image
import os


import os
import cv2
import numpy as np
import csv
from scipy import misc
import pandas as pd

file_path = '/home/imed/qh/REFUGE2-Test/'
file_path1 = '/home/imed/qh/disc_small'
file_path2 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/hy/disk/disc_vgg0/post_process/fina1/'
filename = '/home/imed/qh/roi/det_test_scalpel.txt'
name = []
score = []
hpos = []
wpos = []
xpos = []
ypos = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
            pass
        n_tmp, m_tmp, h_tmp, w_tmp, x_tmp, y_tmp = [str(i) for i in lines.split()]
        print(n_tmp)
        name.append(n_tmp)
        score.append(float(m_tmp))
        hpos.append(float(h_tmp))
        wpos.append(float(w_tmp))
        xpos.append(float(x_tmp))
        ypos.append(float(y_tmp))
        pass

score = np.array(score)
hpos = np.array(hpos)
wpos = np.array(wpos)
xpos = np.array(xpos)
ypos = np.array(ypos)
labels1 = []
labels2 = []
labels3 = []
labels4 = []
coord = []
name_coord = []
x_coord = []
y_coord = []
name_lst = []
# csvFile = open('via_region_data-1.csv', "r")
# reader = csv.reader(csvFile)
# for item in reader:
#     if item[4][0] == '0':
#         item1 = eval(item[5])
#         number = item1['cx']
#         labels1.append(number)
#         number1 = item1['cy']
#         labels2.append(number1)
#     else:
#         item1 = eval(item[5])
#         number2 = item1['cx']
#         labels3.append(number2)
#         number3 = item1['cy']
#         labels4.append(number3)
# ii=0
# k=0
with open(r'3.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(name)):
        # distance = 1000.0
        if name[i] != name[i - 1]:
            # distance0 = []
            # for j in range(0, 25):
            #     x_cer = int((int(hpos[i + j]) + int(xpos[i + j])) / 2)
            #     y_cer = int((int(wpos[i + j]) + int(ypos[i + j])) / 2)
            #     distance1 = np.power((np.power((x_cer - int(labels3[ii])), 2) + np.power((y_cer - int(labels4[ii])), 2)), 0.5)
            #     # print(distance1)
            #     distance0.append(distance1)
            #     if distance1 < distance:
            #         distance = distance1
            # ii += 1
            # print(distance0)
            # k = distance0.index(distance)
            # # print(k)
            img_path = file_path + name[i] + '.jpg'
            img_name = name[i] + '.jpg'
            #
            # img_path1 = file_path1 + name[i + k] + ' ' + name1[i + k] + 'right'+'.jpg'
            print(img_path)
            img = cv2.imread(img_path)
            img = np.asarray(img)
            print(img.shape)
            C_x = int((hpos[i]+xpos[i])/2)
            C_y = int((wpos[i]+ypos[i])/2)
            disc_region = np.zeros((512, 512, 3), dtype=img.dtype)
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
            coord_temp = [name[i] + ".jpg",err_coord[0],err_coord[1],err_coord[2],err_coord[3],crop_coord[0],crop_coord[1],crop_coord[2],crop_coord[3]]
            name_lst.append(img_name)
            x_coord.append(C_x)
            y_coord.append(C_y)

            # csv_writer.writerow(coord_temp)
            cv2.imwrite(os.path.join(file_path1, name[i] + ".png"), disc_region)
dataframe = pd.DataFrame({"ImageName": name_lst, 'Fovea_X': x_coord, 'Fovea_Y': y_coord})
dataframe.to_csv(r"/home/imed/qh/4.csv", sep=',', index=None)




        # img = cv2.line(img, (int(int(hpos[i]+xpos[i])/2) - 25, int(int((wpos[i]+ypos[i])/2))),
        #                (int(int(hpos[i]+xpos[i])/2) + 25, int(int(wpos[i]+ypos[i])/2)), (0,255,0), 5)
        # img1 = cv2.line(img, (int(int(hpos[i]+xpos[i])/2), int(int(wpos[i]+ypos[i])/2) - 25),
        #                 (int(int(hpos[i]+xpos[i])/2), int(int(wpos[i]+ypos[i])/2) + 25),
        #                 (0,255,0), 5)
        # cv2.imwrite(os.path.join(file_path1,name[i]+".png"),disc_region)
        # im1 = np.zeros(shape=[480,480,3])
        # img11 = cv2.imread(img_path1)
        # img11 = cv2.resize(img11, (300,300))
        # img22 = np.zeros(shape=np.shape(img))
        # if int((int(hpos[i])+int(xpos[i]))/2) >=1200:
        #     k=i
        # #     im1 = img[(int((int(wpos[i])+int(ypos[i]))/2-200)):(int((int(wpos[i])+ int(ypos[i]))/2+100)),int((int(hpos[i])+int(xpos[i]))/2)-200:int((int(hpos[i])+int(xpos[i]))/2)+100, :]
        # #     # cv2.imwrite(os.path.join(file_path1, (name[i] +' ' + name1[i]+'right' + '.jpg')), im1)
        #
        #     # for k in range(1,35):
        #     #     if int((int(hpos[i+k]) + int(xpos[i+k])) / 2 )<=800:
        #     #         im1 = img[
        #     #               (int((int(wpos[i+k]) + int(ypos[i+k])) / 2 - 200)):(int((int(wpos[i+k]) + int(ypos[i+k])) / 2 + 100)),
        #     #               int((int(hpos[i+k]) + int(xpos[i+k])) / 2) - 100:int((int(hpos[i+k]) + int(xpos[i+k])) / 2) + 200, :]
        #     #         cv2.imwrite(os.path.join(file_path2, (name[i] + ' ' + name1[i] + 'left' + '.jpg')), im1)
        #     im1 = img[
        #                   (int((int(wpos[i]) + int(ypos[i])) / 2 - 200)):(int((int(wpos[i]) + int(ypos[i])) / 2 + 100)),
        #                   int((int(hpos[i]) + int(xpos[i])) / 2) - 200:int((int(hpos[i]) + int(xpos[i])) / 2) + 100, :]
        #     cv2.imwrite(os.path.join(file_path1, (name[i] + ' ' + name1[i] + 'right' + '.jpg')), im1)
        # else:
        #     im1 = img[(int((int(wpos[k]) + int(ypos[k])) / 2 - 200)):(int((int(wpos[k]) + int(ypos[k])) / 2 + 100)),
        #           int((int(hpos[k]) + int(xpos[k])) / 2) - 200:int((int(hpos[k]) + int(xpos[k])) / 2) + 100, :]
        #     cv2.imwrite(os.path.join(file_path1, (name[i] + ' ' + name1[i] + 'right' + '.jpg')), im1)


        #     im1[0:480, 0:480, :] = img[(int((int(wpos[i]) + int(ypos[i])) / 2 - 240)):(
        #         int((int(wpos[i]) + int(ypos[i])) / 2 + 240)), int((int(hpos[i]) + int(xpos[i])) / 2) - 240:int(
        #         (int(hpos[i]) + int(xpos[i])) / 2) + 240, :]
        #     cv2.imwrite(os.path.join(file_path1, (name[i] + ' ' + name1[i] + 'left' + '.jpg')), im1)

        # img1 = img[
        #        (int((int(wpos[i + k]) + int(ypos[i + k])) / 2 )-180): (int((int(wpos[i + k]) + int(ypos[i + k])) / 2)),
        #        int((int(hpos[i + k]) + int(xpos[i + k])) / 2 ): int((int(hpos[i + k]) + int(xpos[i + k])) / 2)+180]
        # img1 = img[(int((int(wpos[i+k])+int(ypos[i+k]))/2-220)) : (int((int(wpos[i+k])+ int(ypos[i+k]))/2)+80),int((int(hpos[i+k])+int(xpos[i+k]))/2-220) : int((int(hpos[i+k])+int(xpos[i+k]))/2)+80]
        # img1 = cv2.rectangle(img, (int((int(hpos[i+k])+int(xpos[i+k]))/2),int((int(wpos[i+k])+int(ypos[i+k]))/2-150)), (int((int(hpos[i+k])+int(xpos[i+k]))/2-150), int((int(wpos[i+k])+int(ypos[i+k]))/2)), (0, 255, 0), 10)
        # img = np.zeros([1942, 1920])
        # img1 = cv2.rectangle(img, (int(hpos[i]),int(wpos[i])), (int(xpos[i]),int(ypos[i])),  (255, 255, 255), 10)
        # img = cv2.line(img, (
        # int((int(hpos[i + k]) + int(xpos[i + k])) / 2) - 25, int(int(wpos[i + k] + int(ypos[i + k])) / 2)),
        #                (int((int(hpos[i + k]) + int(xpos[i + k])) / 2) + 25,
        #                 int(int(wpos[i + k] + int(ypos[i + k])) / 2)),
        #                (0, 255, 0), 5)
        # img1 = cv2.line(img, (
        # int((int(hpos[i + k]) + int(xpos[i + k])) / 2), int(int(wpos[i + k] + int(ypos[i + k])) / 2) - 25),
        #                 (int((int(hpos[i + k]) + int(xpos[i + k])) / 2),
        #                  int(int(wpos[i + k] + +int(ypos[i + k])) / 2) + 25),
        #                 (0, 255, 0), 5)
        # file_path1 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/郝华颖/disk/disc_vgg0/post_process/pred12'
        # cv.imwrite(os.path.join(file_path1, (name[i] + ' ' + name1[i] + 'right' + '.jpg')), img1)

# if  score[i]>0.1 :
   #     j = 0
   #     img_path = file_path+name[i]+''+name1[i]+'.jpg'
   #     img = cv2.imread(img_path)
   #     #img = np.zeros([1942,1920])
   #     #img1=cv2.rectangle(img, (int(hpos[i]),int(wpos[i])), (int(xpos[i]),int(ypos[i])),  (255,255,255), 10 )
   #     img = cv2.line(img, (int((int(hpos[i])+int(xpos[i]))/2)-25,int(int(wpos[i]+int(ypos[i]))/2)),(int((int(hpos[i])+int(xpos[i]))/2)+25,int(int(wpos[i]+int(ypos[i]))/2)), (255, 0, 0), 5)
   #     img1 = cv2.line(img, (int((int(hpos[i])+int(xpos[i]))/2),int(int(wpos[i]+int(ypos[i]))/2)-25),(int((int(hpos[i])+int(xpos[i]))/2),int(int(wpos[i]+int(ypos[i]))/2)+25), (255, 0, 0), 5)
   #     file_path1 = 'disc_vgg0/post_process/pred1'
   #     cv2.imwrite(os.path.join(file_path1, (name[i] + '.jpg')), img1)
   # else:


#     str1 = str(int(3072*wpos[i])-384) + " " + str(int(2048*hpos[i])-384)
#     str2 = str(int(3072*wpos[i])+384) + " " + str(int(2048*hpos[i])+384)
#     fw.write(name[i] + ".jpg " + "eye_dc " + str1 + " " + str2 + "\n")
