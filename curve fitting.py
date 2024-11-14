import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
import os

img_root = "/home/imed/data_disk/data/qh/line_train"
csv_test = "/home/imed/data_disk/data/qh/train_Result"

for img_name in os.listdir(img_root):
    csv_name = img_name.split('.')[0] + '.bmp-layer-1-lesion.csv'
    # print(img_name)
    csv_path = os.path.join(csv_test,csv_name)
    csv_path = '{}'.format(csv_path)
    pts = np.loadtxt(open(csv_path, "rb"), delimiter=",", skiprows=0)
    # define pts from the question

    tck, u = splprep(pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # plt.plot(pts[:, 0], pts[:, 1], 'ro')
    # plt.plot(x_new, y_new, 'b--')
    # plt.show()

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(x_new, y_new))
















