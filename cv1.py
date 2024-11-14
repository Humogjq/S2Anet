from PIL import Image
from scipy import misc
import numpy as np
import os
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import pprint
import shutil
file_path = '/media/imed/data/qh/disc_cup_seg/data/ORIGA_REFUGE_mix/test/labels'
file_list = os.listdir(file_path)
file_list.sort()
i=0
for item in file_list:
  if item.endswith('.jpg'):
    Xs = []
    Ys = []
    image_path = os.path.join(os.path.abspath(file_path), item)
    image = misc.imresize(misc.imread(image_path), [800, 800, 1])
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
      for col in range(width):
        if image[row, col] == 254:
          Xs.append(col)
          Ys.append(row)

    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    i = i + 1
    xml_file = open("/media/imed/data/qh/test/" + item[:-4] + '.xml', 'wb')
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(i).zfill(6) + '.jpg'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '800'
    node_height = SubElement(node_size, 'height')
    node_height.text = '800'
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    node_seg = SubElement(node_root, 'segmented')
    node_seg.text = '0'
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'disc'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(int(x1))
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(int(y1))
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(int(x2))
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(int(y2))
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    print(xml)
    xml_file.write(xml)
    xml_file.close()




