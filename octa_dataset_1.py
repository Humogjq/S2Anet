# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
import cv2
import random
from PIL import Image
import numpy as np
import torch

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from PIL import Image, ImageEnhance
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.exposure import adjust_log, equalize_adapthist

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

eraser = get_random_eraser()

def elastic_transform(image, label, alpha, sigma, random_state=None, u = 0.5):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if np.random.random() < u:
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        transformed_image = np.zeros(image.shape)
        transformed_label = np.zeros(image.shape)
        print(image.shape, label.shape)
        for i in range(image.shape[-1]):
            transformed_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape)
            if label is not None:
                transformed_label[:, :, i] = map_coordinates(label[:, :, i], indices, order=1).reshape(shape)
            else:
                transformed_label = None
        transformed_image = transformed_image.astype(np.uint8)
        if label is not None:
            transformed_label = transformed_label.astype(np.uint8)
        return transformed_image, transformed_label
    else:
        return image, label

def adjust_light(image, u = 0.5):
    if np.random.random() < u:
        gamma = random.random() * 3 + 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image.astype(np.uint8), table).astype(np.uint8)
    return image


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def ad_blur(image):
    seed = random.random()
    if seed > 0.5:
        image = image / 127.5 - 1
        image[:,:,0] = median(image[:,:,0], disk(3))
        image[:,:,1] = median(image[:,:,1], disk(3))
        image[:,:,2] = median(image[:,:,2], disk(3))
        image = (image + 1) * 127.5
    image = image.astype(np.uint8)
    return image

def adjust_light(image):
    seed = random.random()
    if seed > 0.5:
        gamma = random.random() * 3 + 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image.astype(np.uint8), table).astype(np.uint8)
    return image


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(512, 512)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    
    return image, label

def adjust_contrast(image):
    seed = random.random()
    if seed > 0.5:
        image = adjust_log(image, 1)
    return image

class DrishtiGS(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(512, 512)):
        super(DrishtiGS, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
        print("gt_unique", np.unique(gt))

        gt = cv2.resize(gt, self.scale_size)
        print("gt_unique_resize", np.unique(gt))

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, gt = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
            img = adjust_contrast(img)
            img = ad_blur(img)
            img = adjust_light(img)
            img = get_random_eraser(img)



        gt_ = gt.copy()
        gt = np.zeros((512, 512, 3), dtype=np.uint8)
        # 视盘
        gt[gt_ < 200] = [255, 0, 0]
        # 视杯
        gt[gt_ < 50] = [255, 255, 0]

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.array(gt, np.float32).transpose(2, 0, 1) / 255.0

        img = torch.Tensor(img)
        gt = torch.Tensor(gt)



        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/disc_small/image")
            gt_dir = os.path.join(root + "/train/disc_small/mask")
        else:
            img_dir = os.path.join(root + "/test/disc_small/image")
            gt_dir = os.path.join(root + "/test/disc_small/mask")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name


class ORIGA(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(ORIGA, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)

        gt = cv2.resize(gt, self.scale_size)

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, mask = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.array(gt, np.float32)
        gt_ = np.zeros((512, 512, 3), dtype=np.uint8)
        gt_[gt < 200] = [255, 0, 0]
        gt_[gt < 50] = [255, 255, 0]
        gt = np.array(gt_, np.float32).transpose(2, 0, 1) / 255.0

        img = torch.Tensor(img)
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/disc_small/train/images")
            gt_dir = os.path.join(root + "/disc_small/train/labels")
        else:
            img_dir = os.path.join(root + "/disc_small/test/images")
            gt_dir = os.path.join(root + "/disc_small/test/labels")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name

class REFUGE(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(REFUGE, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)

        gt = cv2.resize(gt, self.scale_size)

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, gt = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
            img = eraser(img)

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.array(gt, np.float32)
        gt_ = np.zeros((512, 512, 3), dtype=np.uint8)
        gt_[gt < 200] = [255, 0, 0]
        gt_[gt < 50] = [255, 255, 0]
        gt = np.array(gt_, np.float32).transpose(2, 0, 1) / 255.0


        img = torch.Tensor(img)
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/images")
            gt_dir = os.path.join(root + "/train/labels")
        else:
            img_dir = os.path.join(root + "/test/images")
            gt_dir = os.path.join(root + "/test/images")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name

class REFUGE1(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(REFUGE1, self).__init__()
        self.img_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)

        img = img[:,450:2526,:]

        img = img[:,:,::-1]  # make image blue
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        img = torch.Tensor(img)


        return img

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = root

        else:
            img_dir = root

        img_lst = sorted(list(map(lambda x: os.path. join(img_dir, x), os.listdir(img_dir))))

        return img_lst

    def getFileName(self):
        return self.name


class ORIGA2(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(ORIGA2, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)

        gt = cv2.resize(gt, self.scale_size)

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, mask = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
        gt = np.expand_dims(gt, axis=2)

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.array(gt, np.float32).transpose(2, 0, 1) / 255.0
        gt[gt > 0.5] = 1
        gt[gt < 0.5] = 0

        img = torch.Tensor(img)
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/images")
            gt_dir = os.path.join(root + "/train/labels")
        else:
            img_dir = os.path.join(root + "/test/images")
            gt_dir = os.path.join(root + "/test/labels")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name

class Mix(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(Mix, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)

        gt = cv2.resize(gt, self.scale_size)

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, gt = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            # img, gt = elastic_transform(img, gt, img.shape[1] * 2, img.shape[1] * 0.08)
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
            img = eraser(img)

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.array(gt, np.float32)
        gt_ = np.zeros((512, 512, 3), dtype=np.uint8)
        gt_[gt < 200] = [255, 0, 0]
        gt_[gt < 50] = [255, 255, 0]
        gt = np.array(gt_, np.float32).transpose(2, 0, 1) / 255.0


        img = torch.Tensor(img)
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/images")
            gt_dir = os.path.join(root + "/train/labels")
        else:
            img_dir = os.path.join(root + "/test/images")
            gt_dir = os.path.join(root + "/test/labels")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name

def RandomEnhance(image):
    value = random.uniform(-2, 2)
    random_seed = random.randint(1, 4)
    if random_seed == 1:
        img_enhanceed = ImageEnhance.Brightness(image)
    elif random_seed == 2:
        img_enhanceed = ImageEnhance.Color(image)
    elif random_seed == 3:
        img_enhanceed = ImageEnhance.Contrast(image)
    else:
        img_enhanceed = ImageEnhance.Sharpness(image)
    image = img_enhanceed.enhance(value)
    return image

def RescalSize(image, label, re_size=512):
    w, h = image.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    image = image.crop(box)
    label = label.crop(box)
    image = image.resize((re_size, re_size))
    label = label.resize((re_size, re_size))
    return image, label

class IOSTAR(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(512, 512)):
        super(IOSTAR, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        image = Image.open(imgPath)
        label = Image.open(gtPath)

        if self.isTraining:
            # augumentation
            angel = random.randint(-40, 40)
            image = image.rotate(angel)
            label = label.rotate(angel)

            if random.random() > 0.5:
                image = RandomEnhance(image)

            image, label = RescalSize(image, label, re_size=512)

            # flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image, label = RescalSize(image, label, re_size=512)
        image = simple_transform(image)
        label = np.array(label)
        label = label.reshape([512,512])
        label = Image.fromarray(np.uint8(label))
        label = simple_transform(label)
        label = label * 255

        return image, label

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/training/images")
            gt_dir = os.path.join(root + "/training/labels")
        else:
            img_dir = os.path.join(root + "/test/images")
            gt_dir = os.path.join(root + "/test/labels")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name


class RIMONE(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(448, 448)):
        super(RIMONE, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)

        gt = cv2.resize(gt, self.scale_size)

        if self.isTraining:
            # augumentation
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

            img, gt = randomShiftScaleRotate(img, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
            img = eraser(img)

        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        gt = np.expand_dims(gt, axis=0)
        gt = np.array(gt, np.float32) / 255.0
        gt[gt >= 0.5] = 1
        gt[gt <= 0.5] = 0

        img = torch.Tensor(img)
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/images")
            gt_dir = os.path.join(root + "/train/RIM-ONE-exp3")
        else:
            img_dir = os.path.join(root + "/valid/images")
            gt_dir = os.path.join(root + "/test/GT")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name

class RIMONE1(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(640, 640)):
        super(RIMONE1, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()

        img = cv2.imread(imgPath)
        img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, self.scale_size)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        img = torch.Tensor(img)

        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, self.scale_size)
        gt = np.array(gt, np.float32) / 255.0
        gt = torch.Tensor(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = "/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/images"
            gt_dir = "/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/GT"

        else:
            img_dir = "/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/images"
            gt_dir = "/home/imed/data_disk/data/qh/disc_cup_seg/data/RIM-ONE/train/GT"

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        return img_lst, gt_lst

    def getFileName(self):
        return self.name



