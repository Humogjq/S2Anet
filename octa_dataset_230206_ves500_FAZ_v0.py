# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
import cv2
import random
from PIL import Image
import numpy as np

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(512, 512)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)

    return image, label




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
        transformed_label = np.zeros(shape)
        #print(image.shape, label.shape)
        for i in range(image.shape[-1]):
            transformed_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape)

        transformed_label[:, :] = map_coordinates(label[:, :], indices, order=1).reshape(shape)

        transformed_image = transformed_image.astype(np.uint8)
        if label is not None:
            transformed_label = transformed_label.astype(np.uint8)
        return transformed_image, transformed_label
    else:
        return image, label

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






class ROSE(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True):
        super(ROSE, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
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
        # deepPath = self.deep_lst[index]
        # superficialPath = self.superficial_lst[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
        # deep = Image.open(deepPath).convert("L")
        # superficial = Image.open(superficialPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)

        # deep = np.array(deep)
        # deep[deep >= 128] = 255
        # deep[deep < 128] = 0
        # deep = Image.fromarray(deep)
        #
        # superficial = np.array(superficial)
        # superficial[superficial >= 128] = 255
        # superficial[superficial < 128] = 0
        # superficial = Image.fromarray(superficial)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
            # deep = deep.rotate(angel)
            # superficial = superficial.rotate(angel)

        img = simple_transform(img)
        gt = simple_transform(gt)
        # deep = simple_transform(deep)
        # superficial = simple_transform(superficial)

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
            img_dir = os.path.join(root + "/train/img")
            gt_dir = os.path.join(root + "/train/gt")
            deep_dir = os.path.join(root + "/train/thin_gt")
            superficial_dir = os.path.join(root + "/train/thick_gt")
        else:
            img_dir = os.path.join(root + "/test/img")
            gt_dir = os.path.join(root + "/test/gt")
            deep_dir = os.path.join(root + "/test/thin_gt")
            superficial_dir = os.path.join(root + "/test/thick_gt")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        # deep_lst = sorted(list(map(lambda x: os.path.join(deep_dir, x), os.listdir(deep_dir))))
        # superficial_lst = sorted(list(map(lambda x: os.path.join(superficial_dir, x), os.listdir(superficial_dir))))

        return img_lst, gt_lst

    def getFileName(self):
        return self.name


class FAZtest(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True):
        super(FAZtest, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
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
        # deepPath = self.deep_lst[index]
        # superficialPath = self.superficial_lst[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
        # deep = Image.open(deepPath).convert("L")
        # superficial = Image.open(superficialPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)

        # deep = np.array(deep)
        # deep[deep >= 128] = 255
        # deep[deep < 128] = 0
        # deep = Image.fromarray(deep)
        #
        # superficial = np.array(superficial)
        # superficial[superficial >= 128] = 255
        # superficial[superficial < 128] = 0
        # superficial = Image.fromarray(superficial)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
            # deep = deep.rotate(angel)
            # superficial = superficial.rotate(angel)

        img = simple_transform(img)
        gt = simple_transform(gt)
        # deep = simple_transform(deep)
        # superficial = simple_transform(superficial)

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
            img_dir = os.path.join(root + "/train/img/SCC")
            gt_dir = os.path.join(root + "/train/gt/FAZ")
            #deep_dir = os.path.join(root + "/train/thin_gt")
            #superficial_dir = os.path.join(root + "/train/thick_gt")
        else:
            img_dir = os.path.join(root + "/test/img/SCC")
            gt_dir = os.path.join(root + "/test/gt/FAZ")
            #deep_dir = os.path.join(root + "/test/thin_gt")
            #superficial_dir = os.path.join(root + "/test/thick_gt")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        # deep_lst = sorted(list(map(lambda x: os.path.join(deep_dir, x), os.listdir(deep_dir))))
        # superficial_lst = sorted(list(map(lambda x: os.path.join(superficial_dir, x), os.listdir(superficial_dir))))

        return img_lst, gt_lst

    def getFileName(self):
        return self.name



class FAZ500(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True):   #CEnet需要第一层卷积需要输入3通道
        super(FAZ500, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
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
        # deepPath = self.deep_lst[index]
        # superficialPath = self.superficial_lst[index]

        simple_transform = transforms.ToTensor()

        #img = Image.open(imgPath)
        #gt = Image.open(gtPath).convert("L")
        img = cv2.imread(imgPath)
        #img = img[:,:,::-1]
        # print("img:{}".format(np.shape(img)))
        img = cv2.resize(img, (304,304))                            #scale_size(304,304)
        gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (304,304))
        # deep = Image.open(deepPath).convert("L")
        # superficial = Image.open(superficialPath).convert("L")

        #if self.channel == 1:
        #    img = img.convert("L")
        #else:
        #    img = img.convert("RGB") #do it

        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        #gt = Image.fromarray(gt)

        # deep = np.array(deep)
        # deep[deep >= 128] = 255
        # deep[deep < 128] = 0
        # deep = Image.fromarray(deep)
        #
        # superficial = np.array(superficial)
        # superficial[superficial >= 128] = 255
        # superficial[superficial < 128] = 0
        # superficial = Image.fromarray(superficial)

        if self.isTraining:
            # augumentation
            #rotate = 10
            #angel = random.randint(-rotate, rotate)
            #img = img.rotate(angel)
            #gt = gt.rotate(angel)

            img, gt = randomShiftScaleRotate(img, gt,
                                             shift_limit=(-0.1, 0.1),
                                             scale_limit=(-0.1, 0.1),
                                             aspect_limit=(-0.1, 0.1),
                                             rotate_limit=(-20, 20))
            img, gt = randomHorizontalFlip(img, gt)
            img, gt = randomVerticleFlip(img, gt)
            img, gt = randomRotate90(img, gt)
            img, gt = elastic_transform(img, gt, img.shape[1] * 2, img.shape[1] * 0.08)
        gt = np.expand_dims(gt, axis=2)

        img = np.array(img, np.float32).transpose( 0, 1,2) / 255.0
        gt = np.array(gt, np.float32).transpose( 0, 1,2)  / 255.0
        # gt[gt > 0.5] = 1
        # gt[gt < 0.5] = 0
        #img = torch.Tensor(img)
        #gt = torch.Tensor(gt)
        img = simple_transform(img)
        gt = simple_transform(gt)
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
            img_dir = os.path.join(root + "/train/img")
            gt_dir = os.path.join(root + "/train/gt_FAZ")
            #deep_dir = os.path.join(root + "/train/thin_gt")
            #superficial_dir = os.path.join(root + "/train/thick_gt")
        else:
            img_dir = os.path.join(root + "/test/img")
            gt_dir = os.path.join(root + "/test/gt_FAZ")
            #deep_dir = os.path.join(root + "/test/thin_gt")
            #superficial_dir = os.path.join(root + "/test/thick_gt")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        # deep_lst = sorted(list(map(lambda x: os.path.join(deep_dir, x), os.listdir(deep_dir))))
        # superficial_lst = sorted(list(map(lambda x: os.path.join(superficial_dir, x), os.listdir(superficial_dir))))

        return img_lst, gt_lst

    def getFileName(self):
        return self.name





class CRIA(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(CRIA, self).__init__()
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

        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)

        img = simple_transform(img)
        gt = simple_transform(gt)

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
            img_dir = os.path.join(root + "/train/original")
            gt_dir = os.path.join(root + "/train/gt")
        else:
            img_dir = os.path.join(root + "/test/original")
            gt_dir = os.path.join(root + "/test/gt")

        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))

        assert len(img_lst) == len(gt_lst)

        return img_lst, gt_lst

    def getFileName(self):
        return self.name