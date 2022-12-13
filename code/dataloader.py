import os

import PIL.Image
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch
import random
from PIL import Image, ImageOps
from utils.transform import Relabel, ToLabel, Colorize

x_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    # transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

NUM_CLASSES = 8

class MyCoTransform(object):
    def __init__(self, augment=False, height=256, width=256):
        self.augment = augment
        self.height = height
        self.width = width
        pass

    def __call__(self, input, target):
        input = Resize([self.height, self.width], Image.BILINEAR)(input)
        target = Resize([self.height, self.width], Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0),
                                     fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        target = ToLabel()(target)
        # print('relabeling 255 as: ', NUM_CLASSES-1)
        target = Relabel(255, NUM_CLASSES - 1)(target)

        return input, target

class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.name = os.listdir(os.path.join(path))
        self.transform = transform

    def __len__(self):
        print("Load data:", len(self.name))
        return len(self.name)      # 数据集的数量

    def __getitem__(self, index):
        segment_name = self.name[index]  #xx.npys
        # print(segment_name)
        data = os.path.join(self.path, segment_name)

        image_label = np.load(data)


        image = image_label[:3, :, :].transpose(1, 2, 0)
        image = image.astype(np.float64)
        image = 255 * image
        image = image.astype(np.uint8)
        image = PIL.Image.fromarray(image)

        label = image_label[-1, :, :]
        label = PIL.Image.fromarray(label)

        if self.transform is not None:
            image, label = self.transform(image, label)

        # image_label = transforms.functional.pil_to_tensor(image_label)
        # image_label = torch.tensor(image_label, dtype=torch.float32)
        # image = image_label[:3, :, :]
        # label = image_label[-1, :, :]
        # plt.imshow(label)
        # plt.show()

        return image, label, segment_name[0:-4]

if __name__ == '__main__':
    data_path = '/Users/dongtianchi/Documents/DL_final/code/data'
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    for i, (image, segment_image) in enumerate(data_loader):
        print(image.shape, segment_image.shape)

    # data = MyDataset(data_path)
    # print(data[0][0].shape) # image.shape = [3, 256, 256]
    # print(data[0][1].shape) # label.shape = [3, 256, 256]
    # plt.subplots(2, 3)
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(data[i][0].permute(1, 2, 0))
    # # print(type(data[0][0]))
    # plt.show()
    #
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(data[i][1].permute(1, 2, 0))
    # plt.show()
    
