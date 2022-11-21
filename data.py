import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
import matplotlib.pyplot as plt

transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)      # 数据集的数量

    def __getitem__(self, index):
        segment_name=self.name[index]  #xx.png
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)
        # print(segment_path)
        image_path=os.path.join(self.path,'JPEGImages',segment_name)
        segment_image=keep_image_size_open(segment_path)
        image=keep_image_size_open(image_path)
        return transform(image),transform(segment_image)


if __name__ == '__main__':
    data=MyDataset('./data')
    # print(data[0][0].shape) # image.shape = [3, 256, 256]
    # print(data[0][1].shape) # label.shape = [3, 256, 256]
    plt.subplots(2, 3)
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(data[i][0].permute(1, 2, 0))
    # print(type(data[0][0]))
    plt.show()
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(data[i][1].permute(1, 2, 0))
    plt.show()
    
