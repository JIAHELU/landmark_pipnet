import numpy as np
import cv2
import sys
import os
from PIL import Image
sys.path.append('..')
from torchvision import transforms

from torch.utils import data
from torch.utils.data import DataLoader

class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        # import pdb
        # pdb.set_trace()
        self.img_path = None
        self.line = None
        self.path = None
        self.img = None
        self.landmark = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        self.imgs_root = os.path.dirname(file_list)
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        print("Sample num:", len(self.lines))
        
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split(' ')
        self.img_path = os.path.join(self.imgs_root, self.line[0])
        self.img = Image.open(os.path.join(self.imgs_root, self.line[0])).convert('RGB')
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        # print(self.landmark)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32) * np.pi / 180.
        self.type_flag = np.asarray(self.line[-1], dtype=np.int32)
        # print("Type flag:", self.type_flag)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img_path, self.img, self.landmark, self.attribute, self.euler_angle, self.type_flag)

    def __len__(self):
        return len(self.lines)

if __name__ == '__main__':
    file_list = '/data/cv/jiahe.lu/nniefacelib/PFPLD-Dataset/test_data/list.txt'
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(file_list, dataset_transform)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    for img, landmark_gt, attribute_gt, euler_angle_gt,type_flag in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark_gt.size())
        print("attrbute size", attribute_gt.size())
        print("euler_angle", euler_angle_gt[0])
