import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import random

class MotionBlur(object):
    def __init__(self, p=1, degree=11, angle=10):
        self.p = p
        self.degree = degree
        self.angle = angle

    def __call__(self, img):
        if random.random() > self.p:
            return img

        angle = np.random.randint(-self.angle, self.angle+1)
        degree = np.random.randint(10, self.degree)
        
        img = np.array(img)
        
        # 确保图像是 2D 或 3D RGB 图像
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree

        blurred = cv2.filter2D(img, -1, motion_blur_kernel)

        # 确保数据类型为 uint8，范围在 [0, 255]
        blurred = np.clip(blurred, 0, 255).astype('uint8')

        img = Image.fromarray(blurred)
        return img
    
if __name__ == '__main__':
    img = Image.open('/data/cv/jiahe.lu/nniefacelib/label_dataset/train/new_label/11frame_81id-0.jpg')
    train_transform = transforms.Compose([
        MotionBlur(),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.ColorJitter(0.6, 0.3, 0.2, 0.1),
        # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.5, 0.5, 0.5)),
    ])
    img = train_transform(img)
    img.save('/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/test_img/1frame.jpg')