import glob
import random
import os

import torch
from torch.utils.data import Dataset
# from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

class RandomCrop(object):

    def __call__(self, im):
        h, w, c = im.shape
        scale = np.random.uniform() / 10. + 1.
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        im = cv2.resize(im, (0,0), fx = scale, fy = scale)
        return im[offy : (offy + h), offx : (offx + w)]

class RandomFlip(object):
    
    def __call__(self, image):
        flip = np.random.binomial(1, .5)
        if flip:
            image = cv2.flip(image, 1)
        return image

class Rescale(object):
    
    def __init__(self, output):
        self.new_h, self.new_w = output
        
    def __call__(self, image):
        h, w, c = image.shape
        new_h = int(self.new_h)
        new_w = int(self.new_w)
        image = cv2.resize(image, (new_w, new_h))
        return image

class Normalize(object):
    
    def __call__(self, image):
        image = np.array(image, np.float64)
        return (image / 255.0 - 0.5) / 0.5


class ToTensor(object):

    def __call__(self, image):


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        imgA = cv2.imread(self.files_A[index % len(self.files_A)])
        item_A = self.transform(imgA)

        if self.unaligned:
            imgB = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            imgB = cv2.imread(self.files_B[index % len(self.files_B)])
        item_B = self.transform(imgB)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))