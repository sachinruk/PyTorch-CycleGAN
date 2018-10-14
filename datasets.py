import glob
import random
import os

from torch.utils.data import Dataset
# from PIL import Image
import cv2
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        imgA = cv2.resize(cv2.imread(self.files_A[index % len(self.files_A)]), (0,0), fx=1.12, fy=1.12)
        item_A = self.transform(imgA)

        if self.unaligned:
            imgB = cv2.resize(cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)]), (0,0), fx=1.12, fy=1.12)
        else:
            imgB = cv2.resize(cv2.imread(self.files_B[index % len(self.files_B)]), (0,0), fx=1.12, fy=1.12)
        item_B = self.transform(imgB)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))