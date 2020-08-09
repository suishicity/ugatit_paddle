import cv2
import os
import glob
import numpy as np


class ImageFolder:
    def __init__(self, root, transform=None):
        self.files = glob.glob(os.path.join(root, '*'))
        self.files.sort()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.load_image(self.files[index])
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def load_image(self, img_path):
        img = cv2.imwrite(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def reader(self):
        for i in len(self.files):
            yield self.__getitem__(i)



