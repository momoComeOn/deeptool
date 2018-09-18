import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import torch

IMSIZE = 224

class ImageDataset(data.Dataset):
    def __init__(self, prefix, root, flag=1, transform=None):
        self._root = root
        self._flag = flag
        self._prefix = prefix
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


        self.im_transform = transforms.Compose([
                transforms.Resize((IMSIZE, IMSIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean, std=self.std
                )
            ])

    def _list_images(self, root):
        self.synsets = []
        self.items = []
        with open(root,'r') as f:
            lines = f.readlines()
        self.items = [(os.path.join(self._prefix,line.strip().split(' ')[0]),float(line.strip().split(' ')[1])) for line in lines]

    def __getitem__(self, idx):
        label = self.items[idx][1]
        img = Image.open(self.items[idx][0])
        if img.mode == 'L':
            img = img.convert('RBG')
        return self.im_transform(img), label

    def __len__(self):
        return len(self.items)

def data_get(batch_size):
    root_prefix = '/home/muyouhang/zkk/gluon/data/AFEW-AV/face/'
    txt = '/home/muyouhang/zkk/gluon/AFEW_AV/train_path.txt'
    datas = ImageDataset(root_prefix,txt)
    dataiter = data.DataLoader(datas,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=False)
    return dataiter
