import torch
from torch.utils.data.dataset import Dataset
import cv2
import pandas as pd
import numpy as np
from skimage.io import imread, imsave

class MyDataset(Dataset):
    def __init__(self, csv_path, num_primary_color, mode=None):
        self.csv_path = csv_path
        if mode == 'train':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[:-50] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-50:]
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)
        if mode == 'test':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)

        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        # img = cv2.imread(self.imgs_path[index])
        img = cv2.imdecode(np.fromfile(self.imgs_path[index], dtype = np.uint8), -1)
        # img = imread(self.imgs_path[index])
        # print(self.imgs_path[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1))
        # print(self.imgs_path.shape)
        target_img = img/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)

    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        return primary_color_layers

# imgs_path = np.array(pd.read_csv("sample.csv", header=None)).reshape(-1)
# print(imgs_path)
# palette_list = np.array(pd.read_csv('palette_%d_%s' % (10, "sample.csv"), header=None)).reshape(-1, 10*3)
# print(palette_list[0][0])
# # img = cv2.imread("J:/FSCS-master/dataset/test/apple.jpg"[0])
# img = cv2.imdecode(np.fromfile("J:/FSCS-master/dataset/test/apple.jpg", dtype = np.uint8), -1)
# print(img.shape)
