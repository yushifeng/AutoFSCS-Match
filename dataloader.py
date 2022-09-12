import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from skimage.io import imsave, imread

class MyDataset2(Dataset):
    def __init__(self, csv_path, num_primary_color, mode = None):
        self.path = csv_path
        if mode == "train":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[:-50]
            self.palette_list = np.array(pd.read_csv("palette_%d_%s"%(num_primary_color, csv_path), header=None))[:-50]
        if mode == "val":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-50:]
            self.palette_list = np.array(pd.read_csv("palette_%d_%s" % (num_primary_color, csv_path), header=None))[-50:]
        if mode == "test":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-10:]
            self.palette_list = np.array(pd.read_csv("palette_%d_%s" % (num_primary_color, csv_path), header=None))[-10:]
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        im = imread(self.img_paths[index])
        im = im.transpose((2,0,1))
        target_img = im/255
        primary_color = self.palette_list[index] / 255
        # print(primary_color.shape)
        # primary_color_layers = self.make_primary_color_layer(self.palette_list[index], target_img)
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color.astype(np.float32))

        return target_img, primary_color_layers

    def __len__(self):
        return len(self.img_paths)

    def make_primary_color_layer(self, palette_values):
        primary_color = palette_values / 255
        return primary_color
