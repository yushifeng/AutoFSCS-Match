import numpy as np
import torch
import torch.utils.data as data
import os
import glob
import imageio


def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg'])

class train_dataset(data.Dataset):
    def __init__(self, img_path='', label_path = '', size_w=512, size_h=512, flip=0, time_series=4, batch_size=1):
        super(train_dataset, self).__init__()
        self.src_list = sorted(glob.glob(img_path + '*.tif'))
        # self.src_list2 = sorted(glob.glob('G:/FSCS_new/results/sample/2012_00/*.png'))
        # print(self.src_list2[:10])
        # self.src_list3 = sorted(glob.glob('alldata/*.tif'))
        self.lab_list = sorted(glob.glob(label_path + '*.tif'))
        # self.lab_list2 = sorted(glob.glob('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2012/splited_images/train/label/*.tif'))
        # print(self.lab_list2[:10])
        # self.lab_list3 = sorted(glob.glob('alldata/*.png'))
        # self.src_list.extend(self.src_list2)
        # self.src_list.extend(self.src_list3)
        # self.lab_list.extend(self.lab_list2)
        # self.lab_list.extend(self.lab_list3)
        self.src_list = np.array(self.src_list)
        self.lab_list = np.array(self.lab_list)
        self.img_path = img_path
        self.label_path = label_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []
        try:
            for i in range(index):
                image = imageio.imread(x_batch[i])
                if image.shape[0] == 4:
                    image = np.transpose(image, [1,2,0])
                data_series.append(image / 255.0)
                # label_series.append(imageio.imread(y_batch[i]) - 1)
                im = imageio.imread(y_batch[i])
                im[im == 255] = 1
                label_series.append(im)
                self.index += 1
                # if label_series[i].max() + 1 > 16 or data_series[i].max() > 255:
                #     print(y_batch[i])
        except OSError:
            return None, None

        # print(np.array(data_series)[0].shape)
        # if np.array(data_series)[0].shape[2] == 4:
        #     data_series = np.array(data_series)[0].transpose(2, 0, 1)
        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        # data_series = data_series.permute(0, 2, 3, 1)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):
                image = imageio.imread(self.src_list[i])
                if image.shape[0] == 4:
                    image = np.transpose(image, [1,2,0])
                data_series.append(image / 255.0)
                # label_series.append(imageio.imread(self.lab_list[i]) - 1)
                label_series.append(imageio.imread(self.lab_list[i]))
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

class test1_dataset(data.Dataset):
    def __init__(self, img_path='', label_path = '', size_w=512, size_h=512, flip=0, time_series=4, batch_size=1):
        # super(train_dataset, self).__init__()
        self.src_list = sorted(glob.glob(img_path + '*.png'))
        # self.src_list2 = sorted(glob.glob(img_path + '*.tif'))
        # self.src_list2 = sorted(glob.glob('G:/FSCS_new/results/sample/2012_00/*.png'))
        # print(self.src_list2[:10])
        # self.src_list3 = sorted(glob.glob('alldata/*.tif'))
        self.lab_list = sorted(glob.glob(label_path + '*.tif'))
        # self.lab_list2 = sorted(glob.glob(label_path + '*.tif'))
        # self.lab_list2 = sorted(glob.glob('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2012/splited_images/train/label/*.tif'))
        # print(self.lab_list2[:10])
        # self.lab_list3 = sorted(glob.glob('alldata/*.png'))
        # self.src_list.extend(self.src_list2)
        # self.src_list.extend(self.src_list3)
        # self.lab_list.extend(self.lab_list2)
        # self.lab_list.extend(self.lab_list3)
        self.src_list = np.array(self.src_list)
        self.lab_list = np.array(self.lab_list)
        self.img_path = img_path
        self.label_path = label_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []
        try:
            for i in range(index):
                image = imageio.imread(x_batch[i])
                if image.shape[0] == 4:
                    image = np.transpose(image, [1,2,0])
                data_series.append(image / 255.0)
                # label_series.append(imageio.imread(y_batch[i]) - 1)
                im = imageio.imread(y_batch[i])
                im[im == 255] = 1
                label_series.append(im)
                self.index += 1
                # if label_series[i].max() + 1 > 16 or data_series[i].max() > 255:
                #     print(y_batch[i])
        except OSError:
            return None, None

        # print(np.array(data_series)[0].shape)
        # if np.array(data_series)[0].shape[2] == 4:
        #     data_series = np.array(data_series)[0].transpose(2, 0, 1)
        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        # data_series = data_series.permute(0, 2, 3, 1)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):
                image = imageio.imread(self.src_list[i])
                if image.shape[0] == 4:
                    image = np.transpose(image, [1,2,0])
                data_series.append(image / 255.0)
                # label_series.append(imageio.imread(self.lab_list[i]) - 1)
                im = imageio.imread(self.lab_list[i])
                im[im == 255] = 1
                label_series.append(im)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

class test_dataset(data.Dataset):
    def __init__(self, img_path='', label_path = '', size_w=512, size_h=512, flip=0, time_series=4, batch_size=1):
        # super(train_dataset, self).__init__()
        self.src_list = np.array(sorted(glob.glob(img_path + '*.tif')))
        self.lab_list = np.array(sorted(glob.glob(label_path + '*.tif')))
        # self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []
        try:
            for i in range(index):
                data_series.append(imageio.imread(x_batch[i]) / 255.0)
                label_series.append(imageio.imread(y_batch[i]) - 1)
                self.index += 1
                # if label_series[i].max() + 1 > 16 or data_series[i].max() > 255:
                #     print(y_batch[i])
        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        # print(data_series.shape)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):
                data_series.append(imageio.imread(self.src_list[i]) / 255.0)
                label_series.append(imageio.imread(self.lab_list[i]) - 1)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter
