from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from measure import SegmentationMetric
from dataset import test_dataset
import segmentation_models_pytorch as smp
from skimage.io import imread, imsave
import numpy as np
from dataset import train_dataset, test_dataset, test1_dataset
from tqdm import tqdm, trange
from measure import SegmentationMetric


net =smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=2,                      # model output channels (number of classes in your dataset)
            )
net.load_state_dict(torch.load('G:/FSCS_new/model/dataⅡ/netG_best.pth'))
# net.load_state_dict(torch.load('G:/FSCS_new/netG_best.pth'))
# val_img_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2012/splited_images/test/image/'
# val_label_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2012/splited_images/test/label/'

val_img_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_test_change1-1-new/'
val_label_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/label_test/'
# val_img_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/image_change/'
# val_label_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/label/'

# val_img_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/test/image/'
# val_img_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016change/hischange/'
# val_label_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/test/label/'

val_datatset_ = test1_dataset(val_img_path, val_label_path, 512, 512, 0, 3)

metric = SegmentationMetric(2)
# i = 0
with torch.no_grad():
    net.eval()
    val_iter = val_datatset_.data_iter()
    for initial_image, semantic_image in tqdm(val_iter, desc='val'):
        initial_image = initial_image
        semantic_image = semantic_image

        semantic_image_pred = net(initial_image).detach()

        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)

        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
        # print(semantic_image_pred.shape)
        # print('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016change/hischange_pred/'+val_datatset_.lab_list[i].split('\\')[-1])
        # imsave('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016change/hischange_pred/'+val_datatset_.lab_list[i].split('\\')[-1].split('.')[0]+'.png',
        #        semantic_image_pred.numpy()*255)
        # i+=1

        metric.addBatch(semantic_image_pred.float(), semantic_image.float())

mIoU = metric.meanIntersectionOverUnion()
# mIoU = metric.pixelAccuracy()
print("mIou: ", mIoU)


# im = imread('G:/FSCS_new/img/0_5_2016.tif')
# # im = imread('G:/FSCS_new/test/image/0_1887.tif')
# im = torch.from_numpy(im).type(torch.FloatTensor)
# im = im.unsqueeze(0).permute(0, 3, 1, 2) / 255.0
# # print(im.size())
# # im = im.cuda()
# res = net(im).detach()
# res = F.softmax(res.squeeze(), dim=0)
# res = res.argmax(dim=0)
# result = torch.squeeze(res.cpu(), 0)
# result = result.numpy()
# imsave('G:/FSCS_new/result_05_2016_2.png', result)
# print(result)
# print(im.size())
# # print(result.size())