from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from measure import SegmentationMetric
from dataset import train_dataset, test_dataset, test1_dataset
# from early_stopping import EarlyStopping
from tqdm import tqdm, trange
import random
import segmentation_models_pytorch as smp
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

batch_size = 8
niter = 200
class_num = 2
learning_rate = 0.00001
beta1 = 0.5
cuda = True
num_workers = 4
size_h = 512
size_w = 512
flip = 0
band = 3
# net = DeepLabV3(band, class_num)
net =smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=2,                      # model output channels (number of classes in your dataset)
            )

# train_img_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/train/image/'
# train_label_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/train/label/'
train_img_path = 'train/image/'
train_label_path = 'train/label/'
val_img_path = 'test/image/'
val_label_path = 'test/label/'
# val_img_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/test/image/'
# val_label_path = 'F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/test/label/'

# train_img_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image/'
# train_label_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/label/'
# val_img_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_val/'
# val_label_path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/label_val/'
out_file = 'Unet_res34_model'
num_GPU = 1
index = 200
# torch.cuda.set_device(0)

try:
    import os
    os.makedirs(out_file)
except OSError:
    pass

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_datatset_ = train_dataset(train_img_path, train_label_path, size_w, size_h, flip, band, batch_size)
val_datatset_ = test1_dataset(val_img_path, val_label_path, size_w, size_h, 0, band)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    os.makedirs(out_file)
    os.makedirs(out_file + '/')
except OSError:
    pass
if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)


###########   LOSS & OPTIMIZER   ##########
# criterion = nn.CrossEntropyLoss(ignore_index=255)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
metric = SegmentationMetric(class_num)
# early_stopping = EarlyStopping(patience=100, verbose=True)

if __name__ == '__main__':
    start = time.time()
    net.train()
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    star = 0
    for epoch in range(1, niter + 1):
        for iter_num in trange(2000 // index, desc='train, epoch:%s' % epoch):
            train_iter = train_datatset_.data_iter_index(index=index)
            for initial_image, semantic_image in train_iter:
                # print(initial_image.shape)
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image)

                loss = criterion(semantic_image_pred, semantic_image.long())
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        lr_adjust.step()

        # early_stopping(1 - mIoU, net, '%s/' % out_file + 'netG.pth')
        torch.save(net.state_dict(), 'netG_dataset2.pth')

        with torch.no_grad():
            net.eval()
            val_iter = val_datatset_.data_iter()

            for initial_image, semantic_image in tqdm(val_iter, desc='val'):
                # print(initial_image.shape)
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image).detach()
                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)

                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                metric.addBatch(semantic_image_pred.float(), semantic_image.float())

        mIoU = metric.meanIntersectionOverUnion()
        # mIoU = metric.meanPixelAccuracy()
        # mIoU = metric.classRecallAccuracy()
        # mIoU = metric.F1Score()
        # mIoU = metric.kappa()
        print('mIoU: ', mIoU)
        if mIoU > star:
            star = mIoU
            print("best performace: ", mIoU)
            print("save model")
            torch.save(net.state_dict(), 'netG_best.pth')
        metric.reset()
        net.train()

        # early_stopping(1 - mIoU, net, '%s/' % out_file + 'netG.pth')
        #
        # if early_stopping.early_stop:
        #     break

    # end = time.time()
    # print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    #
    # test_datatset_ = train_dataset(test_path, time_series=band)
    # start = time.time()
    # test_iter = test_datatset_.data_iter()
    # if os.path.exists('%s/' % out_file + 'netG.pth'):
    #     net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
    #
    # net.eval()
    # for initial_image, semantic_image in tqdm(test_iter, desc='test'):
    #     # print(initial_image.shape)
    #     initial_image = initial_image.cuda()
    #     semantic_image = semantic_image.cuda()
    #
    #     # semantic_image_pred = model(initial_image)
    #     semantic_image_pred = net(initial_image).detach()
    #     semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
    #     semantic_image_pred = semantic_image_pred.argmax(dim=0)
    #
    #     semantic_image = torch.squeeze(semantic_image.cpu(), 0)
    #     semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
    #
    #     metric.addBatch(semantic_image_pred.float(), semantic_image.float())
    #     image = semantic_image_pred
    #
    # end = time.time()
    # print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    # mIoU = metric.meanIntersectionOverUnion()
    # print('mIoU: ', mIoU)
    #
    #     # if early_stopping.early_stop:
    #     #     break

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

