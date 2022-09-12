from __future__ import print_function
from tools import cut_edge, alpha_normalize, proc_guidedfilter
import torch.utils.data
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from color_cnn import ColorCNN
import os
import time
from mydataset import MyDataset
import glob
import numpy as np
from skimage.exposure import match_histograms

run_name = 'sample'
csv_path = 'sample.csv'
img_name = ['2012_00' , '2016_00']
num_primary_color = 7
path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_test_change1-1-new/'
# path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/image_change/'

# path_plan2_generator = 'G:/FSCS_new/results/debug-plan2-9-16/plan2_generator_last.pth'
# path_mask_generator = 'G:/FSCS_new/results/debug-plan2-9-16/mask_generator_last.pth'
# path_residue_predictor = 'G:/FSCS_new/results/debug-plan2-9-16/residue_predictor_last.pth'
path_plan2_generator = 'results/' + run_name + '/plan2_generator_last.pth'
path_mask_generator = 'results/' + run_name + '/mask_generator_last.pth'
path_residue_predictor = 'results/' + run_name + '/residue_predictor_last.pth'

try:
    os.makedirs('results/%s/%s' % (run_name, img_name[0]))
    os.makedirs('results/%s/%s' % (run_name, img_name[1]))
except OSError:
    pass

# img_2012_list = glob.glob('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2012/splited_images/train/image/*.tif')
# img_2016_list = glob.glob('F:/Building change detection dataset_add/Building change detection dataset_add/1. The two-period image data/2016/splited_images/train/image/*.tif')
# img_list = ["a" for i in range(2520)]
# for i in range(0,2520,2):
#     img_list[i] = img_2012_list[int(i/2)]
# for i in range(1, 2520,2):
#     img_list[i] = img_2016_list[int((i-1)/2)]

test_dataset = MyDataset(csv_path, num_primary_color, mode='test')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    )

device = 'cuda'
# define model
plan2_generator = ColorCNN('unet', 7 ).to(device)
mask_generator = MaskGenerator(num_primary_color).to(device)
residue_predictor = ResiduePredictor(num_primary_color).to(device)
# load params
plan2_generator.load_state_dict(torch.load(path_plan2_generator))
mask_generator.load_state_dict(torch.load(path_mask_generator))
residue_predictor.load_state_dict(torch.load(path_residue_predictor))
# eval mode
plan2_generator.eval()
mask_generator.eval()
residue_predictor.eval()

source_list = glob.glob('F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_test/*.tif')
# source_list = glob.glob('F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/test/image/*.tif')
ref_list = glob.glob('F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_new/*.tif')
img_list = ["a" for i in range(674)]
for i in range(0,674,2):
    img_list[i] = source_list[int(i/2)]
for i in range(1, 674,2):
    img_list[i] = ref_list[int((i-1)/2)]

for i in range(674):
    test_dataset.imgs_path[i] = img_list[i]
# test_dataset.imgs_path[1] = "G:/FSCS_new/img/0_4_2012.tif"
# test_dataset.imgs_path[0] = "G:/FSCS_new/img/0_4_2016.tif"


print('Start!')
img_number = 0
mean_estimation_time = 0
with torch.no_grad():
    for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
        # print(primary_color_layers.size())
        print('img #', batch_idx)
        target_img = cut_edge(target_img)
        target_img = target_img.to(device)
        primary_color_layers = primary_color_layers.to(device)
        a, b, primary_color_layers = plan2_generator(target_img, training = False)
        # print('primary_color_layers: ', primary_color_layers.shape)
        primary_color_layers1 = primary_color_layers.repeat(primary_color_layers.size(0), 1, 1, target_img.size(2),target_img.size(3))
        # print('primary_color_layers1: ', primary_color_layers1.shape)
        start_time = time.time()
        primary_color_pack = primary_color_layers1.view(primary_color_layers1.size(0), -1,primary_color_layers1.size(3),primary_color_layers1.size(4))
        # print('primary_color_pack: ', primary_color_pack.size())
        primary_color_pack = cut_edge(primary_color_pack)
        primary_color_layers = primary_color_pack.view(primary_color_pack.size(0), -1, 3, primary_color_pack.size(2),primary_color_pack.size(3))
        # print('primary_color_layers2: ', primary_color_layers.shape)
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2),target_img.size(3))
        ## Alpha Layer Proccessing
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)
        processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img)
        processed_alpha_layers = alpha_normalize(processed_alpha_layers)
        if (batch_idx%2) == 0:
            alpha_save = processed_alpha_layers.clone()
            # print("alpha_save: ", alpha_save.size())
        # for i in range(len(processed_alpha_layers[0])):
        #     save_image(processed_alpha_layers[0, i, :, :, :],'results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name[batch_idx], i))

        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2)
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
        residue_pack = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
        if (batch_idx%2) == 0:
            rgb_save_source = pred_unmixed_rgb_layers.clone()
            rgb_save_source = rgb_save_source.cpu().numpy().transpose(0, 1, 3, 4, 2)

        if (batch_idx%2) == 1:
            rgb_layer = np.ones_like(rgb_save_source)
            rgb_save_ref = pred_unmixed_rgb_layers.clone()
            rgb_save_ref = rgb_save_ref.cpu().numpy().transpose(0, 1, 3, 4, 2)
            for i in range(rgb_save_source.shape[1]):
                # a = match_histograms(rgb_save_2012[0,i,:,:,:], rgb_save_2016[0,i,:,:,:], multichannel=True)
                # print(a.shape)
                rgb_layer[0,i,:,:,:] = match_histograms(rgb_save_source[0,i,:,:,:], rgb_save_ref[0,i,:,:,:], multichannel=True)
            # print("rgb_layer: ", rgb_layer.shape)
            rgb_layer = torch.from_numpy(rgb_layer.transpose(0,1,4,2,3))
            # print("rgb_layer1: ", rgb_layer.size())
            change = rgb_layer * alpha_save.cpu()
            # print(("change_size: ", change.size()))
            change_image = change.sum(dim = 1)
            # print(("change_image_size: ", change_image.size()))
            # save_image(change_image[0, :, :, :],
            #            'G:/FSCS_new/img/0_4_2016change.png')
            save_image(change_image[:, :, :],
                       path + img_list[batch_idx-1].split('\\')[-1].split('.')[0] + '.png')
        # for i in range(len(pred_unmixed_rgb_layers[0])):
        #     save_image(pred_unmixed_rgb_layers[0, i, :, :, :],'results/%s/%s/rgb-00_layer-%02d.png' % (run_name, img_name[batch_idx], i))

        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        end_time = time.time()
        estimation_time = end_time - start_time
        print(estimation_time)
        mean_estimation_time += estimation_time

        # if True:
        #     save_layer_number = 0
        #     save_image(primary_color_layers[save_layer_number, :, :, :, :],'results/%s/%s/test' % (run_name, img_name[batch_idx]) + '_img-%02d_primary_color_layers.png' % batch_idx)
        #     save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),'results/%s/%s/test' % (run_name, img_name[batch_idx]) + '_img-%02d_reconst_img.png' % batch_idx)
        #     save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),'results/%s/%s/test' % (run_name, img_name[batch_idx]) + '_img-%02d_target_img.png' % batch_idx)

        #     RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2)
        #     RGBA_layers = RGBA_layers[0]
        #     for i in range(len(RGBA_layers)):
        #         save_image(RGBA_layers[i, :, :, :],'results/%s/%s/img-%02d_layer-%02d.png' % (run_name, img_name[0], batch_idx, i))
        #     print('Saved to results/%s/%s/...' % (run_name, img_name[0]))

        if batch_idx == 674:
            break  # debug用


