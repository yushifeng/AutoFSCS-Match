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

run_name = 'sample'
csv_path = 'sample.csv'
img_name = '2016_0'
num_primary_color = 7
path = 'F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_newnew_fscs/'

# path_plan2_generator = 'results/' + run_name + '/new_model2/plan2_generator.pth'
# path_mask_generator = 'results/' + run_name + '/new_model2/mask_generator.pth'
# path_residue_predictor = 'results/' + run_name + '/new_model2/residue_predictor.pth'

path_plan2_generator = 'G:/FSCS_new/results/debug-plan2-9-16/plan2_generator_last.pth'
path_mask_generator = 'G:/FSCS_new/results/debug-plan2-9-16/mask_generator_last.pth'
path_residue_predictor = 'G:/FSCS_new/results/debug-plan2-9-16/residue_predictor_last.pth'

# try:
#     os.makedirs('results/%s/%s' % (run_name, img_name))
# except OSError:
#     pass

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

img_list = glob.glob('F:/Satellite dataset Ⅱ (East Asia)/Satellite dataset Ⅱ (East Asia)/1. The cropped image data and raster labels/train/image_new/*.tif')
for i in range(len(img_list)):
    test_dataset.imgs_path[i] = img_list[i]
# test_dataset.imgs_path[0] = "G:/FSCS_new/img/0_8_2012.tif"
# print(len(img_list))

print('Start!')
# img_number = 0
mean_estimation_time = 0
with torch.no_grad():
    for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
        try:
            os.makedirs(path+img_list[batch_idx].split('\\')[-1].split('.')[0])
        except OSError:
            pass
        # if batch_idx != img_number:
        #     print('Skip ', batch_idx)
        #     continue
        print('img #', batch_idx)
        target_img = cut_edge(target_img)
        target_img = target_img.to(device)  # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        a, b, primary_color_layers = plan2_generator(target_img, training = False)  # size(1, 3, 7, 1, 1)
        primary_color_layers1 = primary_color_layers.repeat(primary_color_layers.size(0), 1, 1, target_img.size(2),
                                                            target_img.size(3))  # size(1, 3, 7, 256, 256)
        start_time = time.time()
        primary_color_pack = primary_color_layers1.view(primary_color_layers1.size(0), -1,
                                                        primary_color_layers1.size(3),
                                                        primary_color_layers1.size(4))  # size(1, 21, 256, 256)
        primary_color_pack = cut_edge(primary_color_pack)  # cut edge
        primary_color_layers = primary_color_pack.view(primary_color_pack.size(0), -1, 3, primary_color_pack.size(2),
                                                       primary_color_pack.size(
                                                           3))  # size(1, 7, 3, 256, 256) cut edge 后再还原size
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2),
                                                        target_img.size(3))
        ## Alpha Layer Proccessing
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)
        processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img)  # Option
        processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option

        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
        residue_pack = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)

        reconst_imggg = (pred_unmixed_rgb_layers * processed_alpha_layers)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        pred_unmixed_rgb_layers2016 = pred_unmixed_rgb_layers.clone()
        processed_alpha_layers2016 = processed_alpha_layers.clone()

        end_time = time.time()
        estimation_time = end_time - start_time
        print(estimation_time)
        mean_estimation_time += estimation_time
        # print(path+img_list[batch_idx].split('\\')[-1].split('.')[0]+'/pred-alpha-00_layer-%02d.png' % i)
        for i in range(len(pred_alpha_layers[0])):
            save_image(pred_alpha_layers[0, i, :, :, :],
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0]+'/pred-alpha-00_layer-%02d.png' % i)
        for i in range(len(processed_alpha_layers[0])):
            save_image(processed_alpha_layers[0, i, :, :, :],
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0]+'/proc-alpha-00_layer-%02d.png' % i)
        for i in range(len(pred_unmixed_rgb_layers[0])):
            save_image(pred_unmixed_rgb_layers[0, i, :, :, :],
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0]+'/rgb-00_layer-%02d.png' % i)
        if True:
            # batchsizeは１で計算されているはず．それぞれ保存する．
            save_layer_number = 0
            save_image(primary_color_layers[save_layer_number, :, :, :, :],
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0] + '/_img-%02d_primary_color_layers.png' % batch_idx)
            save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0] + '/_img-%02d_reconst_img.png' % batch_idx)
            save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                       path+img_list[batch_idx].split('\\')[-1].split('.')[0] + '/_img-%02d_target_img.png' % batch_idx)

            # RGBAの４chのpngとして保存する
            RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2)  # out: bn, ln, 4, h, w
            # test ではバッチサイズが１なので，bn部分をなくす
            RGBA_layers = RGBA_layers[0]  # ln, 4. h, w
            # ln ごとに結果を保存する
            for i in range(len(RGBA_layers)):
                save_image(RGBA_layers[i, :, :, :],
                           path+img_list[batch_idx].split('\\')[-1].split('.')[0]+'/img-%02d_layer-%02d.png' % (batch_idx, i))
            print('Saved!')

        if batch_idx == 510:
            break  # debug用

# 処理まえのアルファを保存
# for i in range(len(pred_alpha_layers[0])):
#     save_image(pred_alpha_layers[0, i, :, :, :], 'results/%s/%s/pred-alpha-00_layer-%02d.png' % (run_name, img_name, i))
#
# # 処理後のアルファの保存 processed_alpha_layers
# for i in range(len(processed_alpha_layers[0])):
#     save_image(processed_alpha_layers[0, i, :, :, :],
#                'results/%s/%s/proc-alpha-00_layer-%02d.png' % (run_name, img_name, i))
#
# for i in range(len(pred_unmixed_rgb_layers[0])):
#     save_image(pred_unmixed_rgb_layers[0, i, :, :, :], 'results/%s/%s/rgb-00_layer-%02d.png' % (run_name, img_name, i))
