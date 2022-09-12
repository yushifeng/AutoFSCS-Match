# from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from color_cnn import ColorCNN
from dataloader import MyDataset2
import cv2
import pandas as pd
from tools import reconst_loss, squared_mahalanobis_distance_loss, mono_color_reconst_loss

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--run_name', type=str, default='debug', help='run-name. This name is used for output folder.')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_primary_color', type=int, default=7,
                    help='num of layers')
parser.add_argument('--rec_loss_lambda', type=float, default=1.0,
                    help='reconst_loss lambda')
parser.add_argument('--m_loss_lambda', type=float, default=1.0,
                    help='m_loss_lambda')
parser.add_argument('--sparse_loss_lambda', type=float, default=0.0,
                    help='sparse_loss lambda')
parser.add_argument('--distance_loss_lambda', type=float, default=1.0,
                    help='distance_loss_lambda')
parser.add_argument('--save_layer_train', type=int, default=12,
                    help='save_layer_train')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num_workers of dataloader')
parser.add_argument('--csv_path', type=str, default='sample.csv', help='path to csv of images path')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--reconst_loss_type', type=str, default='l1', help='[mse | l1 | vgg]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

try:
    os.makedirs('results/%s' % args.run_name)
except OSError:
    pass

torch.manual_seed(args.seed)
cudnn.benchmark = True
device = torch.device("cuda:0" if args.cuda else "cpu")

train_dataset = MyDataset2(args.csv_path, args.num_primary_color, mode='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    # worker_init_fn=lambda x: np.random.seed(),
    drop_last=False,
    pin_memory=True
    )
val_dataset = MyDataset2(args.csv_path, args.num_primary_color, mode='val')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    )

def read_backimage():
    img = cv2.imread('G:/FSCS-master/dataset/backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))
    return img.view(1,3,256,256).to(device)
backimage = read_backimage()
def alpha_normalize(alpha_layers):
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

plan2_generator = ColorCNN('unet', 7, ).to(device)
mask_generator = MaskGenerator(7).to(device)
residue_predictor = ResiduePredictor(args.num_primary_color).to(device)
params = list(plan2_generator.parameters())
params += list(mask_generator.parameters())
params += list(residue_predictor.parameters())
optimizer = optim.Adam(params, lr=0.00001, betas=(0.0, 0.99))

def train(epoch, best_now, loss, rloss, mloss, dloss, regularloss):
    plan2_generator.train()
    mask_generator.train()
    residue_predictor.train()

    train_loss = 0
    best_performance = best_now

    for batch_idx, (target_img, primary_color_layers) in enumerate(train_loader):
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers
        primary_color_layers_show = torch.ones((target_img.size(0), 7, 3, 256, 256)) * primary_color_layers.view(target_img.size(0), 7, -1).unsqueeze(3).unsqueeze(4)

        optimizer.zero_grad()

        transformed_img, prob, color_results = plan2_generator(target_img, training = True)

        prob_max, _ = torch.max(prob.view([target_img.size(0), 7, -1]), dim=2)
        avg_max = torch.mean(prob_max)
        regular_loss = np.log2(7) * (1 - avg_max)

        plate_color_layers = color_results.repeat(1, 1, 1, target_img.size(2), target_img.size(3)) # repeat
        primary_color_pack = plate_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))

        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)

        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)
        mono_color_layers = torch.cat((plate_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

        residue_pack  = residue_predictor(target_img, mono_color_layers_pack)

        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((plate_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (plate_color_layers * processed_alpha_layers).sum(dim=1)

        r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda # Lr
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda # La管alpha层的
        d_loss = squared_mahalanobis_distance_loss(plate_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda # Ld：为了在每个颜色层中只收集同质的颜色
        total_loss = 7 * r_loss + 10 * m_loss + 2 * d_loss + 1 * regular_loss
        total_loss.backward()
        train_loss += total_loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if (total_loss.item() / len(target_img)) < best_performance:
                best_performance = (total_loss.item() / len(target_img))
                torch.save(plan2_generator.state_dict(), 'results/%s/new_model/plan2_generator.pth' % args.run_name)
                torch.save(mask_generator.state_dict(), 'results/%s/new_model/mask_generator.pth' % args.run_name)
                torch.save(residue_predictor.state_dict(), 'results/%s/new_model/residue_predictor.pth' % args.run_name)
            loss.append(total_loss.item() / len(target_img))
            rloss.append(r_loss.item() / len(target_img))
            mloss.append(m_loss.item() / len(target_img))
            dloss.append(d_loss.item() / len(target_img))
            regularloss.append(regular_loss.item() / len(target_img))
            print('')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item() / len(target_img)))
            print('reconst_loss *lambda: ', r_loss.item() / len(target_img))
            print('m_loss *lambda: ', m_loss.item() / len(target_img))
            print('squared_mahalanobis_distance_loss *lambda: ', d_loss.item() / len(target_img))
            print('regular_loss:', regular_loss.item() / len(target_img))
            print("best_performance: ", best_performance)

            for save_layer_number in range(args.save_layer_train):
                save_image(primary_color_layers_show[save_layer_number,:,:,:,:],
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_primary_color_layers.png' % save_layer_number)
                save_image(plate_color_layers[save_layer_number,:,:,:,:],
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_plate_color_layers.png' % save_layer_number)
                save_image(pred_unmixed_rgb_layers[save_layer_number,:,:,:,:] * processed_alpha_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_pred_unmixed_rgb_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(mono_color_reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_mono_color_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    # save model
    torch.save(plan2_generator.state_dict(), 'results/%s/new_model/plan2_generator_last.pth' % args.run_name)
    torch.save(mask_generator.state_dict(), 'results/%s/new_model/mask_generator_last.pth' % args.run_name, _use_new_zipfile_serialization=False)
    torch.save(residue_predictor.state_dict(), 'results/%s/new_model/residue_predictor_last.pth' % args.run_name, _use_new_zipfile_serialization=False)

    return best_performance, loss, rloss, mloss, dloss, regularloss

if __name__ == "__main__":
    best_now = 10
    loss = []
    rloss = []
    mloss = []
    dloss = []
    varloss = []
    stdloss = []
    regularloss = []
    for epoch in range(1, args.epochs + 1):
        print('Start training')
        best_now, loss, rloss, mloss, dloss, regularloss = train(epoch, best_now, loss, rloss, mloss, dloss, regularloss)
    log = {"loss":loss, "rloss":rloss, "mloss":mloss, "dloss":dloss, "regularloss":regularloss}
    pd_data = pd.DataFrame(log)
    pd_data.to_csv("G:/FSCS-master/log.csv", index = False, header = False)
