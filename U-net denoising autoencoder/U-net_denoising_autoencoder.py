import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import random
import collections
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import torch.nn.functional as F
import cv2
from scipy import ndimage
import pickle
import argparse



class Dataset(Dataset):
    def __init__(self, images, label):
        self.labels = label
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Down_batch_mid(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(Down_new, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        output = self.maxpool(x)
        output = self.conv1(output)
        output = self.batchnorm1(output)
        output = self.relu(output)
        latent_variable = self.conv2(output)
        output = self.batchnorm2(latent_variable)
        output = self.relu(output)
        return output, latent_variable

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down_batch_mid(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5, lv = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, lv

def training(num_epochs, my_autoencoder, optimizer, criterion, train_loader, validation_loader, test_loader, save_root = args.save_root,
            noise_factor = args.noise_factor):

    for epoch in range(num_epochs):
        epoch_loss_train = 0.0

        my_autoencoder.train()
        for train_x_batch, train_y in train_loader:
            train_x = Variable(train_x_batch).cuda()
            train_x_noise = Variable(train_x_batch + noise_factor*torch.randn(train_x_batch.shape)).cuda()

            optimizer.zero_grad()

            train_output, train_lv = my_autoencoder(train_x_noise)
            train_epoch_loss = criterion(train_output, train_x)

            train_epoch_loss.backward()

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

            optimizer.step()


        train_loss = epoch_loss_train / len(train_x_tr)

        with torch.no_grad():

            epoch_loss_val = 0.0
            epoch_loss_test = 0.0

            my_autoencoder.eval()
            for validation_x_batch, validation_y_batch in validation_loader:
                val_x = Variable(validation_x_batch).cuda()
                val_x_noise = Variable(validation_x_batch + noise_factor*torch.randn(validation_x_batch.shape)).cuda()

                val_output, val_lv = my_autoencoder(val_x_noise)
                val_epoch_loss = criterion(val_output, val_x)

                epoch_loss_val += (val_epoch_loss.data.item() * len(validation_x_batch))


            val_loss = epoch_loss_val / len(val_x_tr)

            my_autoencoder.eval()
            for test_x_batch, test_y_batch in test_loader:

                test_x = Variable(test_x_batch).cuda()
                test_x_noise = Variable(test_x_batch + noise_factor*torch.randn(test_x_batch.shape)).cuda()

                test_output, test_lv = my_autoencoder(test_x_noise)
                test_epoch_loss = criterion(test_output, test_x)

                epoch_loss_test += (test_epoch_loss.data.item() * len(test_x_batch))

            test_loss = epoch_loss_test / len(test_x_tr)

        val_loss_list = np.append(val_loss_list, val_loss)

        if val_loss_list[epoch] == val_loss_list.min():
            print('model_saving ----- epoch : {}, validation_loss : {:.6f}'.format(epoch, val_loss))
            torch.save(my_autoencoder.state_dict(), save_root + '/U-net_checkpoint.pt')

        if (epoch + 1) == 1 :
            print('Epoch [{}/{}], Train loss : {:.4f}, val loss : {:.4f}, test loss : {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss, test_loss))

        if (epoch + 1) % 10 == 0 :
            print('Epoch [{}/{}], Train loss : {:.4f}, val loss : {:.4f}, test loss : {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss, test_loss))


def test_result(my_autoencoder, train_loader, validation_loader, test_loader, save_root = args.save_root):
    fname = save_root + '/U-net_checkpoint.pth'
    checkpoint = torch.load(fname)
    my_autoencoder.load_state_dict(checkpoint)

    with torch.no_grad():
        epoch_loss_train = 0.0
        train_input = np.array([]).reshape(0, 3, 128, 128)
        train_noise = np.array([]).reshape(0, 3, 128, 128)
        train_result = np.array([]).reshape(0, 3, 128, 128)
        train_latent_variable = np.array([]).reshape(0, 512, 8, 8)
        train_label = np.array([])

        epoch_loss_val = 0.0
        val_input = np.array([]).reshape(0, 3, 128, 128)
        val_noise = np.array([]).reshape(0, 3, 128, 128)
        val_result = np.array([]).reshape(0, 3, 128, 128)
        val_latent_variable = np.array([]).reshape(0, 512, 8, 8)
        val_label = np.array([])

        epoch_loss_test = 0.0
        test_input = np.array([]).reshape(0, 3, 128, 128)
        test_noise = np.array([]).reshape(0, 3, 128, 128)
        test_result = np.array([]).reshape(0, 3, 128, 128)
        test_latent_variable = np.array([]).reshape(0, 512, 8, 8)
        test_label = np.array([])

        for train_x_batch, train_y in train_loader:
            train_x = Variable(train_x_batch).cuda()
            train_x_noise = Variable(train_x_batch + noise_factor*torch.randn(train_x_batch.shape)).cuda()

            train_output, train_lv = my_autoencoder(train_x_noise)
            train_lv = train_lv.detach().cpu().numpy()
            train_latent_variable = np.append(train_latent_variable, train_lv, axis = 0)
            train_label = np.append(train_label, train_y)
            train_epoch_loss = criterion(train_output, train_x)

            train_input = np.append(train_input, train_x.data.cpu().numpy(), axis = 0)
            train_noise = np.append(train_noise, train_x_noise.data.cpu().numpy(), axis = 0)
            train_result = np.append(train_result, train_output.data.cpu().numpy(), axis = 0)

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

        train_loss = epoch_loss_train / len(train_x_tr)

        my_autoencoder.eval()
        for validation_x_batch, validation_y_batch in validation_loader:
            val_x = Variable(validation_x_batch).cuda()
            val_x_noise = Variable(validation_x_batch + noise_factor*torch.randn(validation_x_batch.shape)).cuda()

            val_output, val_lv = my_autoencoder(val_x_noise)
            val_lv = val_lv.detach().cpu().numpy()
            val_latent_variable = np.append(val_latent_variable, val_lv, axis = 0)
            val_label = np.append(val_label, validation_y_batch)
            val_epoch_loss = criterion(val_output, val_x)

            val_input = np.append(val_input, val_x.data.cpu().numpy(), axis = 0)
            val_noise = np.append(val_noise, val_x_noise.data.cpu().detach(), axis = 0)
            val_result = np.append(val_result, val_output.data.cpu().numpy(), axis = 0)


        val_loss = val_epoch_loss.data.item()

        for test_x_batch, test_y_batch in test_loader:

            test_x = Variable(test_x_batch).cuda()
            test_x_noise = Variable(test_x_batch + noise_factor*torch.randn(test_x_batch.shape)).cuda()

            test_output, test_lv = my_autoencoder(test_x_noise)
            test_lv = test_lv.detach().cpu().numpy()
            test_latent_variable = np.append(test_latent_variable, test_lv, axis = 0)
            test_label = np.append(test_label, test_y_batch)
            test_epoch_loss = criterion(test_output, test_x)

            epoch_loss_test += (test_epoch_loss.data.item() * len(test_x_batch))

            test_input = np.append(test_input, test_x.data.cpu().numpy(), axis = 0)
            test_noise = np.append(test_noise, test_x_noise.data.cpu().numpy(), axis = 0)
            test_result = np.append(test_result, test_output.data.cpu().numpy(), axis = 0)

        test_loss = epoch_loss_test / len(test_x_tr)

    print('Train loss : {:.7f}, val loss : {:.7f}, test loss : {:.7f}'.format(train_loss, val_loss, test_loss))

    np.save(save_root + '/training_latent_variable_noBatchNorm.npy', train_latent_variable)
    np.save(save_root + '/training_latent_variable_stage_noBatchNorm.npy', train_label)
    np.save(save_root + '/validation_latent_variable_noBatchNorm.npy', val_latent_variable)
    np.save(save_root + '/validation_latent_variable_stage_noBatchNorm.npy', val_label)
    np.save(save_root + '/test_latent_variable_noBatchNorm.npy', test_latent_variable)
    np.save(save_root + '/test_latent_variable_stage_noBatchNorm.npy', test_label)

    np.save(save_root + '/training_input.npy', train_input)
    np.save(save_root + '/training_stage.npy', train_label)
    np.save(save_root + '/validation_input.npy', val_input)
    np.save(save_root + '/validation_stage.npy', val_label)
    np.save(save_root + '/test_input.npy', test_input)
    np.save(save_root + '/test_stage.npy', test_label)

    np.save(save_root + '/training_noise.npy', train_noise)
    np.save(save_root + '/validation_noise.npy', val_noise)
    np.save(save_root + '/test_noise.npy', test_noise)

    np.save(save_root + '/training_result.npy', train_result)
    np.save(save_root + '/validation_input.npy', val_result)
    np.save(save_root + '/test_result.npy', test_result)

def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_root1",
        default='/data_root1/',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root2",
        default='/data_root2/',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root3",
        default='/data_root3/',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root4",
        default='/data_root4/',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_root",
        default='/save_root/',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        required=True,
    )

    parser.add_argument(
        "--noise_factor",
        default=0.1,
        type=float,
        required=True,
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        required=True,
    )

    parser.add_argument(
        "--num_epochs",
        default=300,
        type=int,
        required=True,
    )


    args = parser.parse_args()

    train_x_before1 = np.load(data_root1 + '/ct_total.npy')
    train_y_before1 = np.load(data_root1 + '/stage_total.npy')
    train_x_before2 = np.load(data_root2 + '/ct_total.npy')
    train_y_before2 = np.load(data_root2 + '/stage_total.npy')
    test_x_before1 = np.load(data_root3 + '/ct_total.npy')
    test_y_before1 = np.load(data_root3 + '/stage_total.npy')
    test_x_before2 = np.load(data_root4 + '/ct_total.npy')
    test_y_before2 = np.load(data_root4 + '/stage_total.npy')

    train_x_before = np.append(train_x_before1, train_x_before2, axis = 0)
    train_y_before = np.append(train_y_before1, train_y_before2, axis = 0)

    test_x_before = np.append(test_x_before1, test_x_before2, axis = 0)
    test_y_before = np.append(test_y_before1, test_y_before2, axis = 0)

    train_y_before = train_y_before.astype('int')
    test_y_before = test_y_before.astype('int')

    val_idx = [3, 12, 27, 52, 67, 79, 81, 88]
    train_idx = [i for i in range(len(train_x_before)) if i not in val_idx]

    train_x_re = train_x_before[train_idx]
    train_y_re = train_y_before[train_idx]

    validation_x = train_x_before[val_idx]
    validation_y = train_y_before[val_idx]

    train_x = train_x_re.astype('float16')
    train_y = train_y_re
    val_x = validation_x.astype('float16')
    val_y = validation_y
    test_x = test_x_before.astype('float16')
    test_y = test_y_before

    train_x_tr = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y_tr = torch.from_numpy(train_y)
    val_x_tr = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y_tr = torch.from_numpy(val_y)
    test_x_tr = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y_tr = torch.from_numpy(test_y)

    training_set = Dataset(train_x_tr, train_y_tr)
    train_loader = DataLoader(training_set, batch_size = batch_size, shuffle=True)
    batch_len_train = len(train_loader)

    validation_set = Dataset(val_x_tr, val_y_tr)
    validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle=True)
    batch_len_val = len(validation_loader)

    test_set = Dataset(test_x_tr, test_y_tr)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
    batch_len_test = len(test_loader)

    my_autoencoder = UNet(3, 3);
    my_autoencoder.cuda();

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(my_autoencoder.parameters(),lr = args.learning_rate, momentum = 0.9)

    training(num_epochs, my_autoencoder, optimizer, criterion, train_loader, validation_loader, test_loader, save_root = args.save_root,
            noise_factor = args.noise_factor)
    test_result(my_autoencoder, train_loader, validation_loader, test_loader, save_root = args.save_root)


if __name__ == "__main__":
    main()
