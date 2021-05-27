import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import random
import collections
from torch.optim import lr_scheduler
import pandas as pd
import torch.nn.functional as F
import pickle
import argparse
from U_net_denoising_autoencoder import UNet


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


def training(num_epochs, my_autoencoder, optimizer, criterion, train_loader, validation_loader, test_loader, save_root,
            noise_factor):
    print('Training............')
    val_loss_list = np.array([])

    for epoch in range(num_epochs):
        epoch_loss_train = 0.0

        my_autoencoder.train()
        train_len = 0
        for train_x_batch, train_y in train_loader:
            train_x = Variable(train_x_batch).cuda()
            train_x_noise = Variable(train_x_batch + noise_factor*torch.randn(train_x_batch.shape)).cuda()

            optimizer.zero_grad()

            train_output, train_lv = my_autoencoder(train_x_noise)
            train_epoch_loss = criterion(train_output, train_x)

            train_epoch_loss.backward()

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

            optimizer.step()

            train_len += len(train_x_batch)


        train_loss = epoch_loss_train / train_len

        with torch.no_grad():

            epoch_loss_val = 0.0
            epoch_loss_test = 0.0

            my_autoencoder.eval()
            val_len = 0
            for validation_x_batch, validation_y_batch in validation_loader:
                val_x = Variable(validation_x_batch).cuda()
                val_x_noise = Variable(validation_x_batch + noise_factor*torch.randn(validation_x_batch.shape)).cuda()

                val_output, val_lv = my_autoencoder(val_x_noise)
                val_epoch_loss = criterion(val_output, val_x)

                epoch_loss_val += (val_epoch_loss.data.item() * len(validation_x_batch))
                val_len += len(validation_x_batch)

            val_loss = epoch_loss_val / val_len

            my_autoencoder.eval()
            test_len = 0
            for test_x_batch, test_y_batch in test_loader:

                test_x = Variable(test_x_batch).cuda()
                test_x_noise = Variable(test_x_batch + noise_factor*torch.randn(test_x_batch.shape)).cuda()

                test_output, test_lv = my_autoencoder(test_x_noise)
                test_epoch_loss = criterion(test_output, test_x)

                epoch_loss_test += (test_epoch_loss.data.item() * len(test_x_batch))

                test_len += len(test_x_batch)

            test_loss = epoch_loss_test / test_len

        val_loss_list = np.append(val_loss_list, val_loss)

        if val_loss_list[epoch] == val_loss_list.min():
            print('model_saving ----- epoch : {}, validation_loss : {:.6f}'.format(epoch, val_loss))
            torch.save(my_autoencoder.state_dict(), save_root + '/U-net_checkpoint.pt')

        if (epoch + 1) == 1 :
            print('Epoch [{}/{}], Train loss : {:.4f}, val loss : {:.4f}, test loss : {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss, test_loss))

        if (epoch + 1) % 10 == 0 :
            print('Epoch [{}/{}], Train loss : {:.4f}, val loss : {:.4f}, test loss : {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss, test_loss))


def test_result(my_autoencoder, train_loader, validation_loader, test_loader, criterion, save_root, noise_factor):
    fname = os.path.join(save_root, 'U-net_checkpoint.pt')
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

        train_len = 0
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

            train_len += len(train_x_batch)

        train_loss = epoch_loss_train / train_len

        my_autoencoder.eval()
        val_len = 0
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

            epoch_loss_val += (val_epoch_loss.data.item() * len(validation_x_batch))
            val_len += len(validation_x_batch)

        val_loss = epoch_loss_val / val_len

        test_len = 0
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

            test_len += len(test_x_batch)

        test_loss = epoch_loss_test / test_len

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
        "--train_image_root",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train_stage_root",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val_image_root",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val_stage_root",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test_image_root1",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test_stage_root1",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test_image_root2",
        default= None,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--test_stage_root2",
        default= None,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--independent_val",
        default='Yes',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_root",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--noise_factor",
        default=0.1,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--num_epochs",
        default=3,
        type=int,
        required=False,
    )



    args = parser.parse_args()


    if args.independent_val == 'Yes':
        train_x_re = np.load(args.train_image_root)
        train_y_re = np.load(args.train_stage_root)
        validation_x = np.load(args.val_image_root)
        validation_y = np.load(args.val_stage_root)

        if args.test_image_root2 is not None:
            test_x_before1 = np.load(args.test_image_root1)
            test_y_before1 = np.load(args.test_stage_root1)
            test_x_before2 = np.load(args.test_image_root2)
            test_y_before2 = np.load(args.test_image_root2)
            test_x_before = np.append(test_x_before1, test_x_before2, axis = 0)
            test_y_before = np.append(test_y_before1, test_y_before2, axis = 0)
        else:
            test_x_before = np.load(args.test_image_root1)
            test_y_before = np.load(args.test_stage_root1)

    else:

        train_x_before1 = np.load(args.train_image_root)
        train_y_before1 = np.load(args.train_stage_root)
        train_x_before2 = np.load(args.val_image_root)
        train_y_before2 = np.load(args.val_stage_root)

        if args.test_image_root2 is not None:
            test_x_before1 = np.load(args.test_image_root1)
            test_y_before1 = np.load(args.test_stage_root1)
            test_x_before2 = np.load(args.test_image_root2)
            test_y_before2 = np.load(args.test_stage_root2)
            test_x_before = np.append(test_x_before1, test_x_before2, axis = 0)
            test_y_before = np.append(test_y_before1, test_y_before2, axis = 0)
        else:
            test_x_before = np.load(args.test_image_root1)
            test_y_before = np.load(args.test_stage_root1)

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
    train_loader = DataLoader(training_set, batch_size = args.batch_size, shuffle=True)
    batch_len_train = len(train_loader)

    validation_set = Dataset(val_x_tr, val_y_tr)
    validation_loader = DataLoader(validation_set, batch_size = args.batch_size, shuffle=True)
    batch_len_val = len(validation_loader)

    test_set = Dataset(test_x_tr, test_y_tr)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)
    batch_len_test = len(test_loader)

    my_autoencoder = UNet(3, 3);
    my_autoencoder.cuda();

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(my_autoencoder.parameters(),lr = args.learning_rate, momentum = 0.9)

    training(args.num_epochs, my_autoencoder, optimizer, criterion, train_loader, validation_loader, test_loader, save_root = args.save_root,
            noise_factor = args.noise_factor)
    test_result(my_autoencoder, train_loader, validation_loader, test_loader, criterion, save_root = args.save_root,
            noise_factor = args.noise_factor)


if __name__ == "__main__":
    main()
