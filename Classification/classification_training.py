import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import math

from classification import Model

def labeling(label_bf):
    label = np.array([])
    for i in label_bf:
        if i == 1:
            new_s = 1
        else:
            new_s = 2
        label = np.append(label, new_s)
    return label

def label_onehot(label):
    onehot_label = np.zeros((len(label), 2))
    for i in range(len(label)):
        stage = int(label[i] - 1)
        onehot_label[i, stage] = 1
    return onehot_label

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

def training(num_epochs, model, class_weight, criterion, optimizer, train_loader, validation_loader, test_loader, save_root):
    result = np.array([]).reshape(0,6)
    val_loss_list = np.array([])

    for epoch in range(num_epochs):
        epoch_loss_train = 0.0
        epoch_train_acc = 0.0
        predicted_train_output = np.array([])
        train_real = np.array([])

        my_model.train()
        for train_x_batch, train_y_batch in train_loader:
            train_x = Variable(train_x_batch).cuda()
            train_y = Variable(train_y_batch).cuda()

            optimizer.zero_grad()

            train_output = my_model(train_x)
            train_epoch_loss = criterion(train_output, torch.max(train_y, 1)[1])

            train_epoch_loss.backward()
            optimizer.step()

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

            pred = np.argmax(train_output.data.cpu().numpy(), axis = 1)
            true = np.argmax(train_y.data.cpu().numpy(), axis = 1)
            predicted_train_output = np.append(predicted_train_output, pred)
            train_real = np.append(train_real, true)

        train_loss = epoch_loss_train / len(train_x_tr)
        train_acc = len(np.where(predicted_train_output == train_real)[0]) / len(predicted_train_output)

        with torch.no_grad():

            '''
            validation
            '''
            epoch_loss_val = 0.0
            epoch_acc_val = 0.0
            predicted_val_output = np.array([])
            val_real = np.array([])

            my_model.eval()

            for validation_x_batch, validation_y_batch in validation_loader:
                validation_x = Variable(validation_x_batch).cuda()
                validation_y = Variable(validation_y_batch).cuda()

                validation_output = my_model(validation_x)
                validation_epoch_loss = criterion(validation_output, torch.max(validation_y, 1)[1])

                epoch_loss_val += (validation_epoch_loss.data.item() * len(validation_x_batch))

                pred_val = np.argmax(validation_output.data.cpu().numpy(), axis = 1)
                true_val = np.argmax(validation_y.data.cpu().numpy(), axis = 1)
                correct_val = len(np.where(pred_val == true_val)[0])
                epoch_acc_val += (correct_val / len(pred_val))

                predicted_val_output = np.append(predicted_val_output, pred_val)
                val_real = np.append(val_real, true_val)


            val_loss = epoch_loss_val / len(val_x_tr)
            val_acc = len(np.where(predicted_val_output == val_real)[0]) / len(predicted_val_output)

            '''
            test
            '''
            epoch_loss_test = 0.0
            epoch_acc_test = 0.0
            predicted_test_output = np.array([])
            test_real = np.array([])

            my_model.eval()

            for test_x_batch, test_y_batch in test_loader:
                test_x = Variable(test_x_batch).cuda()
                test_y = Variable(test_y_batch).cuda()

                test_output = my_model(test_x)
                test_epoch_loss = criterion(test_output, torch.max(test_y, 1)[1])

                epoch_loss_test += (test_epoch_loss.data.item() * len(test_x_batch))

                pred_test = np.argmax(test_output.data.cpu().numpy(), axis = 1)
                true_test = np.argmax(test_y.data.cpu().numpy(), axis = 1)
                correct_test = len(np.where(pred_test == true_test)[0])
                epoch_acc_test += (correct_test / len(pred_test))

                predicted_test_output = np.append(predicted_test_output, pred_test)
                test_real = np.append(test_real, true_test)


            test_loss = epoch_loss_test / len(test_x_tr)
            test_acc = len(np.where(predicted_test_output == test_real)[0]) / len(predicted_test_output)

        result_list = [train_loss, train_acc, val_loss, val_acc, test_loss, test_acc]
        result = np.append(result, np.array(result_list).reshape(1, 6), axis = 0)

        val_loss_list = np.append(val_loss_list, val_loss)

        if val_loss_list[epoch] == val_loss_list.min():
            print('model_saving ----- epoch : {}, validation_loss : {:.6f}'.format(epoch, val_loss))
            torch.save(my_model.state_dict(), save_root + '/classification_checkpoint.pt')

        if (epoch + 1) == 1 :
            print('Epoch [{}/{}], Train loss : {:.4f}, Train acc : {:.2f}, Val loss : {:.4f}, Val acc : {:.2f}, Test loss : {:.4f}, Test acc : {:.2f}'.
                  format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))


        if (epoch + 1) % 10 == 0 :
            print('Epoch [{}/{}], Train loss : {:.4f}, Train acc : {:.2f}, Val loss : {:.4f}, Val acc : {:.2f}, Test loss : {:.4f}, Test acc : {:.2f}'.
                  format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

  return result


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
        "--learning_rate",
        default=1e-6,
        type=float,
        required=True,
    )

    parser.add_argument(
        "--num_epochs",
        default=500,
        type=int,
        required=True,
    )

    parser.add_argument(
        "-parms",
        "--list",
        help="delimited list input",
        type=str
    )

    args = parser.parse_args()

    training_latent_variable = np.load(data_root + '/training_latent_variable_noBatchNorm.npy')
    training_label_bf = np.load(data_root + '/training_latent_variable_stage_noBatchNorm.npy')
    validation_latent_variable = np.load(data_root + '/validation_latent_variable_noBatchNorm.npy')
    validation_label_bf = np.load(data_root + '/validation_latent_variable_stage_noBatchNorm.npy')
    test_latent_variable = np.load(data_root + '/test_latent_variable_noBatchNorm.npy')
    test_label_bf = np.load(data_root + '/test_latent_variable_stage_noBatchNorm.npy')

    training_label = labeling(training_label_bf)
    validation_label = labeling(validation_label_bf)
    test_label = labeling(test_label_bf)

    train_label = label_onehot(training_label)
    val_label = label_onehot(validation_label)
    te_label = label_onehot(test_label)


    train_x = training_latent_variable.astype('float16')
    train_y = train_label
    val_x = validation_latent_variable.astype('float16')
    val_y = val_label
    test_x = test_latent_variable.astype('float16')
    test_y = te_label

    train_x_tr = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y_tr = torch.from_numpy(train_y).type(torch.LongTensor)
    val_x_tr = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y_tr = torch.from_numpy(val_y).type(torch.LongTensor)
    test_x_tr = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y_tr = torch.from_numpy(test_y).type(torch.LongTensor)

    training_set = Dataset(train_x_tr, train_y_tr)
    train_loader = DataLoader(training_set, batch_size = batch_size, shuffle=True)
    batch_len_train = len(train_loader)

    validation_set = Dataset(val_x_tr, val_y_tr)
    validation_loader = DataLoader(validation_set, batch_size = batch_size, shuffle=True)
    batch_len_val = len(validation_loader)

    test_set = Dataset(test_x_tr, test_y_tr)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
    batch_len_test = len(test_loader)

    parms = args.parms

    my_model = Model(512, parms[0], parms[1], flatten_size, parms[2], parms[3], parms[4], parms[5], 2)
    my_model.cuda();

    class_weight = torch.FloatTensor([0.36, 0.64]).cuda()
    criterion = nn.CrossEntropyLoss(class_weight)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=1e-5)


    result = training(args.num_epochs, my_model, class_weight, criterion, optimizer, train_loader, validation_loader, test_loader, args.save_root)


if __name__ == "__main__":
    main()
