from tqdm import tqdm
from utils import *
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
from time import time as tm
import matplotlib.pyplot as plt
SMOOTH = 1e-6


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)



def mIOU(label, pred, num_classes=5 ):
    # print("pred:",pred.shape)
    # print("label",label.shape)

    probs = torch.softmax(pred, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze(1)
    # print(pred.shape)
    # label = torch.max(label, dim=1)
    label = torch.argmax(label, dim=1).squeeze(1)

    # print(label.shape)
    # pred = F.softmax(pred, dim=1)
    # pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()
    # print(torch.argmax(probs[0,:,0,0], dim=1))
    # print(label.shape)
    pred = pred.view(-1)
    label = label.view(-1)

    # print(pred)
    # print(pred.shape)
    # print(label.shape)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)

        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    # print("Iou:",iou_list)

    return np.mean(present_iou_list)


def mIOUMulti(label, pred, num_classes=5):
    # print("pred:",pred.shape)
    # print("label",label.shape)

    probs = torch.softmax(pred, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze(1)
    # print(pred.shape)
    # label = torch.max(label, dim=1)
    label = torch.argmax(label, dim=1).squeeze(1)

    # print(label.shape)
    # pred = F.softmax(pred, dim=1)
    # pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()
    # print(torch.argmax(probs[0,:,0,0], dim=1))
    # print(label.shape)
    pred = pred.view(-1)
    label = label.view(-1)
    # print(pred.shape)
    # print(label.shape)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        # print("Iou:", iou_list)

    return present_iou_list

class trainer():
    def __init__(self, train_ds, val_ds, model, optimizer, scheduler,
                 criterion, epochs=500,best_acc=None, num_class = 5,trainflow = 2):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_loss = best_acc
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_class = num_class
        self.trainflow = trainflow

    def training(self):
        for idx in range(self.epochs):
            self.train_epoch(idx)
            if ((idx+1) % self.trainflow == 0):
                self.validate(idx)
        return self.model



    def train_epoch(self,epo):

        #
        torch.set_grad_enabled(True)
        self.model.train()
        total_loss = 0
        total_IoU = 0
        #total_batch = 0
        TrainLoader = tqdm(self.train_ds)
        # start = tm()
        # l.info(f"start {start}")
        for idx, (image, label) in enumerate(TrainLoader):
            # end = tm()
            # l.info(f"{end - start}")
            # l.info('start train0')
            label = label.long()
            # print(type(label))
            image = image.to(self.device)
            label  = F.one_hot(label, self.num_class).float()
            label = label.permute(0,3,1,2).to(self.device)
            # label = label.to(self.device)
            # print(label.size)
            # print(label.shape)
            # plt.imshow(label[0,2:3,:,:].permute(1,2,0).cpu().detach().numpy())
            # plt.show()

            # print(label)

            # l.info('start train')


            with torch.cuda.amp.autocast():

                output = self.model(image)
                # soft = nn.softmax()
                # output = soft(output)
                # print(output[0])
                # print(label[0,0,0])
                loss = self.criterion(output, label)
                # print(loss)


            # for param in self.model.parameters():
            #     param.requires_grad = False

            self.optimizer.zero_grad(set_to_none=True)
            # loss = Variable(loss, requires_grad=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            TrainLoader.set_description('Epoch ' + str(epo + 1))
            # TrainLoader.set_postfix(loss=loss.item() Learning_rate=self.optimizer.state_dict()['param_groups'][0]['lr'])
            total_loss += loss
            # print(label.shape, output.shape)
            total_IoU += mIOU(label,output,num_classes=self.num_class)

        train_loss = total_loss / len(self.train_ds)
        mean_IoU = total_IoU / len(self.train_ds)
        # print(train_loss.cpu().detach().numpy())

        self.scheduler.step()
        l.info(f'Epoch : {epo + 1}, Train_loss : {train_loss}, Mean_ioU: {mean_IoU}')
        # print('Epoch : {}, Train_loss : {}, Mean_ioU: {}'.format(epo + 1, train_loss, mean_IoU))


    def validate(self,epo):

        self.model.eval()
        total_IoU = 0
        total_loss = 0

        with torch.no_grad():
            self.model.eval()
            ValLoader = tqdm(self.val_ds)
            for idx, (image, label) in enumerate(ValLoader):

                label = label.long()
                # print(type(label))
                image = image.to(self.device)
                label = F.one_hot(label, self.num_class).float()
                label = torch.transpose(label, 1, 3)
                label = torch.transpose(label, 2, 3).to(self.device)
                # label = label.to(self.device)

                output = self.model(image)

                loss = self.criterion(output, label)
                miou = mIOU(label,output,num_classes=self.num_class)
                total_loss += loss
                total_IoU += miou
        total_loss = total_loss / len(self.val_ds)
        total_IoU = total_IoU / len(self.val_ds)

        l.info(f'Validation: Loss : {total_loss}, mIoU : {total_IoU}')
        self.best_loss = save_checkpoint(self.model, self.best_loss, total_loss, epo)

