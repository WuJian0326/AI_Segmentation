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
from memory_profiler import profile

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




class trainer():
    def __init__(self, train_ds, val_ds, model, optimizer, scheduler,
                 criterion1, criterion2, epochs=500,best_acc=None, num_class = 5,trainflow = 2):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion1 = criterion1
        self.criterion2 = criterion2
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

        for idx, sample in enumerate(TrainLoader):
            
            image = sample['image']
            label = sample['label']

            label = label.long()
            # image = image.long()
            image = image.to(self.device)
            label  = F.one_hot(label, self.num_class).float()
            label = torch.transpose(label, 1, 3)
            label = torch.transpose(label, 2, 3).to(self.device)

            # print(label.shape)
            # print(image.shape)
            

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                
                
                p1 = self.model(image)
                # print(p1.shape)
                # print(p2.shape)
                # print(p3.shape)

                loss_ce1 = self.criterion1(p1, label)
                # loss_ce2 = self.criterion1(p2, label)
                # loss_ce3 = self.criterion1(p3, label)
                # loss_ce4 = self.criterion1(p4, label)
                loss_dice1 = self.criterion2(p1, label, softmax=True)
                # loss_dice2 = self.criterion2(p2, label, softmax=True)
                # loss_dice3 = self.criterion2(p3, label, softmax=True)
                # loss_dice4 = self.criterion2(p4, label, softmax=True)
                loss1 = 0.3 * loss_ce1 + 0.7*loss_dice1
                # loss2 = 0.3 * loss_ce2 + 0.7*loss_dice2
                # loss3 = 0.3 * loss_ce3 + 0.7*loss_dice3
                # loss4 = 0.3 * loss_ce4 + 0.7*loss_dice4
                loss = loss1
                # loss = loss1 + loss2 + loss3 
                outputs = p1 
                # print(p4.shape)
                # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # out = out.cpu().detach().numpy()
                # print(out.shape)
                # print(label.shape)
                # print(label.shape)
                # loss = self.criterion1(output, label)



            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            TrainLoader.set_description('Epoch ' + str(epo + 1))
            # TrainLoader.set_postfix(loss=loss.item() Learning_rate=self.optimizer.state_dict()['param_groups'][0]['lr'])
            total_loss += loss
            # label = torch.argmax(label, dim=1).squeeze(0)
            # label = label.cpu().detach().numpy()
            # print(label.shape, output.shape)
            total_IoU += mIOU(label,outputs,num_classes=self.num_class)
            # torch.cuda.empty_cache()

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
            for idx, sample in enumerate(ValLoader):
                image = sample['image']
                label = sample['label']

                label = label.long()
                # print(type(label))
                image = image.to(self.device)
                label = F.one_hot(label, self.num_class).float()
                label = torch.transpose(label, 1, 3)
                label = torch.transpose(label, 2, 3).to(self.device)
                # label = label.to(self.device)

                p1 = self.model(image)
                loss_ce1 = self.criterion1(p1, label)
                # loss_ce2 = self.criterion1(p2, label)
                # loss_ce3 = self.criterion1(p3, label)
                # loss_ce4 = self.criterion1(p4, label)
                loss_dice1 = self.criterion2(p1, label, softmax=True)
                # loss_dice2 = self.criterion2(p2, label, softmax=True)
                # loss_dice3 = self.criterion2(p3, label, softmax=True)
                # loss_dice4 = self.criterion2(p4, label, softmax=True)
                loss1 = 0.3 * loss_ce1 + 0.7*loss_dice1
                # loss2 = 0.3 * loss_ce2 + 0.7*loss_dice2
                # loss3 = 0.3 * loss_ce3 + 0.7*loss_dice3
                # loss4 = 0.3 * loss_ce4 + 0.7*loss_dice4
                loss = loss1 
                outputs = p1 
                # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # out = out.cpu().detach().numpy()
                # loss = self.criterion(loss, label)
                miou = mIOU(label,outputs,num_classes=self.num_class)
                total_loss += loss
                total_IoU += miou

        total_loss = total_loss / len(self.val_ds)
        total_IoU = total_IoU / len(self.val_ds)

        l.info(f'Validation: Loss : {total_loss}, mIoU : {total_IoU}')
        self.best_loss = save_checkpoint(self.model, self.best_loss, total_loss, epo)

