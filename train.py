import cv2
import os
from model.model import SwinUnet
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms , models
from trainer import trainer,mIOU,mIOUMulti
from torchinfo import summary
from DataLoader import *
from utils import *
from loss import *
# from semseg.models.segformer import SegFormer
import segmentation_models_pytorch as smp
from Dice import DiceLoss
import argparse
import metric

parser = argparse.ArgumentParser()
parser.add_argument("--train",default=False,help="learning rate",action="store_true")
parser.add_argument("--predict",default=False,help="learning rate",action="store_true")
parser.add_argument("--lr",default=1e-4,help="learning rate",type = float)
parser.add_argument("-b","--batch_size",default=160,help="batch_size",type = int)
parser.add_argument("-e","--epoch",default=100,help="num_epoch",type = int)
parser.add_argument("-worker","--num_worker",default=16,help="num_worker",type = int)
parser.add_argument("-class","--num_class",default=4,help="num_class",type = int)
parser.add_argument("-c","--in_channels",default=4,help="in_channels",type = int)
parser.add_argument("-size","--image_size",default=224,help="image_size",type = int)
parser.add_argument("-flow","--train_flow",default=1,help="image_size",type = int)
# parser.add_argument("--batch_size",default=5e-4,help="batch_size")
# parser.add_argument("--batch_size",default=5e-4,help="batch_size")
# parser.add_argument("--batch_size",default=5e-4,help="batch_size")
# parser.add_argument("--batch_size",default=5e-4,help="batch_size")

args = parser.parse_args()


lr = args.lr
batch_size = args.batch_size
num_epoch = args.epoch
num_worker = args.num_worker
num_class = args.num_class
in_channels = args.in_channels
image_size = args.image_size
trainflow = args.train_flow
doTrain  = args.train
doPredict = args.predict



# allLogger.setLevel(logging.INFO)

l.info(f'learning rate : {lr}')
l.info(f'batch size : {batch_size}')
l.info(f'num epoch : {num_epoch}')
l.info(f'trainflow: {trainflow}')


device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = True
train_image_path = '/home/student/Desktop/3D_CT_Image/brats_new/train/images/'
train_mask_path = '/home/student/Desktop/3D_CT_Image/brats_new/train/annotations/'
val_image_path = '/home/student/Desktop/3D_CT_Image/brats_new/val/images/'
val_mask_path = '/home/student/Desktop/3D_CT_Image/brats_new/val/annotations/'
SMOOTH = 1e-6
color_map = np.array([[0, 0, 0], [128, 128, 128], [255, 0, 0], [0, 0, 255]])

test_image = '/home/student/Desktop/3D_CT_Image/brats/val/images/'
test_mask = '/home/student/Desktop/3D_CT_Image/brats/val/annotations/'
test_name = os.listdir(test_image)






def build_model(Predict = False, loadCheckpoint = True,path='checkpoint/ckpt_0.010712.pth'):
    model = SwinUnet(img_size=image_size, in_chans=in_channels, num_classes=num_class
                     ).to(device)
    if loadCheckpoint:
        model = load_checkpoint(model,path)

    # model = smp.Unet(
    #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=5,  # model output channels (number of classes in your dataset)
    # ).to(device)
    # model = nn.DataParallel(model)

    if Predict:
        model.eval()

    # summary(model, input_size=(1, in_channels, image_size, image_size))
    return model

def train_model():

    l.info("==========Start training=========")
    train_transform = get_transform()  # 取得影像增強方式
    vaild_transform = get_vaild_transform()  # 取得測試影像增強
    # 將資料送進dataloader中
    train_data = NumpyDataLoader(train_image_path, train_mask_path, train_transform)  #
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
    val_data = NumpyDataLoader(val_image_path,val_mask_path, vaild_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
    # 建立模型

    model = build_model(Predict = False, loadCheckpoint= False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train = trainer(train_loader, val_loader, model, optimizer, scheduler, loss_function,
                    epochs=num_epoch, best_acc=None, num_class= num_class, trainflow = trainflow)
    # 訓練


    model = train.training()

    l.info("==========finish train=========")

def predict():

    l.info("==========Start predict=========")
    model = build_model(Predict = True)

    vaild_transform = get_vaild_transform()
    predres = []
    gt = []
    #
    ioulist = [0,0,0,0,0]



    Miou = 0
    Mioumulti = [0,0,0,0,0]
    tatol = [0,0,0,0,0]
    #
    for idx, n in enumerate(tqdm(test_name)):
        path = test_image + n
        pathMask = test_mask + n
        img = np.load(path).astype("float")
        mask = np.load(pathMask)
        mask = torch.tensor(mask).unsqueeze(0).to(device)
        # print(mask.shape)
        # img = cv2.resize(img,(224,224))

        img = vaild_transform(image=img)['image'].unsqueeze(0).to(device)
        # print(img.shape)
        with torch.no_grad():
            # out1 = nn.Softmax()
            # output = out1(output)
            out = model(img)
            # print(mask.shape)
            # print(out.shape)
            # Miou += mIOU(mask,out)
            # print(len(mIOUMulti(mask,out)))
            _, mask = torch.max(mask, dim=1)
            mask_img = mask.cpu().detach().numpy().transpose(1, 2, 0)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(probs, dim=1)
            pred_img = preds.cpu().detach().numpy().transpose(1, 2, 0)
            predres.append(pred_img)
            gt.append(mask_img)


    # print(tatol)
    # print(ioulist)
    # for i in range(len(tatol)):
    #     Mioumulti[i] = ioulist[i] / tatol[i]
    # print(Mioumulti)


            # output_image = np.zeros((pred_img.shape[0], pred_img.shape[1], 3), dtype=np.uint8)
            # for i in range(pred_img.shape[0]):
            #     for j in range(pred_img.shape[1]):
            #         output_image[i, j, :] = color_map[pred_img[i, j]]
            #
            # cv2.imwrite('brat/' + n.replace('npy', 'png'), output_image)
    iou = metric.mean_iou(predres, gt, 5, None,)
    print("intersect ", iou)
    l.info(f"Miou = {Miou/len(test_name)}")
    # l.info(f"Miou class = {Mioumulti}")
    l.info("==========finish predict=========")



if __name__ == '__main__':

    if doTrain:
        train_model()
    if doPredict:
        predict()

    l.info("==========finish work=========")
    # model = SwinUnet(img_size=224
    #                  ).to(device)
    # summary(model, input_size=(1, 3, 224, 224))
    # predict_path = "resultU/"
    # mask_path = 'Microglia/tmp/'
    #
    # print(iou_numpy(predict_path,mask_path))