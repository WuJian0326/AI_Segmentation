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
from trainer import trainer,mIOU
from torchinfo import summary
from DataLoader import *
from utils import *
from loss import *
import segmentation_models_pytorch as smp
from Dice import DiceLoss
import argparse
import metric
from model.FCT import FCT , init_weights
from lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded
from torch.optim.lr_scheduler import LambdaLR


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'

parser = argparse.ArgumentParser()
parser.add_argument("--train",default=True,help="learning rate",action="store_true")
parser.add_argument("--predict",default=False,help="learning rate",action="store_true")
parser.add_argument("--lr",default=1e-3,help="learning rate",type = float)
parser.add_argument("-b","--batch_size",default=1,help="batch_size",type = int)
parser.add_argument("-e","--epoch",default=160,help="num_epoch",type = int)
parser.add_argument("-worker","--num_worker",default=16,help="num_worker",type = int)
parser.add_argument("-class","--num_class",default=4,help="num_class",type = int)
parser.add_argument("-c","--in_channels",default=1,help="in_channels",type = int)
parser.add_argument("-size","--image_size",default=224,help="image_size",type = int)
parser.add_argument("-flow","--train_flow",default=10,help="image_size",type = int)
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




l.info(f'learning rate : {lr}')
l.info(f'batch size : {batch_size}')
l.info(f'num epoch : {num_epoch}')
l.info(f'trainflow: {trainflow}')


device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = True
train_image_path = '/home/student/Desktop/3D_CT_Image/brats_new_v4/train/images/'
train_mask_path = '/home/student/Desktop/3D_CT_Image/brats_new_v4/train/annotations/'
val_image_path = '/home/student/Desktop/3D_CT_Image/brats_new_v4/val/images/'
val_mask_path = '/home/student/Desktop/3D_CT_Image/brats_new_v4/val/annotations/'
SMOOTH = 1e-6
color_map = np.array([[0, 0, 0], [128, 128, 128], [255, 0, 0], [0, 0, 255]])

test_image = '/home/student/Desktop/3D_CT_Image/brats_new_v4/test/images/'
test_mask = '/home/student/Desktop/3D_CT_Image/brats_new_v4/test/annotations/'
test_name = os.listdir(test_image)






def build_model(Predict = False, loadCheckpoint = True,path='checkpoint/Unet.pth'):
    # model = SwinUnet(img_size=image_size, in_chans=in_channels, num_classes=num_class
                    #  ).to(device)


    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_class,  # model output channels (number of classes in your dataset)
    ).to(device)
    
    if loadCheckpoint:
        model = load_checkpoint(model,path)

    if Predict:
        model.eval()

    return model

def train_model():

    l.info("==========Start training=========")
    train_transform = get_transform()  # 取得影像增強方式
    valid_transform = get_vaild_transform()  # 取得測試影像增強
    # 將資料送進dataloader中
    # train_data = NumpyDataLoader(train_image_path, train_mask_path, train_transform)  #
    train_data = h5DataLoader(base_dir="/home/student/Desktop/SSL4MIS/data/ACDC",
                              split="train",num=None, transform=train_transform
                              )
    val_data = h5DataLoader(base_dir="/home/student/Desktop/SSL4MIS/data/ACDC"
                            , split="val", transform=valid_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
    # 建立模型
    
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_class,  # model output channels (number of classes in your dataset)
    ).to(device)
    
    # model = build_model(Predict = False, loadCheckpoint= False)
    # model = FCT_C().to(device)
    # model.apply(init_weights)

    # model = MERIT_Cascaded(n_class=num_class, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    # model = load_checkpoint(model,path='checkpoint/ckpt_latest.pth')

    loss_function１ = nn.CrossEntropyLoss()
    loss_function2 = DiceLoss(n_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def lr_lambda(epoch):
        if epoch < 50:
            # 在 warm up 阶段，使用线性增长的学习率
            return (epoch + 1) / 50
        else:
            # 在 warm up 结束后，使用固定的学习率
            return 1

    scheduler = LambdaLR(optimizer, lr_lambda)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    train = trainer(train_loader, val_loader, model, optimizer, scheduler, loss_function1,loss_function2,
                    epochs=num_epoch, best_acc=None, num_class= num_class, trainflow = trainflow)
    # 訓練


    model = train.training()

    l.info("==========finish train=========")

def predict():

    l.info("==========Start predict=========")
    # model = build_model(Predict = True)
    model = MERIT_Cascaded(n_class=num_class, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    model = load_checkpoint(model,path='checkpoint/ckpt_latest.pth')

    vaild_transform = get_vaild_transform()
    predres = []
    gt = []
    #




    Miou = 0

    #
    val_data = h5DataLoader(base_dir="/home/student/Desktop/SSL4MIS/data/ACDC"
                        , split="val", transform=vaild_transform)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1, pin_memory=True)

    for idx, sample in enumerate(tqdm(val_loader)):
        img = sample['image']
        label = sample['label']
        print(label.shape)

        image = img.cuda()

        numpy_array = label.numpy()

# 
        reshaped_array = np.reshape(numpy_array, (256, 256, 1))

        with torch.no_grad():
            p1, p2, p3 = model(image)

            out = p1 + p2 + p3 
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(probs, dim=1)
            pred_img = preds.cpu().detach().numpy().transpose(1, 2, 0)
            predres.append(pred_img)
            gt.append(reshaped_array)
            print(pred_img.shape)

    iou = metric.mean_iou(predres, gt, 4, None,)
    print("intersect ", iou)





    l.info(f"Miou = {Miou/len(test_name)}")
    l.info("==========finish predict=========")



if __name__ == '__main__':

    train_model()


    l.info("==========finish work=========")