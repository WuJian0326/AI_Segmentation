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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",default=None,help="checkpoint path",type= str)
parser.add_argument("--out",default=None,help="output path",type= str)

args = parser.parse_args()
checkpoint_path = args.checkpoint
out_path = args.out

def predict():

    l.info("==========Start predict=========")


    model = torch.load(checkpoint_path)

    model.eval()

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
        print(numpy_array.shape)
# 
        reshaped_array = np.reshape(numpy_array, (224, 224, 1))

        with torch.no_grad():
            p1 = model(image)

            out = p1  
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
    predict()