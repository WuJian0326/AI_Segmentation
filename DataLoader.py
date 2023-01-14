import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from time import time
# def one_hot_encode(label_image, num_classes):
#     label_image = label_image.astype(int)
#     rows, cols = label_image.shape
#     one_hot_labels = np.zeros((num_classes, rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             one_hot_labels[label_image[i, j], i, j] = 1.0
#     return one_hot_labels

def get_transform():

    transform = A.Compose([
        A.Resize(224,224),
        A.OneOf([

            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                               p=0.3),
            A.RandomRotate90(),
        ], p=0.3),
        # A.OneOf([
            # A.RandomBrightness(limit=0.1, always_apply=False, p=0.5),
            # A.RandomContrast(limit=0.1, always_apply=False, p=0.5),
        # ], p=0.3),
        A.OneOf([
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.3, num_steps=10),
            A.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5)
        ]),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.424],
            std=[0.229, 0.224, 0.225, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    return transform

def get_vaild_transform():
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.424],
            std=[0.229, 0.224, 0.225, 0.225],
            max_pixel_value=255.0
            ),
        ToTensorV2(),
    ])
    return transform

def one_hot_encode(label_image, num_classes):
    label_image = label_image.astype(int)
    rows, cols = label_image.shape
    one_hot_labels = np.zeros((num_classes, rows, cols))
    for i in range(rows):
        for j in range(cols):
            one_hot_labels[label_image[i, j], i, j] = 1.0
    return one_hot_labels

class ImageDataLoader(Dataset):
    def __init__(self,  img_dir,mask_dir, transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]).replace('.jpg','.png')


        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.
        image = cv2.resize(image,(224,224))
        mask = cv2.resize(mask,(224,224))
        mask[mask > 0.8] = 1.0
        mask[mask < 0.8] = 0.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

class NumpyDataLoader(Dataset):
    def __init__(self,  img_dir,mask_dir, transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform

        self.images = os.listdir(img_dir)
        # data = []
        #
        # for i in self.images:
        #     data.append([img_dir + i,mask_dir + i])
        #
        # data = np.array(data)
        #
        # self.out = Parallel(n_jobs=8)(
        #     delayed(self.transform_image)(data[i]) for i in trange(len(data))
        # )


    # def transform_image(self, data_path):
    #
    #     image = np.load(data_path[0])
    #     mask = np.load(data_path[1])
    #     image.astype("float")
    #     return [image,mask]



    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        #
        # # print(img_path)
        image = np.load(img_path)
        # # print("org_size",image.shape)
        # mask = np.load(mask_path)
        mask = cv2.imread(mask_path.replace('npy','png'),0)
        # print(mask.shape)
        # mask = one_hot_encode(mask,4)
        # print(mask.shape)
        # mask = np.eye(4)[mask].transpose(2,0,1)
        # print(mask.shape)
        # image = self.out[index][0]
        # mask = self.out[index][1]
        image = image.astype("float")

        # print(image.shape)
        # mask = cv2.resize(mask,(224,224)).astype("uint8")
        # mask = one_hot_encode(label_image= mask ,num_classes= 5)
        # print(mask.shape)
        # print(mask.shape)
        # mask[mask > 0.8] = 1.0
        # mask[mask < 0.8] = 0.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# train_image_path = '/home/student/Desktop/3D_CT_Image/brats/images/'
# train_mask_path = '/home/student/Desktop/3D_CT_Image/brats/annotations/'
# a = NumpyDataLoader(train_image_path,train_mask_path,get_vaild_transform())
#
# for idx, (image, mask) in enumerate(a):
#     print(image.shape)
#     print(mask.shape)