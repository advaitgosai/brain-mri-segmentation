import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
import os
import torchio as tio
import torch
import random
import math

class GeneralDataset(Dataset):
    def __init__(self, images, masks, center_crop_size, augmentation = False):
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
        self.center_crop_size = center_crop_size
        
        self.prob = 0.8
        
        self.flip_transforms = tio.Compose([
            tio.RandomFlip(axes=(0,), p=self.prob),
            tio.RandomFlip(axes=(1,), p=self.prob),
            tio.RandomFlip(axes=(2,), p=self.prob),
        ])
        
        if self.augmentation:
            self.transforms = tio.OneOf({
                self.flip_transforms,
                tio.RandomAffine(degrees=5, translation=5, scales=(0.95, 1.05), isotropic=True, default_pad_value='minimum'),
                tio.RandomBiasField(p=self.prob),
                tio.RandomElasticDeformation(p=self.prob),
                tio.RandomNoise(std=(0, 0.25), p=self.prob),
                tio.RandomMotion(p=self.prob),
                tio.RandomGhosting(p=self.prob),
                tio.RandomSpike(p=self.prob),
                tio.RandomBlur(p=self.prob),
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        stacked_images = []
      
        for modality in image_path:
            img = self.load_img(modality)

            img = self.crop_or_pad_img(img, self.center_crop_size)

            img = self.normalize(img)
            stacked_images.append(img)
    
        stacked_images = np.stack(stacked_images)
        stacked_images = np.moveaxis(stacked_images, (0, 1, 2, 3), (0, 3, 2, 1))
        
        if mask_path == "neg":
            mask = np.zeros((1, ) + stacked_images.shape[1:])
        else:
            mask = self.load_img(mask_path)
            mask = self.crop_or_pad_img(mask, self.center_crop_size)
            mask = mask[None, ...]
            mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        
        subject = tio.Subject(
            img = tio.ScalarImage(tensor = torch.from_numpy(stacked_images)),
            mask = tio.LabelMap(tensor = torch.from_numpy(mask))
        )
        
        if self.augmentation:
            subject = self.transforms(subject)

        return {
            "image": subject['img'].data,
            "mask": subject['mask'].data,
            "image_path": image_path,
            "mask_path": mask_path
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def crop_or_pad_img(self, img, output_size):
        img = np.expand_dims(img, 0)
        
        # Create a TorchIO subject with the image
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=img),
        )

        # Create a transform for cropping or padding
        transform = tio.CropOrPad(output_size)

        # Apply the transform to the subject
        transformed_subject = transform(subject)

        # Get the transformed image data
        transformed_img = transformed_subject['img'].data.numpy()

        return transformed_img[0]
    
    
class GeneralDatasetImages(Dataset):
    def __init__(self, images, center_crop_size):
        self.images = images
        self.center_crop_size = center_crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        stacked_images = []
      
        for modality in image_path:
            img = self.load_img(modality)

            img = self.crop_or_pad_img(img, self.center_crop_size)

            img = self.normalize(img)
            stacked_images.append(img)
    
        stacked_images = np.stack(stacked_images)
        stacked_images = np.moveaxis(stacked_images, (0, 1, 2, 3), (0, 3, 2, 1))

        return {
            "image": torch.from_numpy(stacked_images),
            "image_path": image_path
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def crop_or_pad_img(self, img, output_size):
        img = np.expand_dims(img, 0)
        
        # Create a TorchIO subject with the image
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=img),
        )

        # Create a transform for cropping or padding
        transform = tio.CropOrPad(output_size)

        # Apply the transform to the subject
        transformed_subject = transform(subject)

        # Get the transformed image data
        transformed_img = transformed_subject['img'].data.numpy()

        return transformed_img[0]