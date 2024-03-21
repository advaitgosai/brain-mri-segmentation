import os
import random
import numpy as np
from utils.GeneralDataset import GeneralDataset
from torch.utils.data import DataLoader
import nibabel as nib
import torchio as tio
import torch

def get_datasets(get_train_images_and_masks, get_test_images_and_masks, center_crop_size, augmentation = False):
    training_images, training_masks = get_train_images_and_masks()
    
    if get_test_images_and_masks is not None:
        test_images, test_masks = get_test_images_and_masks()
    else:
        test_images, test_masks = None, None

    train_dataset = GeneralDataset(training_images, training_masks, center_crop_size, augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size = 1, num_workers = 4, pin_memory = True, shuffle = True)
    
    if test_images is not None:
        test_dataset = GeneralDataset(test_images, test_masks, center_crop_size, augmentation)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, num_workers = 4, pin_memory = True, shuffle = True)
    
    print("----------------------Sanity Check------------------------")
    print("Train")
    print("Total Data Size:", len(train_dataset))

    data = next(iter(train_dataloader))
    print("Image Path:", data['image_path'])
    print("Mask Path:", data['mask_path'])
    print("Image Shape:", data['image'].shape)
    print("Mask Shape:", data['mask'].shape)

    img_tensor = data['image'].squeeze().cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze().squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts = True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts = True))
    
    if test_images is not None:
        print("Test")
        print("Total Data Size:", len(test_dataset))

        data = next(iter(test_dataloader))
        print("Image Path:", data['image_path'])
        print("Mask Path:", data['mask_path'])
        print("Image Shape:", data['image'].shape)
        print("Mask Shape:", data['mask'].shape)

        img_tensor = data['image'].squeeze().cpu().detach().numpy() 
        mask_tensor = data['mask'].squeeze().squeeze().cpu().detach().numpy()
        print("Num uniq Image values :", len(np.unique(img_tensor, return_counts = True)[0]))
        print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
        print("Num uniq Mask values:", np.unique(mask_tensor, return_counts = True))
    
    return train_dataloader, None if test_images is None else test_dataloader