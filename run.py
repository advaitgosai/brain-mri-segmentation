import os
from models.ONet3d import ONet3d
from torch.optim import Adam
from utils.Meter import BasnetHybridLoss
from training import Trainer
from data_processing import get_datasets

import warnings
warnings.simplefilter("ignore")

def get_ATLAS_train_images_and_masks():
    images = [[os.path.join("ATLAS_data", "Train_Brain_ATLAS", brain)]for brain in sorted(os.listdir(os.path.join("ATLAS_data", "Train_Brain_ATLAS")))]
    masks = [os.path.join("ATLAS_data", "Train_Mask_ATLAS", mask) for mask in sorted(os.listdir(os.path.join("ATLAS_data", "Train_Mask_ATLAS")))]
    return images, masks

def get_ATLAS_test_images_and_masks():
    images = [[os.path.join("ATLAS_data", "Test_Brain_ATLAS", brain)]for brain in sorted(os.listdir(os.path.join("ATLAS_data", "Test_Brain_ATLAS")))]
    masks = [os.path.join("ATLAS_data", "Test_Mask_ATLAS", mask) for mask in sorted(os.listdir(os.path.join("ATLAS_data", "Test_Mask_ATLAS")))]
    return images, masks

train_dataloader, val_dataloader = get_datasets(get_ATLAS_train_images_and_masks, get_ATLAS_test_images_and_masks, center_crop_size = (182, 218, 182))

model = {"model": ONet3d(in_channels = 1, n_classes = 1, n_channels = 8).to('cuda'), "name":f"ONet3D_BasnetHybridLoss"}

trainer = Trainer(net = model["model"],
                  net_name = model["name"],
                  criterion = BasnetHybridLoss(),
                  lr = 1e-3,
                  accumulation_steps = 4,
                  num_epochs = 500,
                  optimizer = Adam,
                  load_prev_model = True,
                  train_dataloader = train_dataloader,
                  val_dataloader = val_dataloader)
    
trainer.run()