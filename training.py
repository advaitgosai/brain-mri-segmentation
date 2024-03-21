from tqdm import tqdm
import os
import time
from datetime import datetime
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.Meter import Meter
import pandas as pd
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, net: nn.Module, net_name: str, criterion: nn.Module, lr: float, accumulation_steps: int, num_epochs: int, load_prev_model: bool = True, optimizer: torch.optim = Adam, train_dataloader = None, val_dataloader = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net
        self.net = self.net.to(self.device)
        self.net_name = net_name
        self.load_prev_model = load_prev_model
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size = 50, gamma = 0.75, verbose = True)
        self.accumulation_steps = accumulation_steps
        self.phases = ["train"] if val_dataloader is None else ["train", "val"]
        self.num_epochs = num_epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.parameter_count = count_parameters(self.net)
        self.prev_epoch = 0
        
        checkpoint_directory = os.path.join("saved_models", self.net_name)
        os.makedirs(checkpoint_directory, exist_ok = True)
        
        plots_directory = os.path.join("plots")
        os.makedirs(plots_directory, exist_ok = True)
        
        if self.load_prev_model:
            self.prev_epoch = self.load_model()
        
         
    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits
        
    def _do_epoch(self, epoch: int, phase: str):
        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        print(f"{phase} epoch: {epoch}")
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'].float(), data_batch['mask'].float()
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                        )
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()
        
        print(f"Epoch Loss: {epoch_loss}")
        print(f"Epoch Dice: {epoch_dice}")
        print(f"Epoch IOU: {epoch_iou}")
        print()
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.iou_scores[phase].append(epoch_iou)

        return epoch_loss
        
    def run(self):
        start = datetime.now()
        for epoch in range(self.prev_epoch + 1, self.num_epochs + 1):
            self._do_epoch(epoch, "train")
            if "val" in self.phases:
                with torch.no_grad():
                    val_loss = self._do_epoch(epoch, "val")
                
            self.scheduler.step()
                
            if "val" in self.phases:
                if len(self.losses["val"]) > 1 and val_loss < min(self.losses["val"][:-1]):
                    print("Got the best val loss. Checkpointing the model...")
                    checkpoint = {
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                    }
                    checkpoint_filename = f"{self.net_name}_epoch_{epoch}.pth"
                    torch.save(checkpoint, os.path.join("saved_models", self.net_name, checkpoint_filename))
            else:
                if epoch % 2 == 0:
                    checkpoint = {
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                    }
                    checkpoint_filename = f"{self.net_name}_epoch_{epoch}.pth"
                    torch.save(checkpoint, os.path.join("saved_models", self.net_name, checkpoint_filename))
                
            print()
            self._plot_train_history()
            
    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.iou_scores]
        x_range = np.arange(self.prev_epoch + 1, self.prev_epoch + len(data[0]['train']) + 1)
        colors = ['deepskyblue', "crimson"]
        if "val" in self.phases:
            labels = [
                f"""
                train loss {self.losses['train'][-1]}
                val loss {self.losses['val'][-1]}
                """,

                f"""
                train dice score {self.dice_scores['train'][-1]}
                val dice score {self.dice_scores['val'][-1]} 
                """, 

                f"""
                train iou score {self.iou_scores['train'][-1]}
                val iou score {self.iou_scores['val'][-1]}
                """,
            ]
        else:
            labels = [
                f"""
                train loss {self.losses['train'][-1]}
                """,

                f"""
                train dice score {self.dice_scores['train'][-1]}
                """, 

                f"""
                train iou score {self.iou_scores['train'][-1]}
                """,
            ]
        
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                if "val" in data[i]:
                    ax.plot(x_range, data[i]['val'], c=colors[0], label="val")
                ax.plot(x_range, data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Scores")
                
            plt.tight_layout()
            plt.savefig(os.path.join("plots", f"{self.net_name}.png"))
        
    def load_model(self):
        model_name_pattern = fr"{self.net_name}_epoch_(\d+).pth"
        for file in sorted(os.listdir(os.path.join("saved_models", self.net_name)), key = lambda x: os.path.getctime(os.path.join("saved_models", self.net_name, x)), reverse = True):
            match = re.search(model_name_pattern, file)
            if match:
                previous_epoch = int(match.group(1))
                print(f"Loading {previous_epoch} epoch model...")
                checkpoint = torch.load(os.path.join("saved_models", self.net_name, file))
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.net = self.net.to(self.device)
                return previous_epoch
        print("Previous model not found. Training from begining...")
        return 0