import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from pytorch_msssim import ssim

def dice_coef_metric(predictions: torch.Tensor, 
                     truth: torch.Tensor,
                     num_classes = 1,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice Score for data batch.
    Params:
        predictions: predicted classes (0, 1, 2, ...).
        truth: ground truth classes.
        eps: additive to refine the estimate.
        Returns: dice score.
    """
    if num_classes == 1:
        intersection = 2.0 * (predictions * truth).sum()
        union = predictions.sum() + truth.sum()
        
        if predictions.sum() == 0 and truth.sum() == 0:
            return [1.0]
        else:
            return [(intersection + eps) / union]
    
    total_dice = []
    for i in range(num_classes):
        pred = (predictions == i).float()
        true = (truth == i).float()
        
        intersection = 2.0 * (true * pred).sum()
        union = true.sum() + pred.sum()
        
        if true.sum() == 0 and pred.sum() == 0:
            dice_score = 1.0
        else:
            dice_score = (intersection + eps) / union
            
        total_dice.append(dice_score)

    return total_dice


def jaccard_coef_metric(predictions: torch.Tensor,
                        truth: torch.Tensor,
                        num_classes = 1,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        predictions: predicted classes (0, 1, 2, ...).
        truth: ground truth classes.
        eps: additive to refine the estimate.
        Returns: jaccard score.
    """
    
    if num_classes == 1:
        intersection = (predictions * truth).sum()
        union = (predictions + truth).sum() - intersection

        if predictions.sum() == 0 and truth.sum() == 0:
            return [1.0]
        else:
            return [(intersection + eps) / union]
    
    total_jaccard = []
    for i in range(num_classes):
        pred = (predictions == i).float()
        true = (truth == i).float()
        
        intersection = (pred * true).sum()
        union = (pred + true).sum() - intersection
        
        if true.sum() == 0 and pred.sum() == 0:
            jaccard_score = 1.0
        else:
            jaccard_score = (intersection + eps) / union
        
        total_jaccard.append(jaccard_score)

    return total_jaccard


class Meter:
    '''Factory for storing and updating iou and dice scores.'''

    def __init__(self):
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.num_classes = None  # Initially set as None

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        # Infer number of classes if it's the first call
        if self.num_classes is None:
            self.num_classes = logits.shape[1]

        if self.num_classes == 1:  # Binary segmentation
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
        else:  # Multi-class segmentation
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        dice = dice_coef_metric(predictions, targets, self.num_classes)
        iou = jaccard_coef_metric(predictions, targets, self.num_classes)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(np.array(self.dice_scores))
        iou = np.mean(np.array(self.iou_scores))
        return dice, iou
    
    def get_metrics_by_class(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(np.array(self.dice_scores), axis = 0)
        iou = np.mean(np.array(self.iou_scores), axis = 0)
        return dice, iou
    

class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert (probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score
    

class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert (logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss
    

class BasnetHybridLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(BasnetHybridLoss, self).__init__()
        self.eps = eps

    def jaccard_loss(self, probability: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = (probability * targets).sum()
        union = probability.sum() + targets.sum() - intersection
        return 1 - (intersection + self.eps) / (union + self.eps)

    def ssim_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return 1 - ssim(y_true, y_pred, data_range=1)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        probability = torch.sigmoid(logits)

        bce_loss = F.binary_cross_entropy(probability, targets)
        ms_ssim_loss = self.ssim_loss(targets, probability)
        jaccard_loss = self.jaccard_loss(probability, targets)

        return bce_loss + ms_ssim_loss + jaccard_loss
    
    
class MultiClassDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(MultiClassDiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # Softmax across the classes dimension
        probabilities = torch.softmax(logits, dim=1)
    
        # Reshape the tensors
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=probabilities.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).to(probabilities.dtype)

        assert (probabilities.shape == targets_one_hot.shape)

        intersection = torch.sum(probabilities * targets_one_hot, dim=(0, 2, 3, 4)) * 2.0
        union = torch.sum(probabilities, dim=(0, 2, 3, 4)) + torch.sum(targets_one_hot, dim=(0, 2, 3, 4))
        dice_scores = (intersection + self.eps) / union

        return 1.0 - torch.mean(dice_scores)
    

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        targets = targets.squeeze(1)
        
        # Make sure that logits are [B, C, ...] and targets are [B, ...]
        assert (logits.shape[1] > 1)
        assert (logits.shape[0] == targets.shape[0])

        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())

        return ce_loss + dice_loss
    
    
class BasnetHybridLossMulticlass(nn.Module):
    def __init__(self, num_classes, eps: float = 1e-9):
        super(BasnetHybridLossMulticlass, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def jaccard_loss(self, probability: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        jaccard_loss = 0
        for cls in range(self.num_classes):
            i_target = (targets == cls).float()
            i_prob = probability[:, cls]
            intersection = (i_prob * i_target).sum()
            union = i_prob.sum() + i_target.sum() - intersection
            jaccard_loss += 1 - (intersection + self.eps) / (union + self.eps)
        
        return jaccard_loss / self.num_classes

    def ssim_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        ssim_loss = 0
        for cls in range(self.num_classes):
            i_target = (y_true == cls).unsqueeze(dim=1).float()
            i_prob = y_pred[:, cls].unsqueeze(dim=1)
            ssim_loss += 1 - ssim(i_target, i_prob, data_range=1)

        return ssim_loss / self.num_classes
    

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze(1)

        ce_loss = self.ce(logits, targets.long())
        
        probability = torch.softmax(logits, dim=1)
        
        ms_ssim_loss = self.ssim_loss(targets, probability)
        jaccard_loss = self.jaccard_loss(probability, targets)

        return ce_loss + ms_ssim_loss + jaccard_loss
