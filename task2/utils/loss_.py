import torch
from torch import nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy

class EMDLoss(nn.modules.loss._Loss):
    def __init__(self,r=2,number_of_classes=3):
        super(EMDLoss, self).__init__()
        self.r = r
        self.number_of_classes = number_of_classes

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        y = torch.nn.functional.one_hot(y, self.number_of_classes)

        cdf_y = torch.cumsum(y, dim=-1)        
        cdf_pred = torch.cumsum(F.softmax(y_pred, dim=-1), dim=-1)
        
        cdf_diff = cdf_pred - cdf_y
        emd_loss = torch.mean(torch.pow(torch.abs(cdf_diff) + 1e-7, self.r),axis=-1)**(1. / self.r)
        return emd_loss.mean()
    
class CrossEmdLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 0.5, r: int = 2.0, number_of_classes: int = 3):
        super(CrossEmdLoss,self).__init__()
        self.lambd = λ
        self.r = r
        self.number_of_classes = number_of_classes
        self.EMD = EMDLoss(r, number_of_classes)
        self.CE = nn.CrossEntropyLoss()
    
    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        return self.lambd * self.CE(y_pred, y) + (1 - self.lambd) * self.EMD(y_pred, y)
class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100, number_of_classes: int = 3, reduction: str = "mean"): # number_of_classes 3 yap geri
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.number_of_classes = number_of_classes
        self.reduction = reduction

    def forward(self, x, target, weight = None):
        # doesnt need one hot
        #target = torch.nn.functional.one_hot(target, self.number_of_classes)

        # log(P[class]) = log_softmax(score)[class]
        logpt = F.log_softmax(x, dim=1)
        # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
        logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()

        # Ignore index (set loss contribution to 0)
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
        if self.ignore_index >= 0 and self.ignore_index < x.shape[1]:
            valid_idxs[target.view(-1) == self.ignore_index] = False

        # Get P(class)
        pt = logpt.exp()

        # Weight
        if weight is not None:
            # Tensor type
            if weight.type() != x.data.type():
                weight = weight.type_as(x.data)
            logpt = weight.gather(0, target.data.view(-1)) * logpt

        # Loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Loss reduction
        loss = loss.reshape(-1)
        if self.reduction == "sum":
            loss = loss[valid_idxs].sum()
        elif self.reduction == "mean":
            loss = loss[valid_idxs].mean()
        else:
            # if no reduction, reshape tensor like target
            loss = loss.view(*target.shape)
        
        return loss

class FocalEmdLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 0.5, r: int = 2.0, number_of_classes: int = 3):
        super(FocalEmdLoss,self).__init__()
        self.lambd = λ
        self.r = r
        self.number_of_classes = number_of_classes
        self.EMD = EMDLoss(r, number_of_classes)
        self.FL = FocalLoss(number_of_classes=number_of_classes)
    
    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        return self.lambd * self.FL(y_pred, y) + (1 - self.lambd) * self.EMD(y_pred, y)


LOSS = {
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
    "focal": FocalLoss,
    "emd": EMDLoss,
    "crossemd": CrossEmdLoss,
    "focalemd": FocalEmdLoss,
}