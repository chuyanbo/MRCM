#Dice系数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

#Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss0(nn.Module):
    def __init__(self):
        super(DiceLoss0, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score

#BiseNet中的DiceLoss代码
import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    
    C = tensor.size(1)        #获得图像的维数
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))     
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)                 #将维数的数据转换到第一位
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)              


class BiseNetDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator
        
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, sampling='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.sampling = sampling

    def forward(self, y_pred, y_true):
        alpha = self.alpha
        alpha_ = (1 - self.alpha)
        if self.logits:
            y_pred = torch.sigmoid(y_pred)

        pt_positive = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt_negative = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        pt_positive = torch.clamp(pt_positive, 1e-3, .999)
        pt_negative = torch.clamp(pt_negative, 1e-3, .999)
        pos_ = (1 - pt_positive) ** self.gamma
        neg_ = pt_negative ** self.gamma

        pos_loss = -alpha * pos_ * torch.log(pt_positive)
        neg_loss = -alpha_ * neg_ * torch.log(1 - pt_negative)
        loss = pos_loss + neg_loss

        if self.sampling == "mean":
            return loss.mean()
        elif self.sampling == "sum":
            return loss.sum()
        elif self.sampling == None:
            return loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / N
        return 1 - loss
    
class Multi_FocalLoss(nn.Module):
    def __init__(self,device, gamma = 2, alpha = 0.25, size_average = True):
        super(Multi_FocalLoss, self).__init__()
        self.device=device
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * labels_length * seq_length
        """
        assert(logits.size(0) == labels.size(0))
        # assert(logits.size(1) == labels.size(1))
        # assert(logits.size(2) == labels.size(2))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)*logits.size(3)
        # transpose labels into labels onehot
        label_onehot = labels.view(batch_size,labels_length,-1)
        # label_onehot = torch.zeros(batch_size, labels_length, seq_length).to(self.device).scatter(1, labels , 1)
        logits = logits.view(batch_size,labels_length,-1)

        # calculate log
        soft_p = F.softmax(logits,dim=1)
        log_p = -1 * torch.log(soft_p+self.elipson)
        log_pt = label_onehot * log_p
        soft_pt = label_onehot * soft_p
        
        sub_pt = 1 - soft_pt*label_onehot
        fl = self.alpha *( (sub_pt)**self.gamma )* log_pt
        
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
    
# loss = Multi_FocalLoss("cpu")
# predict = torch.abs(torch.randn( 3, 7 , 4, 4))
# target = torch.abs(torch.randn( 3, 1, 4, 4))
# target = torch.ones_like(target,dtype=torch.int64)
# # predict = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]]).view(1,3,2,2)
# # target = torch.tensor([[2,2,2,2],[2,2,2,2],[2,2,2,2]]).view(1,3,2,2)

# score = loss(predict, target)
# print(predict.shape,score)
