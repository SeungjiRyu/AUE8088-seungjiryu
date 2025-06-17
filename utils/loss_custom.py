# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


# Binary Cross Entropy with label smoothing
def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps

# Binary Cross Entropy with Logits Loss with reduced missing label effects
# 실제로 객체가 있지만 라벨링되지 않은 경우, loss계산이 과도하게 커지는 것을 방지
class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

# Focal Loss, QFocal Loss: 객체 탐지 시에 배경과 객체 간의 불균형을 완화
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

# def bbox_iou_wiou(box1, box2, eps=1e-7):
#     """
#     WIoU v1 구현 (박스 위치, 크기, 비율 차이 반영하여 )
#     box1, box2: [x, y, w, h] 형식
#     """
#     # IoU 계산
#     b1_x1, b1_y1 = box1[:, 0] - box1[:, 2]/2, box1[:, 1] - box1[:, 3]/2
#     b1_x2, b1_y2 = box1[:, 0] + box1[:, 2]/2, box1[:, 1] + box1[:, 3]/2
#     b2_x1, b2_y1 = box2[:, 0] - box2[:, 2]/2, box2[:, 1] - box2[:, 3]/2
#     b2_x2, b2_y2 = box2[:, 0] + box2[:, 2]/2, box2[:, 1] + box2[:, 3]/2

#     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
#     union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
#     iou = inter / union

#     # 중심점 거리
#     c_dist = (box1[:,0]-box2[:,0])**2 + (box1[:,1]-box2[:,1])**2
    
#     # 외접 박스 대각선 길이
#     enclose_x1 = torch.min(b1_x1, b2_x1)
#     enclose_y1 = torch.min(b1_y1, b2_y1)
#     enclose_x2 = torch.max(b1_x2, b2_x2)
#     enclose_y2 = torch.max(b1_y2, b2_y2)
#     enclose_diag = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps
    
#     # 크기 차이
#     w_diff = (box1[:,2] - box2[:,2])**2
#     h_diff = (box1[:,3] - box2[:,3])**2

#     return 1 - iou + (c_dist + w_diff + h_diff)/enclose_diag

class WIoU_Calculator:
    def __init__(self, momentum=0.999):
        self.momentum = momentum
        self.running_mean = 0.4  # L_IoU 

    def update(self, batch_iou):
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_iou.mean().item()

    def __call__(self, pred, target):
        # IoU
        inter_x1 = torch.max(pred[:,0]-pred[:,2]/2, target[:,0]-target[:,2]/2)
        inter_y1 = torch.max(pred[:,1]-pred[:,3]/2, target[:,1]-target[:,3]/2)
        inter_x2 = torch.min(pred[:,0]+pred[:,2]/2, target[:,0]+target[:,2]/2)
        inter_y2 = torch.min(pred[:,1]+pred[:,3]/2, target[:,1]+target[:,3]/2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union = pred[:,2]*pred[:,3] + target[:,2]*target[:,3] - inter_area
        iou = inter_area / (union + 1e-7)

        # WIoU v1 elements
        center_dist = (pred[:,0] - target[:,0])**2 + (pred[:,1] - target[:,1])**2
        enclose_x1 = torch.min(pred[:,0]-pred[:,2]/2, target[:,0]-target[:,2]/2)
        enclose_y1 = torch.min(pred[:,1]-pred[:,3]/2, target[:,1]-target[:,3]/2)
        enclose_x2 = torch.max(pred[:,0]+pred[:,2]/2, target[:,0]+target[:,2]/2)
        enclose_y2 = torch.max(pred[:,1]+pred[:,3]/2, target[:,1]+target[:,3]/2)
        enclose_diag = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-7

        # Gradient gain
        beta = (1 - iou) / (1 - self.running_mean + 1e-7)
        beta = beta.clamp(min=1e-3, max=10.0)  
        r = 4.0 / (1.6 ** (beta - 4.0 + 1e-7))  # α=1.6, δ=4 (Default), 1.6^(beta-4.0)

        return torch.mean(r * (1 - iou + (center_dist + 1) / enclose_diag))

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.wiou = WIoU_Calculator() # Initialize WIoU calculator

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # iou loss
                wiou_loss = self.wiou(pbox, tbox[i])  # WIoU loss
                lbox += wiou_loss
                
                # Update WIoU running mean
                with torch.no_grad():
                    iou = bbox_iou(pbox, tbox[i], CIoU=False).squeeze()
                    self.wiou.update(iou)

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou

                # # If prediction is matched (iou > 0.5) with bounding box marked as ignore,
                # # do not calculate objectness loss
                # ign_idx = (tcls[i] == -1) & (iou > self.hyp["iou_t"])
                # keep = ~ign_idx
                
                # b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]

                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)                
            
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
