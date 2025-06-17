# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """
    Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Example: y = (x - mean) / std
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """Equalizes image histogram, with optional CLAHE, for BGR or RGB image with shape (n,m,3) and range 0-255."""
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates half of the smallest object labels in an image for data augmentation.

    Returns augmented image and labels.
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def letterbox_kaist(im, new_shape=(640, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    KAIST 데이터셋을 위한 letterbox 함수.
    원본 비율 640x512를 유지하여 이미지를 리사이즈하고 패딩 적용.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        # KAIST 비율에 맞게 조정 (640:512 = 1.25:1)
        if new_shape <= 512:
            new_shape = (new_shape, new_shape)  # 작은 경우는 정사각형
        else:
            new_shape = (int(new_shape * 512 / 640), new_shape)  # height, width

    # KAIST 원본 비율 확인 (640x512)
    original_ratio = 640 / 512  # 1.25
    target_ratio = new_shape[1] / new_shape[0]  # width / height
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """
    Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.

    Details at https://arxiv.org/abs/2012.07177.
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """
    Applies cutout augmentation to an image with optional label adjustment, using random masks of varying sizes.

    Details at https://arxiv.org/abs/1708.04552.
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending images and labels.

    See https://arxiv.org/pdf/1710.09412.pdf for details.
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """
    Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold
    `ar_thr`, and area ratio threshold `area_thr`.

    box1(4,n) is before augmentation, box2(4,n) is after augmentation.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Initializes a LetterBox object for YOLOv5 image preprocessing with optional auto sizing and stride
        adjustment.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads input image `im` (HWC format) to specified dimensions, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round(imh, imw) - (h, w)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


# RGBT (Multispectral) Image Augmentation Functions
def augment_hsv_rgbt(imgs, labels=None, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV augmentation to RGBT images (only to RGB, keeps thermal unchanged)."""
    if not imgs or len(imgs) != 2:
        return imgs, labels
    
    # Apply HSV augmentation only to RGB image (index 1), keep thermal (index 0) unchanged
    rgb_img = imgs[1].copy()  # visible spectrum image (make copy to avoid modifying original)
    thermal_img = imgs[0]  # thermal image (keep unchanged)
    
    # Apply HSV augmentation to RGB image only
    if hgain or sgain or vgain:
        augment_hsv(rgb_img, hgain=hgain, sgain=sgain, vgain=vgain)
    
    return [thermal_img, rgb_img], labels


def flip_rgbt_images(imgs, labels, direction='lr'):
    """Flips RGBT images and adjusts labels accordingly."""
    if not imgs or len(imgs) != 2:
        return imgs, labels
    
    flipped_imgs = []
    for img in imgs:
        if direction == 'lr':  # left-right flip
            flipped_imgs.append(np.fliplr(img))
        elif direction == 'ud':  # up-down flip
            flipped_imgs.append(np.flipud(img))
        else:
            flipped_imgs.append(img)
    
    # Adjust labels
    if labels is not None and len(labels) > 0:
        if direction == 'lr':
            labels[:, 1] = 1 - labels[:, 1]  # flip x coordinates
        elif direction == 'ud':
            labels[:, 2] = 1 - labels[:, 2]  # flip y coordinates
    
    return flipped_imgs, labels


def random_perspective_rgbt(imgs, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, 
                          perspective=0.0, border=(0, 0)):
    """
    RGBT 이미지 쌍에 대한 랜덤 원근 변환 적용.
    
    Args:
        imgs: [thermal_img, visible_img] RGBT 이미지 쌍
        targets: 라벨들 (xyxy 형식, 6열: class, x1, y1, x2, y2, occlusion)
        segments: 세그먼트들
        degrees: 회전 각도 범위
        translate: 평행 이동 비율
        scale: 스케일 변화 범위
        shear: 전단 변환 각도
        perspective: 원근 변환 강도
        border: 경계 설정
        
    Returns:
        transformed_imgs: 변환된 RGBT 이미지들
        transformed_targets: 변환된 라벨들
    """
    height = imgs[0].shape[0] + border[0] * 2  # shape(h,w,c)
    width = imgs[0].shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -imgs[0].shape[1] / 2  # x translation (pixels)
    C[1, 2] = -imgs[0].shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            transformed_imgs = []
            for img in imgs:
                img_transformed = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                transformed_imgs.append(img_transformed)
        else:
            transformed_imgs = []
            for img in imgs:
                img_transformed = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                transformed_imgs.append(img_transformed)

        # Transform label coordinates
        n = len(targets)
        if n:
            # Create new targets with same number of columns as input
            new_targets = np.zeros_like(targets)
            new_targets[:, 0] = targets[:, 0]  # Copy class
            if targets.shape[1] > 5:
                new_targets[:, 5] = targets[:, 5]  # Copy occlusion if present
            
            # warp boxes (only coordinate columns 1-4)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_coords = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new_coords[:, [0, 2]] = new_coords[:, [0, 2]].clip(0, width)
            new_coords[:, [1, 3]] = new_coords[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new_coords.T, area_thr=0.01)
            new_targets = new_targets[i]
            new_targets[:, 1:5] = new_coords[i]

            return transformed_imgs, new_targets
        else:
            return transformed_imgs, targets
    else:
        return imgs, targets


# KAIST Dataset Specialized Augmentation Functions
def apply_kaist_horizontal_flip(image, labels=None, probability=0.5):
    """
    KAIST 데이터셋을 위한 수평 뒤집기 증강.
    바운딩 박스 좌표는 [클래스, 좌상단x, 좌상단y, 너비, 높이] 포맷으로 가정.
    
    Args:
        image: 입력 이미지 (numpy array)
        labels: [클래스, x_좌상단, y_좌상단, 너비, 높이] 형식의 라벨
        probability: 뒤집기를 적용할 확률 (기본값: 0.5)
        
    Returns:
        flipped_image: 뒤집힌(또는 원본) 이미지
        flipped_labels: 뒤집기 후 업데이트된 라벨
    """
    import torch
    
    # 원본 데이터 수정 방지를 위한 복사본 생성
    if isinstance(image, torch.Tensor):
        # Tensor의 형태를 확인하고 적절히 변환
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW 형태인 경우
            flipped_image = image.permute(1, 2, 0).clone().numpy()  # CHW -> HWC
        else:
            flipped_image = image.clone().numpy()
    else:
        flipped_image = image.copy()
        
    # 이미지 유효성 검사
    if flipped_image.size == 0 or len(flipped_image.shape) < 2:
        return flipped_image, labels
        
    flipped_labels = labels.copy() if labels is not None else np.zeros((0, 5))
    
    # 주어진 확률로 뒤집기 적용
    if random.random() < probability:
        flipped_image = np.fliplr(flipped_image)
        
        # 라벨이 존재하는 경우 업데이트
        if len(flipped_labels) > 0:
            # 좌상단 x 좌표 조정: 새로운 x_좌상단 = 1.0 - 원래_x_좌상단 - 너비
            flipped_labels[:, 1] = 1.0 - flipped_labels[:, 1] - flipped_labels[:, 3]
    
    return flipped_image, flipped_labels


def apply_kaist_hsv_augmentation(image, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    KAIST 데이터셋을 위한 HSV 색상 증강.
    
    Args:
        image: 입력 BGR 이미지
        hgain: 색상(H) 조정 강도 (기본값: 0.015)
        sgain: 채도(S) 조정 강도 (기본값: 0.7)
        vgain: 명도(V) 조정 강도 (기본값: 0.4)
        
    Returns:
        HSV 증강이 적용된 이미지
    """
    import torch
    
    # 이미지가 Tensor인 경우 numpy로 변환
    if isinstance(image, torch.Tensor):
        # Tensor의 형태를 확인하고 적절히 변환
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW 형태인 경우
            image_aug = image.permute(1, 2, 0).clone().numpy()  # CHW -> HWC
        else:
            image_aug = image.clone().numpy()
    else:
        image_aug = image.copy()
    
    # 이미지 유효성 검사
    if image_aug.size == 0 or len(image_aug.shape) < 3:
        return image_aug
    
    if hgain or sgain or vgain:
        # BGR 이미지를 HSV로 변환
        hsv = cv2.cvtColor(image_aug, cv2.COLOR_BGR2HSV)
        
        # 랜덤 증강 값 생성
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        
        # 각 HSV 채널에 증강 적용
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)
        
        # H 채널(색상) 조정
        hsv_h = (hsv_h * r[0]) % 180
        
        # S 채널(채도) 조정
        hsv_s = np.clip(hsv_s * r[1], 0, 255)
        
        # V 채널(명도) 조정
        hsv_v = np.clip(hsv_v * r[2], 0, 255)
        
        # 채널 병합
        hsv = cv2.merge([hsv_h.astype(np.uint8), hsv_s.astype(np.uint8), hsv_v.astype(np.uint8)])
        
        # HSV에서 BGR로 변환
        image_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image_aug


def apply_kaist_shear_transform(image, labels, shear_range=(-5, 5)):
    """
    KAIST 데이터셋을 위한 전단(shear) 변환과 바운딩 박스 좌표 업데이트.
    
    Args:
        image: 입력 이미지 (numpy array)
        labels: [클래스, x_좌상단, y_좌상단, 너비, 높이] 형식의 라벨
        shear_range: 랜덤 전단 각도 범위 (기본값: -5 ~ 5도)
        
    Returns:
        transformed_image: 전단 변환된 이미지
        transformed_labels: 변환 후 업데이트된 라벨
    """
    import torch
    
    # 이미지가 Tensor인 경우 numpy로 변환
    if isinstance(image, torch.Tensor):
        # Tensor의 형태를 확인하고 적절히 변환
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW 형태인 경우
            image = image.permute(1, 2, 0).clone().numpy()  # CHW -> HWC
        else:
            image = image.clone().numpy()
        
    # 이미지 유효성 검사
    if image.size == 0 or len(image.shape) < 2:
        return image, labels
        
    height, width = image.shape[:2]
    
    # 최소 크기 검사
    if height <= 0 or width <= 0:
        return image, labels
    
    # 변환을 위한 타겟 포맷 [클래스, x1, y1, x2, y2] 생성
    targets = np.zeros((len(labels), 5))
    if len(labels) > 0:
        # 클래스 ID 복사
        targets[:, 0] = labels[:, 0]
        
        # [x_좌상단, y_좌상단, 너비, 높이]를 [x1, y1, x2, y2]로 변환
        x_left = labels[:, 1]
        y_top = labels[:, 2]
        w = labels[:, 3]
        h = labels[:, 4]
        
        targets[:, 1] = x_left            # x1 (top-left x)
        targets[:, 2] = y_top             # y1 (top-left y)
        targets[:, 3] = x_left + w        # x2 (bottom-right x)
        targets[:, 4] = y_top + h         # y2 (bottom-right y)
    
    # 중심 이동 행렬 (이미지 중심으로 이동)
    C = np.eye(3)
    C[0, 2] = -width / 2  # x 이동
    C[1, 2] = -height / 2  # y 이동
    
    # 전단(Shear) 변환 행렬 (x, y 방향 모두 적용)
    S = np.eye(3)
    shear_x = random.uniform(shear_range[0], shear_range[1])
    shear_y = random.uniform(shear_range[0], shear_range[1])
    S[0, 1] = math.tan(shear_x * math.pi / 180)  # x 방향 전단 (도)
    S[1, 0] = math.tan(shear_y * math.pi / 180)  # y 방향 전단 (도)
    
    # 이동 행렬 (원래 위치로 복귀)
    T = np.eye(3)
    T[0, 2] = width / 2   # x 이동
    T[1, 2] = height / 2  # y 이동
    
    # 변환 행렬 결합
    M = T @ S @ C  # 오른쪽에서 왼쪽으로 순서대로 적용
    
    # 이미지에 변환 적용
    transformed_image = cv2.warpAffine(
        image, M[:2], (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)
    )
    
    # 라벨 변환
    n = len(targets)
    if n > 0:
        # 바운딩 박스 모서리 점 변환
        points = np.ones((n * 4, 3))
        # 각 박스의 4개 모서리 추출하고 픽셀 좌표로 변환
        points[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2) * [width, height]
        # 변환 적용
        points = points @ M.T
        # 박스 단위로 다시 재구성
        points = points[:, :2].reshape(n, 8)  # x1y1, x2y2, x1y2, x2y1
        
        # 변환된 좌표로 새로운 바운딩 박스 생성
        x = points[:, [0, 2, 4, 6]]  # 모든 x 좌표
        y = points[:, [1, 3, 5, 7]]  # 모든 y 좌표
        # 최소/최대를 사용하여 축 정렬 바운딩 박스 생성
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        # 이미지 경계로 클립
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        
        # 필터링: 너무 작거나 비정상적인 비율을 가진 박스 제거
        box_w = new[:, 2] - new[:, 0]
        box_h = new[:, 3] - new[:, 1]
        area = box_w * box_h
        original_area = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2]) * width * height
        ar = np.maximum(box_w / (box_h + 1e-16), box_h / (box_w + 1e-16))  # 종횡비
        i = (box_w > 4) & (box_h > 4) & (area / (original_area + 1e-16) > 0.1) & (ar < 10)
        
        # 필터 적용
        targets = targets[i]
        new = new[i]
        
        # 정규화된 [클래스, x, y, w, h] 형식으로 변환
        if len(new) > 0:
            transformed_labels = np.zeros((len(new), 5))
            transformed_labels[:, 0] = targets[:, 0]  # 클래스 ID
            
            # 픽셀 xyxy에서 정규화된 xywh로 변환
            transformed_labels[:, 1] = new[:, 0] / width                      # x_좌상단
            transformed_labels[:, 2] = new[:, 1] / height                     # y_좌상단
            transformed_labels[:, 3] = (new[:, 2] - new[:, 0]) / width        # 너비
            transformed_labels[:, 4] = (new[:, 3] - new[:, 1]) / height       # 높이
            
            return transformed_image, transformed_labels
    
    return transformed_image, labels


def apply_kaist_rgbt_augmentation(visible_img, thermal_img, labels, 
                                flip_prob=0.5, 
                                hsv_params=dict(hgain=0.015, sgain=0.7, vgain=0.4), 
                                shear_range=(-5, 5)):
    """
    KAIST 데이터셋의 RGB-T 이미지와 라벨에 증강 파이프라인 적용.
    
    Args:
        visible_img: RGB 가시광선 스펙트럼 이미지
        thermal_img: 열화상 이미지
        labels: [클래스, x_좌상단, y_좌상단, 너비, 높이] 형식의 라벨
        flip_prob: 수평 뒤집기 적용 확률
        hsv_params: HSV 증강 파라미터 (hgain, sgain, vgain)
        shear_range: 전단 변환 각도 범위 (도)
        
    Returns:
        visible_aug: 증강된 가시광선 이미지
        thermal_aug: 증강된 열화상 이미지
        labels_aug: 증강 후 업데이트된 라벨
    """
    import torch
    
    # 원본 데이터 수정 방지를 위한 복사본 생성
    if isinstance(visible_img, torch.Tensor):
        # Tensor의 형태를 확인하고 적절히 변환
        if visible_img.dim() == 3 and visible_img.shape[0] in [1, 3]:  # CHW 형태인 경우
            visible_aug = visible_img.permute(1, 2, 0).clone().numpy()  # CHW -> HWC
        else:
            visible_aug = visible_img.clone().numpy()
    else:
        visible_aug = visible_img.copy()
        
    if isinstance(thermal_img, torch.Tensor):
        # Tensor의 형태를 확인하고 적절히 변환
        if thermal_img.dim() == 3 and thermal_img.shape[0] in [1, 3]:  # CHW 형태인 경우
            thermal_aug = thermal_img.permute(1, 2, 0).clone().numpy()  # CHW -> HWC
        else:
            thermal_aug = thermal_img.clone().numpy()
    else:
        thermal_aug = thermal_img.copy()
        
    # 이미지 유효성 검사
    if visible_aug.size == 0 or thermal_aug.size == 0:
        return visible_aug, thermal_aug, labels
        
    labels_aug = labels.copy() if labels is not None else np.zeros((0, 5))
    
    # 1. 수평 뒤집기를 flip_prob 확률로 적용
    # 두 이미지에 동일한 랜덤 상태 사용
    flip_state = random.random() < flip_prob
    if flip_state:
        visible_aug = np.fliplr(visible_aug)
        thermal_aug = np.fliplr(thermal_aug)
        
        # 라벨 업데이트
        if len(labels_aug) > 0:
            # 좌상단 x 좌표 조정: 새로운 x_좌상단 = 1.0 - 원래_x_좌상단 - 너비
            labels_aug[:, 1] = 1.0 - labels_aug[:, 1] - labels_aug[:, 3]
    
    # 2. HSV 증강 적용 (가시광선 이미지에만 적용, 열화상은 변경 없음)
    visible_aug = apply_kaist_hsv_augmentation(
        visible_aug, 
        hgain=hsv_params.get('hgain', 0.015),
        sgain=hsv_params.get('sgain', 0.7),
        vgain=hsv_params.get('vgain', 0.4)
    )
    
    # 3. 두 이미지에 동일한 랜덤 시드로 전단 변환 적용
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    # 가시광선 이미지에 적용하고 변환된 라벨 얻기
    visible_aug, labels_aug = apply_kaist_shear_transform(visible_aug, labels_aug, shear_range)
    
    # 열화상 이미지에도 동일한 변환 적용하기 위해 랜덤 상태 초기화
    random.setstate(random_state)
    np.random.set_state(np_state)
    
    # 열화상 이미지에 동일한 변환 적용 (라벨은 이미 업데이트됨)
    thermal_aug, _ = apply_kaist_shear_transform(thermal_aug, labels_aug, shear_range)
    
    return visible_aug, thermal_aug, labels_aug


def visualize_kaist_augmentation(image, labels, class_names=None, class_colors=None):
    """
    시각화를 위해 이미지에 바운딩 박스 그리기.
    
    Args:
        image: 입력 이미지
        labels: [클래스, x_좌상단, y_좌상단, 너비, 높이] 형식의 라벨
        class_names: 클래스명 리스트 (선택사항)
        class_colors: 각 클래스별 색상 리스트 (선택사항)
        
    Returns:
        바운딩 박스가 그려진 이미지
    """
    if class_colors is None:
        class_colors = [
            (255, 0, 0),    # 빨간색
            (0, 255, 0),    # 초록색
            (0, 0, 255),    # 파란색
            (255, 255, 0)   # 노란색
        ]
    
    if class_names is None:
        class_names = ['person', 'cyclist', 'people', 'person?']
    
    img_with_boxes = image.copy()
    height, width = image.shape[:2]
    
    if len(labels) > 0:
        for label in labels:
            cls, x_norm, y_norm, w_norm, h_norm = label[:5]
            
            # 정규화된 좌표를 픽셀 좌표로 변환
            x1 = int(x_norm * width)
            y1 = int(y_norm * height)
            w = int(w_norm * width)
            h = int(h_norm * height)
            x2 = x1 + w
            y2 = y1 + h
            
            # 이미지 경계로 클립
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 클래스별 색상 선택
            color = class_colors[int(cls) % len(class_colors)]
            
            # 박스 그리기
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 클래스명 얻기
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
            
            # 텍스트 배경과 텍스트 그리기
            (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_boxes, 
                         (x1, y1 - text_h - 10), 
                         (x1 + text_w + 5, y1), 
                         color, -1)
            cv2.putText(img_with_boxes, class_name, 
                       (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_with_boxes


# KAIST Dataset Mosaic Augmentation Functions
def load_kaist_mosaic_with_individual_augmentation(dataset, index, s=640):
    """
    각 이미지에 개별적으로 증강을 적용한 후 4개 이미지를 2x2 모자이크로 조합.
    KAIST 1.25:1 비율 유지 (640x512).
    
    Args:
        dataset: KAIST RGBT 데이터셋
        index: 메인 이미지 인덱스
        s: 모자이크 기준 크기 (기본 640)
        
    Returns:
        mosaic_imgs: [thermal_mosaic, visible_mosaic]
        labels4: 모자이크 라벨들 (KAIST 형식)
    """
    labels4 = []
    
    # KAIST 비율에 맞는 셀 크기 계산 (1.25:1 ratio)
    cell_w = s  # 640
    cell_h = int(s * 0.8)  # 512 (640 * 0.8 = 512)
    
    # 모자이크 전체 크기
    mosaic_w = cell_w * 2  # 1280
    mosaic_h = cell_h * 2  # 1024
    
    # 4개 이미지용 인덱스 생성 (메인 이미지 + 3개 랜덤)
    indices = [index] + random.choices(range(len(dataset)), k=3)
    
    # 라벨이 있는 이미지를 우선적으로 선택
    labeled_indices = [i for i in range(len(dataset)) if len(dataset.labels[i]) > 0]
    if len(labeled_indices) >= 3:
        indices[1:] = random.sample(labeled_indices, 3)
    
    # 2x2 모자이크 이미지 초기화 (KAIST 비율)
    img4_thermal = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)  
    img4_visible = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
    
    # 각 이미지에 개별적으로 증강 적용 후 모자이크에 배치
    for i, idx in enumerate(indices):        
        # 1. 원본 이미지와 라벨 로드
        imgs, (h0s, w0s), (hs, ws) = dataset.load_image(idx)
        thermal_img, visible_img = imgs[0], imgs[1]
        h, w = hs[0], ws[0]
        labels = dataset.labels[idx].copy()
        
        # 2. 개별 이미지에 증강 적용 (수평 뒤집기, HSV, 전단 변환 등)
        if dataset.augment and random.random() < 0.5:
            # 수평 뒤집기
            thermal_img = np.fliplr(thermal_img)
            visible_img = np.fliplr(visible_img)
            
            if len(labels) > 0:
                # KAIST 형식 라벨 조정: [class, x_left_top, y_left_top, width, height, occlusion]
                labels[:, 1] = 1.0 - labels[:, 1] - labels[:, 3]  # x_left_top adjustment
        
        # HSV 증강 (visible 이미지에만)
        if dataset.augment and random.random() < 0.8:
            # HSV 증강을 직접 구현
            hsv = cv2.cvtColor(visible_img, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)
            
            # 랜덤 증강 값
            hgain, sgain, vgain = 0.015, 0.7, 0.4
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            
            # 각 채널 조정
            h_channel = (h_channel * r[0]) % 180
            s_channel = np.clip(s_channel * r[1], 0, 255)
            v_channel = np.clip(v_channel * r[2], 0, 255)
            
            # 다시 합치고 BGR로 변환
            hsv_new = cv2.merge([h_channel.astype(np.uint8), s_channel.astype(np.uint8), v_channel.astype(np.uint8)])
            visible_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        
        # 3. 이미지를 KAIST 비율 셀 크기로 리사이즈
        thermal_resized = cv2.resize(thermal_img, (cell_w, cell_h))
        visible_resized = cv2.resize(visible_img, (cell_w, cell_h))
        
        # 4. 2x2 그리드에서의 위치 결정 (KAIST 비율)
        if i == 0:  # top left
            y1, y2 = 0, cell_h
            x1, x2 = 0, cell_w
        elif i == 1:  # top right
            y1, y2 = 0, cell_h
            x1, x2 = cell_w, mosaic_w
        elif i == 2:  # bottom left
            y1, y2 = cell_h, mosaic_h
            x1, x2 = 0, cell_w
        elif i == 3:  # bottom right
            y1, y2 = cell_h, mosaic_h
            x1, x2 = cell_w, mosaic_w
        
        # 5. 모자이크에 이미지 배치
        img4_thermal[y1:y2, x1:x2] = thermal_resized
        img4_visible[y1:y2, x1:x2] = visible_resized
        
        # 6. 라벨 좌표 변환
        if len(labels) > 0:
            # 스케일링 팩터 계산 (원본 이미지 -> 셀 크기)
            scale_x = cell_w / w
            scale_y = cell_h / h
            
            # 라벨 픽셀 좌표로 변환 후 스케일링
            labels_scaled = labels.copy()
            labels_scaled[:, 1] = labels_scaled[:, 1] * w * scale_x + x1  # x_left_top
            labels_scaled[:, 2] = labels_scaled[:, 2] * h * scale_y + y1  # y_left_top
            labels_scaled[:, 3] = labels_scaled[:, 3] * w * scale_x       # width
            labels_scaled[:, 4] = labels_scaled[:, 4] * h * scale_y       # height
            
            # 모자이크 좌표계로 정규화
            labels_normalized = labels_scaled.copy()
            labels_normalized[:, 1] = labels_normalized[:, 1] / mosaic_w  # x_left_top
            labels_normalized[:, 2] = labels_normalized[:, 2] / mosaic_h  # y_left_top
            labels_normalized[:, 3] = labels_normalized[:, 3] / mosaic_w  # width
            labels_normalized[:, 4] = labels_normalized[:, 4] / mosaic_h  # height
            
            labels4.append(labels_normalized)
    
    # 7. 모든 라벨 결합
    if labels4:
        final_labels = np.concatenate(labels4, 0)
        
        # 경계 클리핑
        final_labels[:, 1:5] = np.clip(final_labels[:, 1:5], 0, 1)
        
        # 너무 작은 박스 제거
        box_areas = final_labels[:, 3] * final_labels[:, 4]  # width * height
        valid_mask = box_areas > 0.0001  # 최소 면적 기준
        final_labels = final_labels[valid_mask]
    else:
        final_labels = np.zeros((0, 6))
    
    # 8. 모자이크 이미지들 반환
    mosaic_imgs = [img4_thermal, img4_visible]
    
    return mosaic_imgs, final_labels


def apply_mosaic_final_augmentation_kaist(mosaic_imgs, labels, hyp):
    """
    모자이크 생성 후 최종 증강 적용 (KAIST 비율 유지)
    
    Args:
        mosaic_imgs: [thermal_mosaic, visible_mosaic]
        labels: 모자이크 라벨들
        hyp: 하이퍼파라미터
        
    Returns:
        final_imgs: 최종 증강된 이미지들
        final_labels: 최종 라벨들
    """
    # KAIST 비율 (640x512)로 리사이즈
    target_h, target_w = 512, 640
    final_imgs = []
    
    for img in mosaic_imgs:
        # 현재 모자이크 크기에서 KAIST 비율로 변환
        img_resized, ratio, pad = letterbox_kaist(img, (target_h, target_w), auto=False, scaleup=True)
        final_imgs.append(img_resized)
    
    # 라벨 좌표 조정
    if len(labels) > 0:
        # 스케일링 팩터
        scale_x = target_w / mosaic_imgs[0].shape[1]
        scale_y = target_h / mosaic_imgs[0].shape[0]
        
        # 라벨을 픽셀 좌표로 변환 후 스케일링
        labels_pixel = labels.copy()
        labels_pixel[:, 1] *= mosaic_imgs[0].shape[1] * scale_x  # x_left_top to pixel
        labels_pixel[:, 2] *= mosaic_imgs[0].shape[0] * scale_y  # y_left_top to pixel
        labels_pixel[:, 3] *= mosaic_imgs[0].shape[1] * scale_x  # width to pixel
        labels_pixel[:, 4] *= mosaic_imgs[0].shape[0] * scale_y  # height to pixel
        
        # 다시 정규화
        labels[:, 1] = labels_pixel[:, 1] / target_w  # x_left_top
        labels[:, 2] = labels_pixel[:, 2] / target_h  # y_left_top
        labels[:, 3] = labels_pixel[:, 3] / target_w  # width
        labels[:, 4] = labels_pixel[:, 4] / target_h  # height
        
        # 경계 클리핑
        labels[:, 1:5] = np.clip(labels[:, 1:5], 0, 1)
    
    return final_imgs, labels
    """
    KAIST RGBT 데이터셋용 개선된 2x2 모자이크 생성.
    
    Args:
        dataset: KAIST RGBT 데이터셋
        index: 메인 이미지 인덱스
        s: 모자이크 크기 (기본 640)
        
    Returns:
        mosaic_imgs: [thermal_mosaic, visible_mosaic]
        labels4: 모자이크 라벨들 (xyxy 형식)
    """
    labels4, segments4 = [], []
    
    # 4개 이미지용 인덱스 생성 (메인 이미지 + 3개 랜덤)
    indices = [index] + random.choices(range(len(dataset)), k=3)
    
    # 라벨이 있는 이미지를 우선적으로 선택
    labeled_indices = [i for i in range(len(dataset)) if len(dataset.labels[i]) > 0]
    if len(labeled_indices) >= 3:
        indices[1:] = random.sample(labeled_indices, 3)
    
    # KAIST 비율(1.25:1)을 고려한 모자이크 크기 계산
    # 각 셀의 크기: width = s, height = s * 0.8 (640/512 = 1.25, 역수는 0.8)
    cell_w = s
    cell_h = int(s * 0.8)  # KAIST 비율 적용
    mosaic_w = cell_w * 2
    mosaic_h = cell_h * 2
    
    # 2x2 모자이크 이미지 초기화 (KAIST 비율 유지)
    img4_thermal = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)  # 회색 배경
    img4_visible = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        # Load image
        imgs, (h0s, w0s), (hs, ws) = dataset.load_image(idx)
        thermal_img, visible_img = imgs[0], imgs[1]
        h, w = hs[0], ws[0]  # Use first image dimensions (both should be same)
        
        # 3. 이미지를 KAIST 비율 셀 크기로 리사이즈
        thermal_resized = cv2.resize(thermal_img, (cell_w, cell_h))
        visible_resized = cv2.resize(visible_img, (cell_w, cell_h))
        
        # 4. 2x2 그리드에서의 위치 결정 (KAIST 비율 적용)
        if i == 0:  # top left
            y1, y2 = 0, cell_h
            x1, x2 = 0, cell_w
        elif i == 1:  # top right
            y1, y2 = 0, cell_h
            x1, x2 = cell_w, mosaic_w
        elif i == 2:  # bottom left
            y1, y2 = cell_h, mosaic_h
            x1, x2 = 0, cell_w
        elif i == 3:  # bottom right
            y1, y2 = cell_h, mosaic_h
            x1, x2 = cell_w, mosaic_w
        
        # 모자이크에 이미지 배치
        img4_thermal[y1:y2, x1:x2] = thermal_resized
        img4_visible[y1:y2, x1:x2] = visible_resized
        
        # 라벨 처리
        labels, segments = dataset.labels[idx].copy(), dataset.segments[idx].copy()
        
        if labels.size > 0:
            # KAIST 라벨을 픽셀 좌표로 변환 (원본 이미지 기준)
            labels_pixel = labels.copy()
            labels_pixel[:, 1] = labels_pixel[:, 1] * w  # x_left_top
            labels_pixel[:, 2] = labels_pixel[:, 2] * h  # y_left_top  
            labels_pixel[:, 3] = labels_pixel[:, 3] * w  # width
            labels_pixel[:, 4] = labels_pixel[:, 4] * h  # height
            
            # xyxy 형식으로 변환
            labels_xyxy = np.zeros((len(labels_pixel), 6))
            labels_xyxy[:, 0] = labels_pixel[:, 0]  # class
            labels_xyxy[:, 1] = labels_pixel[:, 1]  # x1 (left)
            labels_xyxy[:, 2] = labels_pixel[:, 2]  # y1 (top)
            labels_xyxy[:, 3] = labels_pixel[:, 1] + labels_pixel[:, 3]  # x2 (right)
            labels_xyxy[:, 4] = labels_pixel[:, 2] + labels_pixel[:, 4]  # y2 (bottom)
            labels_xyxy[:, 5] = labels_pixel[:, 5] if labels_pixel.shape[1] > 5 else 0  # occlusion
            
            # 이미지 리사이징에 따른 스케일링
            scale_x = s / w
            scale_y = s / h
            
            labels_xyxy[:, 1] *= scale_x  # x1
            labels_xyxy[:, 2] *= scale_y  # y1
            labels_xyxy[:, 3] *= scale_x  # x2
            labels_xyxy[:, 4] *= scale_y  # y2
            
            # 모자이크 위치로 이동
            labels_xyxy[:, 1] += x1  # x1 offset
            labels_xyxy[:, 2] += y1  # y1 offset
            labels_xyxy[:, 3] += x1  # x2 offset
            labels_xyxy[:, 4] += y1  # y2 offset
        else:
            labels_xyxy = np.zeros((0, 6))
        
        # 세그먼트 처리
        for seg in segments:
            if len(seg) > 0:
                scale_x = s / w
                scale_y = s / h
                seg[:, 0] = seg[:, 0] * scale_x + x1  # x coordinates
                seg[:, 1] = seg[:, 1] * scale_y + y1  # y coordinates
            
        labels4.append(labels_xyxy)
        segments4.extend(segments)
    
    # 라벨들 합치기
    if labels4:
        labels4 = np.concatenate(labels4, 0)
        
        # 모자이크 경계로 클립 (좌표만)
        if len(labels4) > 0:
            labels4[:, [1, 3]] = labels4[:, [1, 3]].clip(0, 2 * s)  # x coordinates
            labels4[:, [2, 4]] = labels4[:, [2, 4]].clip(0, 2 * s)  # y coordinates
            
            # 너무 작은 박스 제거 (면적이 전체의 0.0001% 미만)
            w_box = labels4[:, 3] - labels4[:, 1]
            h_box = labels4[:, 4] - labels4[:, 2]
            area = w_box * h_box
            min_area = (2 * s) * (2 * s) * 0.000001  # 최소 면적
            valid_indices = area > min_area
            labels4 = labels4[valid_indices]
    else:
        labels4 = np.zeros((0, 6))
    
    # 모자이크 이미지들을 리스트로 반환
    mosaic_imgs = [img4_thermal, img4_visible]
    
    return mosaic_imgs, labels4


def apply_kaist_mosaic_augmentation(mosaic_imgs, labels, hyp):
    """
    KAIST 모자이크 이미지에 추가 증강 적용.
    
    Args:
        mosaic_imgs: [thermal_mosaic, visible_mosaic] 모자이크 이미지들
        labels: 모자이크 라벨들 (xyxy 형식)
        hyp: 하이퍼파라미터
        
    Returns:
        augmented_imgs: 증강된 모자이크 이미지들
        augmented_labels: 증강된 라벨들
    """
    # 랜덤 원근 변환 적용
    mosaic_imgs, labels = random_perspective_rgbt(
        mosaic_imgs,
        labels,
        degrees=hyp.get("degrees", 0),
        translate=hyp.get("translate", 0.1),
        scale=hyp.get("scale", 0.5),
        shear=hyp.get("shear", 0),
        perspective=hyp.get("perspective", 0.0),
        border=[-mosaic_imgs[0].shape[0] // 2, -mosaic_imgs[0].shape[1] // 2]  # 경계 설정
    )
    
    # KAIST 비율로 이미지 크기 조정 (640x512)
    from utils.augmentations import letterbox_kaist
    final_imgs = []
    target_h, target_w = 512, 640  # KAIST aspect ratio
    
    for img in mosaic_imgs:
        # letterbox로 KAIST 비율 유지하면서 크기 조정
        img_resized, ratio, pad = letterbox_kaist(img, (target_h, target_w), auto=False, scaleup=True)
        final_imgs.append(img_resized)
    
    # 라벨 좌표도 새로운 크기에 맞게 조정
    if len(labels) > 0:
        # 기존 모자이크 크기에서 새로운 크기로 스케일링
        scale_x = target_w / mosaic_imgs[0].shape[1]
        scale_y = target_h / mosaic_imgs[0].shape[0]
        
        labels[:, 1] *= scale_x  # x1
        labels[:, 2] *= scale_y  # y1
        labels[:, 3] *= scale_x  # x2
        labels[:, 4] *= scale_y  # y2
        
        # 경계 클리핑
        labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, target_w)
        labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, target_h)
    
    return final_imgs, labels


def convert_mosaic_labels_to_kaist_format(labels, img_w, img_h):
    """
    모자이크 라벨들을 KAIST center-based 정규화 형식으로 변환.
    
    Args:
        labels: xyxy 형식의 모자이크 라벨들 (class, x1, y1, x2, y2, occlusion)
        img_w: 최종 이미지 너비
        img_h: 최종 이미지 높이
        
    Returns:
        kaist_labels: [class, x_center, y_center, width, height, occlusion] 정규화된 라벨들
    """
    if len(labels) == 0:
        return np.zeros((0, 6))
    
    kaist_labels = np.zeros((len(labels), 6))
    kaist_labels[:, 0] = labels[:, 0]  # class
    
    # xyxy를 center-based xywh로 변환하고 정규화
    x_center = (labels[:, 1] + labels[:, 3]) / 2 / img_w
    y_center = (labels[:, 2] + labels[:, 4]) / 2 / img_h
    width = (labels[:, 3] - labels[:, 1]) / img_w
    height = (labels[:, 4] - labels[:, 2]) / img_h
    
    kaist_labels[:, 1] = x_center
    kaist_labels[:, 2] = y_center
    kaist_labels[:, 3] = width
    kaist_labels[:, 4] = height
    kaist_labels[:, 5] = labels[:, 5] if labels.shape[1] > 5 else 0  # occlusion
    
    # 좌표를 [0, 1] 범위로 클립
    kaist_labels[:, 1:5] = np.clip(kaist_labels[:, 1:5], 0, 1)
    
    return kaist_labels
