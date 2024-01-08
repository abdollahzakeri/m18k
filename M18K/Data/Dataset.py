import os
import cv2

import numpy as np
import torch
import torchvision.transforms
from pycocotools.coco import COCO

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import albumentations as A

class M18KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, outputs="torch", train=True):
        self.root = root
        self.transforms = transforms
        self.annotations = COCO(os.path.join(root, "_annotations.coco.json"))
        self.train = train
        self.outputs = outputs

    def augmentation(self, image, masks):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(0.1),
            A.Affine([0.8,1.2],[0.8,1.2],None,[-360, 360],[-15, 15],fit_output=True,p=0.8),
            A.Resize(720,1280)
        ])

        transformed = transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_mask = transformed['masks']
        
        return transformed_image,transformed_mask

    def __getitem__(self, idx):
        # load images and masks
        image_object = self.annotations.imgs[idx]
        img_path = image_object["file_name"]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        masks = self.annotations.loadAnns(self.annotations.getAnnIds([image_object["id"]]))
        mask_list = [self.annotations.annToMask(mask) for mask in masks]
        
        if self.train:
            img, mask_list = self.augmentation(img,mask_list)
        
        # tensor of shape [#objects,h,w] of binary masks
        binary_masks = torch.tensor(np.dstack(mask_list), dtype=torch.uint8).permute([2, 0, 1])

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(binary_masks)

        # there is only one class
        labels = torch.tensor([mask["category_id"] for mask in masks], dtype=torch.int64)

        image_id = idx
        area = torch.tensor([mask["area"] for mask in masks], dtype=torch.float32)

        iscrowd = torch.tensor([mask["iscrowd"] for mask in masks], dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        #img = tv_tensors.Image(img)
        img = torchvision.transforms.ToTensor()(img)

        # if self.transforms is not None and self.train:
        #     img, target = self.transforms(img, target)

        if self.outputs == "torch":
            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            target["masks"] = tv_tensors.Mask(binary_masks)
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            return (img, target)
        elif self.outputs == "hf":
            pixel_mask = torch.ones((h, w)).int()  # Convert to int for binary mask
            return {
                "pixel_values": img,
                "pixel_mask": pixel_mask,
                "mask_labels": binary_masks,
                "class_labels": labels
            }
    def __len__(self):
        return len(self.annotations.imgs)