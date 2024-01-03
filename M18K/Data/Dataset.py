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


class M18KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        self.annotations = COCO(os.path.join(root, "_annotations.coco.json"))

    def __getitem__(self, idx):
        # load images and masks
        image_object = self.annotations.imgs[idx]
        img_path = image_object["file_name"]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = cv2.imread(os.path.join(self.root, img_path))
        masks = self.annotations.loadAnns(self.annotations.getAnnIds([image_object["id"]]))


        # tensor of shape [#objects,h,w] of binary masks
        binary_masks = torch.tensor(np.dstack([self.annotations.annToMask(mask) for mask in masks]),
                                    dtype=torch.uint8).permute([2, 0, 1])

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

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(binary_masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return (img, target)
    def __len__(self):
        return len(self.annotations.imgs)