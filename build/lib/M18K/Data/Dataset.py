import copy
import os

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F
import gdown
import zipfile
from scipy.ndimage import median_filter

class M18KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, outputs="torch", train=True, depth=True, improve_depth=True):
        self.download_dataset()
        parent = os.path.abspath(os.getcwd())
        self.root = os.path.join(parent,root)
        self.transforms = transforms

        self.annotations = COCO(os.path.join(self.root, "_annotations.coco.json"))
        self.train = train
        self.outputs = outputs
        self.depth = depth
        self.improve_depth = improve_depth

    def download_dataset(self):
        parent_folder = os.path.abspath(os.getcwd())
        train_folder = os.path.join(parent_folder,"train")
        test_folder = os.path.join(parent_folder, "test")
        valid_folder = os.path.join(parent_folder, "valid")
        if os.path.exists(train_folder) and os.path.exists(test_folder) & os.path.exists(valid_folder):
            return
        gdown.download("https://drive.google.com/uc?id=1iPANwP1k6tbz1EZRdG2qbst1tqqYsZ_x","m18k.zip")
        with zipfile.ZipFile(os.path.join(parent_folder,"m18k.zip"), 'r') as zip_ref:
            zip_ref.extractall(parent_folder)
        os.remove(os.path.join(parent_folder,"m18k.zip"))
        os.rmdir(os.path.join(parent_folder,"__MACOSX"))

    def augmentation(self, image, masks, depth=None, improve_depth=True):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(0.1),
            A.Affine([0.8, 1.2], [0.8, 1.2], None, [-360, 360], [-15, 15], fit_output=True, p=0.8),
            A.Resize(720, 1280)
        ])
        if depth is not None:
            transformed = transform(image=image, masks=masks, mask=depth)
            transformed_image = transformed['image']
            transformed_mask = transformed['masks']
            transformed_depth = transformed['mask']
            return transformed_image, transformed_mask, transformed_depth
        else:
            transformed = transform(image=image, masks=masks)
            transformed_image = transformed['image']
            transformed_mask = transformed['masks']
            return transformed_image, transformed_mask

    def __getitem__(self, idx):

        image_object = self.annotations.imgs[idx]
        img_path = image_object["file_name"]

        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.depth:
            d = np.load(os.path.join(self.root, img_path[:-4].replace("images","depth") + ".npy"))
            percentiles = np.percentile(d, range(1, 101))
            differences = np.diff(percentiles)
            max_diff_index = np.argmax(differences)
            threshold = percentiles[max_diff_index]
            if max_diff_index > 95 and threshold > 0:
                d = np.clip(d, 0.0, threshold)
            if self.improve_depth:
                d[d == 0] = np.mean(d)
                depth_map = median_filter(d, size=15)
        h, w, _ = img.shape
        masks = self.annotations.loadAnns(self.annotations.getAnnIds([image_object["id"]]))
        mask_list = [self.annotations.annToMask(mask) for mask in masks]
        if self.train:
            if self.depth:
                img, mask_list, d = self.augmentation(img, mask_list, d)
            else:
                img, mask_list = self.augmentation(img, mask_list)

        binary_masks = torch.tensor(np.dstack(mask_list), dtype=torch.uint8).permute([2, 0, 1])

        boxes = masks_to_boxes(binary_masks)

        labels = torch.tensor([mask["category_id"] for mask in masks], dtype=torch.int64)
        image_id = idx
        area = torch.tensor([mask["area"] for mask in masks], dtype=torch.float32)
        iscrowd = torch.tensor([mask["iscrowd"] for mask in masks], dtype=torch.int64)

        img = torchvision.transforms.ToTensor()(img)
        if self.depth:
            img_org = copy.deepcopy(img)
            d = torch.from_numpy(d / np.max(d)).float()
            img = torch.cat((img, d.unsqueeze(0)), dim=0)

        if self.outputs == "torch":
            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            target["masks"] = tv_tensors.Mask(binary_masks)
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_name"] = image_object["file_name"]
            if self.depth:
                target["depth"] = d
                target["img_org"] = img_org
            return (img, target)
        elif self.outputs == "hf":
            non_black_mask = img != 0
            pixel_mask = non_black_mask.all(dim=0).int()
            return {
                "pixel_values": img.float(),
                "pixel_mask": pixel_mask.long(),
                "mask_labels": binary_masks.float(),
                "class_labels": labels.long()
            }

    def __len__(self):
        return len(self.annotations.imgs)
