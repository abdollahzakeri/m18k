import os

import cv2

from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torchvision
import torch
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt
import numpy as np
import time

class MaskRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50"):
        super().__init__()
        self.backbone = backbone
        match backbone:
            case "resnet_50":
                self.model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
            
            case "mobilenet_v3":
                backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
                backbone.out_channels = 960
                anchor_generator = AnchorGenerator(
                    sizes=((32, 64, 128, 256, 512),),
                    aspect_ratios=((0.5, 1.0, 2.0),)
                )

                roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=['0'],
                    output_size=7,
                    sampling_ratio=2
                )

                # put the pieces together inside a Faster-RCNN model
                self.model = torchvision.models.detection.MaskRCNN(
                    backbone,
                    num_classes=self.num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler
                )
            
            case "efficientnet_b1":
                backbone = torchvision.models.efficientnet_b1(weights="DEFAULT").features
                backbone.out_channels = 1280
                anchor_generator = AnchorGenerator(
                    sizes=((32, 64, 128, 256, 512),),
                    aspect_ratios=((0.5, 1.0, 2.0),)
                )

                roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=['0'],
                    output_size=7,
                    sampling_ratio=2
                )

                # put the pieces together inside a Faster-RCNN model
                self.model = torchvision.models.detection.MaskRCNN(
                    backbone,
                    num_classes=self.num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler
                )
            
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )

    def training_step(self, batch):
        losses = self(batch)
        self.model.train()
        loss = sum(losses.values()) / len(losses)

        self.log('train_loss_classifier', losses["loss_classifier"], prog_bar=True, sync_dist=True)
        self.log('train_loss_box_reg', losses["loss_box_reg"], prog_bar=True, sync_dist=True)
        self.log('train_loss_mask', losses["loss_mask"], prog_bar=True, sync_dist=True)
        self.log('train_trailoss_objectnessn_loss', losses["loss_objectness"], prog_bar=True, sync_dist=True)
        self.log('train_loss_rpn_box_reg', losses["loss_rpn_box_reg"], prog_bar=True, sync_dist=True)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch):
        images,targets = batch
        results = self.model(images)
        for image, r in zip(images, results):
            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
            colors = [(255,0,0) if x == 1 else (0,0,255) for x in r["labels"]]
            masks = r["masks"].squeeze(1) > 0.5
            visualized = draw_bounding_boxes(image,boxes= r["boxes"],colors=colors,width=2)
            visualized = draw_segmentation_masks(visualized,masks=masks,alpha=0.6,colors=colors)
            visualized = visualized.permute(1, 2, 0).numpy().astype(np.uint8)
            path = f"tests/maskrcnn_{self.backbone}/visualizations/"
            os.makedirs(path,exist_ok=True)
            cv2.imwrite(f"{path}{time.time()}.jpg", cv2.cvtColor(visualized,cv2.COLOR_BGR2RGB))