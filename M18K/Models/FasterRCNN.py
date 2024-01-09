from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.utils import draw_bounding_boxes
import cv2
import time
import os
import numpy as np
import torch

class FasterRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50"):
        super().__init__()
        self.backbone = backbone

        match backbone:
            case "resnet_50":
                self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

            case "mobilenet_v3":
                self.model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
            
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

                self.model = torchvision.models.detection.FasterRCNN(
                    backbone,
                    num_classes=self.num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler
                )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)


    def test_step(self, batch):
        images,targets = batch
        results = self.model(images)
        for image, r in zip(images, results):
            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
            colors = [(255,0,0) if x == 1 else (0,0,255) for x in r["labels"]]
            visualized = draw_bounding_boxes(image,boxes= r["boxes"],colors=colors,width=2)
            visualized = visualized.permute(1, 2, 0).numpy().astype(np.uint8)
            path = f"tests/fasterrcnn_{self.backbone}/visualizations/"
            os.makedirs(path,exist_ok=True)
            cv2.imwrite(f"{path}{time.time()}.jpg", cv2.cvtColor(visualized,cv2.COLOR_BGR2RGB))