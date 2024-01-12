import os
from typing import Any

import cv2
import pandas
from lightning.pytorch.utilities.types import STEP_OUTPUT

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
import json
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import timm
from TimmWrapper import TimmBackboneWrapper

class MaskRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50", depth=False):
        super().__init__()
        self.results = {"images":[],"masks":[],"boxes":[],"labels":[],"scores":[]}
        self.backbone = backbone
        self.depth = depth
        match backbone:
            case "resnet_50":
                if self.depth:
                    backbone = TimmBackboneWrapper('resnet50', pretrained=True, in_chans=4)
                    self.model = self.create_model(backbone, backbone.out_channels)
                else:
                    self.model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

            case "mobilenet_v3":
                backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
                self.model = self.create_model(backbone, 960)

            
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

            case "densenet121":
                backbone = torchvision.models.densenet121(weights="DEFAULT").features
                backbone.out_channels = 1024
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

    def create_model(self, backbone, out_channels):
        backbone.out_channels = out_channels
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
        model = torchvision.models.detection.MaskRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        return model
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Example Usage
        images,targets = batch

        ds_root = "M18K/Data/test"
        self.results["images"] += [os.path.join(ds_root,x["image_name"]) for x in targets]
        # Replace below with actual masks
        self.results["masks"] += [(r["masks"].squeeze(1) > 0.5).cpu().numpy() for r in outputs]
        self.results["boxes"] += [r["boxes"].cpu().numpy() for r in outputs]  # List of bounding boxes for each image
        self.results["labels"] += [r["labels"].cpu().numpy() for r in outputs]  # List of labels for each bounding box
        self.results["scores"] += [r["scores"].cpu().numpy() for r in outputs]
    def on_test_end(self) -> None:
        # categories = [{"id": 1, "name": "BB"}, {"id": 2, "name": "WB"}]
        # coco_pred = COCO("temp.json")
        coco_gt = COCO("M18K/Data/test/_annotations.coco.json")
        categories = [{'id': 0, 'name': 'Mushrooms', 'supercategory': 'none'}, {'id': 1, 'name': 'BB', 'supercategory': 'Mushrooms'}, {'id': 2, 'name': 'WB', 'supercategory': 'Mushrooms'}]
        pred = self.create_coco_annotations(self.results["images"],self.results["masks"],self.results["boxes"],self.results["labels"], self.results["scores"],categories)
        coco_pred = COCO()
        coco_pred.imgs = pred["images"]
        coco_pred.anns = pred["annotations"]
        coco_pred.cats = pred["categories"]
        coco_pred.dataset = pred
        coco_pred.createIndex()
        print("Segmentation Results: ")
        coco_eval = COCOeval(coco_gt, coco_pred, "segm")  # Use 'segm' for segmentation evaluation
        max_dets = [50, 100, 150]
        coco_eval.params.maxDets = max_dets
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = [
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets[1]} ]",
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets={max_dets[2]} ]",
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets={max_dets[2]} ]",
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets={max_dets[2]} ]",
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={max_dets[2]} ]",
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets={max_dets[2]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets[0]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets[1]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets[2]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets={max_dets[2]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets={max_dets[2]} ]",
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets={max_dets[2]} ]"
        ]
        results = {}
        for m, v in zip(metrics, coco_eval.stats):
            results[f"{m}"] = [v]
        print("Detection Results: ")
        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")  # Use 'segm' for segmentation evaluation
        coco_eval.params.maxDets = [50, 100, 150]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for m, v in zip(metrics, coco_eval.stats):
            results[f"{m}"] += [v]
        path = f"tests/maskrcnn_{self.backbone}/"
        os.makedirs(path, exist_ok=True)
        df = pd.DataFrame.from_dict(results, orient='index', columns=["Segmentation", "Detection"])
        df.to_csv(os.path.join(path, "results.csv"))
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

    def create_coco_annotations(self, images, masks, bboxes, labels, scores, categories):

        coco_output = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        annotation_id = 0

        for image_id, (image_path, image_masks, bbox, label, score) in enumerate(zip(images, masks, bboxes, labels, scores), 0):
            with Image.open(image_path) as img:
                width, height = img.size

            coco_output["images"].append({
                "id": image_id,
                "file_name": image_path,
                "width": width,
                "height": height
            })

            for mask, box, lbl, sc in zip(image_masks, bbox, label, score):
                # Convert mask to polygon
                contours = find_contours(mask, 0.5)
                segmentation = []
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation.append(contour.ravel().tolist())
                x1, y1, x2, y2 = box
                h, w = (x2-x1),(y2-y1)
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": lbl,
                    "bbox": [x1,y1,h,w],
                    "area": h * w,
                    "score": sc,
                    "segmentation": segmentation,
                    "iscrowd": 0
                })
                annotation_id += 1

        return coco_output

    def test_step(self, batch):
        images, targets = batch

        results = self.model(images)
        for image, r, t in zip(images, results, targets):
            name = t["image_name"]
            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
            colors = [(255,0,0) if x == 1 else (0,0,255) for x in r["labels"]]
            masks = r["masks"].squeeze(1) > 0.5
            visualized = draw_bounding_boxes(image,boxes= r["boxes"],colors=colors,width=2)
            visualized = draw_segmentation_masks(visualized,masks=masks,alpha=0.6,colors=colors)
            visualized = visualized.permute(1, 2, 0).numpy().astype(np.uint8)
            path = f"tests/maskrcnn_{self.backbone}/visualizations/"
            os.makedirs(path,exist_ok=True)
            cv2.imwrite(f"{path}{name}.jpg", cv2.cvtColor(visualized,cv2.COLOR_BGR2RGB))

        return results