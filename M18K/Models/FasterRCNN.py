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
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd

class FasterRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50"):
        super().__init__()
        self.results = {"images": [], "boxes": [], "labels": [], "scores": []}
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

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Example Usage
        images,targets = batch

        ds_root = "M18K/Data/test"
        self.results["images"] += [os.path.join(ds_root,x["image_name"]) for x in targets]
        # Replace below with actual masks
        self.results["boxes"] += [r["boxes"].cpu().numpy() for r in outputs]  # List of bounding boxes for each image
        self.results["labels"] += [r["labels"].cpu().numpy() for r in outputs]  # List of labels for each bounding box
        self.results["scores"] += [r["scores"].cpu().numpy() for r in outputs]
    def on_test_end(self) -> None:
        # categories = [{"id": 1, "name": "BB"}, {"id": 2, "name": "WB"}]
        # coco_pred = COCO("temp.json")
        coco_gt = COCO("M18K/Data/test/_annotations.coco.json")
        categories = [{'id': 0, 'name': 'Mushrooms', 'supercategory': 'none'}, {'id': 1, 'name': 'BB', 'supercategory': 'Mushrooms'}, {'id': 2, 'name': 'WB', 'supercategory': 'Mushrooms'}]
        pred = self.create_coco_annotations(self.results["images"],self.results["boxes"],self.results["labels"], self.results["scores"],categories)
        coco_pred = COCO()
        coco_pred.imgs = pred["images"]
        coco_pred.anns = pred["annotations"]
        coco_pred.cats = pred["categories"]
        coco_pred.dataset = pred
        coco_pred.createIndex()
        print("Detection Results: ")
        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")  # Use 'segm' for segmentation evaluation
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
        path = f"tests/fasterrcnn_{self.backbone}/"
        os.makedirs(path, exist_ok=True)
        df = pd.DataFrame.from_dict(results, orient='index', columns=["Detection"])
        df.to_csv(os.path.join(path, "results.csv"))

    def create_coco_annotations(self, images, bboxes, labels, scores, categories):

        coco_output = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        annotation_id = 0

        for image_id, (image_path, bbox, label, score) in enumerate(zip(images, bboxes, labels, scores), 0):
            with Image.open(image_path) as img:
                width, height = img.size

            coco_output["images"].append({
                "id": image_id,
                "file_name": image_path,
                "width": width,
                "height": height
            })

            for box, lbl, sc in zip(bbox, label, score):
                # Convert mask to polygon
                x1, y1, x2, y2 = box
                h, w = (x2-x1),(y2-y1)
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": lbl,
                    "bbox": [x1,y1,h,w],
                    "area": h * w,
                    "score": sc,
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
            visualized = draw_bounding_boxes(image,boxes= r["boxes"],colors=colors,width=2)
            visualized = visualized.permute(1, 2, 0).numpy().astype(np.uint8)
            path = f"tests/fasterrcnn_{self.backbone}/visualizations/"
            os.makedirs(path,exist_ok=True)
            cv2.imwrite(f"{path}{name}.jpg", cv2.cvtColor(visualized,cv2.COLOR_BGR2RGB))

        return results