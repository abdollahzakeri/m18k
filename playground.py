from ultralytics import YOLO
from M18K.Data.Dataset import M18KDataset
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from skimage.measure import find_contours
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import tqdm

model = YOLO('../YOLO/runs/segment/train6/weights/best.pt')
root = "M18K/Data/test"
ds = M18KDataset(root="M18K/Data/test",transforms=None,train=False)
results = {"images":[],"masks":[],"boxes":[],"labels":[],"scores":[]}
for i in tqdm.tqdm(range(len(ds))):
    d = ds[i]
    image, target = d
    filename = os.path.join(root,target["image_name"])
    output = model(filename,save=True)[0]
    results["images"] += [target["image_name"]]
    results["boxes"] += [output.boxes.xyxy.cpu().numpy()]
    results["scores"] += [output.boxes.conf.cpu().numpy()]
    results["masks"] += [F.interpolate(output.masks.data.unsqueeze(1), size=(720, 1280), mode='bilinear').squeeze(1).cpu().numpy()]
    results["labels"] += [(output.boxes.cls+1).cpu().numpy()]

def create_coco_annotations(images, masks, bboxes, labels, scores, categories):

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 0

    for image_id, (image_path, image_masks, bbox, label, score) in enumerate(zip(images, masks, bboxes, labels, scores), 0):
        with Image.open("M18K/Data/test/"+image_path) as img:
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

coco_gt = COCO("M18K/Data/test/_annotations.coco.json")
categories = [{'id': 0, 'name': 'Mushrooms', 'supercategory': 'none'}, {'id': 1, 'name': 'BB', 'supercategory': 'Mushrooms'}, {'id': 2, 'name': 'WB', 'supercategory': 'Mushrooms'}]
pred = create_coco_annotations(results["images"],results["masks"],results["boxes"],results["labels"], results["scores"],categories)
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
path = f"tests/yolov8/"
os.makedirs(path, exist_ok=True)
df = pd.DataFrame.from_dict(results, orient='index', columns=["Segmentation", "Detection"])
df.to_csv(os.path.join(path, "results.csv"))