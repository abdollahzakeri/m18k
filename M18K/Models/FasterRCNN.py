from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
class FasterRCNN(TorchVisionGenericModel):
    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        self.num_classes = 3  # 1 class (person) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
