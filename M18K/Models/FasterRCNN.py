from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision


class FasterRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50"):
        super().__init__()
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