from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

class MaskRCNN(TorchVisionGenericModel):
    def __init__(self, backbone="resnet_50"):
        super().__init__()
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
