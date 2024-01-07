from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
class MaskRCNN(TorchVisionGenericModel):
    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

        self.num_classes = 3  # 1 class (person) + background
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
