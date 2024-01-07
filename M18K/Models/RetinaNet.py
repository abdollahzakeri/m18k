from torchvision.models.detection import retinanet_resnet50_fpn_v2
from .TorchVision import TorchVisionGenericModel
from torchvision.models.detection.retinanet import RetinaNetHead


class FasterRCNN(TorchVisionGenericModel):
    def __init__(self):
        super().__init__()
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT")

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 3  # 1 class (person) + background
        # get number of input features for the classifier
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.head.classification_head.conv[0][0].in_channels
        model.head = RetinaNetHead(in_channels, num_anchors, num_classes)

    def training_step(self, batch):
        losses = self(batch)
        self.model.train()
        loss = sum(losses.values()) / len(losses)

        self.log('train_loss_classifier', losses["classification"], prog_bar=True, sync_dist=True)
        self.log('train_loss_box_reg', losses["bbox_regression"], prog_bar=True, sync_dist=True)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss