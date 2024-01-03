import torch.nn.functional as F
from lightning import LightningModule, Trainer
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from torch.nn import BCELoss
from torchmetrics import Accuracy
class MaskRCNN_ResNet50(LightningModule):
    def __init__(self, model_name="mobilenet_v2", embedding_size=512):
        super(MaskRCNN_ResNet50, self).__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        num_classes = 3  # 1 class (person) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

        # self.criterion = BCELoss()
        # self.acc = Accuracy(task="binary")

    def forward(self, x):
        images, targets = x
        output = self.model(images, targets)
        #print(output)
        return output

    def training_step(self, batch):
        losses = self(batch)
        self.model.train()
        loss = sum(losses.values()) / len(losses)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        self.model.train()
        losses = self(batch)
        loss = sum(losses.values()) / len(losses)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)