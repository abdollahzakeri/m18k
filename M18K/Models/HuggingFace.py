import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from transformers import Mask2FormerForUniversalSegmentation


class HuggingFaceGenericModel(LightningModule):
    def __init__(self):
        super(HuggingFaceGenericModel, self).__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance",
                                                                         id2label={0: 'M', 1: 'BB', 2: 'WB'},
                                                                         ignore_mismatched_sizes=True)

    def forward(self, batch):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        return outputs

    def training_step(self, batch):
        loss = self(batch).loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        loss = self(batch).loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = StepLR(optim, gamma=0.95, step_size=2)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
