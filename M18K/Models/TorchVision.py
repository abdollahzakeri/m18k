import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR


class TorchVisionGenericModel(LightningModule):
    def __init__(self):
        super(TorchVisionGenericModel, self).__init__()
        self.num_classes = 3

    def forward(self, x):
        images, targets = x
        output = self.model(images, targets)

        return output

    def training_step(self, batch):
        losses = self(batch)
        self.model.train()
        loss = sum(losses.values()) / len(losses)
        self.log('train_loss_classifier', losses["loss_classifier"], prog_bar=True, sync_dist=True)
        self.log('train_loss_box_reg', losses["loss_box_reg"], prog_bar=True, sync_dist=True)
        self.log('train_trailoss_objectnessn_loss', losses["loss_objectness"], prog_bar=True, sync_dist=True)
        self.log('train_loss_rpn_box_reg', losses["loss_rpn_box_reg"], prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        self.model.train()
        losses = self(batch)
        loss = sum(losses.values()) / len(losses)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optim, gamma=0.95, step_size=10)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
