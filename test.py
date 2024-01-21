from M18K.Data.Dataset import M18KDataset
from M18K.Data.DataModule import M18KDataModule
from M18K.Models.MaskRCNN import MaskRCNN
from M18K.Models.FasterRCNN import FasterRCNN
from lightning import LightningModule, Trainer
from torchvision import transforms
from lightning.pytorch import loggers as pl_loggers
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.cuda import device_count
from lightning.pytorch.callbacks import LearningRateMonitor


def main(model_name="maskrcnn_resnet50_fpn_v2"):
    # Instantiate the data module
    t = transforms.ToTensor()
    # if model_name == "swin_v2_b":

    dm = M18KDataModule(batch_size=8, outputs="torch", depth=True)

    # Instantiate the model
    match model_name:

        case "maskrcnn_resnet50_fpn_v2":
            model = MaskRCNN(depth=True)
        case "maskrcnn_mobilenet_v3":
            model = MaskRCNN(backbone="mobilenet_v3")
        case "maskrcnn_efficientnet_b1":
            model = MaskRCNN(backbone="efficientnet_b1")

        case "fasterrcnn_resnet50_fpn_v2":
            model = FasterRCNN()
        case "fasterrcnn_mobilenet_v3":
            model = FasterRCNN(backbone="mobilenet_v3")
        case "fasterrcnn_efficientnet_b1":
            model = FasterRCNN(backbone="efficientnet_b1")
        case _:
            model = FasterRCNN()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"tests/{model_name}/")
    trainer = Trainer(max_epochs=1000, devices=1, log_every_n_steps=1, logger=tb_logger)

    # Train the model ⚡
    trainer.test(model, dm, ckpt_path="runs/maskrcnn_resnet50_fpn_v2_RGBD/maskrcnn_resnet50_fpn_v2_RGBD-epoch=265-val_loss=0.0687.ckpt")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    # parser.add_argument('model', type=str, help='model name',default="maskrcnn_resnet50_fpn_v2")
    # args = parser.parse_args()
    # model = args.model
    model = "maskrcnn_resnet50_fpn_v2"
    main(model)