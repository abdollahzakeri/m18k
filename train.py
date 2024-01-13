import torch.cuda

from M18K.Data.Dataset import M18KDataset
from M18K.Data.DataModule import M18KDataModule
from M18K.Models.MaskRCNN import MaskRCNN
from M18K.Models.FasterRCNN import FasterRCNN
from M18K.Models.HuggingFace import HuggingFaceGenericModel
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
    if model_name[:4] in ["mask","fast"]:
        dm = M18KDataModule(batch_size=4,outputs="torch")
    else:
        dm = M18KDataModule(batch_size=2, outputs="hf")

    # Instantiate the model
    match model_name:

        case "maskrcnn_resnet50_fpn_v2":
            model = MaskRCNN(depth=True)
        case "maskrcnn_mobilenet_v3":
            model = MaskRCNN(backbone="mobilenet_v3")
        case "maskrcnn_efficientnet_b1":
            model = MaskRCNN(backbone="efficientnet_b1")
        case "maskrcnn_densenet121":
            model = MaskRCNN(backbone="densenet121")

        case "fasterrcnn_resnet50_fpn_v2":
            model = FasterRCNN()
        case "fasterrcnn_mobilenet_v3":
            model = FasterRCNN(backbone="mobilenet_v3")
        case "fasterrcnn_efficientnet_b1":
            model = FasterRCNN(backbone="efficientnet_b1")
        case "fasterrcnn_densenet121":
            model = FasterRCNN(backbone="densenet121")

        case "hf_mask2former":
            model = HuggingFaceGenericModel()

        case _:
            model = FasterRCNN()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath=f"runs/{model_name}/",
        filename= model_name+"-{epoch:02d}-{val_loss:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    #early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=50, verbose=False, mode="max")

    # Initialize a trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"runs/{model_name}/")
    trainer = Trainer(max_epochs=1000, devices=torch.cuda.device_count(), log_every_n_steps=1, logger=tb_logger, callbacks=[checkpoint_callback, lr_monitor])

    # Train the model ⚡
    trainer.fit(model, dm)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    # parser.add_argument('model', type=str, help='model name')
    # args = parser.parse_args()
    # model = args.model
    model = "maskrcnn_resnet50_fpn_v2"
    main(model)