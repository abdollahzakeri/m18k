import argparse

import torch.cuda
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

from M18K.Data.DataModule import M18KDataModule
from M18K.Models.FasterRCNN import FasterRCNN
from M18K.Models.HuggingFace import HuggingFaceGenericModel
from M18K.Models.MaskRCNN import MaskRCNN


def main(model_name="maskrcnn_resnet50_fpn_v2", batch_size=2, depth=True):
    if model_name[:4] in ["mask", "fast"]:
        dm = M18KDataModule(batch_size=batch_size, outputs="torch", depth=depth)
    else:
        dm = M18KDataModule(batch_size=batch_size, outputs="hf", depth=depth)

    match model_name:
        case "maskrcnn_resnet50_fpn_v2":
            model = MaskRCNN(depth=depth)
        case "maskrcnn_mobilenet_v3":
            model = MaskRCNN(backbone="mobilenet_v3")
        case "maskrcnn_efficientnet_b1":
            model = MaskRCNN(backbone="efficientnet_b1")
        case "maskrcnn_densenet121":
            model = MaskRCNN(backbone="densenet121")
        case "maskrcnn_resnet101":
            model = MaskRCNN(backbone="resnet_101", depth=depth)
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
        dirpath=f"runs/{model_name}" + ("_RGBD" if depth else "") + "/",
        filename=model_name + ("_RGBD" if depth else "") + "-{epoch:02d}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"runs/{model_name}" + ("_RGBD" if depth else "") + "/")
    trainer = Trainer(max_epochs=1000, devices=torch.cuda.device_count(), log_every_n_steps=1, logger=tb_logger,
                      callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for processing (default: 32)')
    parser.add_argument('--depth', action='store_true', help='include depth (set this flag to use depth)')
    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    depth = args.depth
    main(model, batch_size, depth)
