from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from torchvision import transforms

from M18K.Data.DataModule import M18KDataModule
from M18K.Models.FasterRCNN import FasterRCNN
from M18K.Models.MaskRCNN import MaskRCNN


def main(model_name="maskrcnn_resnet50_fpn_v2"):
    t = transforms.ToTensor()

    dm = M18KDataModule(batch_size=8, outputs="torch", depth=True)

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

    trainer.test(model, dm,
                 ckpt_path="runs/maskrcnn_resnet50_fpn_v2_RGBD/maskrcnn_resnet50_fpn_v2_RGBD-epoch=265-val_loss=0.0687.ckpt")


if __name__ == '__main__':
    model = "maskrcnn_resnet50_fpn_v2"
    main(model)
