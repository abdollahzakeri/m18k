from M18K.Data.Dataset import M18KDataset
from M18K.Data.DataModule import M18KDataModule
from M18K.Models.TorchVision import MaskRCNN_ResNet50
from lightning import LightningModule, Trainer
from torchvision import transforms
from lightning.pytorch import loggers as pl_loggers
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def main(model_name="maskrcnn_resnet50_fpn"):
    # Instantiate the data module
    t = transforms.ToTensor()
    # if model_name == "swin_v2_b":
    
    dm = M18KDataModule(batch_size=4)

    
    # Instantiate the model
    model = MaskRCNN_ResNet50()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath=f"runs/{model_name}/",
        filename= model_name+"-{epoch:02d}-{val_loss:.3f}",
    )

    #early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=50, verbose=False, mode="max")

    # Initialize a trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"runs/{model_name}/")
    trainer = Trainer(max_epochs=1000,devices=8,log_every_n_steps=1,logger=tb_logger,callbacks=[checkpoint_callback])

    # Train the model ⚡
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    parser.add_argument('model', type=str, help='model name')
    args = parser.parse_args()
    model = args.model
    main(model)