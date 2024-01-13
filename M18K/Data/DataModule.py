import lightning as pl
from .Dataset import M18KDataset
from torch.utils.data import DataLoader
import torch
class M18KDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=4, transform=None, outputs="torch", depth=True):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.outputs = outputs
        self.collate = self.collate_fn_torch if outputs == "torch" else self.collate_fn_hf
        self.depth = depth
    def setup(self, stage=None):
        pass

    def collate_fn_torch(self, batch):
        return tuple(zip(*batch))

    def collate_fn_hf(self, batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
        class_labels = [example["class_labels"] for example in batch]
        mask_labels = [example["mask_labels"] for example in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels,
                "mask_labels": mask_labels}

    def train_dataloader(self):
        ds = M18KDataset("M18K/Data/train", transforms=self.transform, train=True, outputs=self.outputs, depth=self.depth)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.collate)

    def val_dataloader(self):
        ds = M18KDataset("M18K/Data/valid", transforms=self.transform, train=False, outputs=self.outputs, depth=self.depth)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate)

    def test_dataloader(self):
        ds = M18KDataset("M18K/Data/test", transforms=self.transform, train=False, outputs=self.outputs, depth=self.depth)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate)