import lightning as pl
from .Dataset import M18KDataset
from torch.utils.data import DataLoader

class M18KDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=4, transform=None):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def collate_fn(self,batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        ds = M18KDataset("M18K/Data/train", transforms=self.transform, train=True)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)

    def val_dataloader(self):
        ds = M18KDataset("M18K/Data/valid", transforms=self.transform, train=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn)

    def test_dataloader(self):
        ds = M18KDataset("M18K/Data/test", transforms=self.transform, train=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn)