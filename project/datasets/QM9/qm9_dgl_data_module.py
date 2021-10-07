from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.QM9.qm9_dgl_dataset import QM9DGLDataset
from project.utils.utils import RandomRotation, collate


class QM9DGLDataModule(LightningDataModule):
    """QM9 data module for DGL with PyTorch."""

    # Dataset partition instantiations
    qm9_train = None
    qm9_val = None
    qm9_test = None

    def __init__(self, data_dir='datasets/QM9/QM9_data.pt', task='homo', batch_size=32, num_dataloader_workers=1):
        super().__init__()

        # Dataset parameters
        self.data_dir = data_dir
        self.task = task

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    @property
    def num_node_features(self) -> int:
        return self.qm9_train.num_atom_features

    @property
    def num_pos_features(self) -> int:
        return 3

    @property
    def num_coord_features(self) -> int:
        return 3

    @property
    def num_edge_features(self) -> int:
        return self.qm9_train.num_bonds

    @property
    def num_fourier_features(self) -> int:
        return 0

    @property
    def std(self) -> int:
        return self.qm9_train.std

    @property
    def mean(self) -> int:
        return self.qm9_train.mean

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.qm9_train = QM9DGLDataset(self.data_dir, self.task, mode='train', transform=RandomRotation())
        self.qm9_val = QM9DGLDataset(self.data_dir, self.task, mode='valid')
        self.qm9_test = QM9DGLDataset(self.data_dir, self.task, mode='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.qm9_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=collate)
