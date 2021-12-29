from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transform_lib

from ..datasets.lirmmbase import TripletLIRMMBase256


class TripletLIRMMBase256DataModule(pl.LightningDataModule):
    name = "triplet_lirmmbase_256"

    def __init__(
        self,
        data_dir: str,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.prefetch_factor = prefetch_factor

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = TripletLIRMMBase256(
            self.data_dir, transform=self._default_transforms()
        )
        self.dataset_train, self.dataset_val, self.dataset_test = self._split_dataset(
            dataset
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_test)

    def _default_transforms(self) -> Callable:
        return transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def _split_dataset(self, dataset: Dataset) -> Dataset:
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        splits = [train_len, val_len, test_len]

        return random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
