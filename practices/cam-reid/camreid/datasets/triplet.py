from typing import Tuple

import numpy as np
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import DatasetFolder


class TripletDataset(Dataset):
    def __init__(self, dataset: DatasetFolder):
        super().__init__()
        self.dataset = dataset
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[Tensor, int], Tuple[Tensor, int], Tuple[Tensor, int]]:
        item_anchor = self.dataset.__getitem__(index)
        sample_anchor, target_anchor = item_anchor

        item_positive = self._choice_positive_item(target_anchor)
        item_negative = self._choice_negative_item(target_anchor)

        sample_positive, target_positive = item_positive
        sample_negative, target_negative = item_negative

        return (sample_anchor, sample_positive, sample_negative), (target_anchor, target_positive, target_negative)

    def _choice_item_by_target(self, target):
        indices = [
            idx for idx, item in enumerate(self.dataset.targets) if item == target
        ]
        return self.dataset.__getitem__(np.random.choice(indices))

    def _choice_positive_item(self, target_anchor):
        return self._choice_item_by_target(target_anchor)

    def _choice_negative_item(self, target_anchor):
        indices_possible = [
            idx for idx, cls in enumerate(self.dataset.classes) if idx != target_anchor
        ]
        target_negative = np.random.choice(indices_possible)
        return self._choice_item_by_target(target_negative)
