from typing import Callable, Optional, Tuple

import numpy as np
from torch import Tensor
from torchvision.datasets import ImageFolder


class SPC2018(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.idx_to_class = {y: x for x, y in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, int], Tuple[Tensor, int], Tuple[Tensor, int]]:
        item_anchor = super().__getitem__(index)
        _, target_anchor = item_anchor

        item_positive = self._choice_positive_item(target_anchor)
        item_negative = self._choice_negative_item(target_anchor)

        return item_anchor, item_positive, item_negative

    def _choice_item_by_target(self, target):
        indices = [idx for idx, item in enumerate(self.targets) if item == target]
        return super().__getitem__(np.random.choice(indices))

    def _choice_positive_item(self, target_anchor):
        return self._choice_item_by_target(target_anchor)

    def _choice_negative_item(self, target_anchor):
        indices_possible = [
            idx for idx in self.idx_to_class.keys() if idx != target_anchor
        ]
        target_negative = np.random.choice(indices_possible)
        return self._choice_item_by_target(target_negative)
