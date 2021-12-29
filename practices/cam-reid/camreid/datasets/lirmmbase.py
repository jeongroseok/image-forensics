import bisect
import os
from typing import Callable, List, Optional

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder

if __name__ == "__main__":
    from triplet import TripletDataset
else:
    from .triplet import TripletDataset

class LIRMMBase256(Dataset):
    __datasets: List[ImageFolder]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(Dataset, self).__init__()
        names = ["Dresden", "Columbia", "RAISE"]
        self.__datasets = [
            ImageFolder(os.path.join(root, name), transform, target_transform)
            for name in names
        ]
        self.cumulative_sizes = self.cumsum(self.__datasets)

        self.classes, self.class_to_idx = self.__find_classes()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.__map_item_index(idx)
        dataset = self.__datasets[dataset_idx]
        sample, target = dataset[sample_idx]
        target = self.__map_target(dataset_idx, target)
        return sample, target

    def __map_item_index(self, idx: int):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    def __map_target(self, dataset_idx, target):
        dataset = self.__datasets[dataset_idx]
        target_str = dataset.classes[target]
        return self.class_to_idx[target_str]

    def __find_classes(self):
        classes = []
        for dataset in self.__datasets:
            for klass in dataset.classes:
                classes.append(klass)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __targets(self):
        for dataset_idx, dataset in enumerate(self.__datasets):
            for target in dataset.targets:
                yield self.__map_target(dataset_idx, target)

    def __samples(self):
        for dataset in self.__datasets:
            for path, class_index in dataset.samples:
                target_str = dataset.classes[class_index]
                yield path, self.class_to_idx[target_str]

    @property
    def targets(self):
        return list(self.__targets())

    @property
    def samples(self):
        return list(self.__samples())


class TripletLIRMMBase256(TripletDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(LIRMMBase256(root, transform, target_transform))


def main(args=None):
    DATA_DIR = fr"C:\Users\jeong\Desktop\machine-learning\image-forensics\practices\cam-reid\data\LIRMMBase256x256"
    dataset = TripletLIRMMBase256(DATA_DIR)

    for idx, item in enumerate(dataset):
        anchor, positive, negative = item
        print(
            f"target: {anchor[1], positive[1], negative[1]}, class: {dataset.classes[anchor[1]], dataset.classes[positive[1]], dataset.classes[negative[1]]}"
        )


if __name__ == "__main__":
    main()
