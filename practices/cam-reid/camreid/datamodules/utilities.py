import pytorch_lightning
import torchvision.utils


def make_grid_by_first_train_batch(
    datamodule: pytorch_lightning.LightningDataModule,
    tensor_extractor: callable,
    padding: int = 2,
    normalize: bool = False,
):
    dataloader = datamodule.train_dataloader()
    batch_size = dataloader.batch_size
    batch = next(dataloader)

    tensor = tensor_extractor(batch)
    grid = torchvision.utils.make_grid(
        tensor, nrow=batch_size, padding=padding, normalize=normalize
    )
    return grid
