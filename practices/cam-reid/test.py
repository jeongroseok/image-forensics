import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from camreid.datamodules.lirmmbase import TripletLIRMMBase256DataModule
from camreid.models.mymodel import MyModel

DATA_DIR = fr"C:\Users\jeong\Desktop\machine-learning\image-forensics\practices\cam-reid\data\LIRMMBase256x256"
MODEL_PATH = fr"C:\Users\jeong\Desktop\machine-learning\image-forensics\practices\cam-reid\epoch=26-step=2726.ckpt"


def main():
    pl.seed_everything(0)
    datamodule = TripletLIRMMBase256DataModule(
        data_dir=DATA_DIR, batch_size=8, num_workers=0, persistent_workers=True
    )
    datamodule.setup()

    model = MyModel.load_from_checkpoint(MODEL_PATH)

    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        samples, targets = batch
        sample_anchor, _, sample_negative = samples
        content_code, fingerprint_code = model.encoder(sample_anchor)
        sample_anchor_re = model.decoder(content_code, fingerprint_code)
        grid = make_grid(
            torch.concat([sample_anchor, sample_anchor_re]),
            datamodule.batch_size,
            normalize=True,
        )
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
        break


if __name__ == "__main__":
    main()
