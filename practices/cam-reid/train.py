import pytorch_lightning as pl
from camreid.datamodules.lirmmbase import TripletLIRMMBase256DataModule
from camreid.models.mymodel import MyModel

DATA_DIR = fr"C:\Users\jeong\Desktop\machine-learning\image-forensics\practices\cam-reid\data\LIRMMBase256x256"


def main(args=None):
    # fix seed
    pl.seed_everything(0)

    # define dataset, model, trainer
    datamodule = TripletLIRMMBase256DataModule(
        data_dir=DATA_DIR, batch_size=8, num_workers=4, persistent_workers=True
    )

    model = MyModel(pretrained_encoder=True, num_pretraining=10)

    trainer = pl.Trainer(
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=100,
    )

    # fit
    trainer.fit(model, datamodule=datamodule)
    pass


if __name__ == "__main__":
    main()
