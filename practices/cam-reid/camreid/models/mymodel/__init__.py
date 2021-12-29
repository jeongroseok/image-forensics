from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics.classification.accuracy

from .components import Decoder, Discriminator, Encoder, Head


class MyModel(pl.LightningModule):
    class __HPARAMS:
        img_shape: Tuple[int, int, int]
        num_classes: int
        hidden_dim: int
        num_residual: int
        num_downsample: int
        num_upsample: int
        num_pretraining: int
        lr: float
        beta1: float
        beta2: float
        weight_decay: float
        lambda_adv: float
        lambda_id: float
        lambda_img_recon: float
        lambda_code_recon: float
        pretrained_encoder: bool

    hparams: __HPARAMS

    def __init__(
        self,
        img_shape: Tuple[int, int, int] = (3, 256, 256),
        num_classes: int = 6,  # 6 for LIRMMBase256
        hidden_dim: int = 16,
        num_residual: int = 2,
        num_downsample: int = 1,
        num_upsample: int = 1,
        num_pretraining: int = 10,
        lr=0.0001,
        beta1=0,
        beta2=0.999,
        weight_decay=0.0005,
        lambda_adv: float = 1,
        lambda_id: float = 0.5,
        lambda_img_recon: float = 5,
        lambda_code_recon: float = 5,
        pretrained_encoder: bool = True,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()

        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        # encoder
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_residual=num_residual,
            num_downsample=num_downsample,
            pretrained=pretrained_encoder,
            progress=pretrained_encoder,
        )
        fingerprint_dim = self.encoder.fingerprint_encoder.out_features

        self.head = Head(fingerprint_dim, num_classes=6, hidden_dim=128)

        # decoder
        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            num_residual=num_residual,
            num_upsample=num_upsample,
            fingerprint_dim=fingerprint_dim,
        )

        # discriminator
        self.discriminator = Discriminator(img_shape)

    def configure_optimizers(self):
        hparams = self.hparams
        opt_head = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
        )
        opt_gen = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.head.parameters()),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
        )
        return [opt_head, opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        opt_head, opt_gen, opt_disc = self.optimizers()

        samples, targets = batch
        smp_anc, smp_pos, smp_neg = samples
        tgt_anc, tgt_pos, tgt_neg = targets

        # extract content and fingerprint features
        cnt_anc, fp_anc = self.encoder(smp_anc)
        cnt_pos, fp_pos = self.encoder(smp_pos)
        cnt_neg, fp_neg = self.encoder(smp_neg)

        # pretraining phase for head
        if self.current_epoch < self.hparams.num_pretraining:
            tgt_anc_hat = self.head(fp_anc)
            tgt_pos_hat = self.head(fp_pos)
            tgt_neg_hat = self.head(fp_neg)

            loss = (
                self.criterion_ce(tgt_anc_hat, tgt_anc)
                + self.criterion_ce(tgt_pos_hat, tgt_pos)
                + self.criterion_ce(tgt_neg_hat, tgt_neg)
            ) / 3

            accuracy = (
                self.metric_accuracy(tgt_anc_hat, tgt_anc)
                + self.metric_accuracy(tgt_pos_hat, tgt_pos)
                + self.metric_accuracy(tgt_neg_hat, tgt_neg)
            ) / 3

            self.log(f"pretrain/loss", loss)
            self.log(f"pretrain/accuracy", accuracy)

            opt_head.zero_grad()
            self.manual_backward(loss)
            opt_head.step()

            return

        # generation phase for autoencoder and gan
        smp_anc_anc = self.decoder(cnt_anc, fp_anc)
        smp_anc_pos = self.decoder(cnt_anc, fp_pos)
        smp_anc_neg = self.decoder(cnt_anc, fp_neg)
        smp_neg_anc = self.decoder(cnt_neg, fp_anc)

        ## self-identity generation phase
        tgt_anc_hat = self.head(fp_anc)
        tgt_pos_hat = self.head(fp_pos)
        tgt_neg_hat = self.head(fp_neg)
        loss_img_recon = (
            self.criterion_l1(smp_anc_anc, smp_anc)
            + self.criterion_l1(smp_anc_pos, smp_anc)
        ) / 2
        loss_id = (
            self.criterion_ce(tgt_anc_hat, tgt_anc)
            + self.criterion_ce(tgt_pos_hat, tgt_pos)
            + self.criterion_ce(tgt_neg_hat, tgt_neg)
        ) / 3
        loss_self_identity = (loss_img_recon * self.hparams.lambda_img_recon) + (
            loss_id
        )

        accuracy = (
            self.metric_accuracy(tgt_anc_hat, tgt_anc)
            + self.metric_accuracy(tgt_pos_hat, tgt_pos)
            + self.metric_accuracy(tgt_neg_hat, tgt_neg)
        ) / 3

        self.log(f"train/self-identity/accuracy", accuracy)
        self.log(f"train/self-identity/img_recon", loss_img_recon)
        self.log(f"train/self-identity/id", loss_id)

        ## cross-identity generation phase
        cnt_neg_re, fp_neg_re = self.encoder(smp_anc_neg)
        cnt_anc_re, fp_anc_re = self.encoder(smp_neg_anc)
        tgt_anc_re_hat = self.head(fp_anc_re)
        tgt_neg_re_hat = self.head(fp_neg_re)
        loss_code_recon = (
            self.criterion_l1(fp_anc_re, fp_anc) + self.criterion_l1(fp_neg_re, fp_neg)
        ) / 2
        loss_id = (
            self.criterion_ce(tgt_anc_re_hat, tgt_anc)
            + self.criterion_ce(tgt_neg_re_hat, tgt_neg)
        ) / 2

        accuracy = (
            self.metric_accuracy(tgt_anc_re_hat, tgt_anc)
            + self.metric_accuracy(tgt_neg_re_hat, tgt_neg)
        ) / 2

        loss_cross_identity = (loss_code_recon * self.hparams.lambda_code_recon) + (
            loss_id * self.hparams.lambda_id
        )
        self.log(f"train/cross-identity/accuracy", accuracy)
        self.log(f"train/cross-identity/code_recon", loss_code_recon)
        self.log(f"train/cross-identity/id", loss_id)

        ## discrimination(generator)
        d_anc_anc = self.discriminator(smp_anc_anc)
        d_anc_pos = self.discriminator(smp_anc_pos)
        d_neg_anc = self.discriminator(smp_neg_anc)
        loss_adv = (
            (
                self.criterion_mse(d_anc_anc, torch.ones_like(d_anc_anc))
                + self.criterion_mse(d_anc_pos, torch.ones_like(d_anc_pos))
                + self.criterion_mse(d_neg_anc, torch.ones_like(d_neg_anc))
            )
            / 3
        ) * self.hparams.lambda_adv
        self.log(f"train/discrimination(generator)", loss_adv)

        loss = loss_self_identity + loss_cross_identity + loss_adv

        opt_gen.zero_grad()
        self.manual_backward(loss)
        opt_gen.step()

        ## discrimination(discriminator)
        d_anc_anc = self.discriminator(smp_anc_anc.detach())
        d_anc_pos = self.discriminator(smp_anc_pos.detach())
        d_neg_anc = self.discriminator(smp_neg_anc.detach())
        d_anc = self.discriminator(smp_anc.detach())
        d_pos = self.discriminator(smp_pos.detach())
        d_neg = self.discriminator(smp_neg.detach())
        loss = (
            self.criterion_mse(d_anc_anc, torch.zeros_like(d_anc_anc))
            + self.criterion_mse(d_anc_pos, torch.zeros_like(d_anc_pos))
            + self.criterion_mse(d_neg_anc, torch.zeros_like(d_neg_anc))
            + self.criterion_mse(d_anc, torch.ones_like(d_anc))
            + self.criterion_mse(d_pos, torch.ones_like(d_pos))
            + self.criterion_mse(d_neg, torch.ones_like(d_neg))
        ) / 6
        self.log(f"train/discrimination(discriminator)", loss)

        opt_disc.zero_grad()
        self.manual_backward(loss)
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        smp_anc, smp_pos, smp_neg = samples
        tgt_anc, tgt_pos, tgt_neg = targets

        # extract fingerprint features
        _, fp_anc = self.encoder(smp_anc)
        _, fp_pos = self.encoder(smp_pos)
        _, fp_neg = self.encoder(smp_neg)

        # pretraining phase for head
        tgt_anc_hat = self.head(fp_anc)
        tgt_pos_hat = self.head(fp_pos)
        tgt_neg_hat = self.head(fp_neg)

        loss = (
            self.criterion_ce(tgt_anc_hat, tgt_anc)
            + self.criterion_ce(tgt_pos_hat, tgt_pos)
            + self.criterion_ce(tgt_neg_hat, tgt_neg)
        ) / 3

        accuracy = (
            self.metric_accuracy(tgt_anc_hat, tgt_anc)
            + self.metric_accuracy(tgt_pos_hat, tgt_pos)
            + self.metric_accuracy(tgt_neg_hat, tgt_neg)
        ) / 3

        self.log(f"val/loss", loss, on_epoch=True, on_step=False)
        self.log(f"val/accuracy", accuracy, on_epoch=True, on_step=False)
