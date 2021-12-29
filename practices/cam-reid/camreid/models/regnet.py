import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import regnet_y_400mf
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics.classification.accuracy import Accuracy


class RegNetY_400MF(pl.LightningModule):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        model = regnet_y_400mf(True, True)
        self.body = create_feature_extractor(model, return_nodes=["flatten"])

        input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.body(input)
        in_features = output["flatten"].shape[1]
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

        self.metric_accuracy = Accuracy()
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.body(x)["flatten"]
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.body.parameters()) + list(self.fc.parameters()), lr=0.001
        )

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        output = self.forward(anchor[0])
        loss = self.criterion_ce(output, anchor[1])
        acc = self.metric_accuracy(output, anchor[1])
        self.log(f"train/acc", acc)
        self.log(f"train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        output = self.forward(anchor[0])
        loss = self.criterion_ce(output, anchor[1])
        acc = self.metric_accuracy(output, anchor[1])
        self.log(f"val/acc", acc)
        self.log("val/loss", loss)