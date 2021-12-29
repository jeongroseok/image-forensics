import torch
import torch.nn.functional as F
from kornia.filters import GaussianBlur2d
from torch import nn
from torchvision.models import regnet_y_400mf
from torchvision.models.feature_extraction import create_feature_extractor

from ..cyclegan.components import Discriminator as Discriminator_CYCLEGAN
from ..munit.components import Decoder as Decoder_MUNIT
from ..munit.components import ContentEncoder as ContentEncoder_MUNIT


class HighPassFilter(nn.Module):
    """
    Deep learning for steganalysis is better than a rich model with an ensemble classifier, and is natively robust to the cover source-mismatch
    We observed that CNNs do not converge without this preliminary high-pass filtering.
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(
            [
                [-1, +2, -2, +2, -1],
                [+2, -6, +8, -6, +2],
                [-2, +8, -12, +8, -2],
                [+2, -6, +8, -6, +2],
                [-1, +2, -2, +2, -1],
            ],
            dtype=torch.float,
        )
        self.weights /= 12.0
        self.weights = self.weights.repeat(3, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = F.conv2d(x, self.weights.to(x.device))
        return x


class LowPassFilter(GaussianBlur2d):
    """
    Source camera identification for heavily JPEG compressed low resolution still images
    The same conclusion is reached for the other three camera models used in this research and is easily explained using \sigma = 0.6
    """

    def __init__(self, sigma: float = 0.6):
        kernel_size = int(sigma * 5)
        super().__init__((kernel_size, kernel_size), (sigma, sigma))


class Head(nn.Sequential):
    """
    Deep learning for steganalysis is better than a rich model with an ensemble classifier, and is natively robust to the cover source-mismatch
    """

    def __init__(self, in_features: int, num_classes: int = 6, hidden_dim: int = 1000):
        super().__init__(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(),
        )


class FingerprintEncoder(nn.Module):
    def __init__(
        self, pretrained: bool = False, progress: bool = True, *args: any, **kwargs: any
    ) -> None:
        super().__init__()
        self.filter = HighPassFilter()
        regnet = regnet_y_400mf(pretrained, progress, **kwargs)
        self.body = create_feature_extractor(regnet, return_nodes=["flatten"])

    @property
    def out_features(self):
        input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = self.body(input)
        return output["flatten"].shape[1]

    def forward(self, x):
        x = self.filter(x)
        return self.body(x)["flatten"]


class ContentEncoder(ContentEncoder_MUNIT):
    def __init__(self, in_channels=3, hidden_dim=64, num_residual=2, num_downsample=2):
        super().__init__(
            in_channels=in_channels,
            dim=hidden_dim,
            n_residual=num_residual,
            n_downsample=num_downsample,
        )


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dim=64,
        num_residual=2,
        num_downsample=2,
        **kwargs: any
    ) -> None:
        super().__init__()
        self.content_encoder = ContentEncoder(
            in_channels, hidden_dim, num_residual, num_downsample
        )
        self.fingerprint_encoder = FingerprintEncoder(**kwargs)

    def forward(self, x):
        content_code = self.content_encoder(x)
        fingerprint_code = self.fingerprint_encoder(x)
        return content_code, fingerprint_code


class Decoder(Decoder_MUNIT):
    """
    Multimodal Unsupervised Image-to-Image Translation
    """

    def __init__(
        self,
        out_channels=3,
        hidden_dim=64,
        num_residual=2,
        num_upsample=2,
        fingerprint_dim=8,
    ):
        super().__init__(
            out_channels=out_channels,
            dim=hidden_dim,
            n_residual=num_residual,
            n_upsample=num_upsample,
            style_dim=fingerprint_dim,
        )

    def forward(self, content_code, fingerprint_code):
        return super().forward(content_code, fingerprint_code)


class Discriminator(Discriminator_CYCLEGAN):
    """
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
    """

    def __init__(self, img_shape):
        super().__init__(img_shape)
