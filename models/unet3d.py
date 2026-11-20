
import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.pool = nn.MaxPool3d(2)

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.enc2 = DoubleConv3D(32, 64)
        self.enc3 = DoubleConv3D(64, 128)

        # Dropout in deeper encoder
        self.dropout_enc3 = nn.Dropout3d(p=0.10)

        # Bottleneck
        self.bottleneck = DoubleConv3D(128, 256)

        # Stronger dropout at bottleneck
        self.dropout_bottleneck = nn.Dropout3d(p=0.20)

        # Decoder
        self.up3 = nn.ConvTranspose3d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv3D(256, 128)

        self.up2 = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv3D(128, 64)

        self.up1 = nn.ConvTranspose3d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv3D(64, 32)

        # Final segmentation layer
        self.output = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        enc1 = self.enc1(x)

        enc2 = self.enc2(
            self.pool(enc1)
        )

        enc3 = self.enc3(
            self.pool(enc2)
        )

        enc3 = self.dropout_enc3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool(enc3)
        )

        bottleneck = self.dropout_bottleneck(
            bottleneck
        )

        # Decoder
        dec3 = self.up3(bottleneck)

        dec3 = self.dec3(
            torch.cat((dec3, enc3), dim=1)
        )

        dec2 = self.up2(dec3)

        dec2 = self.dec2(
            torch.cat((dec2, enc2), dim=1)
        )

        dec1 = self.up1(dec2)

        dec1 = self.dec1(
            torch.cat((dec1, enc1), dim=1)
        )

        # Raw logits (no sigmoid)
        return self.output(dec1)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = UNet3D().to(device)

    # Test in training mode
    model.train()

    x = torch.rand(
        1, 1, 8, 128, 128,
        device=device
    )

    output = model(x)

    print("\nTraining mode test")
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    # Test in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(x)

    print("\nEvaluation mode test")
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    assert output.shape == x.shape, (
        "Input and output shapes should match."
    )

    print("\n3D U-Net with Dropout3D test passed!")


if __name__ == "__main__":
    main()
