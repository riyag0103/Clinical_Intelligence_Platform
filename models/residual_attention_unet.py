import torch
import torch.nn as nn

# ----------------------------------
# RESIDUAL BLOCK
# ----------------------------------
class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # ----------------------------------
        # SKIP CONNECTION
        # ----------------------------------
        self.skip = nn.Sequential()

        if in_channels != out_channels:

            self.skip = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

    def forward(self, x):

        identity = self.skip(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out

# ----------------------------------
# ATTENTION BLOCK
# ----------------------------------
class AttentionBlock(nn.Module):

    def __init__(
        self,
        gate_channels,
        skip_channels,
        inter_channels
    ):

        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(

            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.W_skip = nn.Sequential(

            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.psi = nn.Sequential(

            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):

        gate_features = self.W_gate(gate)

        skip_features = self.W_skip(skip)

        attention = self.relu(
            gate_features + skip_features
        )

        attention = self.psi(attention)

        return skip * attention

# ----------------------------------
# RESIDUAL ATTENTION U-NET
# ----------------------------------
class ResidualAttentionUNet(nn.Module):

    def __init__(self):

        super(
            ResidualAttentionUNet,
            self
        ).__init__()

        # ----------------------------------
        # ENCODER
        # ----------------------------------
        self.enc1 = ResidualBlock(
            1,
            64
        )

        self.enc2 = ResidualBlock(
            64,
            128
        )

        self.enc3 = ResidualBlock(
            128,
            256
        )

        self.enc4 = ResidualBlock(
            256,
            512
        )

        self.pool = nn.MaxPool2d(2)

        # ----------------------------------
        # BOTTLENECK
        # ----------------------------------
        self.bottleneck = ResidualBlock(
            512,
            1024
        )

        # ----------------------------------
        # DECODER
        # ----------------------------------
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionBlock(
            512,
            512,
            256
        )

        self.dec4 = ResidualBlock(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionBlock(
            256,
            256,
            128
        )

        self.dec3 = ResidualBlock(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionBlock(
            128,
            128,
            64
        )

        self.dec2 = ResidualBlock(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionBlock(
            64,
            64,
            32
        )

        self.dec1 = ResidualBlock(
            128,
            64
        )

        # ----------------------------------
        # OUTPUT
        # ----------------------------------
        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # ----------------------------------
        # ENCODER
        # ----------------------------------
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ----------------------------------
        # BOTTLENECK
        # ----------------------------------
        b = self.bottleneck(
            self.pool(e4)
        )

        # ----------------------------------
        # DECODER 4
        # ----------------------------------
        d4 = self.up4(b)

        e4_att = self.att4(
            d4,
            e4
        )

        d4 = torch.cat(
            [d4, e4_att],
            dim=1
        )

        d4 = self.dec4(d4)

        # ----------------------------------
        # DECODER 3
        # ----------------------------------
        d3 = self.up3(d4)

        e3_att = self.att3(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, e3_att],
            dim=1
        )

        d3 = self.dec3(d3)

        # ----------------------------------
        # DECODER 2
        # ----------------------------------
        d2 = self.up2(d3)

        e2_att = self.att2(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, e2_att],
            dim=1
        )

        d2 = self.dec2(d2)

        # ----------------------------------
        # DECODER 1
        # ----------------------------------
        d1 = self.up1(d2)

        e1_att = self.att1(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, e1_att],
            dim=1
        )

        d1 = self.dec1(d1)

        # ----------------------------------
        # OUTPUT
        # ----------------------------------
        return torch.sigmoid(
            self.final(d1)
        )