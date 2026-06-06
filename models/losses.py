import torch
import torch.nn as nn

# ----------------------------------
# DICE LOSS
# ----------------------------------
class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2. * intersection + self.smooth
        ) / (
            predictions.sum()
            + targets.sum()
            + self.smooth
        )

        return 1 - dice

# ----------------------------------
# FOCAL LOSS
# ----------------------------------
class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=0.8,
        gamma=2
    ):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.bce = nn.BCELoss(
            reduction='none'
        )

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        pt = torch.exp(-bce_loss)

        focal_loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * bce_loss
        )

        return focal_loss.mean()

# ----------------------------------
# TVERSKY LOSS
# ----------------------------------
class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha=0.7,
        beta=0.3,
        smooth=1
    ):

        super(TverskyLoss, self).__init__()

        self.alpha = alpha

        self.beta = beta

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        true_positive = (
            predictions * targets
        ).sum()

        false_positive = (
            ((1 - targets) * predictions)
        ).sum()

        false_negative = (
            (targets * (1 - predictions))
        ).sum()

        tversky = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return 1 - tversky

# ----------------------------------
# HYBRID MEDICAL LOSS
# ----------------------------------
class HybridMedicalLoss(nn.Module):

    def __init__(self):

        super(HybridMedicalLoss, self).__init__()

        self.focal = FocalLoss()

        self.tversky = TverskyLoss()

    def forward(self, predictions, targets):

        focal_loss = self.focal(
            predictions,
            targets
        )

        tversky_loss = self.tversky(
            predictions,
            targets
        )

        total_loss = (
            focal_loss
            + tversky_loss
        )

        return total_loss