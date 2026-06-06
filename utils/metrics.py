import torch

# ----------------------------------
# DICE SCORE
# ----------------------------------
def dice_score(
    predictions,
    targets,
    smooth=1e-6
):

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        (2. * intersection + smooth)
        /
        (
            predictions.sum()
            +
            targets.sum()
            +
            smooth
        )
    )

    return dice.item()

# ----------------------------------
# IOU SCORE
# ----------------------------------
def iou_score(
    predictions,
    targets,
    smooth=1e-6
):

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions.sum()
        +
        targets.sum()
        -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()