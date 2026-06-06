import torch

from metrics import (
    dice_score,
    iou_score
)

# ----------------------------------
# DUMMY DATA
# ----------------------------------
predictions = torch.rand(
    (1, 1, 256, 256)
)

targets = torch.rand(
    (1, 1, 256, 256)
)


# ----------------------------------
# CALCULATE METRICS
# ----------------------------------
dice = dice_score(
    predictions,
    targets
)

iou = iou_score(
    predictions,
    targets
)

# ----------------------------------
# PRINT RESULTS
# ----------------------------------
print("\nDICE SCORE:")
print(dice)

print("\nIOU SCORE:")
print(iou)

print("\nMETRICS WORKING SUCCESSFULLY!")