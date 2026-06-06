import torch

from losses import BCEDiceLoss

# ----------------------------------
# CREATE LOSS FUNCTION
# ----------------------------------
criterion = BCEDiceLoss()

# ----------------------------------
# DUMMY PREDICTIONS
# ----------------------------------
predictions = torch.rand(
    (1, 1, 256, 256)
)

# ----------------------------------
# DUMMY TARGET MASKS
# ----------------------------------
targets = torch.rand(
    (1, 1, 256, 256)
)

# ----------------------------------
# COMPUTE LOSS
# ----------------------------------
loss = criterion(
    predictions,
    targets
)

print("\nLOSS VALUE:")
print(loss.item())

print("\nLOSS FUNCTION WORKING SUCCESSFULLY!")