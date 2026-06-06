import torch

from unet import UNet

# ----------------------------------
# DEVICE
# ----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = UNet().to(device)

# ----------------------------------
# PRINT MODEL
# ----------------------------------
print("\nMODEL ARCHITECTURE:\n")

print(model)

# ----------------------------------
# CREATE DUMMY INPUT
# ----------------------------------
dummy_input = torch.randn(
    1,      # batch size
    1,      # channels
    256,    # height
    256     # width
).to(device)

# ----------------------------------
# FORWARD PASS
# ----------------------------------
output = model(dummy_input)

# ----------------------------------
# PRINT OUTPUT SHAPE
# ----------------------------------
print("\nINPUT SHAPE:")
print(dummy_input.shape)

print("\nOUTPUT SHAPE:")
print(output.shape)

print("\nMODEL TEST SUCCESSFUL!")