import torch

from attention_unet import AttentionUNet

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
model = AttentionUNet().to(device)

# ----------------------------------
# PRINT MODEL
# ----------------------------------
print("\nATTENTION U-NET ARCHITECTURE:\n")

print(model)

# ----------------------------------
# CREATE DUMMY INPUT
# ----------------------------------
dummy_input = torch.randn(
    1,      # batch size
    1,      # grayscale channel
    256,    # height
    256     # width
).to(device)

# ----------------------------------
# FORWARD PASS
# ----------------------------------
output = model(dummy_input)

# ----------------------------------
# PRINT SHAPES
# ----------------------------------
print("\nINPUT SHAPE:")
print(dummy_input.shape)

print("\nOUTPUT SHAPE:")
print(output.shape)

print("\nATTENTION U-NET TEST SUCCESSFUL!")