import torch

from residual_attention_unet import (
    ResidualAttentionUNet
)

# ----------------------------------
# CREATE MODEL
# ----------------------------------
model = ResidualAttentionUNet()

print("\nMODEL CREATED SUCCESSFULLY!")

# ----------------------------------
# DUMMY INPUT
# ----------------------------------
dummy_input = torch.randn(
    1,
    1,
    256,
    256
)

print("\nINPUT SHAPE:")

print(dummy_input.shape)

# ----------------------------------
# FORWARD PASS
# ----------------------------------
output = model(dummy_input)

print("\nOUTPUT SHAPE:")

print(output.shape)

# ----------------------------------
# SUCCESS
# ----------------------------------
print("\nRESIDUAL ATTENTION U-NET TEST SUCCESSFUL!")