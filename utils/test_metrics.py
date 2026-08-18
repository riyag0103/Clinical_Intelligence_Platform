import torch
from metrics import dice_score, iou_score

# Perfect match
gt = torch.tensor([[1,1,0,0]], dtype=torch.float32)
pred = torch.tensor([[1,1,0,0]], dtype=torch.float32)

print("Perfect Dice:", dice_score(pred, gt))
print("Perfect IoU :", iou_score(pred, gt))

# Completely wrong
gt = torch.tensor([[1,1,0,0]], dtype=torch.float32)
pred = torch.tensor([[0,0,1,1]], dtype=torch.float32)

print("Wrong Dice:", dice_score(pred, gt))
print("Wrong IoU :", iou_score(pred, gt))