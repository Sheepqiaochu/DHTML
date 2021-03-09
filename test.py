import torch
import numpy as np
from torchvision import transforms

# rd1 = np.random.ran(0, 2, 10)
# rd2 = np.random.randint(0, 2, 10)
# print(np.floor(rd2))
# print(np.floor(rd1))
# rd1.resize((2, 5))
# rd2.resize((2, 5))
# rd2 = torch.tensor(rd2)
# rd1 = torch.tensor(rd1)
pre = (True, True, False)
pre = torch.BoolTensor(pre)
pre = pre.byte()
# transforms.ToTensor(rd1)
# transforms.ToTensor(rd2)

# print(torch.sum(rd1 & rd2).item())
print(pre)
# tp = torch.sum(np.floor((predicts + val_matches)/2)).item()
# fp = torch.sum(predicts & ~val_matches).item()
# tn = torch.sum(~predicts & ~val_matches).item()
# fn = torch.sum(~predicts & val_matches).item()

