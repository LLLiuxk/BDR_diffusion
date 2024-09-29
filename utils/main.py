import torch
import pytorch_lightning as pl

print(pl.__version__)


print(torch.__version__)
print(torch.cuda.is_available())

# print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))