import torch
from torchmetrics.clustering import MutualInfoScore
input = torch.tensor([2, 1, 0, 1, 0],dtype=torch.float32)
target = torch.tensor([0, 2, 1, 1, 0],dtype=torch.float32)
weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5],requires_grad=True,dtype=torch.float32)
preds = input*weights
mi_score = MutualInfoScore()
mi_loss = mi_score(preds, target)
mi_loss.backward()