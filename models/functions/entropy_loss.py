import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self, smooth=True) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        targets = targets.contiguous().view(-1)

        if self.smooth:
            eps = 0.2
            n_class = predictions.size(1)

            one_hot = torch.zeros_like(predictions).scatter(1, targets.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = nn.functional.log_softmax(predictions, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = nn.functional.cross_entropy(predictions, targets, reduction='mean')

        return loss
