"""The HW2 network: a small MLP on top of a frozen sentence embedding.

    x (384, MiniLM vector — fixed, not trained)
      -> Linear(384 -> 128) -> ReLU -> Dropout(0.2)     one hidden layer (configurable)
      -> Linear(128 -> 4)                               logits for fact / policy / testimony / value

The output is *logits* (scores before softmax). `nn.CrossEntropyLoss` applies
log-softmax itself, so the model must not. Probabilities are only computed at
prediction time (`predict_proba`).

With the default shape the network has 384*128 + 128 + 128*4 + 4 = 49,796
parameters.
"""

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class SentenceTypeMLP(nn.Module):
    def __init__(self, in_dim: int = 384, hidden: list[int] = (128,), n_classes: int = 4,
                 activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
        self.activation = activation

        # One block per hidden layer: Linear -> activation -> Dropout.
        # hidden=[] gives no hidden layer at all (= logistic regression).
        layers = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, n_classes))  # output layer: one score per class
        self.net = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self) -> None:
        """Training step [1]: give every weight a starting value.

        Hidden layers use Kaiming (He) init, scaled for ReLU so the signal
        neither shrinks nor explodes from layer to layer. The output layer has
        no activation after it, so it uses Xavier (Glorot) init. Biases start
        at 0. The random draw depends on torch.manual_seed, set by the caller.
        """
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            if self.activation == "relu":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training step [2]: (batch, 384) -> (batch, 4) logits."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 384) -> (batch, 4) probabilities that sum to 1. For the app, not for training."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
