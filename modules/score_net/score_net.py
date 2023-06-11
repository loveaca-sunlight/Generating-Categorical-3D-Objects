import torch


class ScoreNet(torch.nn.Module):
    """
    A simple network that convert cosine similarity to positive score
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hidden:int):
        super(ScoreNet, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_hidden, dim_out),
            torch.nn.Sigmoid()
        )

        self._init_layers()

    def _init_layers(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.layers(x)
