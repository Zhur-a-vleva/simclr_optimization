import torch.nn as nn


class SimCLR(nn.Module):
    def __init__(self, base_model, output_dim):
        super(SimCLR, self).__init__()
        self.encoder = base_model(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = nn.functional.normalize(z, dim=1)
        return h, z
