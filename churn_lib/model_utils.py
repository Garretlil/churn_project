from torch import nn


class ChurnModel(nn.Module):

    def __init__(self):
        super(ChurnModel,self).__init__()

        self.layers=nn.Sequential(
            nn.Linear(8,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)