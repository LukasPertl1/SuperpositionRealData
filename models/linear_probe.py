import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        """
        The linear probe maps the input embedding (of size input_dim)
        to a single output (for binary classification).
        """
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.fc(x)