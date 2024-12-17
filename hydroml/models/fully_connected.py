import torch


class FullyConnected(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, activation):
        super(FullyConnected, self).__init__()
        self.layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(in_features, hidden_size))
            if activation == 'tanh':
                self.layers.append(torch.nn.Tanh())
            elif activation == 'linear':
                pass  # Linear activation means no activation
            in_features = hidden_size
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)