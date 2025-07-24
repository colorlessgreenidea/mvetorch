import torch
import torch.nn as nn
import torch.nn.functional as F

class MVEModel(nn.Module):
    def __init__(self,
                 input_dim, 
                 hidden_dims_mean, 
                 hidden_dims_var,
                 output_dim=14,
                 batch_norm = False,
                 dropout_mean=0.0, 
                 dropout_var=0.0,
                 softmax = True,
                 device = 'cpu'):
        
        super().__init__()
        self.output_dim = output_dim
        self.softmax = softmax
        self.device = device

        # Mean branch
        mean_layers = []
        prev_dim = input_dim
        for h in hidden_dims_mean:
            mean_layers.append(nn.Linear(prev_dim, h))
            if batch_norm: mean_layers.append(nn.BatchNorm1d(h))
            mean_layers.append(nn.ELU())
            if dropout_mean > 0.0:
                mean_layers.append(nn.Dropout(p=dropout_mean))
            prev_dim = h
        mean_layers.append(nn.Linear(prev_dim, output_dim))
        self.mean_net = nn.Sequential(*mean_layers)

        # Variance branch
        var_layers = []
        prev_dim = input_dim
        for h in hidden_dims_var:
            var_layers.append(nn.Linear(prev_dim, h))
            if batch_norm: var_layers.append(nn.BatchNorm1d(h))
            var_layers.append(nn.ELU())
            if dropout_var > 0.0:
                var_layers.append(nn.Dropout(p=dropout_var))
            prev_dim = h
        var_layers.append(nn.Linear(prev_dim, output_dim))
        self.var_net = nn.Sequential(*var_layers)
        self.var_net[-1].bias.data.fill_(1.0)

    def forward(self, x):
        x = x.to(self.device)
        mu = self.mean_net(x)
        # If you want the mean to live on the interval [0,1],
        # you need to set softmax to "True"
        if self.softmax: mu = F.softmax(mu, dim=1)
        # To ensure positive variance, we always use the
        # softplus activiation function
        var = F.softplus(self.var_net(x)) + 1e-6
        return mu, var