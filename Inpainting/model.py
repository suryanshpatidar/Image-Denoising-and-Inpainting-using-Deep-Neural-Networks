import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rho=0.05, beta=1e-2, lambda_=1e-4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.rho = rho
        self.beta = beta
        self.lambda_ = lambda_

    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))
        out = torch.sigmoid(self.decoder(h))
        return out, h

    def sparsity_loss(self, h):
        rho_hat = h.mean(0)
        rho = torch.full_like(rho_hat, self.rho)
        kl_div = rho * torch.log(rho / (rho_hat + 1e-8)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
        return self.beta * kl_div.sum()


class SSDA(nn.Module):
    def __init__(self, input_dim, hidden_dims, rho=0.05, beta=1e-2, lambda_=1e-4):
        super().__init__()
        self.dae1 = DenoisingAutoencoder(input_dim, hidden_dims[0], rho, beta, lambda_)
        self.dae2 = DenoisingAutoencoder(hidden_dims[0], hidden_dims[1], rho, beta, lambda_)
        self.lambda_ = lambda_

        self.encoder1 = self.dae1.encoder
        self.encoder2 = self.dae2.encoder
        self.decoder2 = self.dae2.decoder
        self.decoder1 = self.dae1.decoder

    def forward(self, x):
        h1 = torch.sigmoid(self.encoder1(x))
        h2 = torch.sigmoid(self.encoder2(h1))
        out2 = torch.sigmoid(self.decoder2(h2))
        out1 = torch.sigmoid(self.decoder1(out2))
        return out1