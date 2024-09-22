
from efficient_kan import KANLinear
import torch

from torch import nn
from torch.functional import F

from chebykanlayer import ChebyKANLayer
from fourierkanlayer import NaiveFourierKANLayer
from helper_functions import *




class KANMixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_mixtures):
        super().__init__()


        self.hidden_block = nn.Sequential(
            KANLinear(input_dim, hidden_dim),
            nn.ReLU(),
            KANLinear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.pi = KANLinear(hidden_dim, n_mixtures)
        self.mu = KANLinear(hidden_dim, n_mixtures)
        self.sigma = KANLinear(hidden_dim, n_mixtures)

    
    def forward(self, x):

        hidden = self.hidden_block(x)

        pi, mu, sigma = F.softmax(self.pi(hidden), 1), self.mu(hidden), torch.exp(self.sigma(hidden))

        return pi, mu, sigma
    
class MultivariateKANMixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_mixtures, grid_size: int = 5, spline_order: int = 3, n_hidden_layers=2):
        super().__init__()

        layers = [KANLinear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([KANLinear(hidden_dim, hidden_dim, grid_size, spline_order), nn.ReLU()])
        self.hidden_block = nn.Sequential(*layers)

        # Output layers
        self.pi = KANLinear(hidden_dim, n_mixtures, grid_size, spline_order)
        self.mu = KANLinear(hidden_dim, n_mixtures * output_dim,  grid_size, spline_order)
        self.sigma = KANLinear(hidden_dim, n_mixtures * output_dim,  grid_size, spline_order)
        
        self.n_mixtures = n_mixtures
        self.output_dim = output_dim

    def forward(self, x):
        hidden = self.hidden_block(x)
        
        pi = F.softmax(self.pi(hidden), dim=1)
        mu = self.mu(hidden)
        sigma = torch.exp(self.sigma(hidden))
        
        if self.output_dim > 1:
            mu = mu.view(-1, self.n_mixtures, self.output_dim)
            sigma = sigma.view(-1, self.n_mixtures, self.output_dim)
        
        return pi, mu, sigma




class ChebyKANMixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_mixtures, degree):
        super().__init__()


        self.hidden_block = nn.Sequential(
            ChebyKANLayer(input_dim, hidden_dim, degree=degree),
            nn.ReLU(),
            ChebyKANLayer(hidden_dim, hidden_dim, degree=degree),
            nn.ReLU()
        )

        self.pi = ChebyKANLayer(hidden_dim, n_mixtures, degree=degree)
        self.mu = ChebyKANLayer(hidden_dim, n_mixtures, degree=degree)
        self.sigma = ChebyKANLayer(hidden_dim, n_mixtures, degree=degree)

    
    def forward(self, x):

        hidden = self.hidden_block(x)

        pi, mu, sigma = F.softmax(self.pi(hidden), 1), self.mu(hidden), torch.exp(self.sigma(hidden))

        return pi, mu, sigma



class FourierKANMixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_mixtures, gridsize, smooth_init=False):
        super().__init__()


        self.hidden_block = nn.Sequential(
            NaiveFourierKANLayer(input_dim, hidden_dim, gridsize, smooth_init),
            nn.ReLU(),
            NaiveFourierKANLayer(hidden_dim, hidden_dim, gridsize, smooth_init),
            nn.ReLU()
        )

        self.pi = NaiveFourierKANLayer(hidden_dim, n_mixtures, gridsize, smooth_init)
        self.mu = NaiveFourierKANLayer(hidden_dim, n_mixtures, gridsize, smooth_init)
        self.sigma = NaiveFourierKANLayer(hidden_dim, n_mixtures, gridsize, smooth_init)

    
    def forward(self, x):

        hidden = self.hidden_block(x)

        pi, mu, sigma = F.softmax(self.pi(hidden), 1), self.mu(hidden), torch.exp(self.sigma(hidden))

        return pi, mu, sigma
