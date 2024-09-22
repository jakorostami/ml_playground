import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def generate_samples(number_of_samples: int = 1000):
    #Generate some example data
    n_samples = 10000
    epsilon = torch.randn(n_samples)
    x_train = torch.linspace(-10, 10, n_samples)
    y_train = 5*np.sin(0.75*x_train) + 0.3*x_train + epsilon

    y_train, x_train = x_train.view(-1, 1), y_train.view(-1, 1)

    x_train = torch.linspace(-10, 10, n_samples).unsqueeze(1).cpu().numpy()
    y_train = (7*np.sin(0.75*x_train) + 0.5*x_train + np.random.normal())

    # Convert the data to PyTorch tensors
    x_train, y_train = torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device)

    return x_train, y_train



# Define the loss function
def mdn_loss(y, mu, sigma, pi) -> torch.Tensor:
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)



def training_process(model, loss_fnc, optim, epochs, training_data, target):
    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        pi, mu, sigma = model(training_data)
        # print(y_logits)
        # 2. Calculate loss and accuracy
        loss = loss_fnc(target, mu, sigma, pi)

        # 3. Optimizer zero grad
        optim.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optim.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            pi, mu, sigma = model(training_data)
        
        # Print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}")
        
    return pi, mu, sigma


def create_mdn_distribution(pi, mu, sigma):
    """
    Create a MixtureSameFamily distribution from MDN outputs.
    
    Args:
    - pi: tensor of shape [batch_size, num_mixtures] - mixture weights
    - mu: tensor of shape [batch_size, num_mixtures, output_dim] - means
    - sigma: tensor of shape [batch_size, num_mixtures, output_dim] - standard deviations
    
    Returns:
    - mdn_dist: a MixtureSameFamily distribution
    """
    # Create the Categorical distribution for mixture components
    mix = torch.distributions.Categorical(probs=pi)
    
    # Create the MultivariateNormal distribution for each component
    comp = torch.distributions.Normal(loc=mu, scale=sigma)
    
    # Combine into a mixture distribution
    mdn_dist = torch.distributions.MixtureSameFamily(mix, comp)
    
    return mdn_dist

def sample_from_mdn(pi, mu, sigma, num_samples=1):
    """
    Sample from a Mixture Density Network output using MixtureSameFamily.
    
    Args:
    - pi: tensor of shape [batch_size, num_mixtures] - mixture weights
    - mu: tensor of shape [batch_size, num_mixtures, output_dim] - means
    - sigma: tensor of shape [batch_size, num_mixtures, output_dim] - standard deviations
    - num_samples: number of samples to draw for each input
    
    Returns:
    - samples: tensor of shape [batch_size, num_samples, output_dim]
    """
    mdn_dist = create_mdn_distribution(pi, mu, sigma)
    return mdn_dist.sample((num_samples,))