import torch
from torch import nn

# Fixed LatentODEFunc class that properly handles tensor dimensions
class LatentODEFunc(ODEF):  # Assumes ODEF is defined elsewhere
    def __init__(self, latent_dim, hidden_dim):
        super(LatentODEFunc, self).__init__()
        self.latent_dim = latent_dim
        
        # Neural network to approximate the derivative
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z, t):
        # Handle dimension issues - ensure z is 2D and t is properly shaped
        original_shape = z.shape
        if z.dim() > 2:
            z = z.reshape(-1, self.latent_dim)  # Flatten batch dimensions
        
        # Make sure t is a scalar or properly shaped tensor
        if isinstance(t, torch.Tensor):
            if t.dim() > 0:
                t = t.reshape(-1)[0]  # Take the first element if t is not a scalar
        
        # Expand t to match z's dimensions
        t_expanded = torch.ones(z.shape[0], 1, device=z.device) * t
        
        # Concatenate z and t
        z_t = torch.cat([z, t_expanded], dim=1)
        
        # Compute derivative
        return self.net(z_t)

# Fixed Encoder for the Latent ODE model (no changes needed, included for completeness)
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # RNN to process sequential data
        self.gru = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)  # +1 for time delta
        
        # Output mean and log variance for the latent state
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, time_delta):
        # Combine position and time data
        combined = torch.cat([x, time_delta.unsqueeze(-1)], dim=-1)
        
        # Process with GRU
        _, h_n = self.gru(combined.unsqueeze(0))
        h_n = h_n.squeeze(0)
        
        # Get latent parameters
        mean = self.fc_mean(h_n)
        logvar = self.fc_logvar(h_n)
        
        return mean, logvar

# Fixed Decoder (no changes needed, included for completeness)
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)

# Fixed Complete Latent ODE model with proper dimension handling
class LatentODE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, device):
        super(LatentODE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        # Encoder, ODE function, and Decoder
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim).to(device)
        self.ode_func = LatentODEFunc(latent_dim, hidden_dim).to(device)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim).to(device)
        
        # Neural ODE solver
        self.ode_solver = NeuralODE(self.ode_func).to(device)  # Assumes NeuralODE is defined elsewhere
        
    def encode(self, x, time_delta):
        mean, logvar = self.encoder(x, time_delta)
        
        # Reparameterization trick for sampling
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mean + eps * std
        
        return z0, mean, logvar
    
    def forward(self, x, times, time_delta, return_latent=False):
        # Encode to get initial latent state
        z0, mean, logvar = self.encode(x, time_delta)
        
        # Ensure times is properly shaped for ODE solver
        if isinstance(times, torch.Tensor) and times.dim() > 1:
            times = times.reshape(-1)  # Flatten to 1D tensor
        
        # Solve ODE to get latent trajectory
        z_trajectory = self.ode_solver(z0.unsqueeze(0), times, return_whole_sequence=True)
        
        # Decode latent trajectory to observations
        x_reconstructed = self.decoder(z_trajectory)
        
        if return_latent:
            return x_reconstructed, z_trajectory, mean, logvar
        else:
            return x_reconstructed, mean, logvar

# Fixed function for preparing time data
def prepare_time_data(tensor_timestamps):
    # Ensure timestamps is a 1D tensor
    if tensor_timestamps.dim() > 1:
        return tensor_timestamps.reshape(-1)
    return tensor_timestamps

# Example of fixed training loop usage
def fixed_training_loop(model, tensor_positions, tensor_timestamps, tensor_time_deltas, optimizer, loss_function):
    # Use 1D tensor for time
    all_times = prepare_time_data(tensor_timestamps)
    
    # Forward pass using all data
    x_reconstructed, mean, logvar = model(tensor_positions, all_times, tensor_time_deltas)
    
    # Calculate loss (assuming correct dimensions)
    loss, recon_loss, kl_loss = loss_function(
        x_reconstructed.squeeze(1),  # Remove batch dimension
        tensor_positions.unsqueeze(0).expand_as(x_reconstructed).squeeze(1),  # Match dimensions
        mean, 
        logvar
    )
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, recon_loss, kl_loss 