import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

# Define the ODE function for the latent space dynamics
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp and any(p is not None for p in adfdp):
            adfdp = torch.cat([p_grad.flatten() if p_grad is not None else 
                            torch.zeros_like(p).flatten() 
                            for p_grad, p in zip(adfdp, self.parameters())]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        else:
            adfdp = torch.zeros(batch_size, 0).to(z)
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        else:
            adfdt = torch.zeros(batch_size, 1).to(z)
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters) if len(flat_parameters) > 0 else torch.tensor([])

class LatentODEFunc(ODEF):
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

# Data preparation (key fixes)
def prepare_data(tensor_timestamps, tensor_positions, tensor_time_deltas):
    # Use 1D tensor for time
    all_times = tensor_timestamps.clone()
    
    # Or if you need to handle in different ways:
    # all_times = tensor_timestamps.reshape(-1)  # Ensure 1D
    
    return all_times

# Forward method for LatentODE class (fixed version)
def forward_fixed(self, x, times, time_delta, return_latent=False):
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

# Training loop modifications:
def training_loop_fixed(model, tensor_positions, tensor_timestamps, tensor_time_deltas):
    # Ensure all_times is 1D
    all_times = tensor_timestamps.clone()  # Use 1D tensor
    
    # Forward pass using all data with correctly shaped time tensor
    x_reconstructed, mean, logvar = model(tensor_positions, all_times, tensor_time_deltas)
    
    # Rest of the training code...
    return x_reconstructed, mean, logvar 