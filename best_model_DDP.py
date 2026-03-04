import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
from typing import List
from contextlib import nullcontext
import random
import torch.distributed as dist
import os


import matplotlib.pyplot as plt

"""
To test locally distributed training: `torchrun --nproc_per_node=2 best_model_DDP.py`
and use dummy data files with similar shapes.
"""
#########################
# Dataset and Dataloader
#########################

class TrajectoryDataset(Dataset):
    def __init__(self, states_path, actions_path):
        """
        Args:
            states_path (str): Path to the states .npy file.
            actions_path (str): Path to the actions .npy file.
            augmentations (callable, optional): A function or transform to apply to the states and actions.
        """
        self.states = np.load(states_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        states = torch.tensor(self.states[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.float32)
        
        return states, actions

#########################
# Model Components
#########################

#########################
# Encoder
#########################

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=256):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, state_dim)

    def forward(self, x):
        if x.ndimension() == 5:  # (B, T, C, H, W) 
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  
            h = self.conv(x) 
            h = h.view(h.size(0), -1)
            s = self.fc(h)
            s = s.view(B*T,16,4,-1)
        else:  # (B, C, H, W) 
            h = self.conv(x) 
            h = h.view(h.size(0), -1) 
            s = self.fc(h) # (B, D)
            s = s.view(B,16,4,-1)
        return s

#########################
# Recurrent CNN Predictor
#########################

class RecurrentPredictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, hidden_dim=128, cnn_channels=64):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(16 + 16, cnn_channels, kernel_size=3, padding=1),
            nn.LayerNorm([cnn_channels,4,4]),
            nn.GELU(),
            nn.Conv2d(cnn_channels, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16,4,4]),
        )

    def forward(self, prev_state, action):
        """
        Args:
            prev_state: Tensor of shape (B, state_dim, H, W)
            action: Tensor of shape (B, action_dim)
        Returns:
            next_state: Tensor of shape (B, state_dim, H, W)
        """
        B, D, H, W = prev_state.size()
        
        action_embedding = self.action_mlp(action)
        action_embedding = action_embedding.view(B, D, H, W)
        
        x = torch.cat([prev_state, action_embedding], dim=1)  # (B, 2 * state_dim, H, W)
        next_state = self.cnn(x)  # (B, state_dim, H, W)
        
        return next_state

#########################
# JEPA Model (Recurrent)
#########################

class JEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99, cnn_channels=64):
        super().__init__()
        self.repr_dim = state_dim

        # Online encoder (learned)
        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)

        # Target encoder (EMA copy of online encoder)
        self.target_encoder = deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Recurrent CNN Predictor
        self.predictor = RecurrentPredictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, cnn_channels=cnn_channels)

        # EMA update rate
        self.ema_rate = ema_rate

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)."""
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data

    def forward(self, states, actions):
        """
        Args:
            states: Tensor of shape (B, T, 2, 64, 64)
            actions: Tensor of shape (B, T-1, 2)

        Returns:
            predicted_states: Predicted latent states (B, T-1, D)
            target_next_states: Target latent states (B, T-1, D)
            all_states: All latent states including the first online state (B, T, D)
        """
        B, T, _, _, _ = states.shape 

        encoded_states = self.online_encoder(states)  #(B*T, 16, 4, 4) or (B, 16, 4, 4) at inference
        H,W = 4, 4 
        encoded_states = encoded_states.view(B, T, -1, H, W)  # Shape: (B, T, 16, 4, 4)
        
        initial_state = encoded_states[:, 0] # Shape: (B, 16, 4, 4)
        predicted_states = []
        prev_state = initial_state

        for t in range(actions.size(1)):  # T-1 iterations
            action = actions[:, t]  # (B, action_dim)
            next_state = self.predictor(prev_state, action)  # (B, D, H, W)
            predicted_states.append(next_state.view(B, -1))  # Flatten spatial dims for final output
            prev_state = next_state

        predicted_states = torch.stack(predicted_states, dim=1)  # (B, T-1, D)
        
        if T > 1:  # Training scenario
            target_next_states = encoded_states[:, 1:].view(B, T-1, -1)  # (B, T-1, D)
        else:  # Inference scenario
            target_next_states = 0  # Placeholder value for inference

        all_states = torch.cat([initial_state.view(B, 1, -1), predicted_states], dim=1)  # Shape: (B, T, D)

        return predicted_states, target_next_states, all_states

#########################
# Regularization Utilities
#########################

def variance_regularization(latents, epsilon=1e-4):
    var = torch.var(latents, dim=0)
    return torch.mean(torch.clamp(epsilon - var, min=0))

def covariance_regularization(latents):
    latents = latents - latents.mean(dim=0)
    latents = latents.view(latents.size(0), -1)  # Flatten all dimensions except the batch dimension
    cov = torch.mm(latents.T, latents) / (latents.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.sum(off_diag ** 2)

def normalize_latents(latents):
    return latents / (torch.norm(latents, dim=-1, keepdim=True) + 1e-8)

def contrastive_loss(predicted_states, target_states, temperature=0.1):
    """
    Compute contrastive loss between predicted and target states.
    Args:
        predicted_states: Tensor of shape (B, T-1, D)
        target_states: Tensor of shape (B, T-1, D)
        temperature: Temperature scaling factor for contrastive loss
    Returns:
        loss: Contrastive loss value
    """
    B, T_minus_1, D = predicted_states.shape
    predicted_states = predicted_states.reshape(-1, D)
    target_states = target_states.reshape(-1, D)

    # Normalize the embeddings
    predicted_states = normalize_latents(predicted_states)
    target_states = normalize_latents(target_states)

    # Compute similarity scores
    logits = torch.mm(predicted_states, target_states.T) / temperature
    labels = torch.arange(B * T_minus_1, device=predicted_states.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def scheduled_loss_weight(epoch, total_epochs, T, mode="linear"):
    """
    Compute loss weights for timesteps based on a schedule.
    Args:
        epoch (int): Current epoch.
        total_epochs (int): Total training epochs.
        T (int): Total number of timesteps in the sequence.
        mode (str): Schedule mode. Options: ["linear", "exponential"].
    Returns:
        Tensor: Weights for each timestep.
    """
    if mode == "linear":
        weight = torch.linspace(1.0, (epoch / total_epochs), T) 
    elif mode == "exponential":
        factor = epoch / total_epochs
        weight = torch.tensor([(factor ** t) for t in range(1, T + 1)]) 
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    weight /= weight.sum()  # Normalize weights to sum to 1
    return weight

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

#########################
# Training Loop Example
#########################

if __name__ == "__main__":
    # TODO: Read local_rank, rank, world_size from env ("torchrun --nproc_per_node=2 best_model_DDP.py")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # TODO: Initialize communication between devices
    dist.init_process_group(backend="gloo", init_method="env://")

    # TODO: Initialize device for DDP to assign data slices to specific devices
    # Use CPU when CUDA is not available (e.g. local testing on Mac with gloo backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparams
    batch_size = 64
    lr = 3e-4*2
    epochs = 20
    state_dim = 256
    action_dim = 2
    hidden_dim = 128
    cnn_channels = 64
    initial_accumulation_steps = 4  
    final_accumulation_steps = 4   
    
    # Load data
    # TODO: Add distributed sampler here and change shuffle in dataloader == True. Shuffle handled by sampler and also in sampler.set_epoch(epoch)
    train_dataset = TrajectoryDataset("data/train/dummy/states_dummy.npy", "data/train/dummy/actions_dummy.npy")
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=sampler)
    
     # TODO: Wrap model in DDP so gradients are synchronized across processes
    model = nn.parallel.DistributedDataParallel(JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, cnn_channels=cnn_channels).to(device))
    
    # TODO: device -> device.type since now device is cuda:0 / cuda:1
    if device.type == 'cuda':
        model = torch.compile(model)

    torch.set_float32_matmul_precision('high')

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.90, 0.99), eps=1e-8)
    criterion = nn.MSELoss()
    
    loss_history = []

    model.train()
    #TODO: sync all processes in gpus before and that cpu and gpu are synced as well before timer starts
    torch.distributed.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(epochs):
        #TODO: call sampler to reset different shuffle (slice of data) per epoch
        sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        
        accumulation_steps = max(final_accumulation_steps, initial_accumulation_steps - (initial_accumulation_steps - final_accumulation_steps) * epoch // epochs)
        #TODO: log on rank 0 only
        for step, (states, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0))):
            states = states.to(device)
            actions = actions.to(device)

            # Compute losses
            # Autocast (float16) only on CUDA; CPU lacks LayerNorm kernel for Half
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
            with autocast_ctx:
            
                predicted_states, target_states, _ = model(states, actions)

                mse_loss = criterion(predicted_states, target_states)

                # Add variance and covariance regularization
                mse_loss += 0.01 * variance_regularization(predicted_states)
                mse_loss += 0.01 * covariance_regularization(predicted_states)

                # Add contrastive loss
                contrast_loss = contrastive_loss(predicted_states, target_states)
                
                # Compute scheduled loss weights
                T_minus_1 = predicted_states.size(1)
                weights = scheduled_loss_weight(epoch, epochs, T_minus_1, mode="linear").to(device)
            
                # Apply scheduled loss weighting
                weighted_mse_loss = (weights * torch.mean((predicted_states - target_states) ** 2, dim=-1)).mean()
                
                loss = weighted_mse_loss + contrast_loss

            loss.backward()

            dt=0
            if (step + 1) % accumulation_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update target encoder
                with torch.no_grad():
                    model.update_target_encoder()

                # TODO: device -> device.type since now device is cuda:0 / cuda:1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            total_loss += loss.item()
            loss_history.append(loss.item())

            #TODO: log on rank 0 only
            if rank == 0:
                print(f"loss {loss.item()}, dt {dt:.2f}ms")
        
        
        avg_loss = total_loss / len(train_loader)
        #TODO: log on rank 0 only
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    #TODO: sync all processes in gpus before and that cpu and gpu are synced as well before timer ends
    torch.distributed.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    time_taken = end - start
    #TODO: log on rank 0 only
    if rank==0:
        print(f'Training time: {time_taken/60:.2f} min')

    #TODO: Save on rank 0 only
    if rank == 0:
        # Plot the loss over time
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.savefig('./plots/training_loss_Z_tuned_distributed.png')
        # plt.show()

        #TODO: adjust synthax a bit because w DDP model is wrapped so model.state_dict() -> model.module.state_dict()
        torch.save(model.module.state_dict(), "./encoder_outputs/trained_recurrent_jepa_Z_tuned_distributed.pth")
    
    #TODO: Clean shutdown of all processes
    dist.destroy_process_group()