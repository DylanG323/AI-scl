import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message

# ----------------------------
# Dataset
# ----------------------------
class PolarCodeDataset(Dataset):
    """Generate synthetic polar code data for AISCL training."""
    def __init__(self, N, K, num_samples, snr_db):
        self.N = N
        self.K = K
        self.num_samples = num_samples
        self.snr_db = snr_db
        
        # Codec for encoding
        self.codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=8)
        self.bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
        
        # Fix random seed for reproducibility
        np.random.seed(0)
        torch.manual_seed(0)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random message
        msg = generate_binary_message(size=self.K)
        
        # Encode
        encoded = self.codec.encode(msg)
        
        # Transmit over AWGN
        transmitted = self.bpsk.transmit(message=encoded, snr_db=self.snr_db)
        
        # Compute LLRs
        llr = 2 * transmitted / self.bpsk.noise_power
        
        # For training, we use the ground truth bits as labels
        label = torch.tensor(msg, dtype=torch.float32)
        
        return torch.tensor(llr, dtype=torch.float32), label

# ----------------------------
# Neural Network
# ----------------------------
class PathScoringNet(nn.Module):
    """Neural network to score decoding paths."""
    def __init__(self, N):
        super().__init__()
        self.fc1 = nn.Linear(2 * N, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# ----------------------------
# Training Function
# ----------------------------
def train_model(N, K, num_epochs=20, snr_db=1.0, batch_size=32, num_samples=5000):
    """Train the PathScoringNet for AISCL."""
    print(f"Training PathScoringNet for ({N}, {K}) polar code")
    print(f"SNR: {snr_db} dB, Epochs: {num_epochs}, Batch size: {batch_size}")
    
    # Dataset and loader
    dataset = PolarCodeDataset(N=N, K=K, num_samples=num_samples, snr_db=snr_db)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model and device
    model = PathScoringNet(N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for llr_batch, label_batch in dataloader:
            llr_batch = llr_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Input: concatenate LLRs with zero-initialized bits
            bits_batch = torch.zeros_like(llr_batch)
            x = torch.cat([llr_batch, bits_batch], dim=1)
            
            # Target: fraction of ones in the message
            target = label_batch.mean(dim=1, keepdim=True)
            
            # Forward
            output = model(x)
            
            # Loss
            loss = criterion(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("Training complete!")
    return model

# ----------------------------
# Save / Load Helpers
# ----------------------------
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(N, filepath):
    model = PathScoringNet(N)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    N = 128
    K = 64
    model = train_model(N, K, num_epochs=20, snr_db=1.0, batch_size=32)
    save_model(model, f"trained_model_N{N}_K{K}.pt")