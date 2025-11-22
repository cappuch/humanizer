import torch
import torch.nn as nn
import torch.nn.functional as F

class PathClassifier(nn.Module):
    def __init__(self, seq_len=32, in_channels=2, base_channels=64):
        super().__init__()
        self.seq_len = seq_len
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels*2)
        self.pool2 = nn.MaxPool1d(2)  # seq_len -> seq_len/2
        
        self.conv3 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels*4)
        self.pool3 = nn.MaxPool1d(2)  # seq_len/2 -> seq_len/4
        
        self.conv4 = nn.Conv1d(base_channels*4, base_channels*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(base_channels*8)
        self.pool4 = nn.AdaptiveAvgPool1d(1)  # -> 1
        
        # Classification head
        self.fc1 = nn.Linear(base_channels*8, base_channels*2)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(base_channels*2, base_channels)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(base_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 2, L] - path tensor
        Returns:
            logits: [B] - classification logits (human=1, AI=0)
        """
        # Feature extraction
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.pool3(h)
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool4(h)  # [B, C, 1]
        
        # Flatten
        h = h.squeeze(-1)  # [B, C]
        
        # Classification
        h = F.relu(self.fc1(h))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        logits = self.fc3(h).squeeze(-1)  # [B]
        
        return logits
    
    def predict_prob(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict_class(self, x):
        return (self.predict_prob(x) > 0.5).long()
