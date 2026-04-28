import mlx.core as mx
import mlx.nn as nn

class SleepApneaCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Input shape expected: (batch, time_steps, channels) -> (batch, 600, 2)
        
        # 1st Conv block
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm(16)
        self.relu1 = nn.ReLU()
        
        # 2nd Conv block
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm(32)
        self.relu2 = nn.ReLU()
        
        # 3rd Conv block
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm(64)
        self.relu3 = nn.ReLU()
        
        # Fully connected classifier
        self.fc = nn.Linear(64, num_classes)
        
    def __call__(self, x):
        # x is (batch, time, channels)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Global average pooling over time dimension (axis 1)
        x = x.mean(axis=1)
        
        # Classifier
        x = self.fc(x)
        return x
