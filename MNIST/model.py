import torch.nn.functional as F
from config import *

class LeNet5_NAS(nn.Module):
    def __init__(self, norm_config, num_classes=10):
        """
        norm_config: dict specifying normalization at key positions,
                     e.g., {'conv1': 'batch', 'conv2': 'none', 'fc1': 'layer'}
        """
        super(LeNet5_NAS, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.norm1 = normalization_choices[norm_config.get('conv1', 'none')](6) if norm_config.get('conv1', 'none') != 'none' else None
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = normalization_choices[norm_config.get('conv2', 'none')](16) if norm_config.get('conv2', 'none') != 'none' else None
        
        self.fc1 = nn.Linear(16*4*4, 120)
        # For FC layers, you might use LayerNorm or no normalization; adjust accordingly.
        self.norm_fc1 = normalization_choices.get(norm_config.get('fc1', 'none'), lambda x: None)(120) if norm_config.get('fc1', 'none') != 'none' else None
        
        self.fc2 = nn.Linear(120, 84)
        self.norm_fc2 = normalization_choices.get(norm_config.get('fc2', 'none'), lambda x: None)(84) if norm_config.get('fc2', 'none') != 'none' else None
        
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        if self.norm_fc1 is not None:
            if isinstance(self.norm_fc1, nn. InstanceNorm1d):
                x = x.unsqueeze(-1)
            x = self.norm_fc1(x)
            if isinstance(self.norm_fc1, nn. InstanceNorm1d):
                x = x.squeeze(-1)
        x = F.relu(self.fc2(x))
        if self.norm_fc2 is not None:
            if isinstance(self.norm_fc1, nn. InstanceNorm1d):
                x = x.unsqueeze(-1)
            x = self.norm_fc2(x)
            if isinstance(self.norm_fc1, nn. InstanceNorm1d):
                x = x.squeeze(-1)
        x = self.fc3(x)
        return x