import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual

        return F.relu(out)
    
class EchelonNet(nn.Module):
    def __init__(self, in_channels = 13, num_res_blocks = 5, num_filters = 128):
        super(EchelonNet, self).__init__()

        # Initialized convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual tower (9 blocks * 2 layers, 1 initial + 1 head = ~20 layers)
        self.res_tower = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        # Value Head (board evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size = 1, bias = False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size = 1, bias = False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_tower(x)

        # Value head logic
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.value_fc1(v))        
        v = torch.tanh(self.value_fc2(v)) # scale the value between -1 and 1

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 8 * 8)
        p = self.policy_fc(p)
        # Didn't use softmax here beacuse we'll use in MCTS with cross entropy loss

        return v, p
    
    def predict(self, board_tensor):
        """
        Convenience method for prediction during MCTS
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = board_tensor.unsqueeze(0)

            # Forward pass
            value, policy_logits = self.forward(x)

            # Apply softmax
            policy = F.log_softmax(policy_logits, dim = 1)

            value = value.squeeze(0).item()
            policy = policy.squeeze(0)

            return value, policy
        
def count_parameters(model):
    """Count number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """Test the model with dummy inputs"""
    print("Training EchelonNet...")

    model = EchelonNet(in_channels = 13, num_res_blocks = 5, num_filters = 128)
    print("Model created successfully.")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Memory: ~{count_parameters(model) * 4 / 1024**2:.1f} MB")

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 13, 8, 8)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    value, policy = model(dummy_input)
    print(f"\nOutput shapes:")
    print(f"  Value: {value.shape} (should be {batch_size}, 1)")
    print(f"  Policy: {policy.shape} (should be {batch_size}, 4672)")

    # Check value range
    print(f"\nValue statistics:")
    print(f"  Min: {value.min().item():.3f}")
    print(f"  Max: {value.max().item():.3f}")
    print(f"  Mean: {value.mean().item():.3f}")
    
    # Test predict method
    print("\nTesting predict method")
    single_board = torch.randn(13, 8, 8)
    value, policy = model.predict(single_board)
    print(f"  Value: {value:.3f}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Policy sum (log probs): {policy.exp().sum().item():.3f} (should be ~1.0)")
    print("All tests passed!")
    
    return model

if __name__ == "__main__":
    # Run tests
    model = test_model()

    print("\nModel Architecture: ")
    print(model)
