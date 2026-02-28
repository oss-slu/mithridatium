import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mithridatium.defenses.strip import strip_scores
from mithridatium.utils import get_preprocess_config

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # linear layer initialized lazily when we know input size
        self.linear = None

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if needed
        if self.linear is None:
            # create linear layer matching incoming feature dimension
            self.linear = torch.nn.Linear(x.size(1), 4)
        return self.linear(x)

def test_strip_scores():
    print("Testing strip_scores...")
    
    # Setup
    torch.manual_seed(42)
    model = MockModel()
    
    # Get preprocessing configuration for the CIFAR-10 dataset
    dataset_name = "cifar10"
    config = get_preprocess_config(dataset_name)
    
    # Create dummy data: 100 samples, 10 features each
    data = torch.randn(100, 10)  # Simulated input data with 10 features
    labels = torch.randint(0, 4, (100,))  # Random labels (4 classes)
    
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # monkeypatch the utils.dataloader_for function so strip_scores uses our dummy loader
    import mithridatium.utils as utils_mod

    original_dl_for = utils_mod.dataloader_for
    def fake_dataloader_for(dataset_name, split, batch_size):
        # ignore parameters and return our small dataloader
        return dataloader, None
    utils_mod.dataloader_for = fake_dataloader_for
    
    # Test execution
    try:
        # Run strip_scores on the mock model using the preprocess config
        results = strip_scores(model, configs=config, num_bases=5, num_perturbations=10, device='cpu')
        
        # Extract entropies from the results
        entropies = results.get("entropies")
        
        print(f"Entropies: {entropies}")
        
        # Assert that entropies are in the expected format
        assert isinstance(entropies, list), "Entropies should be a list"
        assert len(entropies) == 5, f"Expected 5 entropies, got {len(entropies)}"
        assert all(isinstance(e, float) for e in entropies), "All entropies should be floats"
        
        print("strip_scores test passed!")
        
    except Exception as e:
        print(f"strip_scores test failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_strip_scores()