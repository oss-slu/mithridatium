import torch
import sys
import os

# Add the project root to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mithridatium.defenses.strip import prediction_entropy

def test_prediction_entropy():
    print("Testing prediction_entropy...")

    # Case 1: Uniform distribution (Maximum entropy)
    # Logits being equal implies uniform distribution after softmax
    logits_uniform = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    entropy_uniform = prediction_entropy(logits_uniform)
    
    # Expected entropy for uniform distribution over N classes is ln(N)
    expected_uniform = torch.tensor([torch.log(torch.tensor(4.0))])
    
    print(f"Uniform Logits: {logits_uniform}")
    print(f"Calculated Entropy: {entropy_uniform}")
    print(f"Expected Entropy: {expected_uniform}")
    
    assert torch.allclose(entropy_uniform, expected_uniform, atol=1e-4), "Uniform distribution entropy mismatch"

    # Case 2: One-hot distribution (Minimum entropy)
    # One logit much larger than others
    logits_one_hot = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    entropy_one_hot = prediction_entropy(logits_one_hot)
    
    # Expected entropy is close to 0
    expected_one_hot = torch.tensor([0.0])
    
    print(f"One-hot Logits: {logits_one_hot}")
    print(f"Calculated Entropy: {entropy_one_hot}")
    print(f"Expected Entropy: {expected_one_hot}")
    
    assert torch.allclose(entropy_one_hot, expected_one_hot, atol=1e-4), "One-hot distribution entropy mismatch"

    print("All tests passed!")

if __name__ == "__main__":
    test_prediction_entropy()
