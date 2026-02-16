import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import mithridatium.evaluator as evaluator
import mithridatium.loader as loader
import unittest

class TestEvaluator(unittest.TestCase):
    def test_extract_embeddings_and_evaluate(self):
        # Get model path from environment variable or use default
        """
        export MODEL_PATH=models/resnet18_bd.pth
        export BATCH_SIZE=128
        .venv/bin/python -m unittest tests/test_evaluator.py
        """
        model_path = os.environ.get("MODEL_PATH", "models/resnet18_bd.pth")
        batch_size = int(os.environ.get("BATCH_SIZE", 128))

        # Use a tiny subset of CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        indices = list(range(512))
        subset = torch.utils.data.Subset(testset, indices)
        loader_ = DataLoader(subset, batch_size=batch_size, shuffle=False)

        model, feature_module = loader.load_resnet18(model_path)
        embs, labels = evaluator.extract_embeddings(model, loader_, feature_module)
        print(f"Embeddings shape: {embs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"First 5 labels: {labels[:5].tolist()}")
        loss, accy = evaluator.evaluate(model, loader_)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accy*100:.2f}%")
        self.assertTrue(embs.shape[0] > 0)
        self.assertTrue(labels.shape[0] > 0)
        self.assertTrue(loss >= 0)
        self.assertTrue(accy >= 0)

if __name__ == "__main__":
    unittest.main()
