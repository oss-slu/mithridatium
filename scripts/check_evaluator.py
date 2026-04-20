import argparse

import mithridatium.evaluator as evaluator
import mithridatium.loader as loader
from mithridatium.utils import dataloader_for_config, get_preprocess_config

def test_build_dataloader_one_batch():
    pp = get_preprocess_config("cifar10")
    loader_, _ = dataloader_for_config("cifar10", "test", pp, batch_size=8)
    x, y = next(iter(loader_))
    assert x.ndim == 4 and x.shape[1] == 3   # NCHW RGB
    assert y.ndim == 1
    assert x.shape[-2:] == pp.input_size[-2:]
        
def main():
    parser = argparse.ArgumentParser()
    '''
    .venv/bin/python -m scripts.check_evaluator --model models/resnet18_poison.pth
    '''
    parser.add_argument("--model", type=str, default="models/resnet18_bd.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load model from checkpoint
    model, feature_module = loader.load_resnet18(args.model)

    # Prepare CIFAR-10 test set
    pp = get_preprocess_config("cifar10")
    test_loader, _ = dataloader_for_config("cifar10", "test", pp, batch_size=args.batch_size)

    # Extract embeddings
    embs, labels = evaluator.extract_embeddings(model, test_loader, feature_module)
    print(f"Embeddings shape: {embs.shape}")
    print(f"Labels shape: {labels.shape}")

    # Evaluate accuracy
    loss, accy = evaluator.evaluate(model, test_loader)
    print(f"Test accuracy: {accy*100:.2f}% | Test loss: {loss:.4f}")

if __name__ == "__main__":
    main()
