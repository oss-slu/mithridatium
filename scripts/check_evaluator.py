import argparse
import mithridatium.evaluator as evaluator
import mithridatium.loader as loader
from mithridatium.data import build_dataloader
from mithridatium.io import load_preprocess_config

def test_build_dataloader_one_batch():
    # expects models/resnet18_bd.json from Issue 1
    pp = load_preprocess_config("models/resnet18_bd.pth")
    loader = build_dataloader("cifar10", "test", pp, batch_size=8)
    x, y = next(iter(loader))
    assert x.ndim == 4 and x.shape[1] == 3   # NCHW RGB
    assert y.ndim == 1
    # optional: verify spatial dims match config
    assert x.shape[-2:] == pp.input_size
        
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
    pp = load_preprocess_config(args.model)
    test_loader = build_dataloader("cifar10", "test", pp, batch_size=args.batch_size)

    # Extract embeddings
    embs, labels = evaluator.extract_embeddings(model, test_loader, feature_module)
    print(f"Embeddings shape: {embs.shape}")

    # Evaluate accuracy
    loss, accy = evaluator.evaluate(model, test_loader)
    print(f"Test accuracy: {accy*100:.2f}% | Test loss: {loss:.4f}")

if __name__ == "__main__":
    main()
