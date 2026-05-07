# Hugging Face Model Testing

Mithridatium supports Hugging Face image-classification models through `mithridatium/loader_hf.py`.

## Basic Smoke Test

```bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense strip \
  --out reports/hf_strip.json \
  --force
```

## Try MMBD

```bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense mmbd \
  --out reports/hf_mmbd.json \
  --force
```

## Try AEVA With Small Settings

AEVA can be slow because it makes many model queries. Start with small values:

```bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense aeva \
  --aeva-samples-per-class 2 \
  --aeva-hsja-iterations 2 \
  --aeva-hsja-max-num-evals 200 \
  --aeva-ep 1 \
  --out reports/hf_aeva_smoke.json \
  --force
```

## Compatibility Notes

- The model must load with `AutoModelForImageClassification`.
- CLIP or multimodal repositories usually do not fit the current wrapper.
- The wrapper expects image tensors and returns logits.
- Processor metadata is used for image size and normalization when available.
- FreeEagle is not currently supported through the Hugging Face wrapper.

## Dataset Mismatch

Be careful when pairing datasets and Hugging Face models. For example, `microsoft/resnet-50` is an ImageNet-style classifier, while CIFAR-10 labels are different. `cifar10_for_imagenet` resizes and normalizes CIFAR-10 images in an ImageNet-style way, but it does not make the label space semantically identical.

Dataset mismatch can affect all output interpretation and is especially important for STRIP because STRIP depends on representative inputs and prediction entropy under mixed images.
