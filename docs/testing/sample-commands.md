# CLI Command Cheat Sheet

These commands are short CLI references for testing and debugging. Full scenario walkthroughs live in `examples/`:

- [Demo commands](../../examples/demo_commands.md)
- [End-to-end smoke](../../examples/end_to_end.md)
- [Invisible backdoor](../../examples/invisible_backdoor.md)
- [Semantic backdoor](../../examples/semantic_backdoor.md)

## Inspect the CLI

```bash
mithridatium --version
mithridatium --help
mithridatium defenses
mithridatium detect --help
```

## Run One Local Defense

```bash
mithridatium detect --model models/resnet18_poison.pth --data cifar10 --defense mmbd --out reports/mmbd.json --force
```

```bash
mithridatium detect --model models/resnet18_poison.pth --data cifar10 --defense strip --out reports/strip.json --force
```

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense freeeagle \
  --freeeagle-optimize-steps 100 \
  --out reports/freeeagle.json \
  --force
```

```bash
mithridatium detect \
  --model models/resnet18_poison.pth \
  --data cifar10 \
  --defense aeva \
  --aeva-samples-per-class 5 \
  --aeva-hsja-iterations 5 \
  --aeva-ep 1 \
  --out reports/aeva_smoke.json \
  --force
```

## Run a Hugging Face Model

```bash
mithridatium detect \
  --provider huggingface \
  --hf-model-id microsoft/resnet-50 \
  --data cifar10_for_imagenet \
  --defense strip \
  --out reports/hf_strip.json \
  --force
```

## Work With Report Output

```bash
cat reports/mmbd.json
```

```bash
mithridatium detect \
  --model models/resnet18_clean.pth \
  --data cifar10 \
  --defense freeeagle \
  --out - \
  | python -m json.tool
```

```bash
mithridatium detect \
  --model models/resnet18_clean.pth \
  --data cifar10 \
  --defense freeeagle \
  --out reports/freeeagle.json \
  --force
```

Useful fields:

- `defense`: the defense that ran
- `dataset`: dataset/preprocessing selection
- `results.verdict`: clean/backdoored decision
- `results.thresholds`: threshold values used for the decision
- `results.parameters`: runtime settings
