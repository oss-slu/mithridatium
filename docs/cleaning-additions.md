# Research post-training backdoor removal methods

Summary Statement Contributor:
Elijah researched LMR, Fine-Pruning, and Neural Cleanse. Ricky researched ANP, CLP, NAD. Both Recommendation and CLI Sketch were done together.

Effort: S = fits in one sprint, M = new dependency or slow runtime, L = spans sprints.

# Methods

| Method | Paper / year | Code + license | Box | Extra data needed? | CIFAR/ResNet? | Effort | Recommend |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| LMR (Logit-Margin Repulsion) | Yang et al., CVPR 2026 | [Trusted-LLM/LMR](https://github.com/Trusted-LLM/LMR), no license file | White | Yes — small clean set, ~1% of the data; ablation goes down to 0.02% (10 images) | Yes; CIFAR-10 / ResNet-18 is the main table. Also Tiny-ImageNet and ImageNet / ResNet-34, plus VGG-16, MobileNetV2, ViT | S | Implement |
| Fine-Pruning (FP) | Liu et al., RAID 2018 ([arXiv 1805.12185](https://arxiv.org/pdf/1805.12185)) | [kangliucn/Fine-pruning-defense](https://github.com/kangliucn/Fine-pruning-defense), no license file; reimplemented in [BackdoorBench](https://github.com/SCLBD/BackdoorBench), CC BY-NC 4.0 | White | Yes — clean set to profile activations, plus clean data for the fine-tune | Not in the paper (face recognition, speech, traffic signs). CIFAR-10 / ResNet-18 runs exist in BackdoorBench, and the LMR paper benchmarks FP on CIFAR-10 / ResNet-18 | S | Later |
| Neural Cleanse (NC) | Wang et al., IEEE S&P 2019 | [bolunwang/backdoor](https://github.com/bolunwang/backdoor), MIT; also reimplemented in BackdoorBench, CC BY-NC 4.0 | White | Yes — clean images for trigger reversal, plus 10% of the clean training data for unlearning | Not in the paper. Repo ships a GTSRB example on Keras 2.2 / TF 1.10 (Python 2.7 / 3.6); third-party work reproduces it on CIFAR-10 with ResNet-101 off the same code | M/L | No |
| ANP (Adversarial Neuron Pruning) | Wu & Wang, NeurIPS 2021 ([paper](https://proceedings.neurips.cc/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)) | [csdongxian/ANP_backdoor](https://github.com/csdongxian/ANP_backdoor), no license file; also reimplemented in [BackdoorBench](https://github.com/SCLBD/BackdoorBench), CC BY-NC 4.0 | White | Yes – small clean set; official example uses 500 CIFAR-10 images | Yes; official implementation uses CIFAR-10 / ResNet-18 | M | Implement |
| CLP (Channel Lipschitzness based Pruning) | Zheng et al., ECCV 2022 ([paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650171.pdf)) | [rkteddy/channel-Lipschitzness-based-pruning](https://github.com/rkteddy/channel-Lipschitzness-based-pruning); also reimplemented in [BackdoorBench](https://github.com/SCLBD/BackdoorBench), CC BY-NC 4.0 | White | No – data-free; UCLC is calculated directly from model weights | Yes; evaluated on standard image-classification models and implemented for ResNet-style architectures in BackdoorBench | S | Implement |
| NAD (Neural Attention Distillation) | Li et al., ICLR 2021 ([paper](https://arxiv.org/pdf/2101.05930)) | [bboylyg/NAD](https://github.com/bboylyg/NAD), no clear license file; also reimplemented in [BackdoorBench](https://github.com/SCLBD/BackdoorBench), CC BY-NC 4.0 | White | Yes – small clean set used to fine-tune a teacher model and perform attention distillation | Yes for CIFAR-10; official demo uses WideResNet, with ResNet-style implementations available in broader backdoor-defense frameworks | M/L | Later |

## Recommendation

I would recommend using LMR first since it is the easiest method to implement and has the best performance. It can also use the data collected by our detection method, and only needs a small set of clean data to actually fix it.

If LMR is not feasible, I think would recommend using CLP as a backup method. Since it is data-free and only requires access to model weights, making it easier to integrate.

# How each method works

## 1. LMR — Logit-Margin Repulsion

Every class gets a logit, which is the raw score the model produces before it picks an answer. LMR never tries to find the trigger. Instead it pushes the backdoor class's logit down on clean inputs, so when a trigger fires and adds its boost, the boost is no longer enough to make that class win. Then it checks which weights moved most during that push and singles them out for removal.

Three pieces, all applied to a small clean set:

- **SCE** — normal cross-entropy, but skip images whose label is `c`. Training on real airplanes pushes the airplane logit back up and fights us.
- **DSC** — for clean images not labeled `c`, force `c`'s logit to sit at least `m1` below the top competing logit. If it's already that far down, no penalty.
- **CM** — only fires on images the model isn't confident about, meaning the true class isn't leading its closest competitor by `m2`. Keeps DSC from jittering the boundaries of the other classes.

```
loss = SCE + α·DSC + β·CM
m1 = 3   α = 1.0   m2 = 0.5   β = 0.25
```

Save `W0 = model.fc.weight.clone()` before the loop starts. Stop when accuracy on class `c` falls to about random, which just means the push worked.

To actually remove it, save `W1` and subtract the two snapshots, `c`'s row only:

```
score_j = |W1[c, j] - W0[c, j]|
```

Whichever columns moved most are the backdoor's wiring. Zero them out across every row, freeze them with a gradient hook so training can't bring them back, then fine-tune briefly on clean data to restore class `c`.

**Gap:** these notes start from a known `c`. The paper finds it first: maximize cross-entropy on a small clean batch (anti-learning), then take the class with the highest mean log-probability. Needs adding before this is implementable. The prune ratio (how many columns count as "moved most") is also a parameter we haven't pinned down.

## 2. Fine-Pruning

Looks for backdoor neurons, meaning channels that are dead on clean data and only wake up for the trigger. Find those, cut them, then do a small retraining to repair the damage.

Run clean data through and find the channels that barely react to it. Those are the suspects. Zero them out lowest-first, stopping once clean accuracy starts to drop. Then fine-tune on clean data to get the accuracy back.

```
a_i = (1/N) Σ_n mean(A_i(x_n))
```

In plain English: for every channel `i`, take the clean images, average the activation grid for each one, add them all up, and divide by however many there were.

## 3. Neural Cleanse

Scans every class and reverse-engineers the smallest trigger that would flip any input into it. The class whose trigger is way too cheap, meaning much smaller than the rest, is the backdoor. Then stamp that recovered trigger onto clean images while keeping the labels correct. The model sees the same trigger over and over without the answer changing, so it learns the trigger means nothing; because our trigger runs through the same pathway as the real one, the real one stops working too.

### Step 1 — find the trigger

```
x_adv = (1 - m) * x + m * delta
```

For every pixel, how much to swap the original out for the trigger. `m` is the slider: at 0 we keep the original pixel, at 1 we take the trigger pixel, and in between we blend them. Most of the image has `m` at 0 and passes through untouched; a small patch has it near 1, and that patch is the trigger.

```
loss = cross_entropy(model(x_adv), target_label) + lam * m.abs().sum()
```

This is how it searches for what the trigger actually is. The first half asks whether the stamped image came out as the class we're testing. The second half adds up the mask to measure how big the trigger got. The two fight each other: one wants the trigger to work and would cover the whole image to do it, the other wants it small. `lam` decides who wins.

**Gap:** "way too cheap" needs a rule. The paper runs MAD (median absolute deviation) over the L1 norms of all the reversed triggers and flags a label when the anomaly index goes above 2. Not in these notes, and it's the step that decides whether the model is infected at all.

### Step 2 — unlearn it

- Take 10% of the clean data
- Stamp the trigger on 20% of that
- Keep the labels correct
- Fine-tune for one epoch

## 4. ANP

A backdoor model has specific neurons that are important for a backdoor to work so ANP purposefully perturbs the neurons to find out which ones are suspicious, then prunes them from network. Since it doesn't know what neuron is connected to what pattern it basically messes with the neuron's parameters and sees which part of the model becomes problematic.
ANP learns a mask value for each neuron while also applying the adversarial pertubations to the neurons weight and biases, so the ones with lower values become suspicious. As masks are learned, the neurons are prunedm, modifying the weights. Is a whitebox method, requires small amount of clean data and offical implementations support CIFAR-10 and ResNet-18

## 5. CLP

CLP identifies channels that are unusually sensitive when there is a slight change in the input. Backdoor tends to react stronger when there is a trigger. This method uses channel Lipschitzness as a way to measure this type of sensitivity. Since calculating Channel Lipschitz Constant (CLC) would be difficult to calculate they calculate the upper bound based on the mode's weights (UCLC), so it doesn't need clean data.
For each layer, CLP calculates a score for each channel then finds the means and standard deviations. 
threshold = mean + u * standard_deviation
If channel score is larger than threshold, it is considered outlier and pruned.

## 6. NAD

NAD removes backdoor by retraining the backdoor modeled using a cleaner teacher model. First the model is fine-tuned, using small clean dataset it will produce the teacher model and the original will be the student. Both the teacher and student are given the same clean images. NAD compares their attention maps then the student is trained to make its attention more similar to that of the teacher model while still being able to classify the clean images. 

the training loss used to repair the backdoored student model in NAD
loss = classification_loss + beta * attention_distillation_loss

The idea is that the teacher focuses on legitimate features of the image, while the student may have learned internal features associated with the trigger. By forcing the student to imitate the teacher's attention, the trigger-related behavior is weakened.

# CLI sketch

mithridatium repair --model models/resnet18_poison.pth --method lmr
--data cifar10 --clean-samples 500 --out models/repaired.pth

Proposed flags: --model, --method, --data, --clean-samples, --seed, --out, --report, plus --lmr-target-class and --lmr-prune-ratio.