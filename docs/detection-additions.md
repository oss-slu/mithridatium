# Research inference-time detection methods

Effort: **S** \= fits in one sprint, **M** \= new dependency or slow runtime, **L** \= spans sprints.

# Methods

| Method | Paper / year | Code \+ license | Box | Data at inference? | CIFAR/ResNet? | Effort | Recommend |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| SCALE-UP | Guo et al., ICLR 2023 | [JunfengGo/SCALE-UP](https://github.com/JunfengGo/SCALE-UP), no license; [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox) GPL-2.0 | Black | No | Yes | S | Implement |
| TeCo | Liu et al., CVPR 2023 | [CGCL-codes/TeCo](https://github.com/CGCL-codes/TeCo), no license | Black | No | Yes | M | Later |
| BaDExpert | Xie et al., ICLR 2024 | [vtu81/backdoor-toolbox](https://github.com/vtu81/backdoor-toolbox), no license | White | Yes | Yes | M | Later |
| TED | Mo et al., IEEE S\&P 2024 | [tedbackdoordefense/ted](https://github.com/tedbackdoordefense/ted), no license | White | Yes | Yes | M/L | Later |
| IBD-PSC | IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency (2024) | [https://github.com/THUYimingLi/BackdoorBox](https://github.com/THUYimingLi/BackdoorBox), GNU GENERAL PUBLIC LICENSE Version 2 | White-Box | Yes | Yes; CIFAR 10, ResNet 18 | S/M | Implement |
| SentiNet | SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems (2020) | No official code repository. An implementation exists. [https://github.com/CassiniHuy/trojan-attacks-and-defenses](https://github.com/CassiniHuy/trojan-attacks-and-defenses) / No given license | White | Yes | No | L | No |
| STRIP | STRIP: A Defence Against Trojan Attacks on Deep Neural Networks | [https://github.com/garrisongys/STRIP](https://github.com/garrisongys/STRIP) / No formal license, though creators request citations. | Black | Yes | Yes CIFAR10, ResNet20 tested with not CIFAR10 images | M | Later |

## Recommendation

SCALE-UP first. It needs only hard labels and no clean data, so it runs on local checkpoints and Hugging Face models alike, and the core is about 30 lines: scale pixels, clip, count how often the label holds.

Back-up:  
IBD-PSC. Though slightly more complex than SCALE-UP, requiring white-box access to the model and a set of \~100 non-poisoned images for calibration, IBD-PSC is well documented and intuitive

# CLI sketch

mithridatium detect \--model models/resnet18\_poison.pth \--method scaleup \\

  \--data cifar10 \--num-samples 1000 \--out reports/detect\_scaleup.json

Proposed flags: \--model, \--method, \--data, \--num-samples, \--clean-samples, \--threshold, \--seed, \--out, plus \--scaleup-scales (default 3,5,7,9,11).

Proposed report fields under results: mode: "input-level", method, verdict, num\_inputs, num\_flagged, threshold, parameters, per\_sample (index, predicted\_label, score, flagged).