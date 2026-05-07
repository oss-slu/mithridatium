# Glossary

## Backdoor Attack

A model compromise where the model behaves normally on ordinary inputs but produces attacker-chosen behavior when a specific trigger is present.

## Poisoning Attack

An attack where the training data or training process is modified so the final model learns malicious behavior.

## Trigger

The pattern, feature, phrase, patch, perturbation, or semantic condition that activates a backdoor.

## Invisible Trigger

A trigger designed to be hard for humans to notice, such as a small universal perturbation added to images.

## Black-Box Defense

A defense that mainly uses model queries and outputs. It does not require direct access to model weights or internal activations.

## White-Box Defense

A defense that requires internal model access, such as weights, layers, gradients, or intermediate activations.

## Logits

The raw model output scores before softmax. Most classification models produce one logit per class.

## Entropy

A measure of prediction uncertainty. Low entropy means the model is very confident in one class; high entropy means probability mass is spread across classes.

## Perturbation

A small change to an input. In this project, perturbations can be used to test whether predictions remain stable or to probe decision boundaries.

## Dataset Mismatch

A mismatch between the dataset/preprocessing selected in Mithridatium and the data or normalization used when the model was trained. This can distort results, especially for defenses such as STRIP that rely on representative input data.

## False Positive

A clean model being flagged as suspicious or backdoored.

## False Negative

A compromised model being reported as clean.

## Benchmark Model

A known clean or known backdoored model used to evaluate whether a defense behaves as expected.
