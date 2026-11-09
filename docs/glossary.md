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

## Anomaly Index

A normalized per-class score. AEVA uses it to identify unusually easy or concentrated target-class behavior; exceeding the configured threshold contributes to a backdoor verdict.

## MAD (Median Absolute Deviation)

A robust measure of how far values deviate from the median. MMBD uses MAD to normalize class scores, and research notes also describe MAD in Neural Cleanse-style anomaly detection.

## HSJA (HopSkipJumpAttack)

A query-based decision-boundary attack that finds perturbations capable of moving an input toward a target class. AEVA uses targeted HSJA to measure how easily source samples can be moved to target classes.

## p-value

A metric that measures how surprising the observed score or distribution would be under the assumed statistical model. MMBD reports a p-value from its gamma-distribution analysis, where a p-value < 0.05 supports the backdoored verdict.

## Clean Accuracy

Model accuracy on unmodified test samples. For AEVA specifically, clean accuracy is calculated while collecting correctly classified test samples, and low clean accuracy can make the defense unreliable.

## Target Class

The class an attacker wants a triggered or poisoned input to be classified as. Several defenses analyze behavior toward a target class, and techniques like AEVA and LMR expose target-class concepts directly.

## Checkpoint

A saved model state or weights file (e.g., .pt or .pth). Mithridatium audit and repair operations load and operate directly on these local model checkpoints.

## Pruning

Removing or suppressing selected model parameters, neurons, or channels. Used by repair techniques such as Fine-Pruning or ANP to remove backdoor-associated model behavior while preserving normal performance.

## Dataset Mismatch

A mismatch between the dataset/preprocessing selected in Mithridatium and the data or normalization used when the model was trained. This can distort results, especially for defenses such as STRIP that rely on representative input data.

## False Positive

A clean model being flagged as suspicious or backdoored.

## False Negative

A compromised model being reported as clean.

## Benchmark Model

A known clean or known backdoored model used to evaluate whether a defense behaves as expected.
