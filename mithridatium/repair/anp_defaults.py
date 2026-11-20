"""Shared default settings for Adversarial Neuron Pruning (ANP)."""

ANP_EPS = 0.2
ANP_ALPHA = 0.2
ANP_STEPS = 1
ANP_PRUNE_THRESHOLD = 0.15
ANP_CLEAN_SAMPLES = 500
ANP_ITERATIONS = 2000
# Keep the ANP clean-reference subset fixed across optimization seeds.
ANP_CLEAN_SUBSET_SEED = 1234
