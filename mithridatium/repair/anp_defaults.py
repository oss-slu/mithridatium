"""Shared default settings for Adversarial Neuron Pruning (ANP)."""

ANP_EPS = 0.2
ANP_ALPHA = 0.2
ANP_STEPS = 1
ANP_PRUNE_THRESHOLD = 0.15
ANP_CLEAN_SAMPLES = 500
ANP_ITERATIONS = 2000
# None means the clean-reference subset is selected using the regular ANP seed.
# An explicit --anp-clean-subset-seed overrides that selection.
ANP_CLEAN_SUBSET_SEED = None
