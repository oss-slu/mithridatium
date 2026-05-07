# Future Work

## Defense Improvements

- Add shared benchmark scripts for clean, patched, invisible-trigger, and semantic-trigger models.
- Calibrate thresholds by dataset and architecture.
- Add confidence or severity levels instead of relying only on binary verdicts.
- Extend FreeEagle beyond ResNet if stable feature-stage adapters are added.
- Make MMBD class probing configurable from the CLI.
- Add richer AEVA summaries and progress reporting.

## Hugging Face Support

- Maintain a tested compatibility list for model families.
- Add clearer warnings for label-space mismatch.
- Explore feature extraction adapters for selected Hugging Face ResNet-like models.
- Add cached/offline test fixtures so CI does not depend on network availability.

## Reporting

- Add report comparison tooling for clean vs poisoned benchmark pairs.
- Include optional environment metadata such as device, torch version, and package version.
- Add a human-readable HTML or Markdown report renderer.

## Documentation

- Keep CLI option docs synchronized with `mithridatium/cli.py`.
- Add screenshots or sample report excerpts once benchmark outputs are stable.
- Add contributor notes for adding a new defense.

## Engineering

- Add CI for the fast test suite.
- Separate slow/integration tests from unit tests more clearly.
- Reduce duplicate logic between `service.py` and `cli.py`.
- Add type hints and smaller helper functions around defense dispatch.
