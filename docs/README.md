# Mithridatium Documentation

This folder is the main documentation home for Mithridatium. It is organized for three audiences:

- New contributors who need a plain-English map of the project.
- Users who want to run one supported defense and understand the output.
- Future tech leads who need handoff notes, current limitations, and next steps.

## Start Here

- [Architecture overview](architecture/overview.md): high-level project map and data flow.
- [CLI flow](architecture/cli-flow.md): what happens when `mithridatium detect` runs.
- [Defenses overview](defenses/overview.md): compare FreeEagle, STRIP, MMBD, and AEVA.
- [Testing overview](testing/overview.md): practical testing workflow.
- [Glossary](glossary.md): definitions for common project and backdoor-defense terms.

## Architecture

- [Overview](architecture/overview.md)
- [CLI flow](architecture/cli-flow.md)
- [Model loading](architecture/model-loading.md)
- [Reporting pipeline](architecture/reporting-pipeline.md)

## Defenses

- [Overview](defenses/overview.md)
- [FreeEagle](defenses/freeeagle.md)
- [STRIP](defenses/strip.md)
- [MMBD](defenses/mmbd.md)
- [AEVA](defenses/aeva.md)

## Testing

- [Overview](testing/overview.md)
- [FreeEagle testing](testing/freeeagle-testing.md)
- [Hugging Face model testing](testing/huggingface-model-testing.md)
- [Sample commands](testing/sample-commands.md)

## Handoff

- [Tech lead handoff](handoff/tech-lead-handoff.md)
- [Known issues](handoff/known-issues.md)
- [Future work](handoff/future-work.md)

## Existing Examples

The `examples/` folder still contains runnable scenario notes and sample reports:

- [End-to-end walkthrough](../examples/end_to_end.md)
- [Demo commands](../examples/demo_commands.md)
- [Invisible backdoor example](../examples/invisible_backdoor.md)
- [Semantic backdoor example](../examples/semantic_backdoor.md)
- [Sample report JSON](../examples/sample_report.json)

## Assets

Existing defense graphs and PDF exports were preserved under `docs/assets/`.
