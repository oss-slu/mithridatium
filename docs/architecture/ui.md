# UI and Hosted Demo

Mithridatium has command-line, Streamlit, and Gradio entry points. The CLI is still the clearest source of truth for detection behavior, but the UI paths are useful for demos and exploratory use.

## Hosted Streamlit Demo

The hosted Hugging Face Space is:

```text
https://huggingface.co/spaces/williamphoenix/Mithridatium
```

Use the hosted demo when you want to show the project quickly without setting up the local environment.

## Local Streamlit App

The Streamlit app lives at the repository root:

```text
app.py
```

Run it with:

```bash
pip install -e ".[ui]"
streamlit run app.py
```

The Streamlit app includes provider selection for local torchvision-style models and Hugging Face models.

## Local Gradio App

The Gradio app lives in:

```text
mithridatium/gradio_app.py
```

The CLI exposes it through:

```bash
pip install -e ".[ui]"
mithridatium ui
```

If Gradio is not installed, the CLI prints an optional-dependency error.

## Notes

- UI behavior should stay aligned with `mithridatium/cli.py`.
- The hosted Space may have different dependency, hardware, or runtime constraints than a local install.
- For reproducible testing and report generation, prefer the CLI examples in [sample commands](../testing/sample-commands.md).
