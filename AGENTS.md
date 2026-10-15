# Project language

Keep all maintained project content in English: source comments, docstrings,
documentation, notebook Markdown, cell titles, prompts, error messages, UI labels,
and report text. New and edited project content must follow this convention.
Do not translate user-provided datasets, annotations, filenames, or model labels.

Edit `scripts/build_colab_notebook.py` to change the notebook, then regenerate
`notebooks/DeMoSearch_SAM3_Colab.ipynb` with:

```bash
python scripts/build_colab_notebook.py
```

Run `uv run --extra dev pytest` before committing changes.
