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

Classifier training code lives in `crop_dataset.py`, `learned_classifier.py` and
`classifier_training.py` under `src/hold_detection`. Run optional encoder tests
with `uv run --extra dev --extra training pytest`. Regenerate the training
notebook with `python scripts/build_training_notebook.py`.

NYU course jobs must use Cloud Bursting, not Torch compute. The user selected
account `csci_ga_2271-2026fa`, GPU partition `c24m170-a100-2` with two A100 GPUs,
and `/scratch/NetID` for persistent checkpoints. CPU preparation/head jobs use
`n2c48m24`. Keep `--requeue` and periodic checkpoint/restart support. Never submit
jobs or claim full training results without authorized access and actual execution.
