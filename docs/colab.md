# Standalone Colab notebook

Download `notebooks/DeMoSearch_SAM3_Colab.ipynb` and upload it at
<https://colab.research.google.com/> using **File → Upload notebook**.
Select a Python 3.12 GPU runtime and execute the cells from top to bottom.

The notebook contains the pipeline source in a checksum-protected embedded ZIP.
It does not depend on an unpublished Git branch, another uploaded source archive,
or a running service. The official SAM 3 implementation is installed at commit
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`; its weights are obtained separately
from the gated `facebook/sam3` Hub repository. A Colab Secret named `HF_TOKEN`
or a hidden prompt supplies authentication. Tokens are not embedded or printed.

An isolated, system-site-packages venv reuses the CUDA PyTorch/torchvision/Triton
already provided by Colab, while keeping SAM's NumPy < 2 dependencies out of the
notebook kernel. Python, CUDA, GPU and package versions are exported with results.
SAM uses native BF16 where supported and FP16 otherwise, including on a T4.

Inputs: individual JPEG/PNG photos, a photo ZIP, a Polycam raw-data ZIP, or a
selected Drive folder/archive. OBJ/GLB texture exports and depth-only archives are
rejected. A complete Polycam export retains its RGB-D and camera metadata, while
only corrected RGB images are sent to the detector. Partial exports with RGB
are also supported. ZIP paths and size limits are checked before extraction.

The default run processes one image with prompt `climbing hold`, optional CNN
classification enabled, and score thresholds of 0.5. Set `MAX_IMAGES=0` for all
frames after inspecting the first result. Rotation is previewed before inference;
automatic 90° for Polycam is tailored to the provided wallA scan and can be
overridden. Each run and result ZIP has a fresh name to preserve earlier results.

Results appear inline in Colab and in a downloadable offline HTML gallery, with
full binary mask PNGs, crops, JSON records and runtime metadata. The result ZIP
does not include authentication files or checkpoint weights. Download it before
ending the Colab session unless you have separately retained the results.

## Local verification

The notebook passed nbformat validation and Python parsing for every code cell.
Its embedded bootstrap was executed and checked against the current source,
including repeatability and preservation of local edits. Input helpers were
tested on fixture archives and the actual user-provided Polycam archive: all
38 corrected RGB keyframes were selected, excluding depth/confidence images.
The result-bundle test checked the offline gallery and referenced assets.

**Actual Colab/GPU inference was not executed in this CPU-only environment.**
Successful structural/helper checks do not establish SAM model accuracy or GPU
memory requirements. The existing CNN's sample accuracy is not a measurement of
performance on the user's gym photos.

Regenerate after source edits:

```bash
python scripts/build_colab_notebook.py
uv run --extra dev pytest
```
