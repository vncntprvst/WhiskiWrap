# WhiskiWrap

WhiskiWrap provides tools for running [whisk](https://github.com/vncntprvst/whisk)
(the Janelia whisker tracker) more easily and efficiently. It improves on whisk by:

1. **Robust input** — it uses your system `ffmpeg` to read almost any video and
   to emit simple TIFF stacks that whisk traces reliably.
2. **Speed** — it runs many `trace` processes in parallel on non-overlapping
   chunks of the video.
3. **Portability / memory** — results are collected into HDF5 (or Parquet)
   files readable from Python or MATLAB, and can be read partially.

The codebase is split into modules: `base` (core utilities), `pipeline`
(high-level workflows), `io` (video I/O), and `wfile_io` / `mfile_io` (ctypes
readers for whisk `.whiskers` / `.measurements` files).

## Installation

**Requirements**: Python ≥ 3.10 and `ffmpeg` on your `PATH` (`ffmpeg -version`
should run). Windows, Linux and macOS are supported.

`whisk` is installed automatically as the `whisk-janelia` dependency — its
prebuilt `trace`/`measure`/`classify` binaries (including **native Windows
`.exe`s**) are bundled in the wheel, so no separate whisk install or
`WHISKPATH` setup is needed.

Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
# isolated environment
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e .            # editable; or: uv pip install .
```

```bash
# plain pip
pip install -e .               # or: pip install whiskiwrap
```

Verify:

```bash
python -c "import WhiskiWrap; print('ok')"
```

> Build tools are only needed if a dependency lacks a wheel for your platform
> (Debian/Ubuntu: `python3-dev build-essential`; Fedora: `python3-devel gcc`;
> macOS: `xcode-select --install`; Windows: usually none, since wheels are used).

## Quick start

```python
import WhiskiWrap

# Copy the input next to a working directory — many temporary files are created.
input_video = 'test_video2.mp4'
output_file = 'output.hdf5'

# Trace (and optionally measure) in parallel chunks -> combined HDF5.
WhiskiWrap.pipeline_trace(input_video, output_file, n_trace_processes=4)
```

Read the per-frame summary (tip/follicle/angle/… of every whisker):

```python
import tables, pandas
with tables.open_file(output_file) as fi:
    summary = pandas.DataFrame.from_records(fi.root.summary.read())
```

For bilateral tracking with measurement per side, use
`interleaved_split_trace_and_measure(...)` (see `WhiskiWrap/pipeline.py`), which
crops each face side, traces in chunks, and writes a per-side Parquet/HDF5 file.

## How it works

1. Split the video into **epochs** (~100k frames) read into memory one at a time.
2. For each epoch: split into **chunks** (~1000 frames, optionally cropped),
   write each chunk as a TIFF stack, trace chunks with parallel `trace`
   instances, then append each chunk's `.whiskers` results to the output file.
3. Optionally delete intermediate chunk files.

Key parameters:

* `n_trace_processes` — parallel `trace` instances (≈ number of CPUs).
* `epoch_sz_frames` — frames per epoch; as large as memory allows (e.g. 100000).
* `chunk_sz_frames` — frames per chunk; ideally `epoch_sz_frames / (N * n_trace_processes)`.
* `measure=True`, `face='left'|'right'` — also run `measure` for the given face side.

## Notes

* whisk identity classification (`classify`/HMM reclassify) operates **within a
  video**. Preserving whisker identity **across chunks and across a whole
  session** is handled by a separate linking step (see the project pipeline /
  the linker module); this is being consolidated into WhiskiWrap.
