# Fiduciary-free frame alignment and image stabilization

Command-line tool and supplementary material to the paper: "Fiduciary-free frame alignment for robust time-lapse drift correction estimation in multi-sample cell microscopy"

![overview](images/overview.svg)

This approach for jitter and drift correction is based on ["RAFT: Recurrent All Pairs Field Transforms for Optical Flow"](https://arxiv.org/pdf/2003.12039.pdf)by Zachary Teed and Jia Deng (ECCV 2020).

## Install

- **requirements:**
   - python 3.8+ \
     `pip install gdown torch torchvision opencv-contrib-python` \
     `pip install scipy tqdm tifffile scikit-image joblib`
   - Clone with the RAFT submodule (the pickled model needs `RAFT/core` importable): \
     `git clone --recurse-submodules https://github.com/StefanBaar/cell_align` \
     `cd cell_align` \
     (if already cloned: `git submodule update --init --recursive`)

## Running

```bash
python stabilize.py <input> [options]
```

`<input>` is auto-detected and may be:

| input                              | behaviour                                  |
|------------------------------------|--------------------------------------------|
| a multi-page **TIFF stack**        | aligned and written as a TIFF stack        |
| a **directory of TIFF stacks**     | each stack aligned independently (batch)   |
| a **directory of 2D frame images** | aligned as a single sequence               |
| a **video** (mp4/avi/mov/…)        | aligned as a single sequence               |

Outputs are written to `out/<name>/`:
- `<name>_aligned.tif` – the aligned stack (always; grayscale input stays grayscale)
- `dyx.txt`, `dyx_cumsum.txt`, `dyx_median.txt` – per-frame displacement estimates
- `--video` additionally writes `<name>_aligned.mp4`
- `--png` additionally writes the aligned frames as individual PNGs

**Options**

```
--device auto|cpu|cuda:0   compute device (default: auto-detect CUDA)
--model PATH               RAFT checkpoint (default: models/raftsintel.pth)
--out DIR                  output directory (default: out)
--iters N                  RAFT refinement iterations (default: 11)
--order N                  spline order for warping (default: 5)
--mode none|pad|crop       border handling before warping (default: none)
--anchor                   estimate drift directly against frame 0 instead of
                           accumulating per-step flow (see note below)
--keyframe K               re-anchor every K frames (see note below)
--jobs N                   parallel processes for the warp step (-1 = all cores)
--video                    also write an aligned mp4
--png                      also write aligned frames as PNGs
--fps F                    frame rate for mp4 output
--filter STR              only process inputs whose name contains STR (batch)
```

Examples:

```bash
# single TIFF stack on the GPU, also emit an mp4
python stabilize.py data/sample_stack.tif --video

# batch a folder of TIFF stacks, 8-way parallel warping, periodic re-anchoring
python stabilize.py data/stacks/ --jobs 8 --keyframe 5

# only process inputs whose filename contains a given string
python stabilize.py data/stacks/ --filter control
```

### Drift estimation modes

The global per-frame translation can be estimated three ways:

- **incremental** (default): frame *i* → *i+1*, then summed. Consecutive frames
  are always similar so it never breaks down, but it accumulates a small
  per-step bias over long sequences (slow residual drift, mostly in one axis).
- **`--anchor`**: each frame measured **directly against frame 0**, so there is
  nothing to accumulate (sub-pixel on well-behaved sequences). It collapses,
  however, once the scene changes enough that frame 0 is no longer a good
  optical-flow target for later frames.
- **`--keyframe K`** (recommended): re-anchor every *K* frames. Each estimate
  spans at most *K* similar frames (so it does not collapse like `--anchor`)
  while only ~N/K offsets are chained (so per-step accumulation is cut by ~K).
  `K=1` is equivalent to incremental, very large `K` to `--anchor`; `K≈5` is a
  good general default.

## Samples

- DMSO (left: raw, right stabilized)

[![DMSO](https://img.youtube.com/vi/gazuq-znHJ4/hqdefault.jpg)](https://youtu.be/gazuq-znHJ4)
- RA (left: raw, right stabilized)

[![RA](https://img.youtube.com/vi/PBX6gSWabdU/hqdefault.jpg)](https://youtu.be/PBX6gSWabdU)
- KNK808 (left: raw, right stabilized)

[![KNK808](https://img.youtube.com/vi/OyPupI3irXw/hqdefault.jpg)](https://youtu.be/OyPupI3irXw)

## Done

- [x] **Model: replace old sintel model url** – the Google-Drive checkpoint is
  fetched by file id (gdown ≥5 dropped the `fuzzy` kwarg). The download is a
  DataParallel *state_dict*, so `load_model` now rebuilds a RAFT model from it
  as well as still accepting the older full-object checkpoint.
- [x] **Cuda: confirm cuda works** – device is auto-detected and CUDA inference
  is used when available.
- [x] **CPU: confirm multi core** – RAFT inference already uses all torch CPU
  threads; the warp step is parallelised with `--jobs` (loky processes, since
  scipy's high-order spline warp is not thread-safe).
- [x] **video conversion** – `--video` writes an aligned mp4 via OpenCV's
  bundled ffmpeg (`mp4v`); no external ffmpeg install required.
- [x] **TIFF stack input** – single stacks and directories of stacks are
  supported; grayscale bit-depth is preserved through warping and output.
- [x] **drift estimation** – fixed the alignment to apply the *cumulative*
  (not per-step) displacement, and added `--anchor` / `--keyframe` estimation
  modes to control long-sequence drift accumulation.

<!--- This repo requires RAFT
git submodule add https://github.com/princeton-vl/RAFT -->
