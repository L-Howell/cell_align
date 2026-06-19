"""Fiduciary-free frame alignment / image stabilization.

Estimates per-frame drift/jitter with RAFT optical flow and warps every frame
back onto the first frame's reference, removing global translation.

Supported inputs (auto-detected from the path):
  * a multi-page TIFF stack            -> aligned TIFF stack
  * a directory of multi-page TIFFs    -> each stack aligned independently (batch)
  * a directory of 2D frame images     -> single aligned sequence
  * a video file (mp4/avi/mov/...)     -> single aligned sequence

Run ``python stabilize.py --help`` for options.
"""

import argparse
import sys
import os
import warnings
from glob import glob
from pathlib import Path

warnings.filterwarnings("ignore")

# RAFT is vendored as a git submodule; its core modules use flat imports
# (``from update import ...``) so its core/ dir must be on sys.path *before*
# the pickled model object is unpickled.
_RAFT_CORE = str(Path(__file__).resolve().parent / "RAFT" / "core")
if _RAFT_CORE not in sys.path:
    sys.path.append(_RAFT_CORE)

import numpy as np
import torch
import cv2
import tifffile
from tqdm import tqdm
from skimage import transform
from joblib import Parallel, delayed

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
TIFF_EXTS = (".tif", ".tiff")

# Google-Drive RAFT-Sintel checkpoint (a DataParallel state_dict).
MODEL_FILE_ID = "1fubTHIa_b2C8HqfbPtKXwoRd9QsYxRL6"


class _Args(dict):
    """Attribute-accessible dict; RAFT probes its args with both ``in`` and ``.``."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def check_model(model_path):
    """Download the RAFT model if it is not present locally."""
    if os.path.isfile(model_path):
        print("model found:", model_path)
        return
    import gdown
    print("Downloading model ->", model_path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    # gdown>=5 dropped the ``fuzzy`` kwarg; download by file id instead.
    gdown.download(id=MODEL_FILE_ID, output=model_path, quiet=False)


def _raft_from_state_dict(state_dict, device):
    """Build a large RAFT model from a (possibly DataParallel) state_dict."""
    from raft import RAFT
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = RAFT(_Args(small=False, dropout=0,
                       alternate_corr=False, mixed_precision=False))
    model.load_state_dict(state_dict)
    return model.to(device)


def load_model(model_path, device):
    """Load the RAFT model onto ``device``.

    The checkpoint may be either a full pickled ``raft.RAFT`` object or a plain
    state_dict (the Google-Drive copy is the latter). Both need RAFT/core on
    sys.path (handled at import) and, on torch>=2.6, ``weights_only=False`` to
    unpickle the custom class / OrderedDict.
    """
    check_model(model_path)
    print("Loading model:", model_path)
    obj = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model = obj if callable(obj) else _raft_from_state_dict(obj, device)
    model.eval()
    return model


def resolve_device(arg):
    """Map ``auto`` to cuda when available, otherwise echo the request back."""
    if arg == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        dev = arg
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available - falling back to CPU")
        dev = "cpu"
    if dev.startswith("cuda"):
        print(f"Using device: {dev} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: cpu ({torch.get_num_threads()} torch threads, "
              f"{os.cpu_count()} cores)")
    return dev


# --------------------------------------------------------------------------- #
# frame source (uniform interface over tiff stack / image dir / video)
# --------------------------------------------------------------------------- #
class FrameSource:
    """Sequence of frames abstracted over TIFF stack, image dir or video.

    ``raw(i)`` returns the frame in its native dtype/channels (used for warping
    and output, so nothing is destroyed). ``rgb(i)`` returns a uint8 3-channel
    copy for RAFT.
    """

    def __init__(self, path):
        self.path = str(path)
        self.fps = None
        if os.path.isdir(self.path):
            self.kind = "image_dir"
            self.files = sorted(
                f for f in glob(os.path.join(self.path, "*"))
                if f.lower().endswith(IMAGE_EXTS)
            )
            if not self.files:
                raise FileNotFoundError(f"No image files found in {self.path}")
            self.n = len(self.files)
        elif self.path.lower().endswith(TIFF_EXTS):
            self.kind = "tiff_stack"
            self.data = tifffile.imread(self.path)
            if self.data.ndim < 3:
                raise ValueError(
                    f"{self.path} is not a multi-page stack (shape {self.data.shape})")
            self.n = self.data.shape[0]
        elif self.path.lower().endswith(VIDEO_EXTS):
            self.kind = "video"
            self.cap = cv2.VideoCapture(self.path)
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or None
        else:
            raise ValueError(f"Unsupported input: {self.path}")

    def __len__(self):
        return self.n

    def raw(self, i):
        if self.kind == "tiff_stack":
            return self.data[i]
        if self.kind == "image_dir":
            return _read_image(self.files[i])
        # video: cv2 gives BGR -> RGB
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f"Could not read frame {i} from {self.path}")
        return frame[:, :, ::-1]

    def rgb(self, i):
        return _to_rgb_uint8(self.raw(i))

    def close(self):
        if self.kind == "video":
            self.cap.release()


def _read_image(path):
    """Read a single 2D image, preserving TIFF bit-depth via tifffile."""
    if path.lower().endswith(TIFF_EXTS):
        return tifffile.imread(path)
    # cv2 reads BGR; flip to RGB for consistency with the rest of the pipeline
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = img[:, :, ::-1]
    return img


def _to_rgb_uint8(frame):
    """Normalise an arbitrary frame to a uint8 H x W x 3 array for RAFT."""
    if frame.ndim == 2:
        f = frame.astype(np.float32)
        peak = f.max()
        if peak > 0:
            f = f / peak * 255.0
        return np.dstack([f.astype("uint8")] * 3)
    if frame.shape[2] == 1:
        return _to_rgb_uint8(frame[:, :, 0])
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        peak = f.max()
        if peak > 0:
            f = f / peak * 255.0
        frame = f.astype("uint8")
    return np.ascontiguousarray(frame[:, :, :3])


def detect_inputs(path):
    """Return a list of input paths to process.

    A directory whose image files are themselves multi-page TIFF stacks is
    treated as a *batch* of stacks; otherwise the directory is a single
    sequence of 2D frames.
    """
    path = str(path)
    if os.path.isfile(path):
        return [path]
    tiffs = sorted(
        f for f in glob(os.path.join(path, "*"))
        if f.lower().endswith(TIFF_EXTS)
    )
    if tiffs:
        with tifffile.TiffFile(tiffs[0]) as tf:
            if len(tf.pages) > 1:
                return tiffs  # batch of stacks
    return [path]  # single sequence (frame directory)


# --------------------------------------------------------------------------- #
# displacement estimation
# --------------------------------------------------------------------------- #
def frame_to_tensor(frame_rgb, device):
    t = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1).type(torch.float32)
    return t.unsqueeze(0).to(device)


def median_flow(frame1, frame2, model, iters, device):
    """Median (dy, dx) of the RAFT flow from frame1 to frame2."""
    with torch.no_grad():
        flow_up = model(frame_to_tensor(frame1, device),
                        frame_to_tensor(frame2, device),
                        iters=iters, test_mode=True)[1]
    dx, dy = flow_up.detach().cpu().numpy()[0]
    return [float(np.median(dy)), float(np.median(dx))]


def estimate_displacements(source, model, device, iters):
    """Per-frame incremental (dy, dx); first frame is the reference (0, 0).

    Each step measures frame i -> i+1; the caller cumsums these. Robust to
    large total drift but accumulates per-step bias over long sequences.
    """
    dyx = [[0.0, 0.0]]
    for ind in tqdm(range(len(source) - 1), desc="estimate"):
        dyx.append(median_flow(source.rgb(ind), source.rgb(ind + 1),
                               model, iters, device))
    return np.stack(dyx)


def estimate_displacements_anchored(source, model, device, iters):
    """Cumulative (dy, dx) of each frame vs frame 0, measured directly.

    Avoids the error accumulation of the incremental method (no cumsum), at the
    cost of requiring RAFT to span the full frame-0->frame-i displacement.
    Returns cumulative displacements directly (no cumsum needed downstream).
    """
    ref = source.rgb(0)
    dyxs = [[0.0, 0.0]]
    for ind in tqdm(range(1, len(source)), desc="estimate(anchor)"):
        dyxs.append(median_flow(ref, source.rgb(ind), model, iters, device))
    return np.stack(dyxs)


def estimate_displacements_keyframe(source, model, device, iters, interval):
    """Cumulative (dy, dx) via periodic re-anchoring (keyframes).

    A keyframe is used as reference for up to ``interval`` following frames, then
    advanced. This bounds the reference->frame distance (so RAFT always compares
    similar frames, avoiding the anchor mode's collapse) while chaining only
    ~N/interval offsets (so the incremental method's per-step accumulation is
    cut by ~interval). interval=1 == incremental, interval>=N == anchor.
    """
    n = len(source)
    dyxs = np.zeros((n, 2))
    ref_idx, ref_rgb, ref_off = 0, source.rgb(0), np.zeros(2)
    for i in tqdm(range(1, n), desc=f"estimate(kf={interval})"):
        d = median_flow(ref_rgb, source.rgb(i), model, iters, device)
        dyxs[i] = ref_off + d
        if i - ref_idx >= interval:        # advance the keyframe
            ref_idx, ref_rgb, ref_off = i, source.rgb(i), dyxs[i].copy()
    return dyxs


# --------------------------------------------------------------------------- #
# apply transform
# --------------------------------------------------------------------------- #
def pad_image(image, pad):
    pad_width = [(pad, pad), (pad, pad)] + [(0, 0)] * (image.ndim - 2)
    return np.pad(image, pad_width, mode="median")


def apply_trafo(image, dyx, order=5):
    move_tf = transform.AffineTransform(translation=(dyx[1], dyx[0]))
    out = transform.warp(image, move_tf.inverse, order=order,
                         preserve_range=True, cval=float(np.median(image)))
    return out.astype(image.dtype)


def warp_frame(frame, dyx, frame_mode, pad, order):
    if frame_mode == "pad":
        frame = pad_image(frame, pad)
    elif frame_mode == "crop":
        frame = frame[pad:-pad, pad:-pad]
    return apply_trafo(frame, dyx, order=order)


def align_frames(source, dyx, frame_mode, order, jobs):
    """Warp every frame; returns a list of aligned frames in native dtype."""
    pad = int(np.abs(dyx).max())
    n = len(source)
    if jobs == 1:
        return [warp_frame(source.raw(i), dyx[i], frame_mode, pad, order)
                for i in tqdm(range(n), desc="warp")]
    # Read frames serially (I/O), warp in parallel *processes*: high-order
    # spline warping uses scipy's spline filter, which is not thread-safe
    # (threads segfault), so loky processes are used instead of threads.
    frames = [source.raw(i) for i in range(n)]
    return Parallel(n_jobs=jobs, backend="loky")(
        delayed(warp_frame)(frames[i], dyx[i], frame_mode, pad, order)
        for i in tqdm(range(n), desc="warp")
    )


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_outputs(frames, dyx, dyxs, dyxm, out_dir, stem,
                  write_png, write_video, fps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = np.stack(frames)
    tiff_path = out_dir / f"{stem}_aligned.tif"
    tifffile.imwrite(str(tiff_path), stack)
    print("  wrote", tiff_path)

    np.savetxt(out_dir / "dyx.txt", dyx)
    np.savetxt(out_dir / "dyx_cumsum.txt", dyxs)
    np.savetxt(out_dir / "dyx_median.txt", dyxm)

    if write_png:
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            out = frame[:, :, ::-1] if frame.ndim == 3 else frame
            cv2.imwrite(str(img_dir / f"{i:07d}.png"), out)
        print("  wrote", img_dir)

    if write_video:
        vid_path = out_dir / f"{stem}_aligned.mp4"
        write_mp4(frames, str(vid_path), fps or 25)
        print("  wrote", vid_path)


def write_mp4(frames, path, fps):
    """Encode aligned frames to an mp4 via OpenCV's bundled ffmpeg (mp4v codec)."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")
    for frame in frames:
        writer.write(_to_rgb_uint8(frame)[:, :, ::-1])  # cv2 expects BGR
    writer.release()


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def stabilize_one(input_path, model, device, args):
    print(f"\n=== {input_path} ===")
    source = FrameSource(input_path)
    print(f"  {source.kind}: {len(source)} frames")
    if len(source) < 2:
        print("  skipped (need >= 2 frames)")
        return

    if args.keyframe and args.keyframe > 0:
        dyxs = estimate_displacements_keyframe(source, model, device,
                                               args.iters, args.keyframe)
        dyx = np.vstack([[0, 0], np.diff(dyxs, axis=0)])   # implied per-step
    elif args.anchor:
        dyxs = estimate_displacements_anchored(source, model, device, args.iters)
        dyx = np.vstack([[0, 0], np.diff(dyxs, axis=0)])   # implied per-step
    else:
        dyx = estimate_displacements(source, model, device, args.iters)
        dyxs = np.cumsum(dyx, 0)        # cumulative displacement vs. first frame
    dyxm = dyxs - np.median(dyxs, 0)    # centre the corrected trajectory (less crop loss)

    frames = align_frames(source, -dyxm, args.mode, args.order, args.jobs)

    stem = Path(input_path).stem if os.path.isfile(input_path) \
        else Path(input_path.rstrip("/\\")).name
    out_dir = Path(args.out) / stem
    fps = args.fps or source.fps
    write_outputs(frames, dyx, dyxs, dyxm, out_dir, stem,
                  args.png, args.video, fps)
    source.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="TIFF stack, directory of stacks/frames, or video")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda:0 (default: auto)")
    p.add_argument("--model", default="models/raftsintel.pth",
                   help="path to RAFT model checkpoint")
    p.add_argument("--out", default="out", help="output directory (default: out)")
    p.add_argument("--iters", type=int, default=11,
                   help="RAFT refinement iterations (default: 11)")
    p.add_argument("--order", type=int, default=5,
                   help="spline interpolation order for warping (default: 5)")
    p.add_argument("--mode", choices=["none", "pad", "crop"], default="none",
                   help="border handling before warping (default: none)")
    p.add_argument("--anchor", action="store_true",
                   help="estimate drift directly against frame 0 (no per-step "
                        "accumulation); reduces long-sequence drift")
    p.add_argument("--keyframe", type=int, default=0, metavar="K",
                   help="re-anchor every K frames (keyframes): robust to scene "
                        "change AND low accumulation. K=1 incremental, large=anchor")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel processes for the warp step (default: 1, -1 = all cores)")
    p.add_argument("--video", action="store_true",
                   help="also write an aligned mp4")
    p.add_argument("--png", action="store_true",
                   help="also write aligned frames as individual PNGs")
    p.add_argument("--fps", type=float, default=None,
                   help="frame rate for mp4 output (default: source fps or 25)")
    p.add_argument("--filter", default=None,
                   help="only process inputs whose name contains this string")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "none":
        args.mode = None

    inputs = detect_inputs(args.input)
    if args.filter:
        inputs = [p for p in inputs if args.filter in os.path.basename(p)]
    if not inputs:
        print("No matching inputs.")
        return
    print(f"{len(inputs)} input(s) to process")

    device = resolve_device(args.device)
    model = load_model(args.model, device)

    for input_path in inputs:
        try:
            stabilize_one(input_path, model, device, args)
        except Exception as exc:  # batch: log and continue
            print(f"  ERROR on {input_path}: {exc}")


if __name__ == "__main__":
    main()
