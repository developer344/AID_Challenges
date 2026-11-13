"""Learn to see when a finger touches the table, with the sound of the taps as the teacher.

A camera alone cannot easily tell a finger touching the table from one hovering just above it, but a tap makes a
sound. So: record a video, with sound, of your index finger tapping on the table. The sound says when the finger
touched (the onsets of the taps, found with the spectral flux, as in Part D of the lecture): those are the labels,
frame by frame. A small convolutional network then learns, from the picture alone, what a touching fingertip looks
like: its shadow meeting it, the finger stopping, its shape. Once trained it needs no sound.

    1. python touch_learn.py record taps.mov
           the camera and the microphone, guided (see "How to record" below): R starts, then TAP and HOVER in turn
           (20 s and 10 s) as the window says; it stops after 4 minutes, or with R. The taps heard show as you go.
           (QuickTime Player also works: File > New Movie Recording, with the camera and the microphone chosen in the
           menu next to the record button.)
    2. python touch_learn.py prepare taps.mov
           finds the taps in the sound and the hand in every frame, cuts a patch around the index fingertip and labels
           it; writes taps.touch.npz (the data) and taps.check.png (look at it: taps, sync, labels, example patches)
    3. python touch_learn.py train taps.touch.npz [more.touch.npz ...]
           learns from the first 70% of each recording, chooses the best epoch and threshold on the next 10%, and
           tests on the last 20%; writes touch_model.pt and touch_model.png (the training, and the test part: the
           network's probability against the taps heard)
    4. python touch_learn.py live                       the camera, the network deciding touch or no touch
       python touch_learn.py live --video test.mov      a recording instead: its taps (heard) are shown to compare
       Every touch is printed with where it was. Set area (A), then click the 4 corners of a rectangle on the table
       (a sheet of paper, say) in the picture, in order: top left, top right, bottom right, bottom left. Seen at an
       angle the rectangle is a trapezoid; a perspective transform (a homography) maps it back to a rectangle, and
       every touch gets coordinates on it, u and v from 0 to 1. The area is printed as --area ..., to start with it.

How to record (3 to 5 minutes; more recordings, in more places and lights, make the network more general):
  - The camera looks down at the table at an angle: tilt the laptop screen FORWARD (close it a little) until the table
    in front of the laptop is in the picture, or use an iPhone's Desk View (Continuity Camera). The hand and the table
    around it must be visible. Save the video in this folder (or give its path).
  - A lamp to one side, low, gives the finger a clear shadow, which meets the fingertip when it touches.
  - Tap with the INDEX finger only, at many places, quickly (touch and lift at once), about once a second, so that
    every touch makes a sound. A quiet room: no typing, no talking, no music.
  - In turn with the taps (record says when: 20 s tapping, 10 s hovering), move the finger just above the table,
    down and up, WITHOUT touching it: the network must learn that close is not touching. In turn, not all at the end:
    the end of the recording is the test part.
  - The first time, macOS asks to let the terminal (or VS Code) use the microphone: allow it.
  - Do not rest the finger on the table: a touch that lasts is heard only when it starts.

The labels, per frame: touching from the tap's sound to 80 ms after it; not touching from 250 ms away from any tap;
in between the frame is not used (the finger is landing or lifting). Sound and picture can be shifted by a few tens
of milliseconds (the camera is slower than the microphone): prepare estimates the shift by lining up the taps with
the moments the fingertip stops moving, and prints it; --offset (in seconds) sets it by hand.

The patch: a square around the index fingertip, turned so that the finger points down and scaled by the palm length
(landmark 0 to 9), as in the pose normalisation of Part C; in grey, 64 x 64 pixels; its three channels are the frames
t-2, t-1 and t, so the network also sees the motion.

OSC in live mode (button Send OSC, or --osc; to --osc-to HOST:PORT, default 127.0.0.1:12000):
    /touch x y state     when the finger touches (1) or leaves (0) the table; x, y the fingertip in the picture (0 to 1)
    /touch/prob p        every frame with a hand: the network's probability that the finger touches
    /area/touch u v state    with an area: a touch (1) inside it, and its release (0); u, v on the area, 0 to 1
    /area/finger u v         with an area: every frame with a hand, the fingertip on the area (for a cursor)

Requirements: those of compare_poses.py (in ../Compare_Poses_OSC, which this script uses), plus torch, av (to read and write videos with sound), scipy, matplotlib and
sounddevice (to record).
"""
import argparse
import bisect
import os
import queue
import sys
import threading
import time
from collections import deque
from fractions import Fraction

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Compare_Poses_OSC"))
import compare_poses as cp                   # MediaPipe hands, OSC and drawing helpers, from ../Compare_Poses_OSC
from touch_surface import label, pt          # text with an outline; a point as whole pixels

PATCH = 64                                   # side of the patch, in pixels
WRIST, MIDDLE, DIP, TIP = 0, 9, 7, 8         # the landmarks used: wrist, base of the middle finger, index finger
TOUCH_S, MARGIN_S = 0.08, 0.25               # touching until 80 ms after a tap; not touching from 250 ms away
BAR = 56                                     # height of the toolbar


def need(path, hint):
    """Stop with a clear message (not a traceback) when a file is not there."""
    if not os.path.isfile(path):
        sys.exit("There is no file %s (in %s).\n%s" % (path, os.getcwd(), hint))


# ---- 1. The taps, from the sound ----

def read_audio(path, rate=16000):
    """The sound of a video, mono, and the time of its first sample on the video's clock."""
    import av
    with av.open(path) as f:
        if not f.streams.audio:
            return None, rate, 0.0
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks, start = [], None
        for frame in f.decode(audio=0):
            if start is None:
                start = float(frame.time or 0.0)
            chunks += [r.to_ndarray().ravel() for r in resampler.resample(frame)]
        chunks += [r.to_ndarray().ravel() for r in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32), rate, start or 0.0


def spectral_flux(x, rate, n=512, hop=128):
    """How much the spectrum grows from one short window to the next (Part D): high at every tap."""
    x = np.pad(x, (0, max(n - len(x), 0)))
    window = np.hanning(n).astype(np.float32)
    starts = np.arange(0, len(x) - n + 1, hop)
    flux, prev = np.zeros(len(starts), np.float32), None
    for c in range(0, len(starts), 4096):                       # in pieces, to save memory
        mag = np.log1p(100 * np.abs(np.fft.rfft(x[starts[c:c + 4096, None] + np.arange(n)] * window, axis=1)))
        grow = np.diff(np.vstack([mag[:1] if prev is None else prev, mag]), axis=0)
        flux[c:c + len(mag)] = np.maximum(grow, 0).sum(axis=1)
        prev = mag[-1:]
    return flux, (starts + n / 2) / rate


def tap_onsets(flux, times, k=5.0, gap=0.12, floor=None):
    """The taps: peaks of the flux above an adaptive threshold (the local median plus k median deviations over 1 s,
    and at least a quarter of the loudest taps, or floor), at least 120 ms apart."""
    from scipy.ndimage import median_filter
    w = max(int(1 / (times[1] - times[0])), 3)
    med = median_filter(flux, w, mode="nearest")
    floor = 0.25 * np.percentile(flux, 99.9) if floor is None else floor
    thr = np.maximum(med + k * median_filter(np.abs(flux - med), w, mode="nearest"), floor) + 1e-6
    peaks = []
    for i in np.flatnonzero((flux > thr) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))):
        if peaks and times[i] - times[peaks[-1]] < gap:
            if flux[i] > flux[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    return times[np.array(peaks, int)], thr


# ---- 2. The patch around the fingertip ----

def video_frames(path):
    """The frames of a video file and their times in seconds, on the same clock as its sound."""
    import av
    with av.open(path) as f:
        stream = f.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate or 30)
        for k, frame in enumerate(f.decode(video=0)):
            yield (float(frame.time) if frame.time is not None else k / rate), frame.to_ndarray(format="bgr24")


def geometry(pts):
    """Where to cut: the centre (a little beyond the fingertip, where its shadow is), the finger's direction,
    the side of the square (1.2 palm lengths) and the palm length, from the hand keypoints in pixels."""
    palm = max(float(np.linalg.norm(pts[MIDDLE] - pts[WRIST])), 1.0)
    d = pts[TIP] - pts[DIP]
    d = d / max(float(np.linalg.norm(d)), 1e-6)
    return pts[TIP] + 0.25 * palm * d, d, 1.2 * palm, palm


def patch_matrix(centre, d, side):
    """The affine map from the picture to the patch: centred, turned so that d points down, scaled to PATCH pixels."""
    R = np.array([[d[1], -d[0]], [d[0], d[1]]]) * (PATCH / side)     # turns d into (0, 1), then scales
    return np.c_[R, PATCH / 2 - R @ centre]


def cut_patch(grays, centre, d, side):
    """The patch from the last three grey frames (t-2, t-1, t), at the same place: 3 x PATCH x PATCH, 0 to 255."""
    frames = list(grays)
    frames = [frames[0]] * (3 - len(frames)) + frames
    M = patch_matrix(centre, d, side)
    return np.stack([cv2.warpAffine(g, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                     for g in frames])


# ---- 3. Prepare: labels from the sound, patches from the picture ----

def estimate_offset(frame_t, stops, onsets, most=0.25):
    """The shift of the picture against the sound: the one that best lines up the taps with the fingertip stops."""
    fps = 1 / np.median(np.diff(frame_t))
    at = np.round(np.interp(onsets, frame_t, np.arange(len(frame_t)))).astype(int)   # the frame of each tap
    shifts = np.arange(-int(most * fps), int(most * fps) + 1)
    scores = np.array([np.mean([stops[i + s] for i in at if 0 <= i + s < len(stops)]) for s in shifts])
    b = int(np.argmax(scores))
    best = float(shifts[b])
    if 0 < b < len(scores) - 1:                                  # between frames: the top of a parabola through 3 points
        bend = scores[b - 1] - 2 * scores[b] + scores[b + 1]
        if bend < 0:
            best += 0.5 * (scores[b - 1] - scores[b + 1]) / bend
    return best / fps, shifts / fps, scores


def labels_for(t, taps, fps):
    """1 touching (from the tap to TOUCH_S after it), 0 not touching (MARGIN_S from any tap), -1 not used."""
    if len(taps) == 0:
        return np.zeros(len(t), np.int8)
    j = np.searchsorted(taps, t)
    before, after = taps[np.clip(j - 1, 0, len(taps) - 1)], taps[np.clip(j, 0, len(taps) - 1)]
    y = np.where(np.minimum(np.abs(t - before), np.abs(after - t)) < MARGIN_S, -1, 0).astype(np.int8)
    for tap in (before, after):
        y[(t - tap >= -0.5 / fps) & (t - tap <= TOUCH_S)] = 1
    return y


def prepare(args):
    need(args.video, "Record one first: python touch_learn.py record %s" % args.video)
    x, rate, a0 = read_audio(args.video)
    if x is None:
        sys.exit("%s has no sound: record it with the microphone on." % args.video)
    flux, ft = spectral_flux(x, rate)
    ft = ft + a0
    onsets, thr = tap_onsets(flux, ft, args.threshold)
    print("%d taps heard in %.0f s of sound" % (len(onsets), len(x) / rate))
    if len(onsets) < 10:
        sys.exit("Too few taps heard: tap louder, or lower --threshold (now %g)." % args.threshold)

    det = cp.Detector(cp.Kind(True, False), video=True)
    grays, T, X, keep, tips, palms = deque(maxlen=3), [], [], [], [], []
    shown = time.time()
    for k, (t, frame) in enumerate(video_frames(args.video)):
        frame = cp.fit(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        pts = det.find(frame)
        T.append(t)
        if pts is None:
            tips.append((np.nan, np.nan))
            palms.append(np.nan)
        else:
            centre, d, side, palm = geometry(pts)
            X.append(cut_patch(grays, centre, d, side))
            keep.append(k)
            tips.append(pts[TIP])
            palms.append(palm)
        if time.time() - shown > 3:
            print("  frame %d (%.0f s), hand found in %d" % (k + 1, t, len(keep)))
            shown = time.time()
    T, keep, tips, palms = np.array(T), np.array(keep, int), np.array(tips, float), np.array(palms)
    if len(keep) < 100:
        sys.exit("The hand was found in only %d of %d frames: is it in the picture?" % (len(keep), len(T)))
    fps = 1 / np.median(np.diff(T))

    # The moments the fingertip stops: in the frame where it arrives, its speed in (from the frame before) is high and
    # its speed out (to the next frame) is low. Squared, so that the hard stops of taps count much more than the soft
    # stops of hovering and the jitter of the keypoints. Speeds in palm lengths per frame.
    speed_in = np.r_[np.nan, np.linalg.norm(np.diff(tips, axis=0), axis=1) / palms[1:]]
    stops = np.nan_to_num(np.maximum(speed_in - np.r_[speed_in[1:], np.nan], 0)) ** 2
    shift, shifts, scores = estimate_offset(T, stops, onsets)
    offset = shift if args.offset is None else args.offset
    best = int(np.argmax(scores))
    others = np.delete(scores, [i for i in (best - 1, best, best + 1) if 0 <= i < len(scores)])
    clear = scores[best] / (np.mean(others) + 1e-12)        # how much the best shift stands out
    print("Sound and picture: the picture is %+.0f ms late (%s)" % (offset * 1000, "estimated" if args.offset is None else "given"))
    if args.offset is None and clear < 3:
        print("  The estimate is uncertain (the best shift is only %.1f times the others): look at the sync plot, record more "
              "taps, or give the shift found on a longer recording with --offset (it depends on the camera and the "
              "microphone, not on the recording)." % clear)
    taps = onsets + offset
    y = labels_for(T[keep], taps, fps)
    out = os.path.splitext(args.video)[0] + ".touch.npz"
    np.savez_compressed(out, X=np.array(X, np.uint8), y=y, t=T[keep], taps=taps, fps=fps, offset=offset)
    print("%d frames, hand in %d: %d touching, %d not touching, %d not used. Saved %s"
          % (len(T), len(keep), (y == 1).sum(), (y == 0).sum(), (y == -1).sum(), out))
    check_figure(os.path.splitext(args.video)[0] + ".check.png", ft, flux, thr, onsets, T, stops, taps, keep, y,
                 np.array(X, np.uint8), shifts, scores, offset)


def check_figure(path, ft, flux, thr, onsets, T, stops, taps, keep, y, X, shifts, scores, offset):
    """A picture to check the data: the taps in the sound, the fingertip stops, the labels, the sync, examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = onsets[min(2, len(onsets) - 1)] - 1
    t1 = t0 + 8
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1.25, 1.25], hspace=0.65, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    m = (ft >= t0) & (ft <= t1)
    ax.plot(ft[m], flux[m], color="#555555", lw=0.8, label="spectral flux")
    ax.plot(ft[m], thr[m], "--", color="#E36C0A", lw=1, label="threshold")
    for o in onsets[(onsets >= t0) & (onsets <= t1)]:
        ax.axvline(o, color="#E36C0A", lw=1.5)
    ax.set_title("1. The taps in the sound (orange lines): peaks of the spectral flux above the threshold", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(t0, t1)
    ax = fig.add_subplot(gs[1, :])
    m = (T >= t0) & (T <= t1)
    ax.plot(T[m], stops[m], color="#2E75B6", lw=1.5, label="fingertip stopping (drop of its speed)")
    tk = T[keep]
    for lab, colour in ((1, "#E36C0A"), (-1, "#DDDDDD")):
        for t in tk[(y == lab) & (tk >= t0) & (tk <= t1)]:
            ax.axvspan(t - 0.016, t + 0.016, color=colour, alpha=0.5 if lab == 1 else 0.6, lw=0)
    ax.set_title("2. The labels on the picture's clock (picture %+.0f ms late): orange touching, grey not used, white not touching"
                 % (offset * 1000), loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (s)")
    ax = fig.add_subplot(gs[2:, :2])
    ax.plot(shifts * 1000, scores, "o-", color="#2E75B6")
    ax.axvline(offset * 1000, color="#E36C0A")
    ax.set_title("3. Sync: how well the taps line up\nwith the fingertip stops, per shift", loc="left", fontsize=11)
    ax.set_xlabel("picture late by (ms)")
    rng = np.random.default_rng(0)
    for row, (lab, name) in enumerate(((1, "touching"), (0, "not touching"))):
        idx = np.flatnonzero(y == lab)
        pick = rng.choice(idx, min(4, len(idx)), replace=False) if len(idx) else []
        for c, i in enumerate(pick):
            ax = fig.add_subplot(gs[2 + row, 2 + c])
            ax.imshow(X[i][2], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color="#E36C0A" if lab else "#2E75B6")
            if row == 0 and c == 0:
                ax.set_title("4. Patches (frame t), as the network sees them", loc="left", fontsize=11)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    print("Saved", path)


# ---- 4. The network ----

class TouchNet(nn.Module):
    """Four convolution blocks (64 -> 32 -> 16 -> 8 -> 4 pixels), then one number: touching or not (a logit)."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        """x: patches, (batch, 3, 64, 64), values 0 to 255. Each patch is standardised first (light does not matter)."""
        x = x.float()
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1.0)
        return self.head(self.features(x)).squeeze(1)


def augment(x):
    """More variety for training: mirror, shift by a few pixels, darker or lighter shadows, noise."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dy, dx = np.random.randint(-4, 5, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    gamma = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.7, 1.4)
    x = 255 * (x / 255).clamp(min=1e-4) ** gamma
    return x + 3 * torch.randn_like(x)


def predict(model, X, device, batch=512):
    """The probability of touching for each patch."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.as_tensor(X[i:i + batch], device=device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def touches_found(t, p, thr, refractory=0.15):
    """The touches the network finds: the frames where its probability rises above the threshold."""
    found, before = [], False
    for ti, pi in zip(t, p):
        above = pi >= thr
        if above and not before and (not found or ti - found[-1] > refractory):
            found.append(ti)
        before = above
    return np.array(found)


def score_touches(found, taps, tol=0.1):
    """Taps found (a touch within 100 ms of the sound), and touches found where no tap was heard."""
    hit = sum(bool(len(found)) and np.min(np.abs(found - o)) <= tol for o in taps)
    false = sum(not len(taps) or np.min(np.abs(taps - f)) > tol for f in found)
    return hit, false


def train(args):
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    # Each recording is split in time: the first 70% to learn from, the next 10% to choose the epoch and the threshold,
    # the last 20% to test. Not at random: neighbouring frames are almost the same picture, so a random split would
    # test on frames the network has practically seen. The parts are 1 s apart for the same reason.
    Xtr, ytr, Xva, yva, tests = [], [], [], [], []
    for path in args.data:
        need(path, "Make it from a video first: python touch_learn.py prepare VIDEO.mov (it writes VIDEO.touch.npz).")
    for path in args.data:
        d = np.load(path)
        X, y, t, taps = d["X"], d["y"], d["t"], d["taps"]
        a, b = t.min() + 0.7 * (t.max() - t.min()), t.min() + 0.8 * (t.max() - t.min())
        learn, choose, test = (t < a - 1) & (y >= 0), (t >= a) & (t < b - 1) & (y >= 0), t >= b
        Xtr.append(X[learn])
        ytr.append(y[learn])
        Xva.append(X[choose])
        yva.append(y[choose])
        heard = taps[(taps >= b) & (taps <= t.max())]
        heard = np.array([o for o in heard if np.min(np.abs(t - o)) < 0.1])    # taps with the hand in the picture
        tests.append(dict(X=X[test], y=y[test], t=t[test], taps=heard, name=os.path.basename(path)))
        print("%s: %d frames to learn from (%d touching), %d to choose, %d to test (%d taps)"
              % (path, learn.sum(), (y[learn] == 1).sum(), choose.sum(), test.sum(), len(heard)))
    Xtr, ytr, Xva, yva = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xva), np.concatenate(yva)
    pos, neg = (ytr == 1).sum(), (ytr == 0).sum()
    if pos < 20 or (yva == 1).sum() < 3:
        sys.exit("Too few touching frames (%d to learn from, %d to choose): record more taps." % (pos, (yva == 1).sum()))

    torch.manual_seed(0)
    np.random.seed(0)
    model = TouchNet().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)   # smaller and smaller steps
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg / pos), dtype=torch.float32, device=device))   # touching frames are rare
    Xg, yg = torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device).float()
    history, best = [], (-1.0, None, 0.5)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(Xg), device=device).split(args.batch):
            loss = loss_fn(model(augment(Xg[idx].float())), yg[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        schedule.step()
        f1, thr = best_threshold(predict(model, Xva, device), yva == 1)
        history.append((total / len(Xg), f1))
        print("epoch %2d  training loss %.3f  choosing part: F1 %.2f (threshold %.2f)" % (epoch + 1, total / len(Xg), f1, thr))
        if f1 >= best[0]:
            best = (f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, thr)
    model.load_state_dict(best[1])
    thr = best[2]

    # The test: frames the network has never seen, and with the threshold chosen before
    hits = falses = taps = 0
    frames_p, frames_y = [], []
    for s in tests:
        s["p"] = predict(model, s["X"], device)
        h, f = score_touches(touches_found(s["t"], s["p"], thr), s["taps"])
        hits, falses, taps = hits + h, falses + f, taps + len(s["taps"])
        frames_p.append(s["p"][s["y"] >= 0])
        frames_y.append(s["y"][s["y"] >= 0] == 1)
    f1 = f1_score(np.concatenate(frames_p) >= thr, np.concatenate(frames_y))
    print("Test part, frame by frame: F1 %.2f (threshold %.2f, chosen on the choosing part)" % (f1, thr))
    print("Test part, taps: %d heard, %d found by the network (%.0f%%), %d touches found where no tap was heard"
          % (taps, hits, 100 * hits / max(taps, 1), falses))
    offset = float(np.mean([np.load(f)["offset"] for f in args.data]))
    torch.save({"state": best[1], "threshold": thr, "offset": offset, "patch": PATCH, "data": args.data}, args.out)
    print("Saved", args.out)
    training_figure(os.path.splitext(args.out)[0] + ".png", history, tests[0], thr)


def best_threshold(p, true):
    """The threshold with the best F1; when several are as good, the middle one."""
    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = np.array([f1_score(p >= th, true) for th in ths])
    good = ths[f1s >= f1s.max() - 1e-9]
    return float(f1s.max()), float(good[len(good) // 2])


def f1_score(pred, true):
    tp = np.sum(pred & true)
    return float(2 * tp / max(np.sum(pred) + np.sum(true), 1))


def training_figure(path, history, test, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1, 2.4]))
    h = np.array(history)
    a1.plot(range(1, len(h) + 1), h[:, 0], color="#2E75B6", label="training loss")
    a1.plot(range(1, len(h) + 1), h[:, 1], "--", color="#E36C0A", label="F1, choosing part")
    a1.set_xlabel("epoch")
    a1.legend(fontsize=9)
    a1.set_title("Training", loc="left")
    t, p = test["t"], test["p"]
    t0 = t.min() + 1
    m = (t >= t0) & (t <= t0 + 12)
    a2.plot(t[m], p[m], ".-", color="#2E75B6", lw=1, ms=3, label="probability of touching")
    a2.axhline(thr, ls="--", color="#555555", lw=1, label="threshold %.2f" % thr)
    for o in test["taps"][(test["taps"] >= t0) & (test["taps"] <= t0 + 12)]:
        a2.axvline(o, color="#E36C0A", lw=1.5, alpha=0.8)
    a2.set_ylim(-0.02, 1.02)
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right", fontsize=9)
    a2.set_title("Test part of %s: the network's probability, and the taps heard (orange)" % test["name"], loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    print("Saved", path)


# ---- 5. Live ----

class Area:
    """A rectangle on the table (a sheet of paper, say), seen at an angle as a trapezoid. Its 4 corners in the picture,
    clicked in order, become (0, 0), (1, 0), (1, 1) and (0, 1); a perspective transform (a homography, the same as
    for a document scanned with a phone) maps any point of the picture to the area's coordinates u and v."""

    def __init__(self):
        self.corners, self.H = None, None

    def set(self, corners):
        """False if the corners do not go round the area (crossed, or not in order)."""
        c = np.float32(corners).reshape(4, 2)
        e = np.roll(c, -1, axis=0) - c                       # the four sides, in order
        f = np.roll(e, -1, axis=0)
        turns = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]        # which way each corner turns
        if not (np.all(turns > 0) or np.all(turns < 0)):
            return False
        self.corners = c
        self.H = cv2.getPerspectiveTransform(c, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
        return True

    def uv(self, p):
        return cv2.perspectiveTransform(np.float32(p).reshape(1, 1, 2), self.H)[0, 0]

    def inside(self, uv):
        return uv is not None and bool(np.all((uv >= 0) & (uv <= 1)))

    def picture(self, uvs):
        """From the area back to the picture."""
        return cv2.perspectiveTransform(np.float32(uvs).reshape(-1, 1, 2), np.linalg.inv(self.H))[:, 0]

    def args(self):
        return "--area " + " ".join("%d %d" % (round(x), round(y)) for x, y in self.corners)


CORNERS = ("top left, (0, 0)", "top right, (1, 0)", "bottom right, (1, 1)", "bottom left, (0, 1)")


def where(area, uv, tip, shape):
    """Where a touch is, as text: on the area (u, v), or in the picture (x, y, 0 to 1) when there is no area."""
    if area.H is None:
        return "x %.2f  y %.2f" % (tip[0] / shape[1], tip[1] / shape[0])
    return "u %.2f  v %.2f" % (uv[0], uv[1]) if area.inside(uv) else "outside the area"


def draw_area(frame, area, picking):
    """The area on the picture: its outline, a grid in perspective and its numbered corners; while it is being set,
    the corners clicked so far."""
    if picking is not None:
        for i, q in enumerate(picking):
            if i:
                cv2.line(frame, pt(picking[i - 1]), pt(q), cp.GREEN, 2, cv2.LINE_AA)
            cv2.circle(frame, pt(q), 6, cp.GREEN, -1, cv2.LINE_AA)
            label(frame, str(i + 1), (q[0] + 8, q[1] - 8), 0.6, cp.GREEN, 2)
        return
    if area.H is None:
        return
    for k in (0.25, 0.5, 0.75):
        for a, b in (((k, 0), (k, 1)), ((0, k), (1, k))):
            qa, qb = area.picture([a, b])
            cv2.line(frame, pt(qa), pt(qb), cp.GREEN, 1, cv2.LINE_AA)
    cv2.polylines(frame, [area.corners.astype(np.int32)], True, cp.GREEN, 2, cv2.LINE_AA)
    for i, q in enumerate(area.corners):
        label(frame, str(i + 1), (q[0] + 6, q[1] - 6), 0.55, cp.GREEN, 2)


def area_view(img, x0, y0, area, aspect, finger, touching, touches):
    """The area seen from above, in its own coordinates (u to the right, v down, from 0 to 1): the fingertip and the
    last touches."""
    cp.text(img, "The area, from above", x0, y0 - 12, 0.55, cp.INK, 2)
    if area.H is None:
        for i, s in enumerate(("None yet: click Set area (A), then", "its 4 corners in the picture. Each",
                               "touch then gets coordinates on it,", "from 0 to 1.")):
            cp.text(img, s, x0, y0 + 14 + i * 22, 0.48, cp.GREY)
        return
    w = int(min(212, 150 * aspect))
    h = int(w / aspect)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (246, 246, 246), -1)
    for k in (0.25, 0.5, 0.75):
        cv2.line(img, (x0 + int(k * w), y0), (x0 + int(k * w), y0 + h), (220, 220, 220), 1)
        cv2.line(img, (x0, y0 + int(k * h)), (x0 + w, y0 + int(k * h)), (220, 220, 220), 1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), cp.GREEN, 2)
    P = lambda uv: (x0 + int(uv[0] * w), y0 + int(uv[1] * h))
    for _, _, uv in touches:
        if area.inside(uv):
            cv2.circle(img, P(uv), 4, cp.ORANGE, -1, cv2.LINE_AA)
    tx = x0 + w + 14
    if area.inside(finger):
        cv2.circle(img, P(finger), 8, cp.ORANGE if touching else cp.BLUE, -1 if touching else 2, cv2.LINE_AA)
        cp.text(img, "u %.2f" % finger[0], tx, y0 + 16, 0.5, cp.INK)
        cp.text(img, "v %.2f" % finger[1], tx, y0 + 38, 0.5, cp.INK)
    else:
        cp.text(img, "finger", tx, y0 + 16, 0.5, cp.GREY)
        cp.text(img, "outside" if finger is not None else "not seen", tx, y0 + 38, 0.5, cp.GREY)
    last = next((uv for _, _, uv in reversed(touches) if area.inside(uv)), None)
    if last is not None:
        cp.text(img, "last touch: u %.2f, v %.2f" % (last[0], last[1]), x0, y0 + h + 22, 0.5, cp.ORANGE, 2)


def live_panel(patch, p, thr, touching, count, height, summary, area, aspect, finger, touches):
    """Beside the picture: what the network sees (the three frames of the patch), its answer, and the area."""
    w = 330
    img = np.full((max(height, 420), w, 3), 255, np.uint8)
    cp.text(img, "What the network sees", 12, 24, 0.55, cp.INK, 2)
    for i, name in enumerate(("t-2", "t-1", "t")):
        x = 12 + i * 94
        if patch is not None:
            img[34:118, x:x + 84] = cv2.cvtColor(cv2.resize(patch[i], (84, 84), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        else:
            cv2.rectangle(img, (x, 34), (x + 83, 117), (235, 235, 235), -1)
        cp.text(img, name, x + 30, 134, 0.45, cp.GREY)
    cp.text(img, "Probability of touching", 12, 160, 0.55, cp.INK, 2)
    cv2.rectangle(img, (12, 170), (292, 192), (235, 235, 235), -1)
    if p is not None:
        cv2.rectangle(img, (12, 170), (12 + max(int(p * 280), 2), 192), cp.ORANGE if touching else cp.BLUE, -1)
        cp.text(img, "%.2f" % p, 250, 210, 0.45, cp.INK)
    x = 12 + int(thr * 280)
    cv2.line(img, (x, 165), (x, 197), cp.INK, 2)
    cp.text(img, "threshold %.2f" % thr, 12, 210, 0.45, cp.GREY)
    cp.text(img, "TOUCH" if touching else "no touch", 12, 242, 0.8, cp.ORANGE if touching else cp.GREY, 2)
    cp.text(img, "touches: %d" % count, 190, 240, 0.55, cp.INK)
    if summary:
        cp.text(img, summary, 12, 264, 0.45, cp.INK)
    area_view(img, 12, 302, area, aspect, finger, touching, touches)
    return img


def live_plot(history, width, thr):
    """The probability over the last frames, the threshold, the touches (shaded) and the taps heard (orange)."""
    h, x0, x1, y0, y1 = 140, 40, width - 20, 32, 126
    img = np.full((h, width, 3), 255, np.uint8)
    cp.text(img, "Probability of touching over time" + ("  (orange: taps heard in the video)" if any(e[2] for e in history) else ""),
            12, 20, 0.5, cp.INK)
    n = history.maxlen
    X = lambda i: int(x0 + i * (x1 - x0) / (n - 1))
    Y = lambda v: int(y1 - np.clip(v, 0, 1) * (y1 - y0))
    for i, (p, touching, tap) in enumerate(history):
        if touching:
            cv2.rectangle(img, (X(i), y0), (X(i) + max((x1 - x0) // n, 1), y1), (200, 225, 250), -1)
        if tap:
            cv2.line(img, (X(i), y0), (X(i), y1), cp.ORANGE, 2)
    cv2.line(img, (x0, Y(thr)), (x1, Y(thr)), cp.GREY, 1)
    cv2.line(img, (x0, y1), (x1, y1), (180, 180, 180), 1)
    for i in range(1, len(history)):
        a, b = history[i - 1][0], history[i][0]
        if a is not None and b is not None:
            cv2.line(img, (X(i - 1), Y(a)), (X(i), Y(b)), cp.BLUE, 2, cv2.LINE_AA)
    return img


def toolbar(width, camera, more):
    """The camera buttons, then more: (text, action, on, colour when on). Returns the image and (x0, y0, x1, y1, action)."""
    bar = np.full((BAR, width, 3), 245, np.uint8)
    buttons, x = [], 10

    def button(text, action, on, colour=cp.BLUE):
        nonlocal x
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 22
        cv2.rectangle(bar, (x, 12), (x + w, 44), colour if on else (255, 255, 255), -1)
        cv2.rectangle(bar, (x, 12), (x + w, 44), (170, 170, 170), 1)
        cp.text(bar, text, x + 11, 34, 0.55, (255, 255, 255) if on else cp.INK)
        buttons.append((x, 12, x + w, 44, action))
        x += w + 6

    cp.text(bar, "Camera", x, 34, 0.55)
    x += 72
    for n in range(cp.CAMERAS):
        button(str(n), n, n == camera)
    x += 16
    for b in more:
        button(*b)
    return bar, buttons


def live(args):
    need(args.model, "Train one first: python touch_learn.py train VIDEO.touch.npz (or give another with --model).")
    if args.video:
        need(args.video, "Give the path of a recording, or leave out --video to use the camera.")
    saved = torch.load(args.model, map_location="cpu")
    model = TouchNet()
    model.load_state_dict(saved["state"])
    model.eval()
    thr = args.threshold if args.threshold is not None else saved["threshold"]
    area = Area()
    if args.area and not area.set(args.area):
        sys.exit("--area: the corners must go round the area, in order: top left, top right, bottom right, bottom left.")
    det = cp.Detector(cp.Kind(True, False), video=True)
    taps, camera, cap, frames = np.zeros(0), None, None, None
    if args.video:
        frames = video_frames(args.video)
        x, rate, a0 = read_audio(args.video)
        if x is not None:                                    # the taps heard, to compare with the network
            flux, ft = spectral_flux(x, rate)
            taps = tap_onsets(flux, ft + a0)[0] + (saved["offset"] if args.offset is None else args.offset)
    else:
        camera = args.camera
        cap = cp.open_camera(camera)
    osc, sending = cp.Osc(args.osc_to), args.osc
    grays, history = deque(maxlen=3), deque([(None, False, False)] * 200, maxlen=200)
    touches, picking, down_uv = deque(maxlen=12), None, None   # the last touches: (when, fingertip, u v)
    clicks, buttons = [], []
    cv2.namedWindow("Touch, learned")
    cv2.setMouseCallback("Touch, learned", lambda event, x, y, flags, param: clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    touching, found, message, said = False, [], "" if (cap or frames) else "Camera %s gives no picture" % camera, time.time()
    period, last, t_prev, start = 0.0, time.time(), None, time.time()
    while True:
        if frames is not None:
            nxt = next(frames, None)
            if nxt is None:
                break                                        # the end of the video
            t, frame = nxt
            ok = True
        else:
            ok, frame = cap.read() if cap else (False, None)
            t = time.time() - start
        frame = cp.fit(frame) if ok else np.full((480, 640, 3), 230, np.uint8)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        pts = det.find(frame) if ok else None

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        action = {ord("o"): "osc", ord("a"): "area", ord("c"): ((camera or 0) + 1) % cp.CAMERAS}.get(key)
        while clicks:
            cx, cy = clicks.pop(0)
            hit = next((a for x0, y0, x1, y1, a in buttons if x0 <= cx <= x1 and y0 <= cy <= y1), None)
            if hit is not None:
                action = hit
            elif picking is not None and cx < frame.shape[1] and BAR <= cy < BAR + frame.shape[0]:
                picking.append((cx, cy - BAR))               # a corner of the area, in the picture
                if len(picking) == 4:
                    if area.set(picking):
                        message = "Area set. To start with it: %s" % area.args()
                        print("Area set. To start with it next time: python touch_learn.py live %s" % area.args())
                    else:
                        message = "The corners must go round the area, in order: Set area again"
                    picking, said = None, time.time()
        if action is not None:
            said = time.time()
        if action == "osc":
            sending = not sending
            message = "Sending OSC to %s:%d" % osc.target if sending else "OSC off"
        elif action == "area":
            picking = [] if picking is None else None        # start clicking the corners, or stop
        elif isinstance(action, int) and action != camera:
            new = cp.open_camera(action)
            if new:
                if cap:
                    cap.release()
                cap, camera, frames, taps, message = new, action, None, np.zeros(0), "Camera %d" % action
                start = time.time()
            else:
                message = "Camera %d is not available" % action

        # The network
        p, patch = None, None
        if pts is not None:
            centre, d, side, palm = geometry(pts)
            patch = cut_patch(grays, centre, d, side)
            with torch.no_grad():
                p = float(torch.sigmoid(model(torch.as_tensor(patch[None]))))
        was = touching
        touching = p is not None and p >= thr
        if touching and not was and found and t - found[-1] < 0.15:
            touching = was                                   # too soon after the last touch: the same one
        uv = area.uv(pts[TIP]) if area.H is not None and pts is not None else None   # the fingertip on the area
        if touching != was:
            tip = pts[TIP].copy() if pts is not None else (touches[-1][1] if touches else np.zeros(2))
            if touching:
                found.append(t)
                touches.append((time.time(), tip, uv))
                down_uv = uv if area.inside(uv) else None
            print("%7.2f s  %-7s  %s  (p %.2f)" % (t, "touch" if touching else "release", where(area, uv, tip, frame.shape),
                                                   p if p is not None else 0))
            if sending:
                osc.send("/touch", [tip[0] / frame.shape[1], tip[1] / frame.shape[0], int(touching)])
                if down_uv is not None:                      # a touch inside the area; its release, at the same place
                    osc.send("/area/touch", [float(down_uv[0]), float(down_uv[1]), int(touching)])
            if not touching:
                down_uv = None
        if sending and p is not None:
            osc.send("/touch/prob", [p])
        if sending and uv is not None:
            osc.send("/area/finger", [float(uv[0]), float(uv[1])])
        heard = bool(len(taps)) and t_prev is not None and np.any((taps > t_prev) & (taps <= t))
        t_prev = t
        history.append((p, touching, heard))

        # The picture
        draw_area(frame, area, picking)
        if pts is not None:
            cp.draw_skeleton(frame, pts, cp.Kind(True, False), cp.BLUE)
            corners = cv2.transform(np.array([[[0, 0], [PATCH, 0], [PATCH, PATCH], [0, PATCH]]], np.float32),
                                    cv2.invertAffineTransform(patch_matrix(centre, d, side)))[0]
            cv2.polylines(frame, [corners.astype(np.int32)], True, cp.ORANGE if touching else (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, pt(pts[TIP]), 8, cp.ORANGE if touching else (255, 255, 255), -1 if touching else 2, cv2.LINE_AA)
        now = time.time()
        for when, q, _ in touches:                           # the touches of the last 1.5 s, and where the last one was
            if now - when < 1.5:
                cv2.circle(frame, pt(q), 14, cp.ORANGE, 2, cv2.LINE_AA)
        if touches and now - touches[-1][0] < 1.5:
            q = touches[-1][1]
            label(frame, where(area, touches[-1][2], q, frame.shape),
                  (min(q[0] + 18, frame.shape[1] - 190), max(q[1] - 18, 50)), 0.65, cp.ORANGE, 2)
        period, last = cp.smooth(period, now - last), now
        if ok and period > 0:
            label(frame, "%.0f fps" % (1 / period), (10, 24), 0.6, cp.INK, 1)
        summary = ""
        if len(taps):
            done = taps[taps <= t]
            hit, false = score_touches(np.array(found), done)
            summary = "taps heard %d, found %d, false touches %d" % (len(done), hit, false)
        out = cp.side_by_side(frame, live_panel(patch, p, thr, touching, len(found), frame.shape[0], summary, area,
                                                args.aspect, uv, touches))
        out = np.vstack([out, live_plot(history, out.shape[1], thr)])
        out = np.pad(out, ((0, 70), (0, 0), (0, 0)), constant_values=255)
        bar, buttons = toolbar(out.shape[1], camera, [("Set area (A)", "area", picking is not None, cp.GREEN),
                                                      ("Send OSC (O)", "osc", sending)])
        out = cp.stack(bar, out)
        if message and time.time() - said > 4:
            message = ""
        if picking is not None:
            warning = "Click corner %d of 4 in the picture: %s" % (len(picking) + 1, CORNERS[len(picking)])
        else:
            warning = message if message else ("No hand found" if ok and pts is None else "")
        cp.text(out, warning, 10, out.shape[0] - 38, 0.6, cp.GREEN if picking is not None else cp.RED, 2)
        cp.text(out, "%s   OSC %s   keys: A area, O OSC, C camera, Q quit" % ("video" if frames is not None else "camera %s" % camera,
                "to %s:%d" % osc.target if sending else "off"), 10, out.shape[0] - 12, 0.5, cp.GREY)
        cv2.imshow("Touch, learned", out)
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    if len(taps):
        hit, false = score_touches(np.array(found), taps[taps <= t])
        print("Taps heard: %d, found by the network: %d, touches where no tap was heard: %d" % (len(taps[taps <= t]), hit, false))


# ---- 6. Record: the video with the sound of the taps (step 1) ----

class Camera:
    """The camera, read in a thread of its own so that every frame gets the time it arrived, however long the drawing
    and the writing take. Keeps the newest frame and, while recording (keep), all of them in a queue."""

    def __init__(self, index):
        self.cap, self.newest, self.frames, self.keep, self.running = cp.open_camera(index), None, queue.Queue(), False, True
        self.arrivals = deque(maxlen=31)                         # the last arrival times, for the frame rate
        self.thread = threading.Thread(target=self.read, daemon=True)
        if self.cap:
            self.thread.start()

    def read(self):
        while self.running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            self.newest = (t, frame)
            self.arrivals.append(t)
            if self.keep:
                self.frames.put((t, frame))

    def fps(self):
        a = list(self.arrivals)
        return (len(a) - 1) / (a[-1] - a[0]) if len(a) > 2 and a[-1] > a[0] else 0.0

    def close(self):
        self.running = False
        if self.cap:
            self.thread.join(1)
            self.cap.release()


class Microphone:
    """The microphone, in blocks, each with the time its first sample was taken, on the camera's clock."""

    def __init__(self, device):
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        self.name, self.rate, self.blocks = info["name"], int(info["default_samplerate"]), queue.Queue()

        def heard(data, n, when, status):
            now = time.perf_counter()
            ago = when.currentTime - when.inputBufferAdcTime          # how long ago the first sample was taken
            self.blocks.put((now - (ago if 0 < ago < 0.5 else n / self.rate), data[:, 0].copy(), bool(status.input_overflow)))

        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.rate, dtype="float32", callback=heard)
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


class Ears:
    """The taps in the sound as it comes, to show them while recording (prepare finds them again, in the file)."""

    def __init__(self, rate):
        self.rate, self.n = rate, int(round(0.032 * rate))          # as in prepare: windows of 32 ms, 8 ms apart
        self.hop = self.n // 4
        self.buf, self.t_buf, self.fresh = np.zeros(0, np.float32), None, True
        self.flux, self.times, self.thr, self.taps = [], [], np.zeros(0), []
        self.floor, self.floor_at, self.first, self.peaks = None, 0.0, None, deque()

    def hear(self, start, x):
        self.first = start if self.first is None else self.first
        self.peaks.append((start, float(np.abs(x).max()) if len(x) else 0.0))
        while self.peaks and self.peaks[0][0] < start - 3:
            self.peaks.popleft()
        if self.t_buf is None:
            self.t_buf = start
        self.buf = np.r_[self.buf, x]
        if len(self.buf) >= self.n + self.hop:
            flux, t = spectral_flux(self.buf, self.rate, self.n, self.hop)
            skip = 0 if self.fresh else 1                           # the first window is the last one of the time before
            self.flux += flux[skip:].tolist()
            self.times += (self.t_buf + t[skip:]).tolist()
            self.fresh, last = False, (len(flux) - 1) * self.hop
            self.buf, self.t_buf = self.buf[last:], self.t_buf + last / self.rate

    def find(self, now):
        """The new taps: tap_onsets on the last 10 s, with its least threshold from all the sound so far."""
        if len(self.flux) < 100:
            return
        if self.floor is None or now - self.floor_at > 2:
            self.floor, self.floor_at = 0.25 * np.percentile(self.flux, 99.9), now
        w = int(10 * self.rate / self.hop)
        times = np.array(self.times[-w:])
        onsets, self.thr = tap_onsets(np.array(self.flux[-w:]), times, floor=self.floor)
        for t in onsets:
            if t < times[-1] - 0.05 and (not self.taps or t > self.taps[-1] + 0.12):
                self.taps.append(t)

    def silent(self, now):
        """Only zeros for 2 s: on a Mac, the app is not allowed to use the microphone."""
        return self.first is not None and now - self.first > 2 and max((p for t, p in self.peaks if t > now - 2), default=0) < 1e-6


class Recorder:
    """Writes the frames and the sound into one video file, each at its own times, counted from t0 (the first frame)."""

    def __init__(self, path, frame, rate, t0):
        import av
        h, w = frame.shape[:2]
        s = min(1.0, 720 / h)                                      # at most 720 pixels high: plenty, and smaller files
        self.size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)     # even, for the video codec
        self.av, self.t0, self.rate, self.pts, self.samples, self.frames = av, t0, rate, -1, 0, 0
        self.file = av.open(path, "w")
        codec, options = next((c, o) for c, o in (("libx264", {"preset": "veryfast", "crf": "18"}),
                                                  ("h264_videotoolbox", {"b": "8M"}), ("mpeg4", {"b": "8M"}))
                              if c in av.codecs_available)
        self.video = self.file.add_stream(codec, rate=30, options=options)
        self.video.width, self.video.height, self.video.pix_fmt = self.size[0], self.size[1], "yuv420p"
        self.video.time_base = self.video.codec_context.time_base = Fraction(1, 1000)    # times in milliseconds
        self.audio = self.file.add_stream("pcm_s16le", rate=rate, layout="mono")         # the sound as it is

    def picture(self, t, frame):
        pts = max(int(round((t - self.t0) * 1000)), self.pts + 1)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        f = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        f.pts, f.time_base = pts, Fraction(1, 1000)
        self.file.mux(self.video.encode(f))
        self.pts, self.frames = pts, self.frames + 1

    def sound(self, start, x):
        """A block of sound whose first sample was taken at start. Sample 0 of the file is at t0; samples the microphone
        lost (a gap of more than 20 ms) become silence, so that the sound stays in step with the picture."""
        first = int(round((start - self.t0) * self.rate))           # the number its first sample should have
        if self.samples == 0 and first < 0:
            x, first = x[-first:], 0                                 # only the part from t0 on
        if first - self.samples > (0 if self.samples == 0 else 0.02 * self.rate):
            self.write(np.zeros(first - self.samples, np.float32))
        self.write(x)

    def write(self, x):
        if len(x):
            f = self.av.AudioFrame.from_ndarray((np.clip(x, -1, 1) * 32767).astype(np.int16)[None], format="s16", layout="mono")
            f.sample_rate, f.pts, f.time_base = self.rate, self.samples, Fraction(1, self.rate)
            self.file.mux(self.audio.encode(f))
            self.samples += len(x)

    def close(self):
        self.file.mux(self.video.encode(None))
        self.file.mux(self.audio.encode(None))
        self.file.close()


def free_name(path):
    """path, or path_2, path_3 ... if it is taken: a recording never replaces another."""
    root, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(path):
        path, k = "%s_%d%s" % (root, k, ext), k + 1
    return path


def record_panel(height, rec, elapsed, args, heard, hand):
    """Beside the picture: what to do now, the time, the taps heard, whether the hand is seen."""
    w, white = 330, (255, 255, 255)
    img = np.full((max(height, 420), w, 3), 255, np.uint8)
    total = args.minutes * 60
    if rec:
        cv2.circle(img, (22, 27), 9, cp.RED, -1, cv2.LINE_AA)
        cp.text(img, "REC  %d:%02d / %d:%02d" % (elapsed // 60, elapsed % 60, total // 60, total % 60), 40, 36, 0.75, cp.INK, 2)
        into = elapsed % (args.tap + args.hover)
        tapping = into < args.tap
        left = (args.tap - into) if tapping else (args.tap + args.hover - into)
        cv2.rectangle(img, (12, 56), (w - 12, 136), cp.ORANGE if tapping else cp.BLUE, -1)
        cp.text(img, "TAP" if tapping else "HOVER", 26, 112, 1.5, white, 3)
        cp.text(img, "%d s" % np.ceil(left), w - 84, 112, 0.8, white, 2)
        lines = (["With the index finger: quick", "taps (touch and lift at once),", "about one a second,",
                  "at many places."] if tapping else
                 ["Move the finger just above", "the table, down and up,", "WITHOUT touching it."])
        lines += ["", "Next: %s." % ("hover" if tapping else "tap")]
    else:
        cp.text(img, "Not recording", 12, 36, 0.75, cp.INK, 2)
        lines = ["R or Record starts: %g min," % args.minutes, "%g s tapping and %g s hovering" % (args.tap, args.hover),
                 "in turn, as this panel says.", "", "First:", "- tilt the screen forward until",
                 "  the table and your hand show", "- a lamp low to one side helps", "- tap a few times: the taps",
                 "  heard show in orange below"]
    for i, s in enumerate(lines):
        cp.text(img, s, 14, (164 if rec else 70) + i * 24, 0.55, cp.INK)
    y = max(height, 420) - 62
    cp.text(img, "Taps heard: %d" % heard + ("" if rec else "  (a test)"), 14, y, 0.65, cp.INK, 2)
    cp.text(img, "Hand: found" if hand else "Hand: not found", 14, y + 32, 0.6, cp.GREEN if hand else cp.RED, 2)
    return img


def record_plot(ears, now, width, span=6.0):
    """The sound of the last seconds: its spectral flux (blue), the threshold (grey) and the taps heard (orange)."""
    h, x0, x1, y0, y1 = 140, 40, width - 20, 32, 126
    img = np.full((h, width, 3), 255, np.uint8)
    cp.text(img, "The sound, last %g s: spectral flux (blue), threshold (grey), taps heard (orange)" % span, 12, 20, 0.5, cp.INK)
    cv2.line(img, (x0, y1), (x1, y1), (180, 180, 180), 1)
    k = bisect.bisect_left(ears.times, now - span)
    times, flux = np.array(ears.times[k:]), np.array(ears.flux[k:])
    if len(times) < 2:
        return img
    top = 1.1 * max(4 * (ears.floor or 0), flux.max(), 1e-3)
    X = lambda t: (x0 + (np.asarray(t) - (now - span)) / span * (x1 - x0)).astype(np.int32)
    Y = lambda v: (y1 - np.clip(np.asarray(v) / top, 0, 1) * (y1 - y0)).astype(np.int32)
    for t in ears.taps:
        if t > now - span:
            cv2.line(img, (int(X(t)), y0), (int(X(t)), y1), cp.ORANGE, 2)
    if len(ears.thr):
        tt, thr = np.array(ears.times[-len(ears.thr):]), ears.thr
        keep = tt > now - span
        if keep.sum() > 1:
            cv2.polylines(img, [np.c_[X(tt[keep]), Y(thr[keep])]], False, cp.GREY, 1, cv2.LINE_AA)
    cv2.polylines(img, [np.c_[X(times), Y(flux)]], False, cp.BLUE, 1, cv2.LINE_AA)
    return img


def record(args):
    if not os.path.splitext(args.video)[1]:
        args.video += ".mov"
    path = free_name(args.video)
    if path != args.video:
        print("%s is already there: recording to %s" % (args.video, path))
    try:
        mic = Microphone(int(args.mic) if args.mic and args.mic.isdigit() else args.mic)
    except Exception as e:                                       # no microphone, or not that one
        sys.exit("The microphone does not open (%s).\nThe microphones there are: python -m sounddevice" % e)
    print("Microphone: %s, %d Hz" % (mic.name, mic.rate))
    ears, cam, camera = Ears(mic.rate), Camera(args.camera), args.camera
    det = cp.Detector(cp.Kind(True, False), video=True)
    rec, started, pending, seen, drawn = None, None, [], None, 0.0
    clicks, buttons, message, said = [], [], "" if cam.cap else "Camera %d gives no picture" % camera, time.time()
    title = "Record taps"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, lambda event, x, y, flags, param: clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        # The sound: heard (to show the taps) and, while recording, written
        while not mic.blocks.empty():
            start, x, lost = mic.blocks.get()
            ears.hear(start, x)
            if lost:
                message, said = "The microphone lost some sound (the computer is busy)", time.time()
            if rec:
                rec.sound(start, x)
            elif started:
                pending.append((start, x))                       # until the first frame of the recording
        now = time.perf_counter()
        ears.find(now)

        # The picture: while recording, every frame is written, with the time it arrived
        while started and not cam.frames.empty():
            t, f = cam.frames.get()
            if rec is None:
                rec = Recorder(path, f, mic.rate, t)
                print("Recording to %s" % path)
                for s, x in pending:
                    rec.sound(s, x)
            rec.picture(t, f)
        if started and cam.frames.qsize() > 60:
            message, said = "The computer is too slow: frames are waiting to be written", time.time()
        elapsed = now - rec.t0 if rec else 0.0
        if rec and elapsed >= args.minutes * 60:
            break                                                # the time is up

        newest = cam.newest
        if (newest is not None and newest[0] != seen) or (newest is None and now - drawn > 0.1):
            seen, drawn = (newest[0] if newest else None), now
            frame = cp.fit(newest[1]) if newest else np.full((480, 640, 3), 230, np.uint8)
            if not newest:
                cp.text(frame, "Camera %d gives no picture: choose another" % camera, 40, 240, 0.7, cp.RED, 2)
            pts = det.find(frame) if newest else None
            if pts is not None:
                centre, d, side, palm = geometry(pts)
                cp.draw_skeleton(frame, pts, cp.Kind(True, False), cp.BLUE)
                corners = cv2.transform(np.array([[[0, 0], [PATCH, 0], [PATCH, PATCH], [0, PATCH]]], np.float32),
                                        cv2.invertAffineTransform(patch_matrix(centre, d, side)))[0]
                cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
            if ears.taps and now - ears.taps[-1] < 0.25:          # a tap just heard
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), cp.ORANGE, 8)
                label(frame, "tap", (frame.shape[1] - 64, 36), 0.9, cp.ORANGE, 2)
            if newest:
                label(frame, "%.0f fps" % cam.fps(), (10, 24), 0.6, cp.INK, 1)
            heard = sum(t >= rec.t0 for t in ears.taps) if rec else len(ears.taps)
            out = cp.side_by_side(frame, record_panel(frame.shape[0], rec, elapsed, args, heard, pts is not None))
            out = np.vstack([out, record_plot(ears, now, out.shape[1])])
            out = np.pad(out, ((0, 70), (0, 0), (0, 0)), constant_values=255)
            bar, buttons = toolbar(out.shape[1], camera, [("Stop (R)" if started else "Record (R)", "record", bool(started), cp.RED)])
            out = cp.stack(bar, out)
            if message and time.time() - said > 4:
                message = ""
            warning = message or ("No sound: allow the microphone (System Settings > Privacy & Security > Microphone)"
                                  if ears.silent(now) else "") or ("No hand found" if newest and pts is None else "")
            cp.text(out, warning, 10, out.shape[0] - 38, 0.55, cp.RED, 2)
            cp.text(out, "camera %d   microphone: %s   file: %s   keys: R record / stop, C camera, Q quit"
                    % (camera, mic.name, os.path.basename(path)), 10, out.shape[0] - 12, 0.5, cp.GREY)
            cv2.imshow(title, out)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(5) & 0xFF                          # nothing new to show: wait a little

        action = {ord("r"): "record", ord("q"): "quit", 27: "quit", ord("c"): (camera + 1) % cp.CAMERAS}.get(key)
        while clicks:
            cx, cy = clicks.pop(0)
            action = next((a for x0, y0, x1, y1, a in buttons if x0 <= cx <= x1 and y0 <= cy <= y1), action)
        if action is not None:
            said = time.time()
        if action in ("record", "quit") and started:
            break                                                # stop, and keep what was recorded
        if action == "quit":
            break
        if action == "record":
            if cam.cap:
                started, cam.keep = time.perf_counter(), True
            else:
                message = "No picture to record: choose another camera"
        elif isinstance(action, int) and action != camera:
            if started:
                message = "Not while recording"
            else:
                new = Camera(action)
                if new.cap:
                    cam.close()
                    cam, camera, message = new, action, "Camera %d" % action
                else:
                    new.close()
                    message = "Camera %d is not available" % action
    if rec:
        rec.close()
    cam.close()
    mic.close()
    cv2.destroyAllWindows()
    if rec is None:
        print("Nothing recorded.")
        return

    # The file, read as prepare will read it
    x, rate, a0 = read_audio(path)
    flux, ft = spectral_flux(x, rate)
    onsets = tap_onsets(flux, ft + a0)[0]
    seconds = rec.pts / 1000
    print("Saved %s: %d:%02d, %d frames (%.0f per second), %d x %d, sound at %d Hz; %d tap%s heard."
          % ((path,) + divmod(int(round(seconds)), 60) + (rec.frames, rec.frames / max(seconds, 1e-3), rec.size[0],
             rec.size[1], mic.rate, len(onsets), "" if len(onsets) == 1 else "s")))
    if np.abs(x).max() < 1e-4:
        print("The sound is silent: allow the microphone for this app (System Settings > Privacy & Security > Microphone).")
    elif seconds < 60:
        print("That is short: 3 to 5 minutes give the network enough to learn from.")
    elif len(onsets) < 10:
        print("Few taps heard: tap a little harder, nearer the laptop, in a quieter room.")
    print("Next: python touch_learn.py prepare %s" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Learn when a finger touches the table, from a video with the sound of the taps.")
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("record", help="record a video with sound, guided: tapping and hovering in turn")
    a.add_argument("video", nargs="?", default="taps.mov", help="the file (default taps.mov; if taken, taps_2.mov ...)")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--mic", help="the microphone: its number or part of its name, from: python -m sounddevice (default: the system's)")
    a.add_argument("--minutes", type=float, default=4, help="stop after this long (default 4; R stops before)")
    a.add_argument("--tap", type=float, default=20, help="seconds of tapping (default 20) ...")
    a.add_argument("--hover", type=float, default=10, help="... in turn with seconds of hovering (default 10)")
    a = sub.add_parser("prepare", help="labels from the sound, patches from the picture")
    a.add_argument("video")
    a.add_argument("--threshold", type=float, default=5.0, help="tap detection: higher finds fewer taps (default 5)")
    a.add_argument("--offset", type=float, help="the picture is late by this many seconds (default: estimated)")
    a = sub.add_parser("train", help="train the network")
    a.add_argument("data", nargs="+", help="one or more .touch.npz files made by prepare")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--out", default="touch_model.pt")
    a.add_argument("--device", help="cpu, mps or cuda (default: the fastest there is)")
    a = sub.add_parser("live", help="run the network on the camera or a video")
    a.add_argument("--model", default="touch_model.pt")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--video", help="a recording instead of the camera")
    a.add_argument("--threshold", type=float, help="touching above this probability (default: the one chosen in training)")
    a.add_argument("--offset", type=float, help="video: the picture is late by this many seconds (default: from training)")
    a.add_argument("--osc", action="store_true", help="start with sending by OSC on")
    a.add_argument("--osc-to", default="127.0.0.1:12000", metavar="HOST:PORT")
    a.add_argument("--area", type=float, nargs=8, metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
                   help="the corners of the area in the picture, as printed when it is set with Set area (A)")
    a.add_argument("--aspect", type=float, default=1.41,
                   help="width / height of the area, for its view from above (default 1.41, a sheet of A4 lying wide)")
    args = ap.parse_args()
    {"record": record, "prepare": prepare, "train": train, "live": live}[args.command](args)
