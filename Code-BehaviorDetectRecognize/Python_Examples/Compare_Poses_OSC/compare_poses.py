"""Compare two poses with MediaPipe, as in the lecture: a teacher and a student.

The steps are those of the slides "Comparing two poses":
  1. translate: the hip centre (hands: the wrist, landmark 0; faces: between the eyes) to (0, 0)
  2. scale: divide by the torso length (hands: the palm length, landmark 0 to 9; faces: the distance between the eyes)
  3. rotate: turn the student by the best angle, alpha = atan2(D, C)
  4. score: the distance of every joint to the same joint of the teacher, their mean and the worst one,
     the cosine similarity of the two poses, and the joint angles theta of both.
Faces also have 52 expression scores from MediaPipe (blendshapes: jaw open, smile, blink, brow raise..., each 0 to 1):
the game is to copy the teacher's expression, compared score by score and by the cosine similarity of the scores.
They are movements of the face, not emotions (see the README).

Usage:
    python compare_poses.py teacher.jpg student.jpg   two photos: scores printed, comparison.png saved
    python compare_poses.py                           webcam: press T to take your current pose as the teacher
    python compare_poses.py --teacher teacher.jpg     webcam, against the pose in a photo
Options:
    --hands     compare hands (21 landmarks) instead of bodies
    --face      compare faces (22 of the 478 landmarks, and the 52 expression scores)
    --upper     only the upper body (nose, shoulders, elbows, wrists, hips): when the legs are out of the picture
    --mirror    flip the student first (left and right swapped), for a student facing the teacher
    --camera N  another camera (0 is the first one)
    --out FILE  where to save the comparison of two photos (default comparison.png)
    --osc       also send the keypoints by OSC, to the Processing sketch Pose_Compare in this folder, to
                DTW_Gestures (Processing_Examples: the fingertip or the wrist is its pen) or any program, see below;
                with the webcam, it starts with the Send OSC button on
    --osc-to HOST:PORT   where to send them (default 127.0.0.1:12000)
Webcam: the toolbar at the top chooses the camera (0 to 3) and what is tracked (body, upper body or hand), sets the
teacher and the mirror, and turns sending by OSC on and off. Keys: T teacher, M mirror, O OSC on/off, C next camera,
B body, U upper body, H hand, F face, S save a picture, Q or Esc quit. The picture shows the frame rate reached
(frames per second) and how long MediaPipe takes per frame.
(On a Mac, camera 0 can be an iPhone nearby, through Continuity Camera: choose another camera.)

OSC messages (UDP, no library needed): keypoints in pixels of the picture, after its width and height.
    /pose w h x0 y0 ... x12 y12          the student's 13 body joints (also in upper-body mode)
    /hand w h x0 y0 ... x20 y20          the student's 21 hand landmarks
    /face w h x0 y0 ... x21 y21          the student's 22 face landmarks
    /face/blend s0 ... s51               the 52 expression scores, 0 to 1, in MediaPipe's order (see the README)
    /teacher/pose ...  /teacher/hand ...  /teacher/face ...  /teacher/face/blend ...
                                         the teacher, when it is set and again every 2 seconds
    /scores mean cosine worst alpha      once a teacher is set: the mean distance, the cosine similarity,
                                         the number of the worst joint and the best rotation in degrees
    /face/scores cosine mean worst       faces: the cosine similarity and the mean difference of the expression
                                         scores, and the number of the most different one
Webcam: one /pose (or /hand, or /face and /face/blend) per frame. Two photos: each message once.

The MediaPipe model (a few MB) is downloaded into the models folder the first time.
Requirements: pip install mediapipe opencv-python numpy
"""
import argparse
import os
import socket
import struct
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "hands": ("hand_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    "face": ("face_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
}

# Bodies: 13 of the 33 MediaPipe Pose landmarks (L and R are the person's left and right)
BODY = {"nose": 0, "shoulder L": 11, "shoulder R": 12, "elbow L": 13, "elbow R": 14, "wrist L": 15, "wrist R": 16,
        "hip L": 23, "hip R": 24, "knee L": 25, "knee R": 26, "ankle L": 27, "ankle R": 28}
BODY_BONES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (0, 1), (0, 2)]
BODY_ANGLES = {"elbow L": (1, 3, 5), "elbow R": (2, 4, 6), "shoulder L": (3, 1, 7), "shoulder R": (4, 2, 8),
               "knee L": (7, 9, 11), "knee R": (8, 10, 12)}
BODY_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]    # left <-> right, for the mirror
UPPER = 9                                                 # the first 9 joints: nose to hips

# Hands: the 21 MediaPipe Hands landmarks, 0 wrist, then 4 per finger from the base to the tip
FINGERS = ["thumb", "index", "middle", "ring", "little"]
HAND_NAMES = ["0 wrist"] + [f"{1 + 4 * f + k} {name}{' tip' if k == 3 else ''}" for f, name in enumerate(FINGERS) for k in range(4)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_ANGLES = {"thumb": (2, 3, 4), "index": (5, 6, 7), "middle": (9, 10, 11), "ring": (13, 14, 15), "little": (17, 18, 19)}

# Faces: 22 of the 478 MediaPipe Face Landmarker points (R and L are the person's right and left)
FACE = {"nose tip": 1, "brow R out": 70, "brow R mid": 105, "brow R in": 107, "brow L in": 336, "brow L mid": 334, "brow L out": 300,
        "eye R out": 33, "eye R top": 159, "eye R in": 133, "eye R bottom": 145, "eye L in": 362, "eye L top": 386,
        "eye L out": 263, "eye L bottom": 374, "mouth R": 61, "lip top": 0, "mouth L": 291, "lip bottom": 17,
        "mouth in top": 13, "mouth in bottom": 14, "chin": 152}
FACE_BONES = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 7), (11, 12), (12, 13), (13, 14), (14, 11),
              (15, 16), (16, 17), (17, 18), (18, 15), (15, 19), (19, 17), (15, 20), (20, 17), (0, 16), (18, 21)]
FACE_ANGLES = {"mouth": (16, 15, 18), "eye R": (8, 7, 10), "eye L": (12, 13, 14)}     # how open, at the corners
FACE_SWAP = [0, 6, 5, 4, 3, 2, 1, 13, 12, 11, 14, 9, 8, 7, 10, 17, 16, 15, 18, 19, 20, 21]   # left <-> right

BLUE, ORANGE, GREEN, RED, INK, GREY = (182, 117, 46), (10, 108, 227), (87, 139, 46), (0, 0, 192), (38, 38, 38), (150, 150, 150)   # BGR
TEAL = (104, 89, 33)


class Kind:
    """What is compared: the body (all 13 joints or the upper 9), a hand (21 landmarks) or a face (22 landmarks)."""

    def __init__(self, hands, upper, face=False):
        self.hands, self.face = hands, face
        self.model = "face" if face else ("hands" if hands else "pose")
        if face:
            self.names, self.bones, self.angles = list(FACE), FACE_BONES, FACE_ANGLES
        elif hands:
            self.names, self.bones, self.angles = HAND_NAMES, HAND_BONES, HAND_ANGLES
        else:
            n = UPPER if upper else len(BODY)
            self.names = list(BODY)[:n]
            self.bones = [b for b in BODY_BONES if max(b) < n]
            self.angles = {k: v for k, v in BODY_ANGLES.items() if max(v) < n}
        self.n = len(self.names)
        self.unit = "eye distances" if face else ("palm lengths" if hands else "torso lengths")


# ---- The four steps ----

def normalise(p, kind):
    """Steps 1 and 2: translate and scale."""
    if kind.face:
        centre = (p[7] + p[9] + p[11] + p[13]) / 4                          # between the eyes
        size = np.linalg.norm(p[13] - p[7])                                 # eye distance, outer corner to outer corner
    elif kind.hands:
        centre, size = p[0], np.linalg.norm(p[9] - p[0])                    # wrist; palm length 0 to 9
    else:
        centre = (p[7] + p[8]) / 2                                          # hip centre
        size = np.linalg.norm((p[1] + p[2]) / 2 - centre)                   # torso: hips to shoulders
    return (p - centre) / max(size, 1e-6)


def best_rotation(t, s):
    """Step 3: the angle alpha that best turns s onto t (both normalised), alpha = atan2(D, C)."""
    C = np.sum(s[:, 0] * t[:, 0] + s[:, 1] * t[:, 1])
    D = np.sum(s[:, 0] * t[:, 1] - s[:, 1] * t[:, 0])
    alpha = np.arctan2(D, C)
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return s @ R.T, alpha


def joint_angle(p, a, b, c):
    """Theta, the angle at joint b between its neighbours a and c, in degrees."""
    u, v = p[a] - p[b], p[c] - p[b]
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))


def mirrored(p, kind):
    """Flip x; for bodies and faces also swap left and right (a hand keeps its numbering)."""
    q = p.copy()
    q[:, 0] = -q[:, 0]
    return q if kind.hands else q[(FACE_SWAP if kind.face else BODY_SWAP)[:kind.n]]


def mirror_order(names):
    """For the expression scores: each one's partner on the other side (browDownLeft <-> browDownRight)."""
    other = lambda n: n.replace("Left", "@").replace("Right", "Left").replace("@", "Right")
    return [names.index(other(n)) if other(n) in names else i for i, n in enumerate(names)]


def compare(teacher, student, kind, mirror=False, blend=None):
    """All four steps. Returns the aligned poses, the distances and the scores.
    Faces: blend = (teacher's scores, student's scores, their names), compared one by one and as one list."""
    teacher, student = teacher[:kind.n], student[:kind.n]                  # the upper body: the first 9 joints
    s = mirrored(student, kind) if mirror else student
    T = normalise(teacher, kind)
    S, alpha = best_rotation(T, normalise(s, kind))
    e = np.linalg.norm(T - S, axis=1)                                       # step 4: distance per joint
    res = dict(T=T, S=S, alpha=np.degrees(alpha), e=e, mean=float(e.mean()), worst=int(np.argmax(e)),
               cosine=float(np.sum(T * S) / np.sqrt(np.sum(T ** 2) * np.sum(S ** 2))),
               angles={k: (joint_angle(teacher, *ix), joint_angle(s, *ix)) for k, ix in kind.angles.items()})
    if blend is not None:
        bt, bs, names = blend
        bs = bs[mirror_order(names)] if mirror else bs
        use = np.array([n != "_neutral" for n in names])                    # "_neutral" is not a movement
        d = np.abs(bt - bs) * use
        res.update(blend_t=bt, blend_s=bs, blend_names=names, blend_mean=float(d[use].mean()), blend_worst=int(np.argmax(d)),
                   blend_cosine=float(np.sum(bt[use] * bs[use]) / (np.sqrt(np.sum(bt[use] ** 2) * np.sum(bs[use] ** 2)) + 1e-9)))
    return res


# ---- MediaPipe ----

class Detector:
    """Finds one body, hand or face in a picture; returns its keypoints in pixels, or None.
    For a face it also keeps the 52 expression scores of the last picture (blend) and their names (blend_names)."""

    def __init__(self, kind, video=False):
        name, url = MODELS[kind.model]
        path = os.path.join(HERE, "models", name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("Downloading the MediaPipe model", name, "…")
            urllib.request.urlretrieve(url, path)
        base = mp_tasks.BaseOptions(model_asset_path=path)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        if kind.face:
            self.model = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base, running_mode=mode, num_faces=1, output_face_blendshapes=True))
        elif kind.hands:
            self.model = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base, running_mode=mode, num_hands=1))
        else:
            self.model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base, running_mode=mode, num_poses=1))
        self.kind, self.video, self.t0 = kind, video, time.time()
        self.blend, self.blend_names = None, []

    def find(self, bgr):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            result = self.model.detect_for_video(image, int((time.time() - self.t0) * 1000))
        else:
            result = self.model.detect(image)
        if self.kind.face:
            found = result.face_landmarks
            self.blend = np.array([c.score for c in result.face_blendshapes[0]]) if found else None
            if found:
                self.blend_names = [c.category_name for c in result.face_blendshapes[0]]
        else:
            found = result.hand_landmarks if self.kind.hands else result.pose_landmarks
        if not found:
            return None
        h, w = bgr.shape[:2]
        points = np.array([[lm.x * w, lm.y * h] for lm in found[0]])
        if self.kind.face:
            return points[list(FACE.values())]                              # 22 of the 478
        return points if self.kind.hands else points[list(BODY.values())]   # all 13 body joints


# ---- Drawing ----

def error_colour(e):
    return GREEN if e < 0.15 else (ORANGE if e < 0.4 else RED)


def draw_skeleton(img, p, kind, colour, errors=None):
    """Bones in one colour; joints in green, orange or red by their distance to the teacher's."""
    small = getattr(kind, "face", False)                                    # a face's points are close together
    for a, b in kind.bones:
        cv2.line(img, tuple(map(int, p[a])), tuple(map(int, p[b])), colour, 2 if small else 3, cv2.LINE_AA)
    for i, q in enumerate(p):
        c = colour if errors is None else error_colour(errors[i])
        r0, grow = (3, 8) if small else (5, 14)
        r = r0 if errors is None else int(r0 + min(errors[i], 1) * grow)
        cv2.circle(img, tuple(map(int, q)), r, c, -1, cv2.LINE_AA)


def text(img, s, x, y, scale=0.5, colour=INK, bold=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, bold, cv2.LINE_AA)


def panel(res, kind, height, mirror):
    """The scores beside the picture: the overlay, the distances as bars, the numbers, the joint angles."""
    if kind.face:
        return face_panel(res, kind, height, mirror)
    w, box, row = 460, 210, 24
    needed = 50 + box + 40 + kind.n * row + 215 + 30 * ((len(kind.angles) + 1) // 2) + 60
    img = np.full((max(height, needed), w, 3), 255, np.uint8)
    text(img, "After steps 1-3: teacher blue, student orange" + (" (mirrored)" if mirror else ""), 12, 26, 0.55, INK)
    overlay(img, res, kind, w, 40, box)
    y = 50 + box + 30
    text(img, "Distance per joint (%s)" % kind.unit, 12, y, 0.6, INK, 2)
    top = max(1.0, float(res["e"].max()))
    bx, bw = 150, 230
    for i, (name, e) in enumerate(zip(kind.names, res["e"])):
        yy = y + 10 + i * row
        text(img, name, 12, yy + row - 6, 0.5, INK)
        cv2.rectangle(img, (bx, yy + 4), (bx + max(int(e / top * bw), 1), yy + row - 4), ORANGE if i == res["worst"] else BLUE, -1)
        text(img, "%.2f" % e, bx + bw + 14, yy + row - 6, 0.5, GREY)
    mx = bx + int(res["mean"] / top * bw)                                  # the mean, as a dashed line
    for yy in range(y + 10, y + 10 + kind.n * row, 7):
        cv2.line(img, (mx, yy), (mx, yy + 4), INK, 1)
    y = y + 10 + kind.n * row + 32
    text(img, "Whole pose", 12, y, 0.6, INK, 2)
    phi = np.degrees(np.arccos(np.clip(res["cosine"], -1, 1)))         # the angle between the two poses as lists
    gauge(img, 12, y + 26, "Cosine similarity (1 = same shape)", res["cosine"], 0.5, 1.0, TEAL, "%.3f  (phi %.0f deg)" % (res["cosine"], phi))
    gauge(img, 12, y + 98, "Mean distance (0 = same pose)", res["mean"], 0.0, 1.0, error_colour(res["mean"]), "%.2f" % res["mean"])
    y += 168
    text(img, "Worst: %s, %.2f" % (kind.names[res["worst"]], res["e"][res["worst"]]), 12, y, 0.58, error_colour(res["e"][res["worst"]]), 2)
    text(img, "Best rotation alpha: %+.0f deg" % res["alpha"], 12, y + 28, 0.55, INK)
    y -= 2
    text(img, "Joint angles theta: teacher / student", 12, y + 66, 0.55, INK)
    for k, (name, (ta, sa)) in enumerate(res["angles"].items()):
        text(img, "%s: %d / %d" % (name, round(ta), round(sa)), 12 + (k % 2) * 225, y + 94 + (k // 2) * 28, 0.52,
             RED if abs(ta - sa) > 20 else INK)
    return img


def overlay(img, res, kind, w, y0, box):
    """Teacher and student after steps 1-3, one over the other, in a box of the panel."""
    span = np.max(np.abs(np.r_[res["T"], res["S"]])) + 1e-6
    for P, col in ((res["T"], BLUE), (res["S"], ORANGE)):
        q = np.c_[w // 2 + P[:, 0] / span * box / 2, y0 + 5 + box / 2 + P[:, 1] / span * box / 2]
        draw_skeleton(img, q, kind, col)


def expression_colour(m):
    return GREEN if m < 0.05 else (ORANGE if m < 0.12 else RED)


def face_panel(res, kind, height, mirror):
    """Faces: the overlay after steps 1-3, the expression scores of both, and the scores of the whole face."""
    w, box, row = 460, 170, 26
    names, bt, bs = res["blend_names"], res["blend_t"], res["blend_s"]
    rows = [j for j in np.argsort(-np.maximum(bt, bs)) if names[j] != "_neutral"][:10]   # the 10 most active
    needed = 40 + box + 50 + len(rows) * row + 230
    img = np.full((max(height, needed), w, 3), 255, np.uint8)
    text(img, "After steps 1-3: teacher blue, student orange" + (" (mirrored)" if mirror else ""), 12, 26, 0.55, INK)
    overlay(img, res, kind, w, 40, box)
    y = 40 + box + 36
    text(img, "Expressions (0 to 1): teacher / student", 12, y, 0.6, INK, 2)
    bx, bw = 185, 200
    for i, j in enumerate(rows):
        yy = y + 10 + i * row
        text(img, names[j], 12, yy + 17, 0.45, RED if j == res["blend_worst"] else INK)
        for k, (v, col) in enumerate(((bt[j], BLUE), (bs[j], ORANGE))):
            cv2.rectangle(img, (bx, yy + 4 + k * 10), (bx + bw, yy + 12 + k * 10), (238, 238, 238), -1)
            cv2.rectangle(img, (bx, yy + 4 + k * 10), (bx + max(int(v * bw), 1), yy + 12 + k * 10), col, -1)
        text(img, "%.2f" % abs(bt[j] - bs[j]), bx + bw + 12, yy + 17, 0.45, GREY)
    y = y + 10 + len(rows) * row + 30
    text(img, "Whole face", 12, y, 0.6, INK, 2)
    gauge(img, 12, y + 26, "Cosine similarity of the scores", res["blend_cosine"], 0.0, 1.0, TEAL, "%.3f" % res["blend_cosine"])
    gauge(img, 12, y + 98, "Mean difference (0 = same expression)", res["blend_mean"], 0.0, 0.3, expression_colour(res["blend_mean"]),
          "%.3f" % res["blend_mean"])
    y += 168
    j = res["blend_worst"]
    text(img, "Most different: %s, %.2f" % (names[j], abs(bt[j] - bs[j])), 12, y, 0.58, expression_colour(abs(bt[j] - bs[j]) / 3), 2)
    text(img, "Shape after steps 1-3: %.2f %s" % (res["mean"], kind.unit), 12, y + 28, 0.5, INK)
    return img


def gauge(img, x, y, label, value, lo, hi, colour, note, w=220):
    """A bar from lo to hi filled up to value, with its label above and the value on the right."""
    text(img, label, x, y, 0.52, INK)
    y0 = y + 10
    cv2.rectangle(img, (x, y0), (x + w, y0 + 20), (235, 235, 235), -1)
    f = (min(max(value, lo), hi) - lo) / (hi - lo)
    cv2.rectangle(img, (x, y0), (x + max(int(f * w), 2), y0 + 20), colour, -1)
    text(img, ("%g" % lo), x, y0 + 36, 0.4, GREY)
    text(img, ("%g" % hi), x + w - 10, y0 + 36, 0.4, GREY)
    text(img, note, x + w + 12, y0 + 16, 0.55, INK, 1)


def side_by_side(*images):
    h = max(i.shape[0] for i in images)
    return np.hstack([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=255) for i in images])


def fit(img, height=480):
    return cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))


def report(res, kind):
    print("Best rotation alpha: %+.1f degrees" % res["alpha"])
    print("Distance per joint (%s):" % kind.unit)
    for name, e in zip(kind.names, res["e"]):
        print("  %-14s %.2f %s" % (name, e, "#" * int(round(e * 20))))
    print("Mean %.3f, worst %s (%.2f), cosine similarity %.3f" % (res["mean"], kind.names[res["worst"]], res["e"][res["worst"]], res["cosine"]))
    print("Joint angles theta, teacher / student:")
    for name, (ta, sa) in res["angles"].items():
        print("  %-12s %5.1f / %5.1f deg%s" % (name, ta, sa, "   off by %.0f" % abs(ta - sa) if abs(ta - sa) > 20 else ""))
    if "blend_t" in res:
        names, bt, bs = res["blend_names"], res["blend_t"], res["blend_s"]
        print("Expression scores, teacher / student (the 8 most different):")
        for j in np.argsort(-np.abs(bt - bs))[:8]:
            print("  %-20s %.2f / %.2f" % (names[j], bt[j], bs[j]))
        print("Expressions: cosine similarity %.3f, mean difference %.3f" % (res["blend_cosine"], res["blend_mean"]))


# ---- OSC ----

class Osc:
    """Sends OSC messages by UDP, encoded here (no library): an address, a type tag per number, the numbers."""

    def __init__(self, target):
        host, _, port = target.rpartition(":")
        self.target = (host or "127.0.0.1", int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, address, values):
        pad = lambda b: b + b"\0" * (4 - len(b) % 4)        # strings end with 1 to 4 zeros: a multiple of 4 bytes
        values = [float(v) for v in values]
        msg = pad(address.encode()) + pad(("," + "f" * len(values)).encode()) + struct.pack(">%df" % len(values), *values)
        try:
            self.sock.sendto(msg, self.target)
        except OSError:                                    # nobody listening, or no network: never stop for that
            pass

    def keypoints(self, pts, kind, size, teacher=False):
        """/pose, /hand or /face (/teacher/...): width and height of the picture, then x, y of each keypoint."""
        address = ("/teacher" if teacher else "") + "/" + {"pose": "pose", "hands": "hand", "face": "face"}[kind.model]
        self.send(address, [size[0], size[1]] + list(np.ravel(pts)))

    def blend(self, scores, teacher=False):
        """/face/blend (/teacher/face/blend): the 52 expression scores, 0 to 1."""
        self.send(("/teacher" if teacher else "") + "/face/blend", list(scores))

    def scores(self, res):
        self.send("/scores", [res["mean"], res["cosine"], res["worst"], res["alpha"]])
        if "blend_t" in res:
            self.send("/face/scores", [res["blend_cosine"], res["blend_mean"], res["blend_worst"]])


def picture_size(img):
    return img.shape[1], img.shape[0]


# ---- Two photos, or the webcam ----

def two_photos(args, kind):
    det = Detector(kind)
    pics = [cv2.imread(f) for f in (args.photos[0], args.photos[1])]
    for f, p in zip(args.photos, pics):
        if p is None:
            sys.exit("Cannot read the picture " + f)
    pics = [fit(p) for p in pics]
    teacher = det.find(pics[0])
    teacher_blend = det.blend
    student = det.find(pics[1])
    for who, pts in (("teacher", teacher), ("student", student)):
        if pts is None:
            sys.exit("No %s found in the %s's picture.%s" % ("face" if kind.face else ("hand" if kind.hands else "person"), who,
                                                             " Use a picture where the face is big, a portrait." if kind.face else ""))
    blend = (teacher_blend, det.blend, det.blend_names) if kind.face else None
    if args.osc:                                           # all the keypoints, before keeping the upper body only
        osc = Osc(args.osc_to)
        print("Sending OSC to %s:%d" % osc.target)
        osc.keypoints(teacher, kind, picture_size(pics[0]), teacher=True)
        osc.keypoints(student, kind, picture_size(pics[1]))
        if kind.face:
            osc.blend(teacher_blend, teacher=True)
            osc.blend(det.blend)
    teacher, student = teacher[:kind.n], student[:kind.n]
    res = compare(teacher, student, kind, args.mirror, blend)
    if args.osc:
        osc.scores(res)
    report(res, kind)
    draw_skeleton(pics[0], teacher, kind, BLUE)
    draw_skeleton(pics[1], student, kind, (200, 200, 200), res["e"])
    text(pics[0], "teacher", 10, 25, 0.7, BLUE, 2)
    text(pics[1], "student", 10, 25, 0.7, ORANGE, 2)
    out = side_by_side(pics[0], pics[1], panel(res, kind, pics[0].shape[0], args.mirror))
    cv2.imwrite(args.out, out)
    print("Saved", args.out)


MODES = {"Body": (False, False), "Upper body": (False, True), "Hand": (True, False), "Face": (False, False, True)}   # (hands, upper, face)
CAMERAS = 4                                                                            # cameras 0 to 3 in the toolbar


def open_camera(index):
    """The camera, if it opens and gives a picture; else None."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened() and cap.read()[0]:
        return cap
    cap.release()
    return None


def toolbar(width, camera, mode, mirror, sending):
    """The buttons at the top of the window. Returns the image and the buttons as (x0, y0, x1, y1, action)."""
    bar = np.full((56, width, 3), 245, np.uint8)
    buttons, x = [], 10

    def button(label, action, on):
        nonlocal x
        w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 22
        cv2.rectangle(bar, (x, 12), (x + w, 44), BLUE if on else (255, 255, 255), -1)
        cv2.rectangle(bar, (x, 12), (x + w, 44), (170, 170, 170), 1)
        text(bar, label, x + 11, 34, 0.55, (255, 255, 255) if on else INK)
        buttons.append((x, 12, x + w, 44, action))
        x += w + 6

    text(bar, "Camera", x, 34, 0.55)
    x += 72
    for n in range(CAMERAS):
        button(str(n), n, n == camera)
    x += 16
    text(bar, "Track", x, 34, 0.55)
    x += 54
    for name in MODES:
        button(name, name, name == mode)
    x += 16
    button("Set teacher (T)", "teacher", False)
    button("Mirror (M)", "mirror", mirror)
    x += 16
    button("Send OSC (O)", "osc", sending)
    return bar, buttons


def stack(top, bottom):
    """One image above the other, padded with white to the same width."""
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda img: np.pad(img, ((0, 0), (0, w - img.shape[1]), (0, 0)), constant_values=255)
    return np.vstack([pad(top), pad(bottom)])


def smooth(average, value, k=0.1):
    """A running average that follows value slowly, so the numbers shown do not flicker."""
    return value if average == 0 else (1 - k) * average + k * value


def webcam(args, kind):
    mode = "Face" if kind.face else ("Hand" if kind.hands else ("Upper body" if kind.n == UPPER else "Body"))
    kinds = {name: Kind(*flags) for name, flags in MODES.items()}
    detectors, teachers = {}, {"pose": None, "hands": None, "face": None}  # one model and one teacher per kind of keypoints
    sizes, teacher_blend = {}, None                # the size of each teacher's picture, for OSC; the teacher's expression
    if args.teacher:
        photo = cv2.imread(args.teacher)
        if photo is None:
            sys.exit("Cannot read the picture " + args.teacher)
        photo = fit(photo)
        d = Detector(kind)
        teachers[kind.model], sizes[kind.model], teacher_blend = d.find(photo), picture_size(photo), d.blend
    osc, sending, teacher_sent = Osc(args.osc_to), args.osc, 0.0           # the OSC button turns sending on and off
    camera = args.camera
    cap = open_camera(camera)
    message = "" if cap else "Camera %d gives no picture: choose another one" % camera
    clicks = []
    cv2.namedWindow("Compare poses")
    cv2.setMouseCallback("Compare poses", lambda event, x, y, flags, param: clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    mirror, saved, buttons, said = args.mirror, 0, [], time.time()
    period, detect, last = 0.0, 0.0, time.time()        # smoothed time per frame and per MediaPipe call, in seconds
    while True:
        k = kinds[mode]
        model = k.model
        if model not in detectors:
            detectors[model] = Detector(k, video=True)
        ok, frame = cap.read() if cap else (False, None)
        frame = fit(frame) if ok else np.full((480, 640, 3), 230, np.uint8)
        t0 = time.time()
        pts = detectors[model].find(frame) if ok else None
        detect = smooth(detect, time.time() - t0)

        # What to do: a key, or a click on the toolbar
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        action = {ord("t"): "teacher", ord("m"): "mirror", ord("o"): "osc", ord("c"): (camera + 1) % CAMERAS, ord("s"): "save",
                  ord("b"): "Body", ord("u"): "Upper body", ord("h"): "Hand", ord("f"): "Face"}.get(key)
        while clicks:
            x, y = clicks.pop(0)
            action = next((a for x0, y0, x1, y1, a in buttons if x0 <= x <= x1 and y0 <= y <= y1), action)
        if action is not None:
            said = time.time()
        if action == "teacher":
            if pts is not None:
                teachers[model], sizes[model] = pts.copy(), picture_size(frame)
                if k.face:
                    teacher_blend = detectors["face"].blend.copy()
                message, teacher_sent = "Teacher set", 0.0          # send it now
            else:
                message = "Nobody found: step into the picture first"
        elif action == "mirror":
            mirror = not mirror
        elif action == "osc":
            sending, teacher_sent = not sending, 0.0                     # on: send the teacher at once
            message = "Sending OSC to %s:%d" % osc.target if sending else "OSC off"
        elif action in MODES:
            mode, message, teacher_sent = action, "", 0.0
        elif isinstance(action, int) and action != camera:
            new = open_camera(action)
            if new:
                if cap:
                    cap.release()
                cap, camera, message = new, action, "Camera %d" % action
            else:
                message = "Camera %d is not available" % action

        # OSC: the student every frame, the teacher every 2 seconds (for a program started later)
        teacher = teachers[model]
        blend = detectors[model].blend if k.face else None                 # the student's expression scores
        if sending and pts is not None:
            osc.keypoints(pts, k, picture_size(frame))
            if blend is not None:
                osc.blend(blend)
        if sending and teacher is not None and time.time() - teacher_sent > 2:
            osc.keypoints(teacher, k, sizes[model], teacher=True)
            if k.face and teacher_blend is not None:
                osc.blend(teacher_blend, teacher=True)
            teacher_sent = time.time()

        # The picture and the scores
        if pts is not None and teacher is not None:
            both = k.face and blend is not None and teacher_blend is not None
            res = compare(teacher, pts, k, mirror, (teacher_blend, blend, detectors[model].blend_names) if both else None)
            if sending:
                osc.scores(res)
            draw_skeleton(frame, pts[:k.n], k, (200, 200, 200), res["e"])
            out = side_by_side(frame, panel(res, k, frame.shape[0], mirror))
        else:
            if pts is not None:
                draw_skeleton(frame, pts[:k.n], k, ORANGE)
            out = frame
        now = time.time()                                                  # the frame rate reached: the whole loop
        period, last = smooth(period, now - last), now
        if ok and period > 0:
            label = "%.0f fps   (MediaPipe %.0f ms)" % (1 / period, detect * 1000)
            w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 16
            cv2.rectangle(out, (0, 0), (w, 30), (255, 255, 255), -1)
            text(out, label, 8, 21, 0.55, INK)
        out = np.pad(out, ((0, 95), (0, 0), (0, 0)), constant_values=255)       # room for the hints below
        bar, buttons = toolbar(max(out.shape[1], 1100), camera, mode, mirror, sending)
        out = stack(bar, out)
        if pts is None or teacher is None:
            hint = ("Press Set teacher (T) to take this %s as the teacher" % ("expression" if k.face else "pose") if teacher is None
                    else "Nobody found: step into the picture")
            text(out, hint if ok else "No picture from camera %d: choose another one" % camera, 10, out.shape[0] - 66, 0.6, INK, 2)
        if message and time.time() - said > 3:                             # messages stay for 3 seconds
            message = ""
        if message:
            text(out, message, 10, out.shape[0] - 38, 0.6, RED, 2)
        text(out, "camera %d   %s   OSC %s   keys: T teacher, M mirror, O OSC, C camera, B U H F mode, S save, Q quit"
             % (camera, mode.lower(), "to %s:%d" % osc.target if sending else "off"), 10, out.shape[0] - 12, 0.5, GREY)
        cv2.imshow("Compare poses", out)
        if action == "save":
            saved += 1
            cv2.imwrite(os.path.join(HERE, "webcam_%d.png" % saved), out)
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare a student's pose with a teacher's (MediaPipe).")
    ap.add_argument("photos", nargs="*", help="two pictures: teacher, then student (none: use the webcam)")
    ap.add_argument("--hands", action="store_true", help="compare hands (21 landmarks)")
    ap.add_argument("--face", action="store_true", help="compare faces (22 landmarks and 52 expression scores)")
    ap.add_argument("--upper", action="store_true", help="only the upper body")
    ap.add_argument("--mirror", action="store_true", help="flip the student first")
    ap.add_argument("--teacher", help="webcam: a picture with the teacher's pose")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--osc", action="store_true", help="send the keypoints by OSC (to the sketch Pose_Compare, in this folder); webcam: start with OSC on")
    ap.add_argument("--osc-to", default="127.0.0.1:12000", metavar="HOST:PORT", help="where to send OSC (default 127.0.0.1:12000)")
    args = ap.parse_args()
    kind = Kind(args.hands, args.upper, args.face)
    if len(args.photos) == 2:
        two_photos(args, kind)
    elif not args.photos:
        webcam(args, kind)
    else:
        sys.exit("Give two pictures (teacher, student), or none to use the webcam.")
