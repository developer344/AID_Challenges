"""Detect when a fingertip touches a surface, with MediaPipe Hands and one webcam.

One camera cannot see depth: seen from above or from the front, a finger touching the table and a finger hovering
2 cm above it look the same. So the camera looks ALONG the surface, at its height. Seen edge-on, the whole surface
is a line in the picture, and a fingertip touches the surface when it reaches that line:

        camera                 hand
          [o]                  \\|//
           ' . . . . . . . . . .|. . .   what the camera sees: the surface edge-on,
    =====================================  a line in the picture

Setup: put the camera (a laptop, or a phone through Continuity Camera) at the level of the surface, looking across
it: a table seen from its edge, a box or a book in front of the camera, a wall seen from the side.
Then show the script where the surface is: click two points on its edge in the picture, or touch the surface with
the index finger and press P, at two places (left and right). The closer the camera is to the height of the
surface, the better: a camera above it sees the far part of the surface higher in the picture than the near part.

A fingertip touches when its distance to the line falls below 0.15 palm lengths, and is released when it rises
above 0.3: two thresholds (hysteresis), so that the jitter of the keypoints does not make it flicker. Distances
are divided by the palm length (landmark 0 to 9), so the thresholds work at any distance from the camera.
The line is split into zones, like keys: touching zone 3 presses key 3 (a table piano).

Usage:
    python touch_surface.py                      webcam 0
    python touch_surface.py --camera 1
    python touch_surface.py --video clip.mp4     a recorded video instead of the camera
    python touch_surface.py --zones 8 --osc      8 zones, and send the touches by OSC
    python touch_surface.py --surface 20 400 620 404   the surface line given (x1 y1 x2 y2 in pixels; printed
                                                 each time you set it, to start with it next time)
Options: --down 0.15 and --up 0.3 change the thresholds (palm lengths); --zones 0 for no zones.
Toolbar: camera 0 to 3, Set surface, Flip side, Send OSC. Keys: S set the surface again, P take the index
fingertip as a point of the surface, F flip the side the hand comes from, O OSC on/off, C next camera,
Q or Esc quit.

OSC (button Send OSC, or --osc; to --osc-to HOST:PORT, default 127.0.0.1:12000), no library needed:
    /touch finger zone x state    when a fingertip touches the surface (state 1) or leaves it (state 0); finger 0
                                  thumb, 1 index, 2 middle, 3 ring, 4 little; zone 1 to N (0: outside the zones, or
                                  no zones); x from 0 to 1 along the surface. Sliding into another zone sends a
                                  release of the old zone and a touch of the new one, as on a keyboard.
    /distance d0 d1 d2 d3 d4      every frame with a hand: the distance of each fingertip to the surface, in palm
                                  lengths (for continuous control, like a theremin over the table)

Requirements: the same as compare_poses.py (mediapipe, opencv-python, numpy), which it uses from ../Compare_Poses_OSC.
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Compare_Poses_OSC"))
import compare_poses as cp      # the MediaPipe detector, the OSC sender and the drawing helpers, from ../Compare_Poses_OSC

TIPS = [4, 8, 12, 16, 20]                                    # the fingertips among the 21 hand landmarks
FINGERS = ["thumb", "index", "middle", "ring", "little"]
BLUE, ORANGE, INK, GREY, RED, TEAL = cp.BLUE, cp.ORANGE, cp.INK, cp.GREY, cp.RED, cp.TEAL
BAR = 56                                                     # height of the toolbar, above the picture


def pt(p):
    return int(round(p[0])), int(round(p[1]))


# ---- The surface and the touches ----

class Surface:
    """The surface seen edge-on: the line through two points a and b, and the side of it where the hand is."""

    def __init__(self, a, b, shape):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        if (abs(d[0]) >= abs(d[1]) and d[0] < 0) or (abs(d[0]) < abs(d[1]) and d[1] < 0):
            a, b = b, a                                      # a on the left (or at the top): x goes from 0 at a to 1 at b
        self.a, self.b = a, b
        self.length = max(np.linalg.norm(b - a), 1.0)
        self.u = (b - a) / self.length                       # along the surface
        self.n = np.array([-self.u[1], self.u[0]])           # across it
        centre = np.array([shape[1] / 2, shape[0] / 2])
        if (centre - a) @ self.n < 0:                        # the hand comes from the side where most of the picture is
            self.n = -self.n

    def flip(self):
        self.n = -self.n

    def distance(self, p):
        """Pixels from the surface: positive on the hand's side, negative past the surface."""
        return float((np.asarray(p) - self.a) @ self.n)

    def position(self, p):
        """Where along the surface: 0 at a, 1 at b (outside the segment: below 0 or above 1)."""
        return float((np.asarray(p) - self.a) @ self.u / self.length)

    def args(self):
        return "--surface %d %d %d %d" % (*pt(self.a), *pt(self.b))


class Touches:
    """For each fingertip: is it down on the surface? Down below one threshold, up again above a higher one."""

    def __init__(self, down, up, zones):
        self.down_at, self.up_at, self.zones = down, up, zones
        self.d = [None] * 5                                  # smoothed distances, in palm lengths
        self.down = [False] * 5
        self.zone = [0] * 5                                  # the zone each finger pressed, for its release
        self.x = [0.0] * 5

    def zone_of(self, x):
        if self.zones == 0 or not 0 <= x <= 1:
            return 0
        return min(int(x * self.zones), self.zones - 1) + 1

    def update(self, dists, xs):
        """New distances and positions of the five fingertips. Returns the events: (finger, state, zone, x)."""
        events = []
        for f in range(5):
            self.d[f] = dists[f] if self.d[f] is None else 0.5 * (self.d[f] + dists[f])   # a little smoothing
            self.x[f] = xs[f]
            zone = self.zone_of(xs[f])
            if not self.down[f] and self.d[f] < self.down_at:
                self.down[f], self.zone[f] = True, zone
                events.append((f, 1, zone, xs[f]))
            elif self.down[f] and self.d[f] > self.up_at:
                self.down[f] = False
                events.append((f, 0, self.zone[f], xs[f]))
            elif self.down[f] and zone != self.zone[f]:      # sliding into another zone: leave one, press the next
                events += [(f, 0, self.zone[f], xs[f]), (f, 1, zone, xs[f])]
                self.zone[f] = zone
        return events

    def lost(self):
        """The hand is out of sight: every finger that was down is released."""
        events = [(f, 0, self.zone[f], self.x[f]) for f in range(5) if self.down[f]]
        self.d, self.down = [None] * 5, [False] * 5
        return events

    def closest(self):
        known = [d for d in self.d if d is not None]
        return min(known) if known else None


# ---- Drawing ----

def label(img, s, p, scale=0.55, colour=INK, bold=1):
    """Text with a white outline, readable on any picture."""
    cv2.putText(img, s, pt(p), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), bold + 3, cv2.LINE_AA)
    cv2.putText(img, s, pt(p), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, bold, cv2.LINE_AA)


def draw_surface(img, s, touches):
    """The line across the picture, and the zones on the segment from a to b; the zones pressed in orange."""
    far = 2 * sum(img.shape[:2])
    cv2.line(img, pt(s.a - s.u * far), pt(s.a + s.u * far), (200, 200, 200), 1, cv2.LINE_AA)
    pressed = {touches.zone[f] for f in range(5) if touches.down[f]}
    n = max(touches.zones, 1)
    for z in range(n):
        q0, q1 = s.a + s.u * s.length * z / n, s.a + s.u * s.length * (z + 1) / n
        on = (z + 1) in pressed if touches.zones else any(touches.down)
        cv2.line(img, pt(q0 + s.u * 2), pt(q1 - s.u * 2), ORANGE if on else ((215, 215, 215) if z % 2 else (150, 150, 150)), 7, cv2.LINE_AA)
        if touches.zones:
            label(img, str(z + 1), (q0 + q1) / 2 - s.n * 22 - np.array([5, -6]), 0.6, ORANGE if on else INK, 2)
    for p in (s.a, s.b):
        cv2.circle(img, pt(p), 5, RED, -1, cv2.LINE_AA)


def draw_hand(img, pts, s, touches):
    """The hand; each fingertip a ring, filled in orange when it touches; the closest one's distance as a line."""
    cp.draw_skeleton(img, pts, cp.Kind(True, False), BLUE)
    for f, t in enumerate(TIPS):
        if touches.down[f]:
            cv2.circle(img, pt(pts[t]), 11, ORANGE, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt(pts[t]), 9, (255, 255, 255), 2, cv2.LINE_AA)
    if s is not None and touches.closest() is not None:
        f = int(np.argmin([d if d is not None else 1e9 for d in touches.d]))
        p = pts[TIPS[f]]
        foot = p - s.n * s.distance(p)                         # the nearest point of the surface
        cv2.line(img, pt(p), pt(foot), TEAL, 2, cv2.LINE_AA)
        label(img, "%.2f" % touches.d[f], (p + foot) / 2 + s.u * 10, 0.5, TEAL, 1)


def panel(touches, log, height, t0):
    """Beside the picture: the distance of each fingertip as a bar, with the two thresholds; the last touches."""
    w = 360
    img = np.full((max(height, 460), w, 3), 255, np.uint8)
    cp.text(img, "Distance to the surface", 12, 28, 0.6, INK, 2)
    cp.text(img, "palm lengths: touch below %.2f (orange)," % touches.down_at, 12, 50, 0.45, GREY)
    cp.text(img, "release above %.2f (grey)" % touches.up_at, 12, 70, 0.45, GREY)
    bx, bw, top = 85, 200, 1.0
    for f in range(5):
        y = 82 + f * 34
        cp.text(img, FINGERS[f], 12, y + 22, 0.52, INK)
        cv2.rectangle(img, (bx, y + 8), (bx + bw, y + 28), (235, 235, 235), -1)
        d = touches.d[f]
        if d is not None:
            fill = int(np.clip(max(d, 0) / top, 0, 1) * bw)
            cv2.rectangle(img, (bx, y + 8), (bx + max(fill, 2), y + 28), ORANGE if touches.down[f] else BLUE, -1)
            cp.text(img, "touch" if touches.down[f] else "%.2f" % d, bx + bw + 10, y + 24, 0.5, ORANGE if touches.down[f] else INK,
                    2 if touches.down[f] else 1)
        for th, c in ((touches.down_at, ORANGE), (touches.up_at, GREY)):
            x = bx + int(th / top * bw)
            cv2.line(img, (x, y + 4), (x, y + 32), c, 2)
    y = 82 + 5 * 34 + 34
    cp.text(img, "Touches", 12, y, 0.6, INK, 2)
    if not log:
        cp.text(img, "none yet", 12, y + 26, 0.5, GREY)
    for i, (t, f, state, zone) in enumerate(log):
        s = "%6.1f s  %-6s %s%s" % (t - t0, FINGERS[f], "touch" if state else "release", "  zone %d" % zone if zone else "")
        cp.text(img, s, 12, y + 26 + i * 24, 0.5, (ORANGE if state else INK) if i == 0 else GREY, 1)
    return img


def plot(history, width, touches):
    """The closest fingertip's distance over the last frames, with the two thresholds; touching shaded."""
    h, x0, x1, y0, y1, top = 160, 50, width - 20, 36, 146, 1.0
    img = np.full((h, width, 3), 255, np.uint8)
    cp.text(img, "Closest fingertip to the surface over time (palm lengths)", 12, 20, 0.5, INK)
    n = history.maxlen
    X = lambda i: int(x0 + i * (x1 - x0) / (n - 1))
    Y = lambda d: int(y1 - np.clip(d, 0, top) / top * (y1 - y0))
    for i, (d, down) in enumerate(history):
        if down:
            cv2.rectangle(img, (X(i), y0), (X(i) + max((x1 - x0) // n, 1), y1), (200, 225, 250), -1)
    for th, c in ((touches.down_at, ORANGE), (touches.up_at, GREY)):
        cv2.line(img, (x0, Y(th)), (x1, Y(th)), c, 2)
        cp.text(img, "%.2f" % th, 8, Y(th) + 5, 0.4, c)
    cv2.line(img, (x0, y1), (x1, y1), (180, 180, 180), 1)
    for i in range(1, len(history)):
        a, b = history[i - 1][0], history[i][0]
        if a is not None and b is not None:
            cv2.line(img, (X(i - 1), Y(a)), (X(i), Y(b)), BLUE, 2, cv2.LINE_AA)
    return img


def toolbar(width, camera, sending):
    """The buttons at the top. Returns the image and the buttons as (x0, y0, x1, y1, action)."""
    bar = np.full((BAR, width, 3), 245, np.uint8)
    buttons, x = [], 10

    def button(text, action, on):
        nonlocal x
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 22
        cv2.rectangle(bar, (x, 12), (x + w, 44), BLUE if on else (255, 255, 255), -1)
        cv2.rectangle(bar, (x, 12), (x + w, 44), (170, 170, 170), 1)
        cp.text(bar, text, x + 11, 34, 0.55, (255, 255, 255) if on else INK)
        buttons.append((x, 12, x + w, 44, action))
        x += w + 6

    cp.text(bar, "Camera", x, 34, 0.55)
    x += 72
    for n in range(cp.CAMERAS):
        button(str(n), n, n == camera)
    x += 16
    button("Set surface (S)", "surface", False)
    button("Flip side (F)", "flip", False)
    x += 16
    button("Send OSC (O)", "osc", sending)
    return bar, buttons


# ---- The loop ----

def run(args):
    det = cp.Detector(cp.Kind(True, False), video=True)
    camera = None if args.video else args.camera
    cap = cv2.VideoCapture(args.video) if args.video else cp.open_camera(camera)
    if args.video and not cap.isOpened():
        sys.exit("Cannot read the video " + args.video)
    touches = Touches(args.down, args.up, args.zones)
    surface, picks = None, ([args.surface[:2], args.surface[2:]] if args.surface else [])
    osc, sending = cp.Osc(args.osc_to), args.osc
    history, log = deque([(None, False)] * 200, maxlen=200), deque(maxlen=6)
    clicks, buttons, frame_shape = [], [], (480, 640)
    cv2.namedWindow("Touch a surface")
    cv2.setMouseCallback("Touch a surface", lambda event, x, y, flags, param: clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    message = "" if cap else "Camera %d gives no picture: choose another one" % camera
    said = t0 = last = time.time()
    period = detect = 0.0
    print("Click two points on the surface's edge (or touch it with the index finger and press P, twice).")
    while True:
        ok, frame = cap.read() if cap else (False, None)
        if not ok and args.video:
            break                                            # the end of the video
        frame = cp.fit(frame) if ok else np.full((480, 640, 3), 230, np.uint8)
        frame_shape = frame.shape
        t = time.time()
        pts = det.find(frame) if ok else None
        detect = cp.smooth(detect, time.time() - t)

        # What to do: a key, or a click on the toolbar or on the picture
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        action = {ord("s"): "surface", ord("f"): "flip", ord("o"): "osc", ord("p"): "point",
                  ord("c"): ((camera or 0) + 1) % cp.CAMERAS}.get(key)
        while clicks:
            x, y = clicks.pop(0)
            hit = next((a for x0, y0, x1, y1, a in buttons if x0 <= x <= x1 and y0 <= y <= y1), None)
            if hit is not None:
                action = hit
            elif surface is None and 0 <= x < frame_shape[1] and BAR <= y < BAR + frame_shape[0]:
                picks.append((x, y - BAR))                   # a point of the surface, clicked on the picture
        if action is not None:
            said = time.time()
        events = []
        if action == "surface":
            surface, picks, message = None, [], "Click two points on the surface, or touch it and press P, twice"
            events += touches.lost()
        elif action == "flip" and surface is not None:
            surface.flip()
            message = "The hand is now on the other side of the line"
        elif action == "osc":
            sending = not sending
            message = "Sending OSC to %s:%d" % osc.target if sending else "OSC off"
        elif action == "point":
            if pts is None:
                message = "No hand found: show the hand, touch the surface with the index finger, press P"
            else:
                if surface is not None:                      # start a new surface from this point
                    surface, picks = None, []
                    events += touches.lost()
                picks.append(tuple(pts[8]))
                message = "Surface point %d of 2" % len(picks)
        elif isinstance(action, int) and action != camera:
            new = cp.open_camera(action)
            if new:
                if cap:
                    cap.release()
                cap, camera, message = new, action, "Camera %d" % action
            else:
                message = "Camera %d is not available" % action
        if surface is None and len(picks) >= 2:
            surface = Surface(picks[0], picks[1], frame_shape)
            picks = []
            message = "Surface set"
            print("Surface set. To start with it next time:", surface.args())

        # The touches
        if surface is not None and pts is not None:
            palm = max(np.linalg.norm(pts[9] - pts[0]), 1.0)
            events += touches.update([surface.distance(pts[t]) / palm for t in TIPS], [surface.position(pts[t]) for t in TIPS])
            if sending:
                osc.send("/distance", touches.d)
        else:
            events += touches.lost()
        for f, state, zone, x in events:
            log.appendleft((time.time(), f, state, zone))
            print("%-6s %s%s  (x %.2f)" % (FINGERS[f], "touch" if state else "release", "  zone %d" % zone if zone else "", x))
            if sending:
                osc.send("/touch", [f, zone, x, state])
        history.append((touches.closest(), any(touches.down)))

        # The picture
        if surface is not None:
            draw_surface(frame, surface, touches)
        for p in picks:
            cv2.circle(frame, pt(p), 6, RED, -1, cv2.LINE_AA)
        if pts is not None:
            draw_hand(frame, pts, surface, touches)
        down = [FINGERS[f] + (" zone %d" % touches.zone[f] if touches.zone[f] else "") for f in range(5) if touches.down[f]]
        label(frame, "TOUCH: " + ", ".join(down) if down else "no touch", (10, 62), 0.8, ORANGE if down else GREY, 2)
        now = time.time()                                    # the frame rate reached, as in compare_poses.py
        period, last = cp.smooth(period, now - last), now
        if ok and period > 0:
            s = "%.0f fps   (MediaPipe %.0f ms)" % (1 / period, detect * 1000)
            cv2.rectangle(frame, (0, 0), (cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 16, 30), (255, 255, 255), -1)
            cp.text(frame, s, 8, 21, 0.55, INK)
        out = cp.side_by_side(frame, panel(touches, log, frame.shape[0], t0))
        out = np.vstack([out, plot(history, out.shape[1], touches)])
        out = np.pad(out, ((0, 95), (0, 0), (0, 0)), constant_values=255)
        bar, buttons = toolbar(out.shape[1], camera, sending)
        out = cp.stack(bar, out)
        if surface is None:
            hint = "Click 2 points on the surface edge, or touch it and press P twice"
        elif pts is None:
            hint = "No hand found" if ok else "No picture from camera %s: choose another one" % camera
        else:
            hint = ""
        cp.text(out, hint, 10, out.shape[0] - 66, 0.55, INK, 2)
        if message and time.time() - said > 3:
            message = ""
        cp.text(out, message, 10, out.shape[0] - 38, 0.6, RED, 2)
        cp.text(out, "%s   zones %d   OSC %s   keys: S surface, P point, F flip, O OSC, C camera, Q quit"
                % ("video" if args.video else "camera %d" % camera, args.zones, "to %s:%d" % osc.target if sending else "off"),
                10, out.shape[0] - 12, 0.5, GREY)
        cv2.imshow("Touch a surface", out)
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect when a fingertip touches a surface seen edge-on (MediaPipe Hands).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="a recorded video instead of the camera")
    ap.add_argument("--zones", type=int, default=5, help="zones (keys) along the surface; 0 for none")
    ap.add_argument("--down", type=float, default=0.15, help="touch below this distance (palm lengths)")
    ap.add_argument("--up", type=float, default=0.3, help="release above this distance (palm lengths)")
    ap.add_argument("--surface", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="the surface line, in pixels")
    ap.add_argument("--osc", action="store_true", help="start with sending by OSC on")
    ap.add_argument("--osc-to", default="127.0.0.1:12000", metavar="HOST:PORT", help="where to send OSC (default 127.0.0.1:12000)")
    run(ap.parse_args())
