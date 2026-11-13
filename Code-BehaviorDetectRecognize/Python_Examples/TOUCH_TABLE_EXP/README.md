# The touch experiment: a finger touching the table

Part of the Python examples for the lecture *Recognising Behaviour* (`../../Behavioral-Data-Recognition.pptx`). Two ways
to tell when a fingertip touches a surface, with one camera and MediaPipe:

| File | What it does |
|---|---|
| `touch_surface.py` | the camera looks along the surface, edge-on: a fingertip touches when it reaches the line of the surface. No training |
| `touch_learn.py` | the camera looks down at the table: it records taps, learns to see touches from the sound of the taps, and then runs live, with a touch area that gives each touch its coordinates |

Both use the MediaPipe helpers of `../Compare_Poses_OSC/compare_poses.py` (and its `models` folder): keep the two
folders side by side. Run them from this folder, so that recordings, data and models are written here:

```bash
cd TOUCH_TABLE_EXP
pip install -r ../requirements.txt                    # mediapipe, opencv-python, numpy
pip install torch av scipy matplotlib sounddevice     # also, for touch_learn.py
```

Here now: `taps.mov` (a first recording), `taps.touch.npz` and `taps.check.png` (its data, from prepare), and
`touch_model.pt` and `touch_model.png` (the network trained on it).

## Touching a surface: `touch_surface.py`

One camera cannot see depth: from above or from the front, a finger touching the table and a finger hovering 2 cm
above it look the same (MediaPipe's z for a hand is only relative to the wrist). So the camera looks **along** the
surface, at its height: seen edge-on, the whole surface is a line in the picture, and a fingertip touches the surface
when it reaches that line.

```
     camera                 hand
       [o]                  \|//
        ' . . . . . . . . . .|. . .     the camera sees the surface edge-on: a line in the picture
 =====================================
```

1. Put the camera (the laptop, or a phone through Continuity Camera) at the level of the surface, looking across it:
   a table seen from its edge, a box or a book in front of the camera, a wall seen from the side. The closer the
   camera is to the height of the surface, the better.
2. `python touch_surface.py` (or `--camera 1`, or `--video clip.mp4` for a recorded video).
3. Show where the surface is: click two points on its edge in the picture, or touch the surface with the index finger
   and press **P**, at its left and at its right. The script prints the line as `--surface X1 Y1 X2 Y2`, to start with
   it next time.
4. Touch the surface: the fingertip turns orange and the zone under it lights up.

How it decides: the distance of each fingertip to the line, divided by the palm length (landmark 0 to 9) so that it
works at any distance from the camera. A fingertip touches below 0.15 palm lengths and is released above 0.3 (two
thresholds, hysteresis, so that the jitter of the keypoints does not make it flicker); `--down` and `--up` change them.
The window shows the distance of every fingertip as bars with both thresholds, the last touches, and the closest
fingertip's distance over time. The line is split into zones like keys (`--zones 5` by default, `--zones 0` for none):
a table piano. **Set surface (S)** starts again, **Flip side (F)** says the hand is on the other side of the line,
**Send OSC (O)** sends:

| Message | When |
|---|---|
| `/touch finger zone x state` | a fingertip touches (state 1) or leaves (state 0) the surface: finger 0 thumb … 4 little; zone 1 to N (0: outside the zones, or none); x from 0 to 1 along the surface. Sliding into another zone releases the old one and touches the new one, as on a keyboard |
| `/distance d0 d1 d2 d3 d4` | every frame with a hand: each fingertip's distance to the surface in palm lengths, for continuous control (a theremin over the table) |

Other ways to sense touch, often more reliable than a camera: capacitive touch on an ESP32 (its touch pins with
aluminium foil or conductive paint on the surface), a contact microphone or a piezo on the table for taps (the onset
detection of Part D), or a depth camera (Kinect, RealSense) that measures the distance of the fingertip to the surface.
A camera from above can also compare the fingertip with its shadow, which meets it when it touches.

## Learning touch from sound: `touch_learn.py`

With the camera looking **down** at the table (the laptop screen tilted forward, or an iPhone's Desk View), a finger
touching and a finger just above the table look almost the same. But a tap makes a sound. So the sound of a recording
is the teacher: it says when the finger touched, frame by frame, and a small convolutional network learns to see it
from the picture alone: the shadow meeting the fingertip, the finger stopping, its shape. Once trained, the network
needs no sound. (Using one sense to label another is called cross-modal or self-supervised learning.)

```bash
python touch_learn.py record taps.mov              # the camera and the microphone, guided: R, then tap and hover in turn
python touch_learn.py prepare taps.mov             # labels from the sound, patches from the picture: taps.touch.npz, taps.check.png
python touch_learn.py train taps.touch.npz         # (more recordings: add them) touch_model.pt, touch_model.png
python touch_learn.py live                         # the camera; or --video test.mov to compare with its taps
```

**Recording** (3 to 5 minutes). `record` shows the camera, a panel that says what to do, and the sound with the taps
it hears. First tilt the laptop screen forward, closing it a little, until the table in front of the laptop and your
hand are in the picture, and tap a few times to see that the taps are heard (in orange). Then press **R**: the panel
says **TAP** for 20 seconds and **HOVER** for 10, in turn (`--tap`, `--hover`), and the recording stops after 4 minutes
(`--minutes`) or when you press **R** again. Tap with the **index finger only**, quickly (touch and lift at once),
about once a second, at many places; when hovering, move the finger just above the table **without** touching it, so
that the network learns that close is not touching. Hovering in turn, not all at the end, matters: the end of the
recording is the test part. Do not rest the finger on the table (a touch that lasts is heard only when it starts).
A quiet room; a lamp low to one side gives a clear shadow. The first time, macOS asks to let the terminal (or VS Code)
use the microphone. A recording never replaces another (`taps_2.mov`, …); more recordings, with other light and other
tables, make the network more general. The file keeps every frame with the time it arrived and the sound uncompressed,
both timed from the first frame. QuickTime Player also works (File → New Movie Recording, with the camera and the
microphone chosen in the menu next to the record button), with the same protocol.

**How it works.**

1. *The taps in the sound*: the spectral flux (how much the spectrum grows from one 32 ms window to the next, as in
   Part D) peaks at every tap; peaks above an adaptive threshold are the taps (`--threshold` to find fewer or more).
2. *Sync*: the camera is usually slower than the microphone. The script lines up the taps with the moments the
   fingertip stops (its speed into a frame high, out of it low) and prints the shift; `--offset` sets it by hand. It
   depends on the camera and the microphone, not on the recording, so a shift found on a long recording can be reused.
3. *Labels*: a frame is *touching* from the tap to 80 ms after it, *not touching* 250 ms or more from any tap, and
   not used in between (landing or lifting).
4. *Patches*: MediaPipe finds the hand; a square of 1.2 palm lengths around the index fingertip is cut, turned so that
   the finger points down and scaled by the palm (the normalisation of Part C), in grey, 64 × 64 pixels. Its three
   channels are the frames t−2, t−1 and t, so the network also sees the motion.
5. *The network*: four convolution blocks and one output, the probability of touching. Each recording is split in
   time, not at random (neighbouring frames are nearly the same picture): the first 70% to learn from, the next 10%
   to choose the epoch and the threshold, the last 20% to test. Touching frames are rare, so they weigh more in the loss.
6. *Live*: the patch of every frame goes through the network; a touch starts when the probability crosses the
   threshold. The window shows what the network sees, its probability over time and, for a video, the taps heard.
   Every touch is printed, and shown on the picture for a moment, with where it was.

**A touch area.** Click **Set area (A)**, then the four corners of a rectangle on the table (a sheet of paper, tape
marks) in the picture, in order: top left, top right, bottom right, bottom left. Seen at an angle, the rectangle is a
trapezoid; a perspective transform (a homography, as when a phone straightens a photographed document) maps it back
to a rectangle, so every touch gets coordinates on it: u from left to right and v from top to bottom, from 0 to 1.
The window draws the area with a grid in perspective and shows it from above, with the fingertip and the last
touches. The script prints the area as `--area X1 Y1 … X4 Y4`, to start with it next time; `--aspect` (width ÷
height, 1.41 by default, a sheet of A4 lying wide) only sets the shape of the view from above.

**Send OSC (O)**, or `--osc`, sends:

| Message | When |
|---|---|
| `/touch x y state` | the finger touches (1) or leaves (0) the table; x, y the fingertip in the picture, 0 to 1 |
| `/touch/prob p` | every frame with a hand: the network's probability that the finger touches |
| `/area/touch u v state` | with an area: a touch inside it (1) and its release (0, at the same u, v); u, v on the area, 0 to 1 |
| `/area/finger u v` | with an area: every frame with a hand, the fingertip on the area (outside it, below 0 or above 1), for a cursor |

Look at `taps.check.png` before training: the taps in the sound, the labels on the picture's clock, the sync curve
(one clear peak) and example patches of both kinds. Wrong labels teach the network wrong things: in testing, labels
one frame late made the training unstable.

**What to expect.** In the same place and light as the recordings, a network like this can find most taps; in another
light or on another table it may not, until it has been trained there too. At 30 frames per second a tap lasts two or
three frames, and fast fingers blur: more light (a shorter exposure) helps. For timing, the sound itself stays more
precise than the camera; the camera says where.

## Checked

`touch_surface.py` was tested without a camera: the touch logic with numbers checked by hand (the hysteresis keeps a
touch through jitter between 0.18 and 0.27, sliding changes zone, a lost hand releases), and the whole window with a
synthetic hand tapping on a table line while sliding sideways (taps found in zones 1, 2, 3, 3; the surface set by
clicks and by P; OSC sent), and once with MediaPipe on a sample photo (55 fps, 15 ms per frame). Not yet with a real
camera at the height of a table.
`touch_learn.py` was tested on synthetic recordings made for it (a finger with a shadow over a textured table, taps
with a click in the sound and silent hovers, the picture 67 ms late) with the hand positions given instead of
MediaPipe: prepare found every tap (within 4 to 10 ms) and the shift (66 ms); trained on 4 minutes (in 23 s on the
Mac's GPU, 2 min on the CPU), the network found all 28 taps of the test part and all 26 of another recording, with no
false touches, and live sent OSC. Synthetic pictures are much easier than real ones: these numbers check the pipeline,
not how well it works with a real camera. prepare also ran with MediaPipe on a video without hands, and stopped with a
message. `record` was tested with a stand-in camera and microphone (a synthetic recording played in real time, every
frame arriving 50 ms late, with its sound), not with real ones: 90 s were written at 30 frames per second with no frame
lost or repeated and frame times right within a few milliseconds, and all 44 taps were heard; prepare then found the
picture 131 ms late against the taps heard (the frames themselves say 127 ms), and the network trained on it found
all 7 taps of the test part. Also checked: no camera, no microphone, stopping early, a file name already taken.
The touch area was checked on that recording, with the corners clicked by a script: every touch printed with its
u, v, and `/area/touch` and `/area/finger` received by a UDP listener.
A first real recording (1 min 51 s, 232 taps) gave one clear sync peak in prepare.
