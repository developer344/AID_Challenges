# Python examples: poses, objects and touch

Python scripts for the lecture *Recognising Behaviour* (`../Behavioral-Data-Recognition.pptx`). A script that sends
its results to a Processing sketch by OSC is in a folder together with that sketch.

| Folder or file | What it does | Receiver |
|---|---|---|
| `Compare_Poses_OSC/` | `compare_poses.py` finds a body, a hand or a face with MediaPipe and compares a student's pose (or expression) with a teacher's (Part C): [below](#comparing-poses-compare_poses_osc) | the sketch `Pose_Compare`, next to it (it also works alone, with the mouse) |
| `YOLO_OSC/` | `yolo_osc.py` finds objects on the webcam with YOLO (Part F); its first lines explain how to run it | the sketch `YOLO_Receiver`, next to it, draws the boxes |
| `TOUCH_TABLE_EXP/` | the touch experiment: `touch_surface.py` detects a fingertip touching a surface seen edge-on; `touch_learn.py` records taps on a table and learns to see touches, the sound of the taps giving the labels, with a touch area and its coordinates. See [its README](TOUCH_TABLE_EXP/README.md) | OSC to any program |

The touch scripts use the MediaPipe helpers of `Compare_Poses_OSC/compare_poses.py`: keep the folders side by side.

## Install

```bash
pip install -r requirements.txt                     # mediapipe, opencv-python, numpy
pip install torch av scipy matplotlib sounddevice   # also, for TOUCH_TABLE_EXP/touch_learn.py
pip install ultralytics python-osc                  # also, for YOLO_OSC/yolo_osc.py
```

The MediaPipe models are in `Compare_Poses_OSC/models/` (they are downloaded there automatically if missing).

## Comparing poses: `Compare_Poses_OSC`

The camera version of the Processing sketch `Pose_Compare`, which is in the same folder.
`compare_poses.py` finds a body, a hand or a face with MediaPipe and compares a student's pose with a teacher's,
with the same four steps as the slides:

| Step | Body | Hand (`--hands`) | Face (`--face`) |
|---|---|---|---|
| 1. Translate | hip centre to (0, 0) | wrist, landmark 0, to (0, 0) | the point between the eyes to (0, 0) |
| 2. Scale | ÷ torso length (hips to shoulders) | ÷ palm length (landmark 0 to 9) | ÷ eye distance (outer corner to outer corner) |
| 3. Rotate | the best angle α = atan2(D, C) | the same | the same (a tilted head) |
| 4. Score | distance per joint, mean, worst joint, cosine similarity, joint angles θ | the same, 21 landmarks; θ at the middle joint of each finger | the same, 22 of the 478 landmarks; and the 52 expression scores |

It shows (and, for two photos, prints) the distance of every joint as bars, the joints coloured by their error
(green < 0.15, orange < 0.4, red), the teacher and student overlaid after steps 1–3, two bars for the whole pose
(cosine similarity, with the angle φ between the poses, and the mean distance), and the joint angles.

**Faces.** MediaPipe's Face Landmarker gives 478 points and 52 *expression scores* (blendshapes), each from 0 to 1:
how much the jaw is open, each side smiles, each eye blinks, each brow is raised, and so on. The face mode is a game:
copy the teacher's expression. The panel shows the 10 most active scores of both, their cosine similarity (the
expression as one list of 52 numbers, as in the cosine slide) and their mean difference; the face's shape is compared
with the four steps on 22 landmarks (brows, eyes, mouth, nose, chin). The scores are good continuous controls: send
them by OSC to a sound engine, an avatar or Wekinator.
They measure movements of the face, not emotions. A smile score says the corners of the mouth are pulled up, not that
someone is happy: people smile when they are not, and feel without showing it, and the same movement means different
things in different contexts and cultures (Barrett et al., 2019). Use expressions as inputs that people make on
purpose, tell them, and store nothing. In the EU, systems that infer the emotions of people at work or in education are
prohibited since February 2025 (AI Act, article 5), and face data used to identify people needs explicit consent (GDPR,
article 9).

### Use

```bash
cd Compare_Poses_OSC

# Two photos: teacher, then student. Prints the scores and saves comparison.png
python compare_poses.py samples/left_man.png samples/right_man.png --upper

# The same person, turned 15° and smaller: the distances stay small, α ≈ 15°
python compare_poses.py samples/woman.png samples/woman_turned.png --upper

# Webcam: stand in front of the camera, press T to take your pose as the teacher, then move
python compare_poses.py
python compare_poses.py --teacher samples/woman.png      # against the pose in a photo
python compare_poses.py --hands                           # hands instead of bodies
python compare_poses.py --face                            # faces: copy the teacher's expression

# Send the keypoints by OSC to the sketch Pose_Compare, in this folder (press O in the sketch too):
# click Send OSC in the toolbar (or press O), or start with it on
python compare_poses.py --osc
python compare_poses.py --osc --hands
```

Options: `--upper` compares only the upper body (nose, shoulders, elbows, wrists, hips), for when the legs are out of
the picture; `--mirror` flips the student first, for a student facing the teacher; `--camera N` picks another camera;
`--out FILE` names the saved comparison; `--osc` sends the keypoints by OSC (with the webcam: starts with the
**Send OSC** button on), to `--osc-to HOST:PORT` (default `127.0.0.1:12000`).

Webcam: the **toolbar** at the top of the window chooses the **camera** (0 to 3) and what is **tracked** (Body,
Upper body, Hand or Face), and has **Set teacher**, **Mirror** and **Send OSC** buttons (Send OSC turns sending the keypoints
on and off, see [OSC](#osc)). Body and upper body share the same teacher; the hand and the face have their own. Keys do the same:
**T** teacher, **M** mirror, **O** OSC on/off, **C** next camera, **B** / **U** / **H** / **F** body, upper body, hand, face,
**S** save a picture, **Q** or Esc quit.
The top-left corner of the picture shows the frame rate reached (frames per second, for the whole loop) and how long
MediaPipe takes per frame: when that time is close to 1000 / fps, MediaPipe is what limits the rate; most webcams also
give at most 30 frames per second.

`samples/example_comparison.png` (in `Compare_Poses_OSC`) shows what the comparison of two photos looks like.

### Tips

- Stand back so that the whole body is in the picture, or use `--upper`: joints outside the picture are guessed, and
  their distances are meaningless.
- On macOS the first webcam run asks to let the terminal use the camera; run it again after allowing it.
- On a Mac with an iPhone nearby, camera 0 can be the iPhone (Continuity Camera). Click another camera in the toolbar
  (or press **C**), or start with `--camera 1`. To stop it, turn off Continuity Camera on the iPhone (Settings, General,
  AirPlay & Continuity).
- `L` and `R` are the person's left and right, as MediaPipe gives them; so are `Left` and `Right` in the expression
  scores. With **Mirror**, the student's left and right scores are swapped too.
- Faces: come close to the camera (the face detector is made for faces within about 2 m), with light on your face. In
  photos, the face must be big: a portrait, not a whole body.

### OSC

With the **Send OSC** button on (or `--osc`) the script sends, by UDP and with no extra library, the keypoints in pixels of the picture, after its
width and height:

| Message | When |
|---|---|
| `/pose w h x0 y0 … x12 y12` | every webcam frame with a body (all 13 joints, also with `--upper`) |
| `/hand w h x0 y0 … x20 y20` | every webcam frame with a hand (21 landmarks) |
| `/face w h x0 y0 … x21 y21` | every webcam frame with a face (22 landmarks, in the order of the table below) |
| `/face/blend s0 … s51` | with every `/face`: the 52 expression scores, 0 to 1, in the order below |
| `/teacher/pose …`, `/teacher/hand …`, `/teacher/face …`, `/teacher/face/blend …` | when the teacher is set, and again every 2 seconds for a program started later |
| `/scores mean cosine worst alpha` | once there is a teacher: mean distance, cosine similarity, number of the worst joint, best rotation in degrees |
| `/face/scores cosine mean worst` | faces, once there is a teacher: the cosine similarity and mean difference of the expression scores, and the number of the most different one |

The 22 face landmarks: nose tip; brow R outer, middle, inner; brow L inner, middle, outer; eye R outer, top, inner,
bottom; eye L inner, top, outer, bottom; mouth R, lip top, mouth L, lip bottom, mouth inside top, mouth inside bottom;
chin (MediaPipe's 1, 70, 105, 107, 336, 334, 300, 33, 159, 133, 145, 362, 386, 263, 374, 61, 0, 291, 17, 13, 14, 152).

The 52 expression scores, numbered from 0: `_neutral`, `browDownLeft`, `browDownRight`, `browInnerUp`,
`browOuterUpLeft`, `browOuterUpRight`, `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight`, `eyeBlinkLeft`,
`eyeBlinkRight`, `eyeLookDownLeft`, `eyeLookDownRight`, `eyeLookInLeft`, `eyeLookInRight`, `eyeLookOutLeft`,
`eyeLookOutRight`, `eyeLookUpLeft`, `eyeLookUpRight`, `eyeSquintLeft`, `eyeSquintRight`, `eyeWideLeft`, `eyeWideRight`,
`jawForward`, `jawLeft`, `jawOpen` (25), `jawRight`, `mouthClose`, `mouthDimpleLeft`, `mouthDimpleRight`,
`mouthFrownLeft`, `mouthFrownRight`, `mouthFunnel`, `mouthLeft`, `mouthLowerDownLeft`, `mouthLowerDownRight`,
`mouthPressLeft`, `mouthPressRight`, `mouthPucker`, `mouthRight`, `mouthRollLower`, `mouthRollUpper`,
`mouthShrugLower`, `mouthShrugUpper`, `mouthSmileLeft` (44), `mouthSmileRight` (45), `mouthStretchLeft`,
`mouthStretchRight`, `mouthUpperUpLeft`, `mouthUpperUpRight`, `noseSneerLeft`, `noseSneerRight`.

Two photos send each message once. The sketch `Pose_Compare`, in the same folder, listens for them after you press **O**, and
switches to bodies or hands by itself (it does not draw faces). `DTW_Gestures`, in `../Processing_Examples`, listens too:
the index fingertip (or, for a body, the right wrist) becomes its pen, and holding Space draws a gesture to record or
recognise. Run one of the two sketches at a time: both listen on port 12000. Also, another program (Max, Pure Data, Wekinator, a sound
engine) can use `/scores`, and `/face/blend` as 52 inputs.
While the script sends a teacher, it replaces the one in the sketch.

## Checked

Tested with mediapipe 0.10.21 and OpenCV 4.11 on the sample photos (crops of `AID-LLMs/images/people-dancing.jpeg`):
the same person turned 15° and scaled gives α = +14.6° and a mean distance of 0.07 torso lengths (MediaPipe's own
jitter); two different people give a mean of 0.22, the right wrist worst at 0.63. The hand mode ran on the same photo,
where the hands are small. The webcam mode was tested with the sample photos standing in for cameras 0 and 1 and with
scripted keys and clicks (teacher, camera, body, upper body, hand, a missing camera), not with a real camera.
OSC was checked with a UDP listener (the messages above, 28 numbers for a body, 44 for a hand) and with `Pose_Compare`
receiving bodies and hands from the sample photos.
The face mode was tested on head crops of the sample photos, enlarged (the faces in the whole photos are only about
30 pixels wide, too small for the face detector at every scale): two people compared found the smile of one as the
largest difference (mouthSmileLeft 0.08 against 0.75); OSC sent `/face` (46 numbers), `/face/blend` (52), the teacher's
and `/face/scores`; the webcam mode, with a photo standing in for the camera and scripted keys, switched to Face, set a
teacher, compared, mirrored and switched back to hand and body. A face against its own mirror image gave a cosine
similarity of the scores of 0.84 with Mirror on, 0.71 without: MediaPipe reads a small, blurry face a little
differently when it is flipped. Not yet with a real webcam.
