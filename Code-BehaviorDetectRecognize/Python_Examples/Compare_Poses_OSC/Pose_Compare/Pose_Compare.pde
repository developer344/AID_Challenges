// Comparing two poses, bodies or hands: drag the joints of the student (middle) to copy the teacher (left).
// Right: both poses after normalising (bodies: centred on the hips, scaled by the torso; hands: centred on the
// wrist, landmark 0, scaled by the palm, 0 to 9) and, if on, rotated to overlay best (Procrustes).
// Each student joint is drawn bigger the further it is off.
// Below left: the distance of every student joint to the same teacher joint, as bars (the worst in orange,
// the dashed line is the mean). Below right: two scores for the whole pose (cosine similarity and mean distance),
// the worst joint, and the joint angles of teacher and student, which need no normalisation at all.
// Notice how the cosine similarity stays high while one arm (or finger) is clearly wrong.
// Keys: H body or hand, R rotation on/off, M mirror the student (bodies: left and right swapped),
// T new teacher pose (with OSC: take the current pose as the teacher), O OSC input on/off.
// OSC input (no library needed): listens on port 12000 for keypoints in pixels, as sent by
// compare_poses.py --osc, in the folder above this sketch (MediaPipe and a webcam):
//   /pose w h x0 y0 ... x12 y12           the student's 13 body joints (image width and height first)
//   /hand w h x0 y0 ... x20 y20           the student's 21 hand landmarks
//   /teacher/pose ...  /teacher/hand ...  the same, for the teacher
// The sketch switches to bodies or hands by itself, following what arrives.

import java.net.*;
import java.nio.ByteBuffer;

// Bodies: 13 joints
String[] BODY_NAMES = {"head", "shoulder L", "shoulder R", "elbow L", "elbow R", "wrist L", "wrist R",
                       "hip L", "hip R", "knee L", "knee R", "ankle L", "ankle R"};
int[][] BODY_BONES = {{1, 2}, {1, 3}, {3, 5}, {2, 4}, {4, 6}, {1, 7}, {2, 8}, {7, 8}, {7, 9}, {9, 11}, {8, 10}, {10, 12}, {0, 1}, {0, 2}};
int[][] BODY_ANGLES = {{1, 3, 5}, {2, 4, 6}, {3, 1, 7}, {4, 2, 8}, {7, 9, 11}, {8, 10, 12}};   // joint in the middle
String[] BODY_ANGLE_NAMES = {"elbow L", "elbow R", "shoulder L", "shoulder R", "knee L", "knee R"};
float[][] BODY_BASE = {{0, -1.7}, {-0.2, -1.45}, {0.2, -1.45}, {-0.45, -1.45}, {0.45, -1.45}, {-0.7, -1.45}, {0.7, -1.45},
                       {-0.13, -0.95}, {0.13, -0.95}, {-0.15, -0.5}, {0.15, -0.5}, {-0.16, -0.05}, {0.16, -0.05}};
int[] BODY_SWAP = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11};   // left <-> right, for the mirror

// Hands: the 21 MediaPipe landmarks, 0 the wrist, then 4 per finger from the base to the tip
String[] HAND_NAMES = {"0 wrist", "1 thumb", "2 thumb", "3 thumb", "4 thumb tip", "5 index", "6 index", "7 index", "8 index tip",
                       "9 middle", "10 middle", "11 middle", "12 middle tip", "13 ring", "14 ring", "15 ring", "16 ring tip",
                       "17 little", "18 little", "19 little", "20 little tip"};
int[][] HAND_BONES = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10}, {10, 11}, {11, 12},
                      {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}};
int[][] HAND_ANGLES = {{2, 3, 4}, {5, 6, 7}, {9, 10, 11}, {13, 14, 15}, {17, 18, 19}};
String[] HAND_ANGLE_NAMES = {"thumb", "index", "middle", "ring", "little"};
float[][] HAND_BASE = {{0, 0}, {-0.35, 0.25}, {-0.62, 0.5}, {-0.82, 0.75}, {-0.98, 0.98}, {-0.3, 1.0}, {-0.35, 1.45}, {-0.38, 1.76}, {-0.4, 2.02},
                       {0, 1.05}, {0, 1.55}, {0, 1.88}, {0, 2.16}, {0.28, 1.0}, {0.32, 1.45}, {0.35, 1.76}, {0.37, 1.99},
                       {0.53, 0.88}, {0.63, 1.22}, {0.69, 1.45}, {0.73, 1.66}};   // y upwards, palm length 1.05

// What is compared now
boolean hands = false;
String[] NAMES, ANGLE_NAMES;
int[][] BONES, ANGLES;
PVector[] teacher, student;                // screen coordinates: teacher in the left panel, student in the middle one
boolean rotateOn = true, mirror = false;
int dragging = -1;

// OSC input
int OSC_PORT = 12000;
boolean oscOn = false;
DatagramSocket socket;
float[] inStudent, inTeacher;              // the last keypoints received, waiting to be used in draw()
boolean inStudentHand, inTeacherHand;
int lastOsc = -100000;
String oscNote = "OSC off (O to listen on port " + OSC_PORT + ")";

color BLUE = #2E75B6, ORANGE = #E36C0A, INK = #262626, GREY = #AAAAAA, RED = #C00000, TEAL = #215968;

void setup() {
  size(1000, 900);
  textFont(createFont("Arial", 14));
  setMode(false);
}

// Bodies or hands: the names, bones and angles of the keypoints, and new default poses
void setMode(boolean h) {
  hands = h;
  NAMES = h ? HAND_NAMES : BODY_NAMES;
  BONES = h ? HAND_BONES : BODY_BONES;
  ANGLES = h ? HAND_ANGLES : BODY_ANGLES;
  ANGLE_NAMES = h ? HAND_ANGLE_NAMES : BODY_ANGLE_NAMES;
  newTeacher();
}

// A teacher pose and a student copy that is moved, smaller, tilted and wrong in one place
void newTeacher() {
  int n = NAMES.length;
  teacher = new PVector[n];
  student = new PVector[n];
  if (!hands) {                            // a random body pose, y upwards negative
    PVector[] p = new PVector[n];
    for (int i = 0; i < n; i++) p[i] = new PVector(BODY_BASE[i][0], BODY_BASE[i][1]);
    float a1 = random(-PI, 0.3), a2 = random(-PI, 0.3);       // the two arms
    float b1 = a1 + random(-0.6, 0.6), b2 = a2 + random(-0.6, 0.6);   // the forearms
    p[3] = PVector.add(p[1], new PVector(-0.25 * cos(a1), 0.25 * sin(a1)));
    p[5] = PVector.add(p[3], new PVector(-0.25 * cos(b1), 0.25 * sin(b1)));
    p[4] = PVector.add(p[2], new PVector(0.25 * cos(a2), 0.25 * sin(a2)));
    p[6] = PVector.add(p[4], new PVector(0.25 * cos(b2), 0.25 * sin(b2)));
    for (int i = 0; i < n; i++) {
      teacher[i] = new PVector(165 + p[i].x * 140, 450 + p[i].y * 140);
      PVector q = p[i].copy().rotate(0.2).mult(0.8);
      student[i] = new PVector(500 + q.x * 140, 440 + q.y * 140);
    }
    student[6] = PVector.add(student[4], new PVector(30, 30));   // the wrong arm
  } else {                                 // an open hand; the student's is smaller, turned, with the index finger bent
    float[][] bent = {{-0.38, 1.42}, {-0.65, 1.5}, {-0.92, 1.58}};   // landmarks 6, 7, 8: the index bends towards the thumb
    for (int i = 0; i < n; i++) {
      teacher[i] = new PVector(165 + HAND_BASE[i][0] * 120, 400 - HAND_BASE[i][1] * 120);
      PVector q = (i >= 6 && i <= 8) ? new PVector(bent[i - 6][0], bent[i - 6][1]) : new PVector(HAND_BASE[i][0], HAND_BASE[i][1]);
      q.rotate(-0.4).mult(0.85);
      student[i] = new PVector(480 + q.x * 120, 410 - q.y * 120);
    }
  }
}

void draw() {
  useOsc();
  background(255);
  stroke(230);
  line(330, 20, 330, 470);
  line(670, 20, 670, 470);
  fill(INK);
  textSize(20);
  text("Teacher" + (hands ? " (hand)" : ""), 20, 35);
  text(oscOn ? "Student: from OSC" : "Student: drag the joints", 345, 35);
  text("Normalised" + (rotateOn ? " and rotated" : "") + (mirror ? ", mirrored" : ""), 685, 35);
  textSize(15);
  fill(GREY);
  text("circle size = distance to the teacher's joint", 685, 58);
  fill(oscOn && millis() - lastOsc < 2000 ? TEAL : GREY);
  text(oscOn ? (millis() - lastOsc < 2000 ? "OSC: receiving on port " + OSC_PORT : "OSC: waiting on port " + OSC_PORT) : oscNote, 345, 58);

  int n = NAMES.length;
  drawPose(teacher, BLUE, null);
  drawPose(student, ORANGE, null);

  PVector[] A = normalise(teacher);
  PVector[] S = new PVector[n];
  for (int i = 0; i < n; i++) {
    int j = (mirror && !hands) ? BODY_SWAP[i] : i;
    S[i] = mirror ? new PVector(1000 - student[j].x, student[j].y) : student[i].copy();
  }
  PVector[] B = normalise(S);
  if (rotateOn) B = rotateOnto(A, B);
  float[] err = new float[n];
  float mean = 0;
  int worst = 0;
  for (int i = 0; i < n; i++) {
    err[i] = PVector.dist(A[i], B[i]);
    mean += err[i] / n;
    if (err[i] > err[worst]) worst = i;
  }
  // Where the normalised poses are drawn: bodies at a fixed place and size; hands, which can point any way,
  // centred and fitted into the right panel
  float cx = 0, cy = 0, vs = 75;
  if (hands) {
    float x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (int i = 0; i < n; i++) {
      x0 = min(x0, min(A[i].x, B[i].x));
      x1 = max(x1, max(A[i].x, B[i].x));
      y0 = min(y0, min(A[i].y, B[i].y));
      y1 = max(y1, max(A[i].y, B[i].y));
    }
    cx = (x0 + x1) / 2;
    cy = (y0 + y1) / 2;
    vs = min(110, min(280 / max(x1 - x0, 0.1), 360 / max(y1 - y0, 0.1)));
  }
  float vx = 835, vy = hands ? 275 : 300;
  PVector[] aS = new PVector[n], bS = new PVector[n];
  for (int i = 0; i < n; i++) {
    aS[i] = new PVector(vx + (A[i].x - cx) * vs, vy + (A[i].y - cy) * vs);
    bS[i] = new PVector(vx + (B[i].x - cx) * vs, vy + (B[i].y - cy) * vs);
  }
  drawPose(aS, color(BLUE, 90), null);
  drawPose(bS, GREY, err);

  // Distance per joint, as bars: the same distances as the circles
  String unit = hands ? "palm lengths" : "torso lengths";
  fill(INK);
  textSize(19);
  text("Distance per joint (" + unit + ")", 20, 515);
  fill(GREY);
  textSize(14);
  text("dashed: mean " + nf(mean, 1, 2), 345, 515);
  float bx = 130, bw = 330, top = max(1, max(err)), y0 = 530, row = min(22, 340.0 / n);
  float labelSize = row >= 20 ? 15 : 13;
  for (int i = 0; i < n; i++) {
    float y = y0 + i * row, w = err[i] / top * bw;
    fill(INK);
    textSize(labelSize);
    textAlign(RIGHT);
    text(NAMES[i], bx - 8, y + row - 5);
    textAlign(LEFT);
    noStroke();
    fill(i == worst ? ORANGE : BLUE);
    rect(bx, y + 2, max(w, 1), row - 5);
    fill(GREY);
    textSize(labelSize - 1);
    text(nf(err[i], 1, 2), bx + bw + 14, y + row - 5);   // the values in a column, clear of the mean line
  }
  float mx = bx + mean / top * bw;               // the mean, as a dashed line
  stroke(INK);
  strokeWeight(1.5);
  for (float y = y0; y < y0 + n * row; y += 8) line(mx, y, mx, y + 4);
  strokeWeight(1);
  noStroke();

  // Scores
  fill(INK);
  textSize(19);
  text("Scores (student vs teacher)", 560, 515);
  float cs = cosine(A, B);
  gauge(560, 545, "Cosine similarity (1 = same shape)", cs, 0.5, 1, TEAL,
        nf(cs, 1, 3) + "  (φ " + round(degrees(acos(constrain(cs, -1, 1)))) + "°)");
  gauge(560, 610, "Mean distance (0 = same pose)", mean, 0, 1, mean > 0.3 ? RED : ORANGE, nf(mean, 1, 2));
  textSize(17);
  fill(err[worst] > 0.3 ? RED : INK);
  text("Worst: " + NAMES[worst] + ", " + nf(err[worst], 1, 2), 560, 678);

  // Joint angles
  fill(INK);
  textSize(19);
  text(hands ? "Finger bend θ (teacher / student)" : "Joint angles θ (teacher / student)", 560, 718);
  textSize(17);
  for (int k = 0; k < ANGLES.length; k++) {
    float ta = angle(A, ANGLES[k]), sa = angle(B, ANGLES[k]);
    float x = 560, y = 746 + k * 24;
    fill(abs(ta - sa) > 20 ? RED : INK);
    text(ANGLE_NAMES[k] + ": " + round(ta) + "° / " + round(sa) + "°" + (abs(ta - sa) > 20 ? "  off by " + round(abs(ta - sa)) + "°" : ""), x, y);
  }
  fill(GREY);
  textSize(15);
  text("H: " + (hands ? "hand" : "body") + "   R: rotation " + (rotateOn ? "on" : "off") + "   M: mirror " + (mirror ? "on" : "off")
       + "   T: new teacher   O: OSC " + (oscOn ? "on" : "off"), 20, 885);
}

// A bar from lo to hi filled up to value: the label above, the value on the right
void gauge(float x, float y, String label, float value, float lo, float hi, color c, String note) {
  float w = 230, f = constrain((value - lo) / (hi - lo), 0, 1);
  fill(INK);
  textSize(16);
  text(label, x, y);
  noStroke();
  fill(235);
  rect(x, y + 8, w, 20);
  fill(c);
  rect(x, y + 8, max(f * w, 2), 20);
  fill(INK);
  textSize(16);
  text(note, x + w + 10, y + 24);
  fill(GREY);
  textSize(12);
  text(nf(lo, 1, 1), x, y + 42);
  text(nf(hi, 1, 0), x + w - 8, y + 42);
}

void drawPose(PVector[] p, color c, float[] err) {
  stroke(c);
  strokeWeight(hands ? 3 : 4);
  for (int[] b : BONES) line(p[b[0]].x, p[b[0]].y, p[b[1]].x, p[b[1]].y);
  strokeWeight(1);
  noStroke();
  for (int i = 0; i < p.length; i++) {
    if (err != null) {
      fill(ORANGE);
      circle(p[i].x, p[i].y, hands ? 5 + 25 * min(err[i], 1) : 8 + 40 * min(err[i], 1));
    } else {
      fill(c);
      circle(p[i].x, p[i].y, i == dragging ? 16 : (hands ? 9 : 11));
    }
  }
}

// Steps 1 and 2. Bodies: centre on the middle of the hips, divide by the torso length (hips to shoulders).
// Hands: centre on the wrist (0), divide by the palm length (0 to 9).
PVector[] normalise(PVector[] p) {
  PVector centre = hands ? p[0].copy() : PVector.add(p[7], p[8]).div(2);
  float size = hands ? PVector.dist(p[0], p[9]) : PVector.dist(centre, PVector.add(p[1], p[2]).div(2));
  size = max(size, 1e-6);
  PVector[] q = new PVector[p.length];
  for (int i = 0; i < p.length; i++) q[i] = PVector.sub(p[i], centre).div(size);
  return q;
}

// Step 3, Procrustes in 2D: the rotation of b that best overlays a (both already centred and scaled)
PVector[] rotateOnto(PVector[] a, PVector[] b) {
  float s = 0, c = 0;
  for (int i = 0; i < a.length; i++) {
    s += b[i].x * a[i].y - b[i].y * a[i].x;
    c += b[i].x * a[i].x + b[i].y * a[i].y;
  }
  float th = atan2(s, c);
  PVector[] r = new PVector[b.length];
  for (int i = 0; i < b.length; i++) r[i] = b[i].copy().rotate(th);
  return r;
}

float cosine(PVector[] a, PVector[] b) {
  float ab = 0, aa = 0, bb = 0;
  for (int i = 0; i < a.length; i++) {
    ab += a[i].dot(b[i]);
    aa += a[i].dot(a[i]);
    bb += b[i].dot(b[i]);
  }
  return ab / sqrt(aa * bb);
}

// The angle at joint j[1] between j[0] and j[2], in degrees
float angle(PVector[] p, int[] j) {
  PVector u = PVector.sub(p[j[0]], p[j[1]]), v = PVector.sub(p[j[2]], p[j[1]]);
  return degrees(PVector.angleBetween(u, v));
}

// ---- Mouse and keys ----

void mousePressed() {
  dragging = -1;
  for (int i = 0; i < student.length; i++) if (dist(mouseX, mouseY, student[i].x, student[i].y) < 12) dragging = i;
}

void mouseDragged() {
  if (dragging >= 0) student[dragging].set(constrain(mouseX, 340, 660), constrain(mouseY, 70, 470));
}

void mouseReleased() {
  dragging = -1;
}

void keyPressed() {
  if (key == 'r' || key == 'R') rotateOn = !rotateOn;
  if (key == 'm' || key == 'M') mirror = !mirror;
  if (key == 'h' || key == 'H') setMode(!hands);
  if (key == 't' || key == 'T') {
    if (oscOn && millis() - lastOsc < 2000)              // the pose being received becomes the teacher
      for (int i = 0; i < student.length; i++) teacher[i] = new PVector(student[i].x - 330, student[i].y);
    else
      newTeacher();
  }
  if (key == 'o' || key == 'O') {
    if (oscOn) stopOsc();
    else startOsc();
  }
}

// ---- OSC input, with Java's own UDP ----

void startOsc() {
  try {
    socket = new DatagramSocket(OSC_PORT);
    oscOn = true;
    Thread t = new Thread(new Runnable() {
      public void run() {
        listen();
      }
    });
    t.setDaemon(true);
    t.start();
  } catch (Exception e) {
    oscNote = "OSC: cannot listen on port " + OSC_PORT + " (" + e.getMessage() + ")";
    oscOn = false;
  }
}

void stopOsc() {
  oscOn = false;
  if (socket != null) socket.close();
  socket = null;
}

void listen() {
  byte[] buf = new byte[8192];
  while (oscOn && socket != null) {
    try {
      DatagramPacket packet = new DatagramPacket(buf, buf.length);
      socket.receive(packet);
      receive(buf, packet.getLength());
    } catch (Exception e) {
      // the socket was closed: stop listening
    }
  }
}

// Read one OSC message: the address, the type tags, then the numbers (floats or ints)
void receive(byte[] b, int length) {
  int[] pos = {0};
  String address = oscString(b, pos);
  String tags = oscString(b, pos);
  float[] v = new float[max(tags.length() - 1, 0)];
  for (int k = 1; k < tags.length() && pos[0] + 4 <= length; k++) {
    char t = tags.charAt(k);
    if (t == 'f') v[k - 1] = ByteBuffer.wrap(b, pos[0], 4).getFloat();
    else if (t == 'i') v[k - 1] = ByteBuffer.wrap(b, pos[0], 4).getInt();
    pos[0] += 4;
  }
  synchronized (this) {
    if (address.equals("/pose") || address.equals("/hand")) {
      inStudent = v;
      inStudentHand = address.equals("/hand");
    } else if (address.equals("/teacher/pose") || address.equals("/teacher/hand")) {
      inTeacher = v;
      inTeacherHand = address.equals("/teacher/hand");
    }
  }
}

String oscString(byte[] b, int[] pos) {
  int start = pos[0];
  while (pos[0] < b.length && b[pos[0]] != 0) pos[0]++;
  String s = new String(b, start, pos[0] - start);
  pos[0] = (pos[0] + 4) & ~3;               // strings are padded to a multiple of 4 bytes
  return s;
}

// Use the keypoints received since the last frame; switch to bodies or hands if needed
void useOsc() {
  float[] st, te;
  boolean sh, th;
  synchronized (this) {
    st = inStudent;
    te = inTeacher;
    sh = inStudentHand;
    th = inTeacherHand;
    inStudent = null;
    inTeacher = null;
  }
  if (te != null && place(te, th, teacher, 10)) lastOsc = millis();
  if (st != null && place(st, sh, student, 340)) lastOsc = millis();
}

// Keypoints in pixels (after the image width and height) into a panel (px to px + 310, y 70 to 470), keeping
// their proportions. Bodies: the whole picture is shown, grown to hold the joints MediaPipe guessed outside it
// (legs out of the picture). Hands: zoomed on the hand, which is small in a camera picture.
boolean place(float[] v, boolean isHand, PVector[] target, float px) {
  int n = isHand ? 21 : 13;
  if (v.length < 2 + 2 * n || v[0] <= 0 || v[1] <= 0) return false;
  if (isHand != hands) {
    setMode(isHand);
    target = (px < 330) ? teacher : student;
  }
  float x0 = isHand ? 1e9 : 0, y0 = isHand ? 1e9 : 0, x1 = isHand ? -1e9 : v[0], y1 = isHand ? -1e9 : v[1];
  for (int i = 0; i < n; i++) {
    x0 = min(x0, v[2 + 2 * i]);
    x1 = max(x1, v[2 + 2 * i]);
    y0 = min(y0, v[3 + 2 * i]);
    y1 = max(y1, v[3 + 2 * i]);
  }
  float s = min(290 / max(x1 - x0, 1), 380 / max(y1 - y0, 1));
  if (isHand) s = min(s, 250 / max(x1 - x0, y1 - y0));    // a little margin around the hand, whatever its shape
  float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  for (int i = 0; i < n; i++) target[i] = new PVector(px + 155 + (v[2 + 2 * i] - cx) * s, 270 + (v[3 + 2 * i] - cy) * s);
  return true;
}
