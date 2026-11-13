// DTW gesture recogniser: record a few gestures with the mouse, then draw new ones to recognise them.
// Press a number key (1-5), then draw: the drawing is saved as a template for that gesture.
// Draw without pressing a number first to recognise: the nearest template wins, if it is close enough.
// Gestures are centred and scaled before comparing, so where and how big you draw does not matter;
// DTW takes care of drawing faster or slower. The direction does matter: a circle drawn clockwise
// and one drawn anticlockwise are different gestures.
// Middle: each template and its DTW distance to the last drawing. LOWER IS BETTER: 0 means identical,
// and a distance under the threshold (the orange line) counts as a match.
// Right: how DTW matched the last drawing (blue) with the best template (orange): the grey lines join the
// points it paired, and below, the cost table (dark = similar points) with the cheapest path through it.
// Keys: 1-5 record the next drawing as that gesture, C clears the templates, + and - change the threshold.
// Space works like the mouse button: hold it to draw (with the mouse, or with OSC), release it to finish.
// With a camera: run compare_poses.py (Python_Examples/Compare_Poses_OSC) with its Send OSC button on. This sketch
// listens on port 12000 (no library needed): with a hand (/hand) the index fingertip is the pen, with a body (/pose)
// the right wrist. Hold Space while you draw in the air. The pen is smoothed with the 1€ filter (as in Filter_Lag).
// M mirrors the camera (on: move your hand to the right and the pen goes right), O turns OSC off and on.

import java.net.*;
import java.nio.ByteBuffer;

ArrayList<PVector> current = new ArrayList<PVector>();
ArrayList<PVector[]> templates = new ArrayList<PVector[]>();
ArrayList<Integer> labels = new ArrayList<Integer>();
float[] lastDist = new float[0];
String result = "Press 1, then draw a gesture";
String detail = "";
float threshold = 0.25;
int recordAs = 0;                        // the gesture the next drawing is saved as, 0 to recognise
int MAX_POINTS = 64;                     // gestures are downsampled to at most this many points
PVector[] lastGesture = null;            // the last drawing, normalised
float[][] lastCost = null;               // distances between its points and the best template's
ArrayList<int[]> lastPath = new ArrayList<int[]>();
int lastBest = -1;

color BLUE = #2E75B6, ORANGE = #E36C0A, INK = #262626, GREY = #AAAAAA, GREEN = #2E8B57, RED = #C00000, TEAL = #215968;

boolean drawing = false, spaceDown = false;   // a stroke in progress (mouse button or Space held)

// OSC input
int OSC_PORT = 12000;
boolean oscOn = false, mirror = true, newPen = false, penHand = false;
DatagramSocket socket;
String oscNote = "";
float[] inPen = null;                    // the newest keypoints received, in pixels after the picture's size
boolean inHand = false;
PVector pen = null;                      // the pen from OSC, in the drawing area
float camW = 640, camH = 480;            // the size of the camera's picture
int lastOsc = -10000;
OneEuroFilter fx = new OneEuroFilter(1.0, 0.01), fy = new OneEuroFilter(1.0, 0.01);

void setup() {
  size(1250, 580);
  textFont(createFont("Arial", 14));
  startOsc();
}

boolean oscLive() {                      // keypoints arrived in the last second
  return oscOn && pen != null && millis() - lastOsc < 1000;
}

void draw() {
  useOsc();
  if (spaceDown) {                         // Space held: the pen draws (OSC), or the mouse if there is no OSC
    if (oscLive()) {
      if (newPen) addPoint(pen.x, pen.y);
    } else if (mouseX != pmouseX || mouseY != pmouseY) addPoint(mouseX, mouseY);
  }
  background(255);
  // Drawing area
  noFill();
  stroke(220);
  rect(20, 20, 560, 540);
  if (oscLive()) {                         // the camera's picture in the drawing area, and the pen
    float s = min(560 / camW, 540 / camH);
    stroke(236);
    rect(20 + (560 - camW * s) / 2, 20 + (540 - camH * s) / 2, camW * s, camH * s);
    stroke(spaceDown ? (recordAs > 0 ? ORANGE : BLUE) : GREY);
    strokeWeight(2);
    if (spaceDown) fill(recordAs > 0 ? ORANGE : BLUE, 90);
    else noFill();
    circle(pen.x, pen.y, 18);
    strokeWeight(1);
    noFill();
  }
  stroke(recordAs > 0 ? ORANGE : BLUE);
  strokeWeight(3);
  beginShape();
  for (PVector p : current) vertex(p.x, p.y);
  endShape();
  strokeWeight(1);
  textSize(18);
  boolean match = lastDist.length > 0 && lastDist[argmin(lastDist)] < threshold;
  fill(recordAs > 0 ? ORANGE : lastDist.length == 0 ? INK : match ? GREEN : RED);
  text(recordAs > 0 ? "Recording gesture " + recordAs + ": draw it" : result, 35, 50);
  fill(INK);
  textSize(13);
  text(detail, 35, 72);
  fill(oscLive() ? TEAL : GREY);
  text(oscLive() ? "OSC: the " + (penHand ? "index fingertip" : "right wrist") + " is the pen" + (mirror ? " (mirrored)" : "") + ". Hold Space to draw."
       : oscOn ? "OSC: listening on port " + OSC_PORT + " for compare_poses.py (its Send OSC button)" : oscNote, 35, 509);
  fill(GREY);
  text("1-5: record next drawing   C: clear   +/-: threshold " + nf(threshold, 1, 2), 35, 527);
  text("Space: hold to draw (mouse or OSC)   O: OSC on/off   M: mirror the camera", 35, 545);

  drawTemplates();
  drawMatch();
}

// Middle panel: templates and their distances (lower is better)
void drawTemplates() {
  int x0 = 600, bx = 715, bw = 150;
  float tx = bx + bw / 2;                      // the threshold sits in the middle of the scale
  fill(INK);
  textSize(15);
  text("Templates (" + templates.size() + ")", x0, 38);
  textSize(11);
  fill(GREY);
  text("DTW distance to your drawing: shorter bar = more similar", x0, 54);
  if (templates.size() > 0) {                  // the scale: 0, the threshold, twice the threshold
    fill(GREEN);
    textAlign(CENTER);
    text("match", (bx + tx) / 2, 72);
    fill(GREY);
    text("no match", (tx + bx + bw) / 2, 72);
    text("0", bx, 84);
    text(nf(threshold, 1, 2), tx, 84);
    text(nf(2 * threshold, 1, 2) + "+", bx + bw, 84);
    textAlign(LEFT);
  }
  int best = argmin(lastDist);
  textSize(14);
  for (int k = 0; k < templates.size() && k < 10; k++) {
    float y = 92 + k * 44;
    boolean has = k < lastDist.length, ok = has && lastDist[k] < threshold;
    if (has && k == best) {                    // the nearest template gets a frame
      noFill();
      stroke(ok ? GREEN : GREY);
      strokeWeight(2);
      rect(x0 - 6, y - 4, 336, 42, 6);
      strokeWeight(1);
    }
    drawThumb(templates.get(k), x0, y, 34, has && k == best && ok ? ORANGE : BLUE);
    fill(INK);
    textSize(13);
    text("Gesture " + labels.get(k), x0 + 42, y + 22);
    textSize(14);
    // The track: light green up to the threshold, light grey beyond
    noStroke();
    fill(#E2F2E8);
    rect(bx, y + 9, bw / 2, 16);
    fill(#EEEEEE);
    rect(tx, y + 9, bw / 2, 16);
    if (has) {
      float w = constrain(lastDist[k] / (2 * threshold), 0, 1) * bw;
      fill(ok ? GREEN : #8C8C8C);
      rect(bx, y + 12, max(w, 3), 10);
      fill(INK);
      text(nf(lastDist[k], 1, 3), bx + bw + 8, y + 22);
    }
  }
}

// Right panel: the pairs of points DTW matched, and the cost table with its path
void drawMatch() {
  int x0 = 960, size = 260;
  fill(INK);
  textSize(15);
  text("How DTW matched them", x0, 38);
  if (lastGesture == null || lastBest < 0) {
    fill(GREY);
    textSize(13);
    text("Draw a gesture to see\nwhich points DTW pairs.", x0, 70);
    return;
  }
  PVector[] a = lastGesture, b = templates.get(lastBest);
  // Side by side: your drawing on the left, the template on the right
  float s = size * 0.4, ax = x0 + size * 0.22, bx = x0 + size * 0.8, cy = 175;
  // Pairs of matched points: every second pair, so the lines stay readable
  stroke(190);
  for (int k = 0; k < lastPath.size(); k += 2) {
    int[] ij = lastPath.get(k);
    line(ax + a[ij[0]].x * s, cy + a[ij[0]].y * s, bx + b[ij[1]].x * s, cy + b[ij[1]].y * s);
  }
  drawCurve(a, ax, cy, s, BLUE);
  drawCurve(b, bx, cy, s, ORANGE);
  textSize(12);
  textAlign(CENTER);
  fill(BLUE);
  text("your drawing", ax, 275);
  fill(ORANGE);
  text("gesture " + labels.get(lastBest), bx, 275);
  fill(GREY);
  textSize(11);
  text("grey lines: points DTW paired (dot = start)", x0 + size / 2, 292);
  textAlign(LEFT);

  // Cost table: rows = your drawing, columns = the template; dark = similar
  int ty = 315, n = lastCost.length, m = lastCost[0].length, th = 220;
  float maxc = 0;
  for (float[] row : lastCost) for (float c : row) maxc = max(maxc, c);
  noStroke();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
      fill(lerpColor(color(#1F497D), color(255), constrain(lastCost[i][j] / maxc, 0, 1)));
      rect(x0 + j * size / (float) m, ty + th - (i + 1) * th / (float) n, size / (float) m + 0.5, th / (float) n + 0.5);
    }
  stroke(ORANGE);
  strokeWeight(2.5);
  noFill();
  beginShape();
  for (int[] ij : lastPath) vertex(x0 + (ij[1] + 0.5) * size / m, ty + th - (ij[0] + 0.5) * th / n);
  endShape();
  strokeWeight(1);
  fill(GREY);
  textSize(11);
  text("template →", x0 + size - 60, ty + th + 14);
  text("your drawing ↑", x0, ty + th + 14);
  text("off the diagonal = one was faster", x0, ty + th + 28);
}

void drawCurve(PVector[] g, float cx, float cy, float s, color c) {
  noFill();
  stroke(c);
  strokeWeight(2.5);
  beginShape();
  for (PVector p : g) vertex(cx + p.x * s, cy + p.y * s);
  endShape();
  strokeWeight(1);
  noStroke();
  fill(c);
  circle(cx + g[0].x * s, cy + g[0].y * s, 8);   // where it starts
}

void mousePressed() {
  startStroke();
}

void mouseDragged() {
  addPoint(mouseX, mouseY);
}

void mouseReleased() {
  finishStroke();
}

void startStroke() {
  current.clear();
  drawing = true;
}

void addPoint(float x, float y) {
  if (drawing && x > 20 && x < 580 && y > 20 && y < 560) current.add(new PVector(x, y));
}

// The end of a stroke (mouse button or Space released): save it as a template, or recognise it
void finishStroke() {
  drawing = false;
  if (current.size() < 5) return;
  PVector[] g = normalise(current);
  if (recordAs > 0) {
    templates.add(g);
    labels.add(recordAs);
    result = "Saved as gesture " + recordAs;
    detail = templates.size() == 1 ? "Record more gestures, or draw to test." : "Draw a gesture to recognise it.";
    recordAs = 0;
    lastDist = new float[0];
    lastGesture = null;
    return;
  }
  if (templates.size() == 0) {
    result = "No templates yet: press 1, then draw";
    return;
  }
  lastDist = new float[templates.size()];
  for (int k = 0; k < templates.size(); k++) lastDist[k] = dtw(g, templates.get(k));
  lastBest = argmin(lastDist);
  lastGesture = g;
  dtwPath(g, templates.get(lastBest));
  float d = lastDist[lastBest];
  result = d < threshold ? "Gesture " + labels.get(lastBest) : "None of these";
  detail = "Nearest: gesture " + labels.get(lastBest) + ", distance " + nf(d, 1, 3)
         + (d < threshold ? " < " : " ≥ ") + "threshold " + nf(threshold, 1, 2) + " (lower = more similar)";
}

void keyPressed() {
  if (key == ' ' && !spaceDown) {          // held down, the key repeats: start only once
    spaceDown = true;
    startStroke();
  }
  if (key == 'o' || key == 'O') {
    if (oscOn) stopOsc();
    else startOsc();
  }
  if (key == 'm' || key == 'M') mirror = !mirror;
  if (key >= '1' && key <= '5') recordAs = key - '0';
  if (key == 'c' || key == 'C') {
    templates.clear();
    labels.clear();
    lastDist = new float[0];
    lastGesture = null;
    result = "Cleared. Press 1, then draw";
    detail = "";
  }
  if (key == '+' || key == '=') threshold += 0.05;
  if (key == '-') threshold = max(0.05, threshold - 0.05);
}

void keyReleased() {
  if (key == ' ' && spaceDown) {
    spaceDown = false;
    finishStroke();
  }
}

// Centre on the mean, scale so the larger side is 1, and keep at most MAX_POINTS points
PVector[] normalise(ArrayList<PVector> pts) {
  int n = min(pts.size(), MAX_POINTS);
  PVector[] g = new PVector[n];
  for (int i = 0; i < n; i++) g[i] = pts.get(round(i * (pts.size() - 1) / max(1.0, n - 1))).copy();
  PVector c = new PVector();
  float x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (PVector p : g) {
    c.add(p);
    x0 = min(x0, p.x); x1 = max(x1, p.x);
    y0 = min(y0, p.y); y1 = max(y1, p.y);
  }
  c.div(n);
  float s = max(max(x1 - x0, y1 - y0), 1);
  for (PVector p : g) p.sub(c).div(s);
  return g;
}

// DTW distance between two gestures, divided by their total length
float dtw(PVector[] a, PVector[] b) {
  int n = a.length, m = b.length;
  float[][] D = new float[n + 1][m + 1];
  for (float[] row : D) java.util.Arrays.fill(row, Float.MAX_VALUE);
  D[0][0] = 0;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++) {
      float cost = PVector.dist(a[i-1], b[j-1]);
      D[i][j] = cost + min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
    }
  return D[n][m] / (n + m);
}

// The same, keeping the cost table and the cheapest path (for the right panel)
void dtwPath(PVector[] a, PVector[] b) {
  int n = a.length, m = b.length;
  lastCost = new float[n][m];
  float[][] D = new float[n + 1][m + 1];
  for (float[] row : D) java.util.Arrays.fill(row, Float.MAX_VALUE);
  D[0][0] = 0;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++) {
      lastCost[i-1][j-1] = PVector.dist(a[i-1], b[j-1]);
      D[i][j] = lastCost[i-1][j-1] + min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
    }
  // Walk back from the end, always to the cheapest of the three cells before
  lastPath.clear();
  int i = n, j = m;
  while (i > 0 && j > 0) {
    lastPath.add(0, new int[] {i - 1, j - 1});
    float diag = D[i-1][j-1], up = D[i-1][j], left = D[i][j-1];
    if (diag <= up && diag <= left) { i--; j--; }
    else if (up < left) i--;
    else j--;
  }
}

int argmin(float[] v) {
  int best = -1;
  for (int k = 0; k < v.length; k++) if (best < 0 || v[k] < v[best]) best = k;
  return best;
}

void drawThumb(PVector[] g, float x, float y, float size, color c) {
  noFill();
  stroke(c);
  strokeWeight(2);
  beginShape();
  for (PVector p : g) vertex(x + size / 2 + p.x * size, y + size / 2 + p.y * size);
  endShape();
  strokeWeight(1);
}

// ---- OSC input from compare_poses.py, with Java's own UDP (as in Pose_Compare) ----

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
    oscNote = "OSC: port " + OSC_PORT + " is taken (is Pose_Compare listening?): O to try again";
    oscOn = false;
  }
}

void stopOsc() {
  oscOn = false;
  oscNote = "OSC off: O to listen on port " + OSC_PORT;
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
  if (!address.equals("/pose") && !address.equals("/hand")) return;    // the student's keypoints only
  String tags = oscString(b, pos);
  float[] v = new float[max(tags.length() - 1, 0)];
  for (int k = 1; k < tags.length() && pos[0] + 4 <= length; k++) {
    char t = tags.charAt(k);
    if (t == 'f') v[k - 1] = ByteBuffer.wrap(b, pos[0], 4).getFloat();
    else if (t == 'i') v[k - 1] = ByteBuffer.wrap(b, pos[0], 4).getInt();
    pos[0] += 4;
  }
  synchronized (this) {
    inPen = v;
    inHand = address.equals("/hand");
  }
}

String oscString(byte[] b, int[] pos) {
  int start = pos[0];
  while (pos[0] < b.length && b[pos[0]] != 0) pos[0]++;
  String s = new String(b, start, pos[0] - start);
  pos[0] = (pos[0] + 4) & ~3;               // strings are padded to a multiple of 4 bytes
  return s;
}

// The newest keypoints: the index fingertip (hand landmark 8) or the right wrist (body joint 6 of the 13),
// from the camera's picture into the drawing area (keeping its proportions), smoothed
void useOsc() {
  float[] v;
  boolean hand;
  synchronized (this) {
    v = inPen;
    hand = inHand;
    inPen = null;
  }
  newPen = false;
  if (v == null) return;
  int i = hand ? 8 : 6;
  if (v.length < 4 + 2 * i || v[0] <= 0 || v[1] <= 0) return;
  float xn = v[2 + 2 * i] / v[0], yn = v[3 + 2 * i] / v[1];
  if (mirror) xn = 1 - xn;
  float s = min(560 / v[0], 540 / v[1]);
  float x = 20 + (560 - v[0] * s) / 2 + xn * v[0] * s, y = 20 + (540 - v[1] * s) / 2 + yn * v[1] * s;
  int now = millis();
  if (now - lastOsc > 500 || hand != penHand) {   // after a pause, or another body part: start the filter afresh
    fx.reset();
    fy.reset();
  }
  float dt = max(now - lastOsc, 1) / 1000.0;
  pen = new PVector(fx.filter(x, dt), fy.filter(y, dt));
  penHand = hand;
  camW = v[0];
  camH = v[1];
  lastOsc = now;
  newPen = true;
}

// The 1€ filter (Casiez, Roussel & Vogel, 2012), as in Filter_Lag: smooths the pen when it is still, follows it when fast
class OneEuroFilter {
  float minCutoff, beta, dCutoff = 1;
  float x = Float.NaN, dx = 0;
  OneEuroFilter(float minCutoff, float beta) {
    this.minCutoff = minCutoff;
    this.beta = beta;
  }
  void reset() {
    x = Float.NaN;
    dx = 0;
  }
  float alpha(float cutoff, float dt) {
    float tau = 1 / (TWO_PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  float filter(float v, float dt) {
    if (Float.isNaN(x)) {
      x = v;
      return v;
    }
    float d = (v - x) / dt;
    dx += alpha(dCutoff, dt) * (d - dx);
    float cutoff = minCutoff + beta * abs(dx);
    x += alpha(cutoff, dt) * (v - x);
    return x;
  }
}
