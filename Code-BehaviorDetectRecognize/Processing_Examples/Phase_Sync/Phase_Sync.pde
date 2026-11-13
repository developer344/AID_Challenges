// Moving with a partner: move the mouse up and down together with the orange ball.
// Each movement gets a phase from its position and speed (where it is in its cycle).
// Middle: the relative phase (you minus the ball) over the last 10 s, as a histogram on a circle,
// with the phase-locking value (PLV) as an arrow: long arrow = a stable relation.
// Near 0°: in phase. Try moving opposite to the ball: the cluster moves to 180° (anti-phase).
// Right: the lag from cross-correlation of the two positions. Positive: you follow the ball.
// Bottom: both movements over the last 10 s.
// Keys: Up/Down change the ball's frequency, Space clears.
// The same measures work on two people: replace the ball with a second sensor or a second mouse.

int HISTORY = 600;                         // 10 s at 60 frames per second
float[] ball = new float[HISTORY], you = new float[HISTORY], rel = new float[HISTORY];
boolean[] valid = new boolean[HISTORY];
float freq = 0.6, t = 0;

color BLUE = #2E75B6, ORANGE = #E36C0A, TEAL = #215968, INK = #262626, GREY = #AAAAAA;

void setup() {
  size(1000, 640);
  frameRate(60);
  textFont(createFont("Arial", 14));
}

void draw() {
  t += 1 / 60.0;
  float b = sin(TWO_PI * freq * t);
  float y = constrain((230 - mouseY) / 170.0, -1.3, 1.3);
  push(ball, b);
  push(you, y);
  float omega = TWO_PI * freq;
  float pb = phase(ball, omega), py = phase(you, omega);
  boolean moving = amplitude(you) > 0.15;
  push(rel, wrap(py - pb));
  pushB(valid, moving);

  background(255);
  // The ball and the mouse
  stroke(230);
  line(60, 60, 60, 400);
  line(160, 60, 160, 400);
  noStroke();
  fill(ORANGE);
  circle(60, 230 - 170 * b, 34);
  fill(BLUE);
  circle(160, 230 - 170 * y, 34);
  fill(INK);
  textAlign(CENTER);
  text("Ball", 60, 430);
  text("You", 160, 430);
  text(nf(freq, 1, 1) + " Hz", 110, 455);
  textAlign(LEFT);

  // Relative phase histogram and PLV
  int cx = 440, cy = 230, R = 150, BINS = 36;
  float[] hist = new float[BINS];
  float sx = 0, sy = 0, n = 0;
  for (int i = 0; i < HISTORY; i++) if (valid[i]) {
    int k = floor((rel[i] + PI) / TWO_PI * BINS) % BINS;
    hist[k]++;
    sx += cos(rel[i]);
    sy += sin(rel[i]);
    n++;
  }
  float hmax = max(max(hist), 1);
  noFill();
  stroke(220);
  circle(cx, cy, 2 * R);
  line(cx - R, cy, cx + R, cy);
  line(cx, cy - R, cx, cy + R);
  noStroke();
  fill(BLUE);
  for (int k = 0; k < BINS; k++) {
    float a0 = -PI + k * TWO_PI / BINS, a1 = a0 + TWO_PI / BINS, r = hist[k] / hmax * (R - 10);
    triangle(cx, cy, cx + r * cos(a0), cy - r * sin(a0), cx + r * cos(a1), cy - r * sin(a1));
  }
  float plv = n > 0 ? sqrt(sx * sx + sy * sy) / n : 0, mean = atan2(sy, sx);
  stroke(ORANGE);
  strokeWeight(4);
  line(cx, cy, cx + plv * R * cos(mean), cy - plv * R * sin(mean));
  strokeWeight(1);
  fill(GREY);
  text("0° in phase", cx + R + 6, cy + 5);
  text("180°", cx - R - 38, cy + 5);
  fill(INK);
  textSize(18);
  text("PLV = " + nf(plv, 1, 2) + (n > 0 ? "   mean " + round(degrees(mean)) + "°" : ""), cx - 100, 420);
  textSize(14);
  text(n < 60 ? "Move the mouse up and down" : plv > 0.7 ? (abs(mean) < PI / 3 ? "In phase" : abs(mean) > 2 * PI / 3 ? "Anti-phase" : "Locked, with a phase shift")
                                         : "No stable relation", cx - 100, 445);

  // Cross-correlation lag over the last 5 s
  int x0 = 690, y0 = 80, w = 280, h = 200, L = 60;
  float[] xc = new float[2 * L + 1];
  float best = -2;
  int bestLag = 0;
  for (int lag = -L; lag <= L; lag++) {
    xc[lag + L] = correlation(ball, you, lag, 300);
    if (xc[lag + L] > best) {
      best = xc[lag + L];
      bestLag = lag;
    }
  }
  stroke(220);
  line(x0, y0 + h / 2, x0 + w, y0 + h / 2);
  line(x0 + w / 2, y0, x0 + w / 2, y0 + h);
  noFill();
  stroke(TEAL);
  strokeWeight(2);
  beginShape();
  for (int i = 0; i < xc.length; i++) vertex(x0 + i * w / (float) (xc.length - 1), y0 + h / 2 - xc[i] * h / 2);
  endShape();
  strokeWeight(1);
  fill(INK);
  text("Cross-correlation, lag −1 s to +1 s", x0, y0 - 15);
  textSize(18);
  int ms = round(bestLag * 1000 / 60.0);
  text("Lag: " + (ms > 0 ? "+" : "") + ms + " ms", x0, y0 + h + 40);
  textSize(14);
  text(abs(ms) < 50 ? "Together" : ms > 0 ? "You follow the ball" : "You lead (you anticipate)", x0, y0 + h + 65);

  // Both movements over time
  int py0 = 548, ph = 52;
  stroke(230);
  line(20, py0, 980, py0);
  plot(ball, ORANGE, py0, ph);
  plot(you, BLUE, py0, ph);
  fill(GREY);
  textSize(13);
  text("Up/Down: ball frequency   Space: clear", 20, 630);
}

// Phase from position and speed, after removing the mean and scaling by the amplitude of the last 2 s
float phase(float[] s, float omega) {
  int n = 120;
  float m = 0;
  for (int i = HISTORY - n; i < HISTORY; i++) m += s[i] / n;
  float a = max(amplitude(s), 1e-3);
  float x = (s[HISTORY - 1] - m) / a;
  float v = (s[HISTORY - 1] - s[HISTORY - 5]) / (4 / 60.0) / a;
  return atan2(-v / omega, x);
}

float amplitude(float[] s) {
  int n = 120;
  float m = 0, v = 0;
  for (int i = HISTORY - n; i < HISTORY; i++) m += s[i] / n;
  for (int i = HISTORY - n; i < HISTORY; i++) v += sq(s[i] - m) / n;
  return sqrt(2 * v);
}

// Correlation of a and b shifted by lag samples (positive: b comes later), over the last n samples
float correlation(float[] a, float[] b, int lag, int n) {
  float ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
  int start = HISTORY - n;
  for (int i = start; i < HISTORY; i++) {
    ma += a[i - max(lag, 0)] / n;
    mb += b[i + min(lag, 0)] / n;
  }
  for (int i = start; i < HISTORY; i++) {
    float da = a[i - max(lag, 0)] - ma, db = b[i + min(lag, 0)] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return sab / max(sqrt(saa * sbb), 1e-9);
}

float wrap(float a) {
  return atan2(sin(a), cos(a));
}

void push(float[] h, float v) {
  arrayCopy(h, 1, h, 0, HISTORY - 1);
  h[HISTORY - 1] = v;
}

void pushB(boolean[] h, boolean v) {
  arrayCopy(h, 1, h, 0, HISTORY - 1);
  h[HISTORY - 1] = v;
}

void plot(float[] h, color c, float y0, float hh) {
  noFill();
  stroke(c);
  strokeWeight(2);
  beginShape();
  for (int i = 0; i < HISTORY; i++) vertex(map(i, 0, HISTORY - 1, 20, 980), y0 - h[i] * hh / 1.3);
  endShape();
  strokeWeight(1);
}

void keyPressed() {
  if (keyCode == UP) freq = min(2.0, freq + 0.1);
  if (keyCode == DOWN) freq = max(0.2, freq - 0.1);
  if (key == ' ') java.util.Arrays.fill(valid, false);
}
