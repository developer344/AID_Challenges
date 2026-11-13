// Filters and their delay: move the mouse up and down over the window.
// The sketch adds noise to the mouse height, as a sensor would, then filters it live.
// Three panels, each with the noisy signal in grey and the mouse itself as a thin dark line: a moving average
// (orange); the 1€ filter ("one euro filter", blue; Casiez, Roussel & Vogel, 2012), an adaptive low-pass filter
// that smooths strongly when you move slowly and lightly when you move fast; and a Kalman filter (green;
// Kalman, 1960), which tracks position and speed with a model of the movement (constant velocity) and
// weighs each new value by how noisy the sensor is (taken from the noise slider).
// The lag of each filter is measured by cross-correlation with the mouse.
// Sliders, top right: the noise added; "History shown", the seconds stretched across the window (fewer = easier
// to see fast movements); and "Bring the lines together", which slides the three panels onto the
// same axes: then you see directly which filtered line is late (the moving average) and which is not.
// Keys: A changes the moving-average window, B the 1€ filter's speed coefficient (beta),
// K how freely the Kalman filter lets the movement change (its motion noise).

int HISTORY = 600;
float[] mouse = new float[HISTORY], noisy = new float[HISTORY], avg = new float[HISTORY], euro = new float[HISTORY];
float[] kalman = new float[HISTORY];
int[] WINDOWS = {3, 8, 15, 30};
int windowIndex = 2;
float[] BETAS = {0, 0.002, 0.007, 0.02};
int betaIndex = 1;
OneEuroFilter filter = new OneEuroFilter(0.5, BETAS[betaIndex]);   // 0.5 Hz when still
float[] MOTIONS = {500, 3000, 15000};    // Kalman: how fast the movement may change (px/s²)
int motionIndex = 1;
Kalman kf = new Kalman();
float[] recent = new float[64];          // for the moving average
int count = 0;
Slider noise = new Slider(480, 32, 380, 0, 100, 40, "Noise", "px");
Slider history = new Slider(480, 77, 380, 1, 10, 4, "History shown", "s");
Slider together = new Slider(480, 122, 380, 0, 100, 0, "Bring the lines together", "%");

color BLUE = #2E75B6, ORANGE = #E36C0A, GREEN = #2E8B57, INK = #262626, GREY = #8C8C8C;

void setup() {
  size(900, 810);
  frameRate(60);
  textFont(createFont("Arial", 14));
}

void draw() {
  float y = mouseY + noise.value * randomGaussian();
  if (count == 0)                         // start the plots from here, not from zero
    for (float[] h : new float[][] {mouse, noisy, avg, euro, kalman}) java.util.Arrays.fill(h, mouseY);
  recent[count % recent.length] = y;
  count++;
  int w = min(WINDOWS[windowIndex], count);
  float sum = 0;
  for (int i = 0; i < w; i++) sum += recent[(count - 1 - i) % recent.length];
  push(mouse, mouseY);
  push(noisy, y);
  push(avg, sum / w);
  push(euro, filter.filter(y, 1 / 60.0));
  push(kalman, kf.filter(y, 1 / 60.0, MOTIONS[motionIndex], max(sq(noise.value), 1)));

  background(255);
  // One panel per filter: the noise in grey, the filtered signal on top, the mouse as a thin line.
  // "Bring the lines together" slides the panels onto the middle one, so the lines can be compared directly.
  float[][] filtered = {avg, euro, kalman};
  color[] colours = {ORANGE, BLUE, GREEN};
  String[] names = {"Moving average", "1€ filter", "Kalman filter"};
  float f = together.value / 100, ph = lerp(200, 630, f);   // together, one big plot
  float[] tops = new float[3];
  for (int k = 0; k < 3; k++) tops[k] = lerp(162 + k * 216, 162, f);
  for (int k = 0; k < 3; k++) {                  // first the frames, the noise and the mouse of every panel
    noFill();
    stroke(225, 255 * (1 - f));
    rect(10, tops[k] - 6, width - 20, ph + 12);
    plot(noisy, GREY, 1.4, tops[k], ph);
    plot(mouse, INK, 1, tops[k], ph);
  }
  for (int k = 0; k < 3; k++) {                  // then the filtered lines, on top of all of them
    plot(filtered[k], colours[k], 3.5, tops[k], ph);
    fill(colours[k]);
    textSize(13);
    text(names[k], 20 + k * 150 * f, tops[k] + 14);
  }

  textSize(14);
  fill(ORANGE);
  text("Moving average of " + WINDOWS[windowIndex] + " samples (A)   lag " + lagMs(avg) + " ms", 20, 28);
  fill(BLUE);
  text("1€ filter, beta " + BETAS[betaIndex] + " (B)   lag " + lagMs(euro) + " ms", 20, 50);
  fill(GREEN);
  text("Kalman filter, motion " + round(MOTIONS[motionIndex]) + " px/s² (K)   lag " + lagMs(kalman) + " ms", 20, 72);
  fill(INK);
  text("Move the mouse up and down: slowly, then fast", 20, 98);
  fill(150);
  textSize(12);
  text("Bring the lines together (slider, right): which one is late?", 20, 120);
  noise.display();
  history.display();
  together.display();
}

void mousePressed() {
  noise.press();
  history.press();
  together.press();
}

void mouseDragged() {
  noise.drag();
  history.drag();
  together.drag();
}

void mouseReleased() {
  noise.dragging = history.dragging = together.dragging = false;
}

void keyPressed() {
  if (key == 'a' || key == 'A') windowIndex = (windowIndex + 1) % WINDOWS.length;
  if (key == 'k' || key == 'K') motionIndex = (motionIndex + 1) % MOTIONS.length;
  if (key == 'b' || key == 'B') {
    betaIndex = (betaIndex + 1) % BETAS.length;
    filter.beta = BETAS[betaIndex];
  }
}

void push(float[] h, float v) {
  arrayCopy(h, 1, h, 0, HISTORY - 1);
  h[HISTORY - 1] = v;
}

// A history in window pixels, drawn into a panel from top to top + ph
void plot(float[] h, color c, float weight, float top, float ph) {
  noFill();
  stroke(c);
  strokeWeight(weight);
  beginShape();
  int first = HISTORY - round(history.value * 60);     // only the last seconds, stretched across the window
  for (int i = first; i < HISTORY; i++) vertex(map(i, first, HISTORY - 1, 10, width - 10), top + constrain(h[i], 0, height) / height * ph);
  endShape();
  strokeWeight(1);
}

// The delay (in frames, shown in ms) that best lines up a filtered signal with the mouse, over the last 5 s
int lagMs(float[] f) {
  int n = 300, bestLag = 0;
  float best = -1e9, ma = 0, mf = 0;
  for (int i = HISTORY - n - 40; i < HISTORY; i++) ma += mouse[i] / (n + 40);
  for (int i = HISTORY - n; i < HISTORY; i++) mf += f[i] / n;
  for (int lag = 0; lag <= 40; lag++) {
    float s = 0;
    for (int i = HISTORY - n; i < HISTORY; i++) s += (mouse[i - lag] - ma) * (f[i] - mf);
    if (s > best) {
      best = s;
      bestLag = lag;
    }
  }
  return round(bestLag * 1000 / 60.0);
}

// The 1€ filter (Casiez, Roussel & Vogel, 2012): a low-pass filter whose cutoff rises with the speed of the signal,
// so it smooths a still hand and follows a fast one. The name says how cheap it is to compute.
class OneEuroFilter {
  float minCutoff, beta, dCutoff = 1;
  float x = Float.NaN, dx = 0;
  OneEuroFilter(float minCutoff, float beta) {
    this.minCutoff = minCutoff;
    this.beta = beta;
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

// A horizontal slider with its name and value above it
class Slider {
  float x, y, w, lo, hi, value;
  String name, unit;
  boolean dragging = false;
  Slider(float x, float y, float w, float lo, float hi, float value, String name, String unit) {
    this.x = x; this.y = y; this.w = w; this.lo = lo; this.hi = hi; this.value = value;
    this.name = name; this.unit = unit;
  }
  void display() {
    fill(INK);
    textSize(14);
    text(name + ": " + round(value) + " " + unit, x, y - 14);
    stroke(170);
    strokeWeight(2);
    line(x, y, x + w, y);
    strokeWeight(1);
    noStroke();
    fill(dragging ? BLUE : 60);
    circle(map(value, lo, hi, x, x + w), y, 16);
  }
  void press() {
    dragging = abs(mouseY - y) < 12 && mouseX > x - 10 && mouseX < x + w + 10;
    drag();
  }
  void drag() {
    if (dragging) value = constrain(map(mouseX, x, x + w, lo, hi), lo, hi);
  }
}

// Kalman filter with a constant-velocity model: it tracks position and speed. Each step it predicts where the
// movement goes, then corrects the prediction by a fraction (the gain) of the error with the new value. The gain is
// large when the sensor is precise compared with how freely the movement may change, small when it is noisy.
class Kalman {
  float x = Float.NaN, v = 0;                     // position and speed
  float p00 = 1e4, p01 = 0, p10 = 0, p11 = 1e4;   // how uncertain they are (covariance)
  float filter(float z, float dt, float motion, float sensorVar) {
    if (Float.isNaN(x)) {
      x = z;
      return z;
    }
    // Predict: move by the speed; uncertainty grows with the motion noise
    x += v * dt;
    float q = motion * motion;
    float n00 = p00 + dt * (p01 + p10) + dt * dt * p11 + q * pow(dt, 4) / 4;
    float n01 = p01 + dt * p11 + q * pow(dt, 3) / 2;
    float n10 = p10 + dt * p11 + q * pow(dt, 3) / 2;
    float n11 = p11 + q * dt * dt;
    // Correct: move towards the measurement by the gain
    float k0 = n00 / (n00 + sensorVar), k1 = n10 / (n00 + sensorVar);
    float err = z - x;
    x += k0 * err;
    v += k1 * err;
    p00 = (1 - k0) * n00;
    p01 = (1 - k0) * n01;
    p10 = n10 - k1 * n00;
    p11 = n11 - k1 * n01;
    return x;
  }
}
