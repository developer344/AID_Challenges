// Comparing theoretical signals: point by point (Euclidean), with DTW, and with cross-correlation.
// Choose two signals (sine, square, triangle, sawtooth, pulse, chirp or line) and move their sliders.
// "Speed warp" plays the same signal faster at the start and slower at the end (or the opposite):
// the shape is the same, only the timing changes. This is what DTW is made for.
// Top plot: A and B on the same axes; the grey lines join the samples that point-by-point compares
// (same time). Middle plot: A above, B below; the grey lines join the samples that DTW pairs.
// Right: the cost table, |A(i) − B(j)| for every pair (dark = similar), and DTW's cheapest path.
// The grey margin is outside the DTW band: the path may not go there.
// Bottom: four measures. Euclidean and DTW are distances: LOWER IS BETTER (0 = identical; both are
// the mean |A − B| over the pairs compared, so they can be compared with each other). The DTW card also
// says how much timing DTW absorbed: how far, on average, its pairs are apart in time. Correlation is a
// similarity: HIGHER IS BETTER (1 = same shape), at the best lag (shift in samples). Spectrum compares
// which frequencies each signal contains (small plot), ignoring when they happen: 0 = same frequencies.
// "Compare slopes" runs DTW on the slopes of the signals instead of their values (derivative DTW,
// Keogh & Pazzani, 2001): it stops DTW pairing one point with a long stretch just to make heights fit,
// but slopes amplify noise: add noise and see the cost table turn into stripes.
// "Find A in B": the first part of A (its length set by the slider that replaces the band) is slid along B,
// and each window of B is compared with it by DTW. The middle plot shows the best window and, below, the
// DTW cost of every window: this is how a gesture is found inside a continuous stream. While it is on,
// the four cards compare the part of A with its best window of B; off, the whole signals.
// Keys: 1 same shape, other speed   2 same shape, shifted   3 other amplitude   4 other shape   5 noise
// Space: a random frequency and speed warp for each signal (the rest stays). Then move the sliders
// to bring them together and watch the three measures. DTW forgives a different warp completely,
// a different frequency only in part (one signal has more cycles than the other has to match).

int N = 200;
float[] sigA = new float[N], sigB = new float[N];   // as generated
float[] a = new float[N], b = new float[N];         // as compared (normalised if the toggle is on)
float[] noiseA = new float[N], noiseB = new float[N];

String[] TYPES = {"Sine", "Square", "Triangle", "Sawtooth", "Pulse", "Chirp", "Line"};
DropList typeA, typeB;
Slider p1A, p2A, shiftA, warpA, noiseAmtA;
Slider p1B, p2B, shiftB, warpB, noiseAmtB;
Slider band;
ToggleButton normalizeBtn, pairsBtn, slopesBtn, findBtn;
int measN = N;                       // length of what the cards compare: N, or the window
Slider winLen;
float[] runCost = new float[0];      // DTW cost of each window of B (Find A in B)
int bestStart = 0, winL = 0;
ArrayList<int[]> winPath = new ArrayList<int[]>();

// Results
float euclid, dtwMean, corrR, timing, specDist;
int K = 20;                                          // frequencies compared: 1 to K cycles
float[] specA = new float[K + 1], specB = new float[K + 1];
int corrLag, pathLen;
float[][] cost = new float[N][N];
ArrayList<int[]> path = new ArrayList<int[]>();
PImage costImage;
String lastKey = "";

color BLUE = #2E75B6, ORANGE = #E36C0A, INK = #262626, GREY = #999999, GREEN = #2E8B57, TEAL = #215968;

void setup() {
  size(1280, 790);
  surface.setTitle("Comparing signals: Euclidean, DTW and correlation");
  textFont(createFont("Arial", 13));
  randomSeed(3);
  for (int i = 0; i < N; i++) {                // fixed noise, so it does not flicker
    noiseA[i] = randomGaussian();
    noiseB[i] = randomGaussian();
  }
  typeA = new DropList(30, 40, 130, 24, "Signal A", TYPES, 0);
  p1A = new Slider(185, 30, 200, 10, 0.5, 6, 2, "");
  p2A = new Slider(185, 58, 200, 10, 0.1, 2, 1, "");
  shiftA = new Slider(185, 86, 200, 10, -0.5, 0.5, 0, "Shift");
  warpA = new Slider(185, 114, 200, 10, 0.5, 2, 1, "Speed warp");
  noiseAmtA = new Slider(185, 142, 200, 10, 0, 0.5, 0, "Noise");
  typeB = new DropList(430, 40, 130, 24, "Signal B", TYPES, 0);
  p1B = new Slider(585, 30, 200, 10, 0.5, 6, 2, "");
  p2B = new Slider(585, 58, 200, 10, 0.1, 2, 1, "");
  shiftB = new Slider(585, 86, 200, 10, -0.5, 0.5, 0, "Shift");
  warpB = new Slider(585, 114, 200, 10, 0.5, 2, 1.6, "Speed warp");
  noiseAmtB = new Slider(585, 142, 200, 10, 0, 0.5, 0, "Noise");
  normalizeBtn = new ToggleButton(840, 28, 95, 24, "Normalise");
  pairsBtn = new ToggleButton(941, 28, 95, 24, "Show pairs");
  slopesBtn = new ToggleButton(1042, 28, 95, 24, "Slopes");
  findBtn = new ToggleButton(1143, 28, 97, 24, "Find A in B");
  pairsBtn.active = true;
  band = new Slider(840, 86, 400, 10, 0.02, 1, 1, "DTW band (fraction of the length; 1 = no limit)");
  winLen = new Slider(840, 86, 400, 10, 0.1, 0.6, 0.3, "Find A in B: length of the part of A searched for (fraction)");
  costImage = createImage(N, N, RGB);
}

void draw() {
  background(250);
  updateLabels();
  String now = state();
  if (!now.equals(lastKey)) {                  // recompute only when something changed
    compute();
    lastKey = now;
  }

  drawOverlay(30, 175, 760, 210);
  drawPairs(30, 400, 760, 230);
  drawCost(830, 175, 410, 455);
  drawMetrics(30, 650);

  for (Slider s : sliders()) s.display();
  normalizeBtn.display();
  pairsBtn.display();
  slopesBtn.display();
  findBtn.display();
  fill(GREY);
  textSize(12);
  text("Keys: 1 same shape, other speed   2 shifted   3 other amplitude", 840, 130);
  text("         4 other shape   5 other speed and noise   Space: random frequency + warp", 840, 148);
  fill(INK);                                    // how far apart they are: bring both to 0 with the sliders
  text("A − B:  frequency " + nfs(p1A.value - p1B.value, 1, 2) + "    speed warp " + nfs(warpA.value - warpB.value, 1, 2), 840, 167);
  textSize(13);
  typeB.display();                              // last, so an open list is drawn on top
  typeA.display();
}

// ------------------- Signals -------------------
float generate(int type, float p1, float p2, float shift, float warp, float t) {
  float u = pow(t, warp) + shift;               // the warp changes the speed, not the shape
  float ph = (p1 * u) % 1;
  if (ph < 0) ph += 1;
  switch (type) {
    case 0: return p2 * sin(TWO_PI * p1 * u);                           // Sine
    case 1: return p2 * (ph < 0.5 ? 1 : -1);                            // Square
    case 2: return p2 * (1 - 4 * abs(ph - 0.5));                        // Triangle
    case 3: return p2 * (2 * ph - 1);                                    // Sawtooth
    case 4: return p2 * exp(-sq((u - 0.5) * p1 * 4));                    // Pulse: p1 = narrowness
    case 5: return p2 * sin(TWO_PI * p1 * (u + u * u));                  // Chirp: frequency rises
    default: return p1 * (u - 0.5) * 0.5 + (p2 - 1);                     // Line: slope, offset
  }
}

void updateLabels() {
  setLabels(typeA.sel, p1A, p2A);
  setLabels(typeB.sel, p1B, p2B);
}

void setLabels(int type, Slider p1, Slider p2) {
  p1.label = type == 6 ? "Slope" : type == 4 ? "Narrowness" : type == 5 ? "Start frequency (cycles)" : "Frequency (cycles)";
  p2.label = type == 6 ? "Offset" : "Amplitude";
}

// ------------------- Measures -------------------
void compute() {
  for (int i = 0; i < N; i++) {
    float t = i / (float) (N - 1);
    sigA[i] = generate(typeA.sel, p1A.value, p2A.value, shiftA.value, warpA.value, t) + noiseAmtA.value * noiseA[i];
    sigB[i] = generate(typeB.sel, p1B.value, p2B.value, shiftB.value, warpB.value, t) + noiseAmtB.value * noiseB[i];
  }
  arrayCopy(sigA, a);
  arrayCopy(sigB, b);
  if (normalizeBtn.active) {
    normalize(a);
    normalize(b);
  }
  computeDTW();
  if (findBtn.active) computeRunning();
  // The cards compare the whole signals, or (Scores: window) the part of A with its best window of B
  float[] p = a, q = b;
  if (local()) {
    p = subset(a, 0, winL);
    q = subset(b, bestStart, winL);
    dtwMean = runCost[bestStart];
    timing = 0;
    for (int[] ij : winPath) timing += abs(ij[0] - (ij[1] - bestStart)) / (float) winL / winPath.size();
  }
  measN = p.length;
  euclid = 0;                                   // point by point: mean |A(i) − B(i)|
  for (int i = 0; i < measN; i++) euclid += abs(p[i] - q[i]) / measN;
  computeCorrelation(p, q);
  computeSpectra(p, q);
  // The cost table as an image: dark blue = similar, white = different, grey = outside the band
  float maxc = 1e-6;
  for (float[] row : cost) for (float c : row) maxc = max(maxc, c);
  int w = bandWidth();
  costImage.loadPixels();
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      color c = lerpColor(color(#1F497D), color(255), constrain(cost[i][j] / maxc, 0, 1));
      if (abs(i - j) > w) c = lerpColor(c, color(200), 0.75);
      costImage.pixels[(N - 1 - i) * N + j] = c;   // row 0 (A's start) at the bottom
    }
  costImage.updatePixels();
}

int bandWidth() {
  return max(1, round(band.value * N));
}

// DTW limited to a band around the diagonal; the result is the mean cost along the cheapest path
void computeDTW() {
  int w = bandWidth();
  float[][] xy = compared();
  float[] x = xy[0], y = xy[1];
  float[][] D = new float[N + 1][N + 1];
  for (float[] row : D) java.util.Arrays.fill(row, Float.MAX_VALUE);
  D[0][0] = 0;
  for (int i = 1; i <= N; i++)
    for (int j = 1; j <= N; j++) {
      cost[i-1][j-1] = abs(x[i-1] - y[j-1]);
      if (abs(i - j) > w) continue;
      D[i][j] = cost[i-1][j-1] + min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
    }
  path.clear();
  int i = N, j = N;
  while (i > 0 && j > 0) {
    path.add(0, new int[] {i - 1, j - 1});
    float diag = D[i-1][j-1], up = D[i-1][j], left = D[i][j-1];
    if (diag <= up && diag <= left) { i--; j--; }
    else if (up < left) i--;
    else j--;
  }
  pathLen = path.size();
  dtwMean = D[N][N] / pathLen;
  timing = 0;                                   // how far apart in time the paired samples are
  for (int[] ij : path) timing += abs(ij[0] - ij[1]) / (float) N / pathLen;
}

// The cards show the matched window while Find A in B is on, otherwise the whole signals
boolean local() {
  return findBtn.active;
}

// The values compared: the signals, or their slopes (both on the same scale, [−1, 1])
float[][] compared() {
  if (!slopesBtn.active) return new float[][] {a, b};
  float[] x = slopes(a), y = slopes(b);
  float m = max(max(abs(min(x)), max(x)), max(abs(min(y)), max(y)), 1e-9);
  for (int k = 0; k < N; k++) { x[k] /= m; y[k] /= m; }
  return new float[][] {x, y};
}

// Find A in B: DTW between the first winL samples of A and every window of B of the same length
void computeRunning() {
  float[][] xy = compared();
  winL = max(10, round(winLen.value * N));
  runCost = new float[N - winL + 1];
  bestStart = 0;
  for (int s = 0; s < runCost.length; s++) {
    runCost[s] = dtwWindow(xy[0], xy[1], s, null);
    if (runCost[s] < runCost[bestStart]) bestStart = s;
  }
  winPath.clear();
  dtwWindow(xy[0], xy[1], bestStart, winPath);
}

// Mean DTW cost between x[0..winL) and y[s..s+winL); fills path (pairs i, j in y) if given
float dtwWindow(float[] x, float[] y, int s, ArrayList<int[]> path) {
  float[][] D = new float[winL + 1][winL + 1];
  for (float[] row : D) java.util.Arrays.fill(row, Float.MAX_VALUE);
  D[0][0] = 0;
  for (int i = 1; i <= winL; i++)
    for (int j = 1; j <= winL; j++)
      D[i][j] = abs(x[i-1] - y[s + j-1]) + min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
  int i = winL, j = winL, steps = 0;
  while (i > 0 && j > 0) {
    if (path != null) path.add(0, new int[] {i - 1, s + j - 1});
    steps++;
    float diag = D[i-1][j-1], up = D[i-1][j], left = D[i][j-1];
    if (diag <= up && diag <= left) { i--; j--; }
    else if (up < left) i--;
    else j--;
  }
  return D[winL][winL] / steps;
}

// Slope at each sample, as in derivative DTW (Keogh & Pazzani, 2001)
float[] slopes(float[] s) {
  float[] d = new float[N];
  for (int i = 1; i < N - 1; i++) d[i] = ((s[i] - s[i-1]) + (s[i+1] - s[i-1]) / 2) / 2;
  d[0] = d[1];
  d[N-1] = d[N-2];
  return d;
}

// How strong each frequency (1 to K cycles) is in each signal, as shares that add up to 1;
// the distance is the share that would have to move to turn one spectrum into the other
void computeSpectra(float[] a, float[] b) {
  spectrum(a, specA);
  spectrum(b, specB);
  specDist = 0;
  for (int k = 1; k <= K; k++) specDist += abs(specA[k] - specB[k]) / 2;
}

void spectrum(float[] s, float[] out) {
  int N = s.length;
  float m = 0, total = 0;
  for (float v : s) m += v / N;
  for (int k = 1; k <= K; k++) {
    float re = 0, im = 0;
    for (int i = 0; i < N; i++) {
      re += (s[i] - m) * cos(TWO_PI * k * i / N);
      im -= (s[i] - m) * sin(TWO_PI * k * i / N);
    }
    out[k] = sqrt(re * re + im * im);
    total += out[k];
  }
  for (int k = 1; k <= K; k++) out[k] = total > 1e-9 ? out[k] / total : 0;
}

// The highest normalised correlation over lags up to half the length, and its lag
void computeCorrelation(float[] a, float[] b) {
  int N = a.length;
  float ma = 0, mb = 0;
  for (int i = 0; i < N; i++) { ma += a[i] / N; mb += b[i] / N; }
  corrR = -2;
  corrLag = 0;
  for (int lag = -N / 2; lag <= N / 2; lag++) {
    double sab = 0, saa = 0, sbb = 0;
    for (int i = max(0, -lag); i < min(N, N - lag); i++) {
      double da = a[i] - ma, db = b[i + lag] - mb;
      sab += da * db; saa += da * da; sbb += db * db;
    }
    float r = (float) (sab / Math.sqrt(Math.max(saa * sbb, 1e-12)));
    if (r > corrR) { corrR = r; corrLag = lag; }
  }
}

void normalize(float[] x) {
  float minV = min(x), maxV = max(x);
  if (maxV - minV < 1e-9) {
    java.util.Arrays.fill(x, 0);
    return;
  }
  for (int i = 0; i < x.length; i++) x[i] = map(x[i], minV, maxV, -1, 1);
}

// ------------------- Drawing -------------------
float lo() { return min(min(min(a), min(b)), -1); }
float hi() { return max(max(max(a), max(b)), 1); }

void drawOverlay(int x, int y, int w, int h) {
  panel(x, y, w, h, "A and B on the same axes. Point by point compares samples at the same time (grey lines)");
  float l = x + 10, r = x + w - 10, t = y + 30, bt = y + h - 10, lv = lo(), hv = hi();
  if (pairsBtn.active) {
    stroke(185);
    for (int i = 0; i < N; i += 4) {
      float xx = map(i, 0, N - 1, l, r);
      line(xx, map(a[i], hv, lv, t, bt), xx, map(b[i], hv, lv, t, bt));
    }
  }
  if (findBtn.active) {                         // the part of A searched for, and where it was found in B
    noStroke();
    fill(BLUE, 28);
    rect(l, t, (r - l) * (winL - 1) / (N - 1), bt - t);
    fill(ORANGE, 28);
    rect(map(bestStart, 0, N - 1, l, r), t, (r - l) * (winL - 1) / (N - 1), bt - t);
  }
  plotSignal(a, l, r, t, bt, lv, hv, BLUE);
  plotSignal(b, l, r, t, bt, lv, hv, ORANGE);
}

void drawPairs(int x, int y, int w, int h) {
  if (findBtn.active) {
    drawRunning(x, y, w, h);
    return;
  }
  panel(x, y, w, h, "A above, B below. DTW pairs each sample with its best match, keeping the order (grey lines)");
  float l = x + 10, r = x + w - 10, half = (h - 40) / 2.0, lv = lo(), hv = hi();
  float ta = y + 30, ba = ta + half - 8, tb = ta + half + 8, bb = y + h - 10;
  if (pairsBtn.active) {
    stroke(185);
    int step = max(1, path.size() / 60);
    for (int k = 0; k < path.size(); k += step) {
      int[] ij = path.get(k);
      line(map(ij[0], 0, N - 1, l, r), map(a[ij[0]], hv, lv, ta, ba), map(ij[1], 0, N - 1, l, r), map(b[ij[1]], hv, lv, tb, bb));
    }
  }
  plotSignal(a, l, r, ta, ba, lv, hv, BLUE);
  plotSignal(b, l, r, tb, bb, lv, hv, ORANGE);
  fill(BLUE);
  text("A", r - 12, ta + 12);
  fill(ORANGE);
  text("B", r - 12, tb + 12);
}

// Find A in B: the part of A over its best window of B, and the DTW cost of every window
void drawRunning(int x, int y, int w, int h) {
  panel(x, y, w, h, "Find A in B: first " + winL + " samples of A vs every window of B. Best: "
        + bestStart + "–" + (bestStart + winL - 1));
  float l = x + 10, r = x + w - 10, lv = lo(), hv = hi();
  float ta = y + 26, ba = y + 92, tb = y + 92, bb = y + 158, ct = y + 178, cb = y + h - 8;
  float wx0 = map(bestStart, 0, N - 1, l, r), wx1 = map(bestStart + winL - 1, 0, N - 1, l, r);
  noStroke();
  fill(ORANGE, 28);
  rect(wx0, ta, wx1 - wx0, bb - ta);
  if (pairsBtn.active) {
    stroke(185);
    for (int k = 0; k < winPath.size(); k += max(1, winPath.size() / 40)) {
      int[] ij = winPath.get(k);
      line(map(bestStart + ij[0], 0, N - 1, l, r), map(a[ij[0]], hv, lv, ta, ba), map(ij[1], 0, N - 1, l, r), map(b[ij[1]], hv, lv, tb, bb));
    }
  }
  float[] part = subset(a, 0, winL);             // the part of A, drawn over its best window
  noFill();
  stroke(BLUE);
  strokeWeight(2.2);
  beginShape();
  for (int k = 0; k < winL; k++) vertex(map(bestStart + k, 0, N - 1, l, r), map(part[k], hv, lv, ta, ba));
  endShape();
  strokeWeight(1);
  plotSignal(b, l, r, tb, bb, lv, hv, ORANGE);
  // The DTW cost of each window, at the window's centre; lower = better match
  float top = max(max(runCost), 1e-6);
  stroke(220);
  line(l, cb, r, cb);
  noFill();
  stroke(TEAL);
  strokeWeight(2);
  beginShape();
  for (int s = 0; s < runCost.length; s++) vertex(map(s + winL / 2.0, 0, N - 1, l, r), map(runCost[s], 0, top, cb, ct));
  endShape();
  strokeWeight(1);
  noStroke();
  fill(ORANGE);
  float bx = map(bestStart + winL / 2.0, 0, N - 1, l, r), by = map(runCost[bestStart], 0, top, cb, ct);
  circle(bx, by, 9);
  fill(TEAL);
  textSize(11);
  text("DTW cost of each window (lower = better match). Best: " + nf(runCost[bestStart], 1, 3), l + 4, ct - 6);
  textSize(13);
}

void drawCost(int x, int y, int w, int h) {
  panel(x, y, w, h, slopesBtn.active ? "Cost table of the slopes and DTW path" : "Cost table |A(i) − B(j)| and DTW path");
  int s = min(w - 50, h - 75), x0 = x + 35, y0 = y + 30;
  image(costImage, x0, y0, s, s);
  stroke(255);
  line(x0, y0 + s, x0 + s, y0);                  // the diagonal: same time
  stroke(ORANGE);
  strokeWeight(2.5);
  noFill();
  beginShape();
  for (int[] ij : path) vertex(x0 + (ij[1] + 0.5) * s / N, y0 + s - (ij[0] + 0.5) * s / N);
  endShape();
  strokeWeight(1);
  fill(GREY);
  textSize(11);
  text("B →", x0 + s - 22, y0 + s + 14);
  pushMatrix();
  translate(x0 - 8, y0 + s);
  rotate(-HALF_PI);
  text("A →", 0, 0);
  popMatrix();
  text("white line: same time (what point by point uses)", x0, y0 + s + 30);
  text("off the diagonal: one signal is ahead of the other", x0, y0 + s + 44);
  textSize(13);
}

void drawMetrics(int x, int y) {
  String of = local() ? "% of the window" : "% of the length", w = local() ? " · window" : "";
  card(x, y, "Point by point" + (local() ? w : " (Euclidean)"), "mean |A(i) − B(i)|", euclid, 0, 1, false,
       "lower is better · 0 = identical");
  card(x + 310, y, (slopesBtn.active ? "DTW on slopes" : "DTW") + w,
       "timing absorbed: " + round(100 * timing) + of, dtwMean, 0, 1, false,
       "lower is better · 0 = same shape, any timing");
  card(x + 620, y, "Cross-correlation" + w, "best r, at lag " + corrLag + " (" + round(100.0 * corrLag / measN) + of + ")",
       corrR, -1, 1, true, "higher is better · 1 = same shape");
  card(x + 930, y, "Spectrum" + (local() ? w : " (1–" + K + " cycles)"), "", specDist, 0, 1, false, "lower is better · 0 = same frequencies");
  // The two spectra, small: one bar pair per frequency (1 to K cycles)
  float sx = x + 944, sy = y + 54, bw = 272.0 / K, top = max(max(specA), max(specB), 1e-9);
  noStroke();
  for (int k = 1; k <= K; k++) {
    fill(BLUE);
    rect(sx + (k - 1) * bw, sy - specA[k] / top * 22, bw * 0.45, specA[k] / top * 22);
    fill(ORANGE);
    rect(sx + (k - 0.5) * bw, sy - specB[k] / top * 22, bw * 0.45, specB[k] / top * 22);
  }
}

// A measure with its bar on a scale from lo to hi; the good end is green
void card(int x, int y, String title, String sub, float v, float lo, float hi, boolean higherBetter, String hint) {
  int w = 300, h = 125;
  stroke(215);
  fill(255);
  rect(x, y, w, h, 8);
  fill(INK);
  textSize(14);
  text(title, x + 14, y + 24);
  textSize(24);
  textAlign(RIGHT);
  text(nf(v, 1, 3), x + w - 14, y + 30);
  textAlign(LEFT);
  textSize(12);
  fill(GREY);
  text(sub, x + 14, y + 44);
  int bx = x + 14, by = y + 60, bw = w - 28;
  noStroke();
  for (int k = 0; k < bw; k++) {                 // the scale: green at the good end, grey at the other
    float f = k / (float) bw;
    fill(lerpColor(color(#DCEFE3), color(#EEEEEE), higherBetter ? 1 - f : f));
    rect(bx + k, by, 1, 18);
  }
  float px = map(constrain(v, lo, hi), lo, hi, bx, bx + bw);
  fill(INK);
  triangle(px - 7, by - 8, px + 7, by - 8, px, by + 2);
  rect(px - 1.5, by, 3, 18);
  fill(GREY);
  textSize(11);
  text(nf(lo, 1, 0), bx, by + 32);
  textAlign(RIGHT);
  text(nf(hi, 1, 0) + (hi == 1 && !higherBetter ? "+" : ""), bx + bw, by + 32);
  textAlign(LEFT);
  fill(GREEN);
  text(hint, bx + 18, by + 32);
  textSize(13);
}

void panel(int x, int y, int w, int h, String title) {
  stroke(215);
  fill(255);
  rect(x, y, w, h, 6);
  fill(INK);
  textSize(12);
  text(title, x + 10, y + 18);
  textSize(13);
}

void plotSignal(float[] s, float l, float r, float t, float bt, float lv, float hv, color c) {
  noFill();
  stroke(c);
  strokeWeight(2.2);
  beginShape();
  for (int i = 0; i < N; i++) vertex(map(i, 0, N - 1, l, r), map(s[i], hv, lv, t, bt));
  endShape();
  strokeWeight(1);
}

// ------------------- Presets and interaction -------------------
void keyPressed() {
  if (key == ' ') {                               // random frequency and speed warp for both signals
    for (Slider s : new Slider[] {p1A, warpA, p1B, warpB}) s.value = random(s.minVal, s.maxVal);
    return;
  }
  if (key < '1' || key > '5') return;
  for (Slider s : new Slider[] {shiftA, shiftB}) s.value = 0;
  for (Slider s : new Slider[] {warpA, warpB}) s.value = 1;
  noiseAmtA.value = noiseAmtB.value = 0;
  typeA.sel = typeB.sel = 0;
  p1A.value = p1B.value = 2;
  p2A.value = p2B.value = 1;
  band.value = 1;
  if (key == '1') warpB.value = 1.6;              // same shape, other speed: DTW ≈ 0, Euclidean large
  if (key == '2') shiftB.value = 0.1;             // shifted: correlation finds the lag
  if (key == '3') p2B.value = 0.5;                // other amplitude: try Normalise
  if (key == '4') typeB.sel = 1;                  // other shape: all three see it
  if (key == '5') { warpB.value = 1.6; noiseAmtB.value = 0.25; }
}

Slider[] sliders() {
  return new Slider[] {p1A, p2A, shiftA, warpA, noiseAmtA, p1B, p2B, shiftB, warpB, noiseAmtB, findBtn.active ? winLen : band};
}

String state() {
  String s = typeA.sel + "," + typeB.sel + "," + normalizeBtn.active + "," + pairsBtn.active + "," + slopesBtn.active + "," + findBtn.active + "," + band.value + "," + winLen.value;
  for (Slider sl : sliders()) s += "," + sl.value;
  return s;
}

void mousePressed() {
  if (typeA.open || typeB.open) {                 // a click on an open list only chooses
    typeA.handleClick(mouseX, mouseY);
    typeB.handleClick(mouseX, mouseY);
    return;
  }
  typeA.handleClick(mouseX, mouseY);
  typeB.handleClick(mouseX, mouseY);
  for (Slider s : sliders()) s.mousePressed();
  normalizeBtn.mousePressed();
  pairsBtn.mousePressed();
  slopesBtn.mousePressed();
  findBtn.mousePressed();
}

void mouseDragged() {
  for (Slider s : sliders()) s.mouseDragged();
}

void mouseReleased() {
  for (Slider s : sliders()) s.mouseReleased();
}

// ------------------- UI Classes -------------------
class Slider {
  int x, y, w, h;
  float minVal, maxVal, value;
  String label;
  boolean dragging = false;

  Slider(int x, int y, int w, int h, float minVal, float maxVal, float initVal, String label) {
    this.x = x; this.y = y; this.w = w; this.h = h;
    this.minVal = minVal; this.maxVal = maxVal; this.value = initVal; this.label = label;
  }

  void display() {
    fill(INK);
    textSize(12);
    text(label + ": " + nf(value, 1, 2), x, y - 5);
    textSize(13);
    stroke(150);
    line(x, y + h / 2, x + w, y + h / 2);
    float knobX = map(value, minVal, maxVal, x, x + w);
    noStroke();
    fill(dragging ? BLUE : 60);
    ellipse(knobX, y + h / 2, 12, 12);
  }

  void mousePressed() {
    // Clicking anywhere on the line moves the knob there
    if (mouseX > x - 8 && mouseX < x + w + 8 && abs(mouseY - (y + h / 2)) < 10) {
      dragging = true;
      mouseDragged();
    }
  }

  void mouseDragged() {
    if (dragging) value = constrain(map(mouseX, x, x + w, minVal, maxVal), minVal, maxVal);
  }

  void mouseReleased() { dragging = false; }
}

class DropList {
  int x, y, w, h;
  String title;
  String[] items;
  int sel;
  boolean open = false;

  DropList(int x, int y, int w, int h, String title, String[] items, int def) {
    this.x = x; this.y = y; this.w = w; this.h = h;
    this.title = title; this.items = items; this.sel = def;
  }

  void display() {
    fill(this == typeA ? BLUE : ORANGE);
    textSize(14);
    text(title, x, y - 8);
    textSize(13);
    stroke(150);
    fill(255);
    rect(x, y, w, h);
    fill(0);
    text(items[sel], x + 8, y + h - 7);
    triangle(x + w - 18, y + h / 2 - 4, x + w - 8, y + h / 2 - 4, x + w - 13, y + h / 2 + 4);
    if (open) {
      stroke(150);
      fill(255);
      rect(x, y + h, w, h * items.length);
      for (int i = 0; i < items.length; i++) {
        int yy = y + h + i * h;
        if (mouseX > x && mouseX < x + w && mouseY > yy && mouseY < yy + h) {
          fill(230);
          rect(x, yy, w, h);
        }
        fill(0);
        text(items[i], x + 8, yy + h - 7);
      }
    }
  }

  boolean handleClick(int mx, int my) {
    if (open) {
      for (int i = 0; i < items.length; i++) {
        int yy = y + h + i * h;
        if (mx > x && mx < x + w && my > yy && my < yy + h) {
          sel = i;
          open = false;
          return true;
        }
      }
      open = false;
      return false;
    } else if (mx > x && mx < x + w && my > y && my < y + h) {
      open = true;
    }
    return false;
  }
}

class ToggleButton {
  int x, y, w, h;
  String label;
  boolean active = false;

  ToggleButton(int x, int y, int w, int h, String label) {
    this.x = x; this.y = y; this.w = w; this.h = h; this.label = label;
  }

  void display() {
    stroke(150);
    fill(active ? color(80, 180, 80) : 255);
    rect(x, y, w, h, 5);
    fill(active ? 255 : 0);
    textAlign(CENTER, CENTER);
    text(label, x + w / 2, y + h / 2 + 1);
    textAlign(LEFT);
  }

  void mousePressed() {
    if (mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h) active = !active;
  }
}
