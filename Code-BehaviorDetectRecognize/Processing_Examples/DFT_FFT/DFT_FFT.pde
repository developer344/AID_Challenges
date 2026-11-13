// DFT and FFT on a signal made here (three waves: the example of the slides) or on the microphone.
// Top: x(t), the last N samples. Middle: A(f), the size of the wave of each frequency in x(t). The bars are
// Δf = fs / N apart: the only frequencies that N samples can tell apart. Orange circles: the waves set with the sliders.
// Bottom: A(f) over time (a spectrogram), one column per frame, dark = strong.
// Buttons:
//   Made here / Microphone: the input (made here: 50 samples a second, like a movement; microphone: 44100).
//   DFT / FFT: the same A(f). The DFT compares x(t) with every frequency: N × N/2 products. The FFT gets the same
//     numbers in N/2 × log2(N) steps, but only for N a power of 2, so other N are padded with zeros. Compare the times.
//   N: more samples give a finer Δf, but a longer window (a slower answer).
//   No window / Hann: a wave that does not fit a whole number of times in the window spreads into the bars next to it
//     (leakage). The Hann window fades the ends of the window and narrows that spread. With N = 200 at 50 Hz,
//     Δf = 0.25 Hz and the waves of the slides (0.5, 2 and 7 Hz) fall exactly on bars: no leakage. With N = 256 they don't.
//   Linear / dB: dB shows the small values too (leakage, noise, harmonics).
// Sliders: the frequency and size of each wave, and noise. A wave above fs / 2 = 25 Hz shows up lower down (aliasing).
// Mouse wheel over A(f): zoom the frequency axis. Space: pause.
// The slides' formula used the sine only (waves starting at 0). Here the cosine is used too, so the phase does not
// matter: A(f) = 2/N × the length of (Σ x·cos, Σ x·sin).
// Uses Java's own audio, so no library is needed. The microphone asks for permission the first time.

import javax.sound.sampled.*;

float MIC_RATE = 44100, SYN_RATE = 50;
int[] SIZES = {200, 256, 512, 1024, 2048, 4096};
float[][] EXAMPLE = {{0.5, 0.6}, {2, 1.0}, {7, 0.3}};    // the slides: 0.6 sin(2π·0.5t) + 1.0 sin(2π·2t) + 0.3 sin(2π·7t)
int N = 256, M = 256;                                     // M: the samples the transform used (N, or N padded for the FFT)
boolean useMic = false, useFFT = true, hann = false, dB = false, paused = false, clicked = false, micFailed = false;
float fShow = 25;                                         // the highest frequency shown
float[] ring = new float[8192];                           // the microphone's newest samples
int ringPos = 0;
Object lock = new Object();
String status = "";
float[] x = new float[0], A = new float[1];
float binHz = 1, msAvg = 0, topMic = 0.01, yTopMic = 0.05, synTime = 0;
int lastMs = 0;
PImage specImage;
int PX = 70, PW = 800, RX = 910, RW = 270;                // the plots, and the panel on the right
color BLUE = #2E75B6, ORANGE = #E36C0A, INK = #262626, GREY = #8C8C8C, DARK = #1F497D;
Slider[] freq = new Slider[3], amp = new Slider[3];
Slider noise;

void setup() {
  size(1200, 800);
  textFont(createFont("Arial", 14));
  for (int i = 0; i < 3; i++) {
    freq[i] = new Slider(RX, 135 + i * 92, RW, 0, 40, EXAMPLE[i][0], "Wave " + (i + 1) + ": f", " Hz", 1);
    amp[i] = new Slider(RX, 177 + i * 92, RW, 0, 1.5, EXAMPLE[i][1], "A", "", 2);
  }
  noise = new Slider(RX, 411, RW, 0, 1, 0, "Noise", "", 2);
  specImage = createImage(PW, 170, RGB);
  clearSpectrogram();
  lastMs = millis();
  new Mic().start();
}

void draw() {
  background(255);
  int now = millis();
  if (!paused) synTime += (now - lastMs) / 1000.0;
  lastMs = now;
  if (micFailed) {                                        // back to the signal made here, keeping the message
    String s = status;
    micFailed = false;
    setSource(false);
    status = s;
  }
  float fs = useMic ? MIC_RATE : SYN_RATE;
  if (!useMic) x = madeHere(N);
  else if (!paused || x.length != N) x = fromMic(N);

  long t0 = System.nanoTime();
  float[] w = windowOf(x.length);
  A = useFFT ? fftAmplitudes(x, w) : dftAmplitudes(x, w);
  float ms = (System.nanoTime() - t0) / 1e6;
  msAvg = msAvg == 0 ? ms : 0.9 * msAvg + 0.1 * ms;
  binHz = fs / M;

  drawSignal(fs, w);
  int peak = drawSpectrum(fs);
  if (!paused) addColumn();
  drawSpectrogram();
  drawRight(fs, peak);
  drawControls();                                         // last: a click changes what the next frame shows
  clicked = false;
}

// ---- The signal ----

float[] madeHere(int n) {
  float[] v = new float[n];
  long newest = (long) Math.floor(synTime * SYN_RATE);
  for (int i = 0; i < n; i++) {
    long s = newest - (n - 1) + i;                        // the sample number: its time is s / fs
    double sum = 0;
    for (int j = 0; j < 3; j++) sum += amp[j].value * Math.sin(2 * Math.PI * freq[j].value * s / SYN_RATE);
    v[i] = (float) sum + noise.value * noiseAt(s);
  }
  return v;
}

float noiseAt(long s) {                                   // the same noise for the same sample, frame after frame
  long h = s * 0x9E3779B97F4A7C15L;
  h ^= h >>> 31;
  h *= 0xBF58476D1CE4E5B9L;
  h ^= h >>> 29;
  return ((h & 0xFFFFFF) / (float) 0xFFFFFF - 0.5) * 3.46;   // uniform, standard deviation 1
}

float[] fromMic(int n) {
  float[] v = new float[n];
  synchronized (lock) {
    for (int i = 0; i < n; i++) v[i] = ring[(ringPos - n + i + ring.length) % ring.length];
  }
  return v;
}

float[] windowOf(int n) {
  float[] w = new float[n];
  for (int i = 0; i < n; i++) w[i] = hann ? 0.5 - 0.5 * cos(TWO_PI * i / (n - 1)) : 1;
  return w;
}

// ---- The two transforms: the same A(f) ----

float[] cosT = new float[0], sinT = new float[0];         // the angles, worked out once for each N

void tables(int n) {
  if (cosT.length == n) return;
  cosT = new float[n];
  sinT = new float[n];
  for (int i = 0; i < n; i++) {
    cosT[i] = cos(TWO_PI * i / n);
    sinT[i] = sin(TWO_PI * i / n);
  }
}

// DFT: each frequency k·fs/N against each of the N samples
float[] dftAmplitudes(float[] v, float[] w) {
  int n = v.length;
  M = n;
  tables(n);
  float[] vw = new float[n];
  float norm = 0;
  for (int i = 0; i < n; i++) {
    vw[i] = v[i] * w[i];
    norm += w[i];
  }
  float[] out = new float[n / 2 + 1];
  for (int k = 0; k <= n / 2; k++) {
    float re = 0, im = 0;
    int idx = 0;                                          // k·i mod n: the angle 2π·k·i/n, read from the table
    for (int i = 0; i < n; i++) {
      re += vw[i] * cosT[idx];
      im -= vw[i] * sinT[idx];
      idx += k;
      if (idx >= n) idx -= n;
    }
    out[k] = 2 * sqrt(re * re + im * im) / norm;
  }
  out[0] /= 2;                                            // 0 Hz (the mean) and fs/2 appear once, not twice
  if (n % 2 == 0) out[n / 2] /= 2;
  return out;
}

// FFT: the same sums, split in halves again and again (N must be a power of 2: pad with zeros)
float[] fftAmplitudes(float[] v, float[] w) {
  int n = v.length;
  M = 1;
  while (M < n) M <<= 1;
  tables(M);
  float[] re = new float[M], im = new float[M];
  float norm = 0;
  for (int i = 0; i < n; i++) {                           // re[n..M-1] stay 0: the padding
    re[i] = v[i] * w[i];
    norm += w[i];
  }
  fft(re, im);
  float[] out = new float[M / 2 + 1];
  for (int k = 0; k <= M / 2; k++) out[k] = 2 * sqrt(re[k] * re[k] + im[k] * im[k]) / norm;
  out[0] /= 2;
  out[M / 2] /= 2;
  return out;
}

// In-place radix-2 FFT (Cooley–Tukey), the angles from the tables
void fft(float[] re, float[] im) {
  int n = re.length;
  for (int i = 1, j = 0; i < n; i++) {                    // reorder the samples by their bit-reversed index
    int bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {                // join pairs, then fours, eights...
    int step = n / len, half = len / 2;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half; k++) {
        float wr = cosT[k * step], wi = -sinT[k * step];
        int a = i + k, b = a + half;
        float xr = re[b] * wr - im[b] * wi;
        float xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

// ---- The plots ----

void panel(float x, float y, float w, float h, String title) {
  noFill();
  stroke(225);
  rect(x, y, w, h);
  fill(INK);
  textSize(14);
  textAlign(LEFT, BASELINE);
  text(title, x, y - 8);
}

void drawSignal(float fs, float[] w) {
  float y = 105, h = 160, mid = y + h / 2;
  int n = x.length;
  panel(PX, y, PW, h, "x(t): the last N = " + n + " samples, " + duration(n / fs) + " at fs = " + hz(fs));
  float top;
  if (useMic) {
    float m = 0;
    for (float v : x) m = max(m, abs(v));
    yTopMic = max(0.01, max(m * 1.1, yTopMic * 0.98));
    top = yTopMic;
  } else {
    top = noise.value * 3;
    for (Slider a : amp) top += a.value;
    top = max(top, 0.3) * 1.1;
  }
  stroke(225);
  line(PX, mid, PX + PW, mid);
  if (hann) {                                             // the window: what the transform keeps of each sample
    stroke(GREY);
    noFill();
    for (int s = -1; s <= 1; s += 2) {
      beginShape();
      for (int i = 0; i < n; i += max(1, n / 200)) vertex(map(i, 0, n - 1, PX, PX + PW), mid - s * w[i] * h / 2);
      endShape();
    }
  }
  stroke(BLUE);
  strokeWeight(1.5);
  noFill();
  beginShape();
  for (int i = 0; i < n; i++) vertex(map(i, 0, n - 1, PX, PX + PW), constrain(mid - x[i] / top * h / 2, y, y + h));
  endShape();
  strokeWeight(1);
  if (n <= 256) {                                         // few enough to see each sample
    noStroke();
    fill(BLUE);
    for (int i = 0; i < n; i++) circle(map(i, 0, n - 1, PX, PX + PW), constrain(mid - x[i] / top * h / 2, y, y + h), 3.5);
  }
  fill(GREY);
  textSize(11);
  textAlign(RIGHT, CENTER);
  text(nf(top, 1, 2), PX - 6, y + 4);
  text("0", PX - 6, mid);
  text("−" + nf(top, 1, 2), PX - 6, y + h - 4);
  textAlign(LEFT, TOP);
  text("−" + duration((n - 1) / fs), PX, y + h + 4);
  textAlign(RIGHT, TOP);
  text(hann ? "grey: the Hann window    now" : "now", PX + PW, y + h + 4);
  textAlign(LEFT, BASELINE);
}

float dbLo() { return useMic ? -100 : -80; }
float dbHi() { return useMic ? 0 : 10; }
float topLin() { return useMic ? topMic : 1.6; }

float level(float a) {                                    // 0 at the bottom of the plot, 1 at the top
  if (dB) return constrain((20 * log(max(a, 1e-12)) / log(10) - dbLo()) / (dbHi() - dbLo()), 0, 1);
  return constrain(a / topLin(), 0, 1);
}

int drawSpectrum(float fs) {
  float y = 310, h = 220;
  panel(PX, y, PW, h, "A(f): the size of the wave of each frequency. A bar every Δf = fs / " + M + " = " + hz(binHz));
  int kMax = min(A.length - 1, floor(fShow / binHz));
  int peak = min(1, kMax);
  for (int k = 1; k <= kMax; k++) if (A[k] > A[peak]) peak = k;
  if (useMic) topMic = max(0.002, max(A[peak] * 1.2, topMic * 0.98));

  float step = tickStep(fShow);                           // the grid
  textSize(11);
  for (float f = 0; f <= fShow + 1e-4; f += step) {
    float px = PX + f / fShow * PW;
    stroke(240);
    line(px, y, px, y + h);
    fill(GREY);
    textAlign(CENTER, TOP);
    text(step >= 1000 ? num(f / 1000, step % 1000 == 0 ? 0 : 1) : num(f, step >= 1 ? 0 : 1), px, y + h + 4);
  }
  textAlign(LEFT, TOP);
  text(step >= 1000 ? "kHz" : "Hz", PX + PW + 8, y + h + 4);
  textAlign(RIGHT, CENTER);
  if (dB) for (float d = dbLo(); d <= dbHi() + 0.1; d += 20) text(num(d, 0) + " dB", PX - 6, y + h - (d - dbLo()) / (dbHi() - dbLo()) * h);
  else for (int i = 0; i <= 2; i++) text(nf(topLin() * i / 2, 1, useMic ? 3 : 1), PX - 6, y + h - i * h / 2);

  float barW = binHz / fShow * PW;                        // the bars, or a line when they are too thin to see
  if (barW >= 3) {
    noStroke();
    fill(BLUE);
    for (int k = 0; k <= kMax; k++) {
      float px = PX + k * binHz / fShow * PW, top = y + h - level(A[k]) * h;
      rect(max(PX, px - barW * 0.35), top, barW * 0.7, y + h - top);
    }
  } else {
    stroke(BLUE);
    noFill();
    beginShape();
    for (int k = 0; k <= kMax; k++) vertex(PX + k * binHz / fShow * PW, y + h - level(A[k]) * h);
    endShape();
  }
  if (!useMic) {                                          // the waves set with the sliders, where they should show up
    for (int j = 0; j < 3; j++) {
      if (amp[j].value < 0.005) continue;
      float fa = aliased(freq[j].value);
      if (fa > fShow) continue;
      float px = PX + fa / fShow * PW, py = y + h - level(amp[j].value) * h;
      stroke(ORANGE);
      strokeWeight(2);
      noFill();
      circle(px, py, 12);
      strokeWeight(1);
      if (freq[j].value > SYN_RATE / 2) {
        fill(ORANGE);
        textAlign(CENTER, BOTTOM);
        text(num(freq[j].value, 1) + " Hz shows at " + num(fa, 1), px, py - 8);
      }
    }
    fill(ORANGE);
    textAlign(RIGHT, TOP);
    text("○ the waves set with the sliders", PX + PW - 8, y + 6);
  }
  if (kMax >= 1) {                                        // the strongest frequency
    float px = PX + peak * binHz / fShow * PW, py = y + h - level(A[peak]) * h;
    fill(INK);
    noStroke();
    circle(px, py, 6);
    textAlign(px > PX + PW - 200 ? RIGHT : LEFT, BOTTOM);
    text("peak " + hz(peak * binHz), px + (px > PX + PW - 200 ? -8 : 8), max(y + 14, py - 4));
  }
  textAlign(LEFT, BASELINE);
  return peak;
}

float aliased(float f) {                                  // where a sampled wave of frequency f shows up
  float r = f % SYN_RATE;
  return min(r, SYN_RATE - r);
}

void clearSpectrogram() {
  specImage.loadPixels();
  java.util.Arrays.fill(specImage.pixels, color(255));
  specImage.updatePixels();
}

void addColumn() {                                        // shift left by one pixel, the new A(f) on the right
  int W = specImage.width, H = specImage.height;
  specImage.loadPixels();
  for (int r = 0; r < H; r++) {
    arrayCopy(specImage.pixels, r * W + 1, specImage.pixels, r * W, W - 1);
    float f0 = fShow * (H - 1 - r) / H, f1 = fShow * (H - r) / H;
    int k0 = round(f0 / binHz), k1 = max(k0, round(f1 / binHz) - 1);
    float a = 0;
    for (int k = k0; k <= k1 && k < A.length; k++) a = max(a, A[k]);
    specImage.pixels[r * W + W - 1] = lerpColor(color(255), DARK, level(a));
  }
  specImage.updatePixels();
}

void drawSpectrogram() {
  float y = 575, h = specImage.height;
  image(specImage, PX, y);
  panel(PX, y, PW, h, "A(f) over time: one column per frame, the newest on the right (dark = strong)");
  fill(GREY);
  textSize(11);
  textAlign(RIGHT, CENTER);
  text(hz(fShow), PX - 6, y + 6);
  text("0", PX - 6, y + h - 4);
  textAlign(LEFT, TOP);
  text("about " + round(PW / max(frameRate, 1)) + " s ago", PX, y + h + 4);
  textAlign(RIGHT, TOP);
  text("now", PX + PW, y + h + 4);
  textAlign(LEFT, BASELINE);
}

// ---- The panel on the right ----

void drawRight(float fs, int peak) {
  if (useMic) {
    panel(RX, 105, RW, 0, "The microphone");
    float rms = 0;
    for (float v : x) rms += v * v / max(1, x.length);
    float lv = constrain((10 * log(max(rms, 1e-12)) / log(10) + 70) / 70, 0, 1);   // -70 to 0 dB
    noStroke();
    fill(240);
    rect(RX, 120, RW, 14, 4);
    fill(BLUE);
    rect(RX, 120, RW * lv, 14, 4);
    fill(INK);
    textSize(13);
    text("Whistle, sing or play a note: the peak is its pitch (" + note(peak * binHz) + ").\n\n"
      + "Sing a vowel: the bars at 2, 3, 4… times the pitch are its harmonics; change the vowel and they change.\n\n"
      + "Clap: every frequency at once (a vertical line below).", RX, 150, RW, 300);
  } else {
    panel(RX, 105, RW, 0, "The signal made here (fs = 50 Hz)");
    for (int j = 0; j < 3; j++) {
      boolean over = freq[j].value > SYN_RATE / 2;
      freq[j].display(over ? ORANGE : INK, over ? "   above fs/2: aliasing" : "");
      amp[j].display(INK, "");
    }
    noise.display(INK, "");
    if (button(RX, 436, 190, "The slides' example", false)) {
      for (int j = 0; j < 3; j++) {
        freq[j].value = EXAMPLE[j][0];
        amp[j].value = EXAMPLE[j][1];
      }
      noise.value = 0;
    }
  }
  float y = 520;                                          // the numbers
  panel(RX, y, RW, 0, "The numbers");
  int n = x.length;
  String[] lines = {
    "fs = " + hz(fs) + ": samples a second",
    "N = " + n + " samples: a window of " + duration(n / fs),
    (M != n ? "FFT pads " + n + " to " + M + " samples: " : "") + "Δf = fs / " + M + " = " + hz(binHz),
    "Up to fs / 2 = " + hz(fs / 2),
    "DFT: " + n + " × " + (n / 2 + 1) + " = " + nfc(n * (n / 2 + 1)) + " products",
    "FFT: " + (pow2(n) / 2) + " × " + log2(pow2(n)) + " = " + nfc(pow2(n) / 2 * log2(pow2(n))) + " steps",
    "Now: " + (useFFT ? "FFT" : "DFT") + ", " + nf(msAvg, 1, 3) + " ms per frame",
    "Peak: " + hz(peak * binHz) + ", A = " + nf(A[peak], 1, useMic ? 4 : 2)
  };
  textSize(13);
  for (int i = 0; i < lines.length; i++) {
    fill(i == 6 ? BLUE : INK);
    text(lines[i], RX, y + 22 + i * 24);
  }
}

// ---- The buttons along the top ----

void drawControls() {
  fill(INK);
  textSize(18);
  text("DFT and FFT", 20, 30);
  fill(GREY);
  textSize(12);
  textAlign(RIGHT, BASELINE);
  text(status + (paused ? "   Paused (space)" : "   Mouse wheel over A(f): zoom.  Space: pause"), width - 20, 30);
  textAlign(LEFT, BASELINE);
  if (button(20, 46, 100, "Made here", !useMic) && useMic) setSource(false);
  if (button(125, 46, 100, "Microphone", useMic) && !useMic) setSource(true);
  if (button(250, 46, 60, "DFT", !useFFT)) { useFFT = false; msAvg = 0; }
  if (button(315, 46, 60, "FFT", useFFT)) { useFFT = true; msAvg = 0; }
  fill(INK);
  textSize(14);
  text("N", 398, 64);
  for (int i = 0; i < SIZES.length; i++) {
    if (button(415 + i * 60, 46, 56, str(SIZES[i]), N == SIZES[i]) && N != SIZES[i]) {
      N = SIZES[i];
      msAvg = 0;
      clearSpectrogram();
    }
  }
  if (button(795, 46, 90, "No window", !hann)) hann = false;
  if (button(890, 46, 70, "Hann", hann)) hann = true;
  if (button(985, 46, 70, "Linear", !dB)) dB = false;
  if (button(1060, 46, 50, "dB", dB)) dB = true;
  if (button(1120, 46, 60, paused ? "Go on" : "Pause", paused)) paused = !paused;
}

void setSource(boolean mic) {
  useMic = mic;
  N = mic ? 2048 : 256;
  fShow = mic ? 4000 : SYN_RATE / 2;
  dB = mic;
  msAvg = 0;
  x = new float[0];
  status = mic ? "Opening the microphone…" : "";
  clearSpectrogram();
}

boolean button(float x, float y, float w, String label, boolean on) {
  stroke(170);
  fill(on ? color(80, 160, 90) : 245);
  rect(x, y, w, 26, 5);
  fill(on ? 255 : INK);
  textSize(13);
  textAlign(CENTER, CENTER);
  text(label, x + w / 2, y + 13);
  textAlign(LEFT, BASELINE);
  return clicked && inside(x, y, w, 26);
}

boolean inside(float x, float y, float w, float h) {
  return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h;
}

void mousePressed() {
  clicked = true;
  if (!useMic) {
    for (int j = 0; j < 3; j++) {
      freq[j].press();
      amp[j].press();
    }
    noise.press();
  }
}

void mouseDragged() {
  for (int j = 0; j < 3; j++) {
    freq[j].drag();
    amp[j].drag();
  }
  noise.drag();
}

void mouseReleased() {
  for (int j = 0; j < 3; j++) freq[j].dragging = amp[j].dragging = false;
  noise.dragging = false;
}

void mouseWheel(MouseEvent e) {
  if (!inside(PX, 310, PW, 435)) return;
  float fs = useMic ? MIC_RATE : SYN_RATE;
  fShow = constrain(fShow * pow(1.2, e.getCount()), useMic ? 200 : 1, fs / 2);
  clearSpectrogram();
}

void keyPressed() {
  if (key == ' ') paused = !paused;
}

// ---- Small helpers ----

int pow2(int n) { int m = 1; while (m < n) m <<= 1; return m; }
int log2(int n) { int b = 0; while ((1 << b) < n) b++; return b; }
String num(float v, int digits) { return digits == 0 ? str(round(v)) : nf(v, 1, digits); }

float tickStep(float range) {
  float[] steps = {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
  for (float s : steps) if (range / s <= 10) return s;
  return 5000;
}

String hz(float f) {
  if (f >= 1000) return nf(f / 1000, 1, 2) + " kHz";
  if (f >= 100) return round(f) + " Hz";
  if (f >= 10) return nf(f, 1, 1) + " Hz";
  return nf(f, 1, f < 1 ? 3 : 2) + " Hz";
}

String duration(float s) {
  return s >= 1 ? nf(s, 1, 2) + " s" : nf(s * 1000, 1, 1) + " ms";
}

String note(float f) {                                    // the nearest note: A4 = 440 Hz
  if (f < 20) return "–";
  String[] names = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
  int midi = round(69 + 12 * log(f / 440) / log(2));
  return names[(midi % 12 + 12) % 12] + (midi / 12 - 1);
}

// ---- The microphone, read in its own thread ----

class Mic extends Thread {
  Mic() {
    setDaemon(true);
  }
  public void run() {
    AudioFormat format = new AudioFormat(MIC_RATE, 16, 1, true, false);
    TargetDataLine in = null;
    byte[] buf = new byte[1024];
    while (true) {
      try {
        if (!useMic) {                                    // the microphone is only open while it is chosen
          if (in != null) {
            in.close();
            in = null;
          }
          Thread.sleep(30);
          continue;
        }
        if (in == null) {
          in = AudioSystem.getTargetDataLine(format);
          in.open(format, 8192);
          in.start();
          status = "Listening to the microphone";
        }
        int got = 0;
        while (got < buf.length) got += in.read(buf, got, buf.length - got);
        synchronized (lock) {
          for (int i = 0; i < buf.length / 2; i++) {
            ring[ringPos] = ((buf[2 * i + 1] << 8) | (buf[2 * i] & 0xff)) / 32768.0;
            ringPos = (ringPos + 1) % ring.length;
          }
        }
      } catch (Exception e) {
        if (in != null) in.close();
        in = null;
        status = "No microphone: " + e.getMessage();
        micFailed = true;
        try { Thread.sleep(500); } catch (Exception e2) { }
      }
    }
  }
}

// A horizontal slider with its name and value above it
class Slider {
  float x, y, w, lo, hi, value;
  String name, unit;
  int digits;
  boolean dragging = false;
  Slider(float x, float y, float w, float lo, float hi, float value, String name, String unit, int digits) {
    this.x = x; this.y = y; this.w = w; this.lo = lo; this.hi = hi; this.value = value;
    this.name = name; this.unit = unit; this.digits = digits;
  }
  void display(color c, String extra) {
    fill(c);
    textSize(13);
    text(name + " = " + nf(value, 1, digits) + unit + extra, x, y - 10);
    stroke(170);
    strokeWeight(2);
    line(x, y, x + w, y);
    strokeWeight(1);
    noStroke();
    fill(dragging ? BLUE : 60);
    circle(map(value, lo, hi, x, x + w), y, 15);
  }
  void press() {
    dragging = abs(mouseY - y) < 12 && mouseX > x - 10 && mouseX < x + w + 10;
    drag();
  }
  void drag() {
    if (dragging) value = constrain(map(mouseX, x, x + w, lo, hi), lo, hi);
    if (dragging) value = round(value * pow(10, digits)) / pow(10, digits);
  }
}
