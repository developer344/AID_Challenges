// Spectral flux and autocorrelation, step by step, on drums made by this sketch (or on the microphone).
// 1. Top left: the spectrogram, a spectrum every 11.6 ms (time across, frequency up, dark = strong).
// 2. Top right: at the last onset, that frame (orange) against the frame before (grey). The shaded increases,
//    added up, are the spectral flux of that moment. Decreases do not count: a sound fading is not an onset.
// 3. Middle: the flux over time. Its peaks above the moving threshold (orange line) are the onsets.
// 4. Bottom left: the flux curve (blue) and a copy shifted by τ (orange). Move the τ slider: when the peaks
//    line up, the sum of their products is high.
// 5. Bottom right: that sum, R(τ), for every shift. The highest point between 0.3 and 1 s (60-200 BPM) is the
//    beat period; the tempo is 60 / τ. Twice the period also scores high: an octave error.
// Controls: the buttons choose the sound (drums made here, or the microphone), sound on or off, and the
// drum pattern; the Tempo slider sets the drums' tempo. Up/Down raise or lower the onset threshold.
// Uses Java's own audio, so no library is needed. The microphone asks for permission the first time.

import javax.sound.sampled.*;

int RATE = 44100, N_FFT = 1024, HOP = 512, BINS = 128;   // BINS shown: up to about 5.5 kHz
float HOP_S = HOP / (float) RATE;                        // 11.6 ms between frames
int HISTORY = 520;                                       // about 6 s of flux
int SPEC_W = 260;                                        // about 3 s of spectrogram
float[] flux = new float[HISTORY], thresh = new float[HISTORY];
boolean[] onset = new boolean[HISTORY];
float[][] spec = new float[SPEC_W][BINS];                // log spectra, a ring buffer
int specPos = 0;
float[] prevAt = new float[BINS], curAt = new float[BINS];   // the two frames compared at the last onset
float fluxAt = 0;
int written = 0, lastOnset = -100;
float delta = 1.0;                                       // threshold above the local median
Object lock = new Object();
boolean useMic = false, soundOn = true;
int pattern = 0;
String[] PATTERNS = {"kick + hi-hat", "kick only", "with gaps"};
String status = "Starting the sound…";
float[] R = new float[0];                                // R(τ) for every shift, in frames
int bestL = 0;
PImage specImage;

Slider tempo = new Slider(640, 34, 240, 60, 180, 120, "Tempo of the drums", "BPM");
Slider tau = new Slider(360, 578, 280, 0.1, 1.5, 0.37, "Shift τ", "s");
color BLUE = #2E75B6, ORANGE = #E36C0A, TEAL = #215968, INK = #262626, GREY = #8C8C8C;

void setup() {
  size(1200, 800);
  textFont(createFont("Arial", 14));
  specImage = createImage(SPEC_W, BINS, RGB);
  new Audio().start();
}

void draw() {
  background(255);
  float[] f, th, pa, ca;
  boolean[] on;
  float[][] sp = new float[SPEC_W][];
  int pos;
  synchronized (lock) {
    f = flux.clone();
    th = thresh.clone();
    on = onset.clone();
    pa = prevAt.clone();
    ca = curAt.clone();
    for (int i = 0; i < SPEC_W; i++) sp[i] = spec[i].clone();
    pos = specPos;
  }
  if (frameCount % 10 == 0) autocorrelation(f);

  drawControls();
  drawSpectrogram(sp, pos, 20, 100, 560, 220);
  drawCompare(pa, ca, 620, 100, 560, 220);
  drawFlux(f, th, on, 20, 360, 1160, 150);
  drawShifted(f, 20, 548, 660, 230);
  drawR(720, 548, 460, 230);
}

// ---- The panels ----

void panel(float x, float y, float w, float h, String title) {
  noFill();
  stroke(225);
  rect(x, y, w, h);
  fill(INK);
  textSize(14);
  text(title, x, y - 8);
}

void drawSpectrogram(float[][] sp, int pos, float x, float y, float w, float h) {
  panel(x, y, w, h, "1. A spectrum every 11.6 ms (time →, frequency ↑)");
  float top = 1e-6;
  for (float[] col : sp) for (float v : col) top = max(top, v);
  specImage.loadPixels();
  for (int i = 0; i < SPEC_W; i++) {
    float[] col = sp[(pos + i) % SPEC_W];          // oldest first
    for (int k = 0; k < BINS; k++)
      specImage.pixels[(BINS - 1 - k) * SPEC_W + i] = lerpColor(color(255), color(#1F497D), constrain(col[k] / top, 0, 1));
  }
  specImage.updatePixels();
  image(specImage, x, y, w, h);
  fill(GREY);
  textSize(11);
  text("5.5 kHz", x + 4, y + 12);
  text("0", x + 4, y + h - 4);
}

void drawCompare(float[] pa, float[] ca, float x, float y, float w, float h) {
  panel(x, y, w, h, "2. At the last onset: this frame vs the frame before");
  float top = 1e-6;
  for (int k = 0; k < BINS; k++) top = max(top, max(pa[k], ca[k]));
  float bw = w / BINS;
  noStroke();
  fill(ORANGE, 90);                                // the increases, which are added up
  for (int k = 0; k < BINS; k++)
    if (ca[k] > pa[k]) rect(x + k * bw, y + h - ca[k] / top * (h - 30), bw, (ca[k] - pa[k]) / top * (h - 30));
  spectrumLine(pa, x, y, w, h, top, GREY);
  spectrumLine(ca, x, y, w, h, top, ORANGE);
  fill(INK);
  textSize(13);
  text("flux = sum of the shaded increases = " + round(fluxAt), x + 10, y + 20);
  fill(GREY);
  textSize(11);
  text("grey: frame before   orange: this frame   frequency → 5.5 kHz", x + 10, y + h - 6);
}

void spectrumLine(float[] v, float x, float y, float w, float h, float top, color c) {
  noFill();
  stroke(c);
  strokeWeight(1.8);
  beginShape();
  for (int k = 0; k < BINS; k++) vertex(x + (k + 0.5) * w / BINS, y + h - v[k] / top * (h - 30));
  endShape();
  strokeWeight(1);
}

void drawFlux(float[] f, float[] th, boolean[] on, float x, float y, float w, float h) {
  panel(x, y, w, h, "3. Spectral flux over the last 6 s: peaks above the threshold (orange) are onsets   (Up/Down: threshold)");
  float top = 1e-6;
  for (float v : f) top = max(top, v);
  noFill();
  stroke(ORANGE);
  beginShape();
  for (int i = 0; i < HISTORY; i++) vertex(x + i * w / HISTORY, y + h - min(th[i] / top, 1) * (h - 10));
  endShape();
  stroke(BLUE);
  strokeWeight(1.6);
  beginShape();
  for (int i = 0; i < HISTORY; i++) vertex(x + i * w / HISTORY, y + h - f[i] / top * (h - 10));
  endShape();
  strokeWeight(1);
  noStroke();
  fill(ORANGE);
  for (int i = 0; i < HISTORY; i++) if (on[i]) circle(x + i * w / HISTORY, y + h - f[i] / top * (h - 10), 8);
}

void drawShifted(float[] f, float x, float y, float w, float h) {
  int L = round(tau.value / HOP_S);
  float[] z = normalised(f);
  float r = L < R.length ? R[L] : 0;
  panel(x, y, w, h, "4. The flux and a copy shifted by τ = " + nf(tau.value, 1, 2) + " s:  sum of products R(τ) = " + nf(r, 1, 2));
  int n = HISTORY - L;
  float top = 1e-6;
  for (float v : z) top = max(top, abs(v));
  plotLine(z, 0, n, x, y + 40, w, h - 50, top, BLUE);
  plotLine(z, L, n, x, y + 40, w, h - 50, top, ORANGE);
  fill(GREY);
  textSize(11);
  text("blue: flux   orange: the copy, moved back by τ", x + 300, y + h - 6);
}

void plotLine(float[] z, int from, int n, float x, float y, float w, float h, float top, color c) {
  noFill();
  stroke(c);
  strokeWeight(1.5);
  beginShape();
  for (int i = 0; i < n; i++) vertex(x + i * w / HISTORY, y + h / 2 - z[i + from] / top * h / 2);
  endShape();
  strokeWeight(1);
}

void drawR(float x, float y, float w, float h) {
  float bt = bestL * HOP_S;
  panel(x, y, w, h, R.length == 0 ? "5. R(τ) for every shift" : "5. R(τ) for every shift: best τ = " + nf(bt, 1, 3) + " s → 60 / τ = " + round(60 / bt) + " BPM");
  if (R.length == 0) return;
  float t0 = 0.1, t1 = 1.5;
  noStroke();
  fill(240);                                         // the range searched: 60-200 BPM
  rect(map(0.3, t0, t1, x, x + w), y, map(1.0, t0, t1, x, x + w) - map(0.3, t0, t1, x, x + w), h);
  stroke(220);
  float zy = y + h - (0 + 0.4) / 1.4 * h;            // R from -0.4 to 1
  line(x, zy, x + w, zy);
  noFill();
  stroke(TEAL);
  strokeWeight(2);
  beginShape();
  for (int L = round(t0 / HOP_S); L <= round(t1 / HOP_S) && L < R.length; L++)
    vertex(map(L * HOP_S, t0, t1, x, x + w), y + h - (constrain(R[L], -0.4, 1) + 0.4) / 1.4 * h);
  endShape();
  strokeWeight(1);
  int L = round(tau.value / HOP_S);
  noStroke();
  fill(ORANGE);
  if (L < R.length) circle(map(L * HOP_S, t0, t1, x, x + w), y + h - (constrain(R[L], -0.4, 1) + 0.4) / 1.4 * h, 11);
  fill(INK);
  circle(map(bestL * HOP_S, t0, t1, x, x + w), y + h - (constrain(R[bestL], -0.4, 1) + 0.4) / 1.4 * h, 8);
  fill(GREY);
  textSize(11);
  text("shaded: the range searched, 60–200 BPM.  τ from 0.1 to 1.5 s", x + 10, y + h - 6);
}

void drawControls() {
  fill(INK);
  textSize(18);
  text("Spectral flux and autocorrelation", 20, 30);
  button(20, 46, 190, "Sound: " + (useMic ? "microphone" : "drums made here"), useMic);
  button(220, 46, 120, "Play: " + (soundOn ? "on" : "off"), soundOn && !useMic);
  button(350, 46, 200, "Pattern: " + PATTERNS[pattern], false);
  tempo.display();
  fill(GREY);
  textSize(12);
  text(status, 900, 38);
  tau.display();
}

void button(float x, float y, float w, String label, boolean on) {
  stroke(170);
  fill(on ? color(80, 160, 90) : 245);
  rect(x, y, w, 26, 5);
  fill(on ? 255 : INK);
  textSize(13);
  textAlign(CENTER, CENTER);
  text(label, x + w / 2, y + 13);
  textAlign(LEFT, BASELINE);
}

boolean inside(float x, float y, float w, float h) {
  return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h;
}

void mousePressed() {
  if (inside(20, 46, 190, 26)) useMic = !useMic;
  if (inside(220, 46, 120, 26)) soundOn = !soundOn;
  if (inside(350, 46, 200, 26)) pattern = (pattern + 1) % PATTERNS.length;
  tempo.press();
  tau.press();
}

void mouseDragged() {
  tempo.drag();
  tau.drag();
}

void mouseReleased() {
  tempo.dragging = tau.dragging = false;
}

void keyPressed() {
  if (keyCode == UP) delta += 0.2;
  if (keyCode == DOWN) delta = max(0.2, delta - 0.2);
}

// ---- Autocorrelation ----

float[] normalised(float[] f) {
  float m = 0;
  for (float v : f) m += v / f.length;
  float[] z = new float[f.length];
  for (int i = 0; i < f.length; i++) z[i] = f[i] - m;
  return z;
}

// R(τ) = sum of z(t) · z(t + τ), divided by R(0) so that it is 1 with no shift
void autocorrelation(float[] f) {
  float[] z = normalised(f);
  int maxL = round(1.6 / HOP_S);
  float[] r = new float[maxL + 1];
  float e = 0;
  for (float v : z) e += v * v;
  if (e <= 0) return;
  for (int L = 0; L <= maxL; L++) {
    float s = 0;
    for (int i = 0; i + L < z.length; i++) s += z[i] * z[i + L];
    r[L] = s / e;
  }
  int best = round(0.3 / HOP_S);
  for (int L = round(0.3 / HOP_S); L <= round(1.0 / HOP_S); L++) if (r[L] > r[best]) best = L;
  R = r;
  bestL = best;
}

// ---- Sound: drums made here (and played), or the microphone; spectral flux of every frame ----

float drums(long sample) {
  float t = sample / (float) RATE, eighth = 30 / tempo.value;   // half a beat, in seconds
  int k = floor(t / eighth);
  float ph = t - k * eighth;                                     // seconds since this eighth began
  float v = 0.004 * random(-1, 1);
  if (k % 2 == 0 && ph < 0.2)                                     // a kick on every beat
    v += 0.8 * sin(TWO_PI * (55 + 90 * exp(-ph / 0.03)) * ph) * exp(-ph / 0.08);
  boolean hat = k % 2 == 1 && pattern != 1 && !(pattern == 2 && (k * 7919) % 5 == 0);
  if (hat && ph < 0.1) v += 0.3 * random(-1, 1) * exp(-ph / 0.015);   // a hi-hat between the beats
  return v;
}

class Audio extends Thread {
  Audio() {
    setDaemon(true);
  }
  public void run() {
    AudioFormat format = new AudioFormat(RATE, 16, 1, true, false);
    SourceDataLine out = null;
    TargetDataLine in = null;
    try {
      out = AudioSystem.getSourceDataLine(format);
      out.open(format, 4096);
      out.start();
    } catch (Exception e) {
      out = null;
    }
    byte[] buf = new byte[HOP * 2];
    float[] block = new float[HOP], frame = new float[N_FFT], window = new float[N_FFT], prev = new float[N_FFT / 2];
    float[] re = new float[N_FFT], im = new float[N_FFT], s1 = new float[BINS], s2 = new float[BINS];
    for (int i = 0; i < N_FFT; i++) window[i] = 0.5 - 0.5 * cos(TWO_PI * i / (N_FFT - 1));
    long sample = 0;
    while (true) {
      try {
        if (useMic) {
          if (in == null) {
            in = AudioSystem.getTargetDataLine(format);
            in.open(format, 8192);
            in.start();
          }
          status = "Listening to the microphone";
          int got = 0;
          while (got < buf.length) got += in.read(buf, got, buf.length - got);
          for (int i = 0; i < HOP; i++) block[i] = ((buf[2 * i + 1] << 8) | (buf[2 * i] & 0xff)) / 32768.0;
        } else {
          if (in != null) {
            in.close();
            in = null;
          }
          for (int i = 0; i < HOP; i++) block[i] = drums(sample++);
          if (out != null) {                         // play them; writing also keeps the pace of real time
            for (int i = 0; i < HOP; i++) {
              short v = (short) (constrain(block[i], -1, 1) * (soundOn ? 20000 : 0));
              buf[2 * i] = (byte) v;
              buf[2 * i + 1] = (byte) (v >> 8);
            }
            out.write(buf, 0, buf.length);
            status = soundOn ? "Playing the drums" : "Drums (muted)";
          } else {
            Thread.sleep(11);
            status = "Drums (no audio output)";
          }
        }
      } catch (Exception e) {
        status = "No microphone: " + e.getMessage();
        useMic = false;
        continue;
      }
      // The spectrum of the newest 1024 samples, and its flux against the frame before
      arrayCopy(frame, HOP, frame, 0, N_FFT - HOP);
      arrayCopy(block, 0, frame, N_FFT - HOP, HOP);
      for (int i = 0; i < N_FFT; i++) {
        re[i] = frame[i] * window[i];
        im[i] = 0;
      }
      fft(re, im);
      float sum = 0;
      float[] cur = new float[BINS];
      for (int k = 1; k < N_FFT / 2; k++) {
        float mag = log(1 + 100 * sqrt(re[k] * re[k] + im[k] * im[k]));
        sum += max(0, mag - prev[k]);                // only the increases count
        prev[k] = mag;
        if (k < BINS) cur[k] = mag;
      }
      synchronized (lock) {
        spec[specPos] = cur;
        specPos = (specPos + 1) % SPEC_W;
        if (addFlux(sum)) {                          // an onset one frame back: keep the two frames it compared
          arrayCopy(s2, prevAt);
          arrayCopy(s1, curAt);
        }
      }
      arrayCopy(s1, s2);
      arrayCopy(cur, s1);
    }
  }
}

// Store the new flux value and its threshold (local median + delta × mean); true if the previous value is a peak
boolean addFlux(float v) {
  arrayCopy(flux, 1, flux, 0, HISTORY - 1);
  arrayCopy(thresh, 1, thresh, 0, HISTORY - 1);
  arrayCopy(onset, 1, onset, 0, HISTORY - 1);
  flux[HISTORY - 1] = v;
  onset[HISTORY - 1] = false;
  written++;
  float[] recent = subset(flux, HISTORY - 16);
  float mean = 0;
  for (float r : recent) mean += r / recent.length;
  thresh[HISTORY - 1] = sort(recent)[8] + delta * mean;
  int c = HISTORY - 2;
  if (flux[c] > thresh[c] && flux[c] >= flux[c - 1] && flux[c] > flux[c + 1] && written - lastOnset > 4) {
    onset[c] = true;
    lastOnset = written;
    fluxAt = flux[c];
    return true;
  }
  return false;
}

// In-place radix-2 FFT
void fft(float[] re, float[] im) {
  int n = re.length;
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -TWO_PI / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < len / 2; k++) {
        float wr = cos(ang * k), wi = sin(ang * k);
        float xr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        float xi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k + len / 2] = re[i + k] - xr;
        im[i + k + len / 2] = im[i + k] - xi;
        re[i + k] += xr;
        im[i + k] += xi;
      }
    }
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
    textSize(13);
    text(name + ": " + (hi - lo > 10 ? str(round(value)) : nf(value, 1, 2)) + " " + unit, x, y - 10);
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
  }
}
