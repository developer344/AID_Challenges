// Onsets and tempo from the microphone: clap, tap the table or play some music near the computer.
// Every 11.6 ms the sketch takes the spectrum of the last 23 ms of sound and adds up how much each
// frequency grew since the previous one: the spectral flux (top). New sounds make it jump.
// A peak above a moving threshold (orange line) is an onset: the dot flashes.
// The tempo comes from the autocorrelation of the flux over the last 6 s (bottom): it peaks at the beat
// period. Half or double the true tempo can also win: an octave error.
// Keys: Up/Down raise or lower the threshold.
// Uses Java's own audio input, so no library is needed. macOS asks for microphone permission
// the first time (for Processing, or for the app if you export it).

import javax.sound.sampled.*;

int RATE = 44100, N_FFT = 1024, HOP = 512;
float HOP_S = HOP / (float) RATE;
int HISTORY = 520;                         // about 6 s of flux
float[] flux = new float[HISTORY], thresh = new float[HISTORY];
boolean[] onset = new boolean[HISTORY];
int written = 0;                           // flux values computed so far
float delta = 1.0;                         // threshold above the local median
int lastOnsetFrame = -100, flashFrame = -100;
String status = "Starting the microphone…";
float[] ac = new float[0];
float tempo = 0;

color BLUE = #2E75B6, ORANGE = #E36C0A, TEAL = #215968, INK = #262626, GREY = #AAAAAA;

void setup() {
  size(900, 560);
  textFont(createFont("Arial", 14));
  new MicIn().start();
}

void draw() {
  background(255);
  float[] f, th;
  boolean[] on;
  synchronized (flux) {
    f = flux.clone();
    th = thresh.clone();
    on = onset.clone();
  }
  if (frameCount % 30 == 0) estimateTempo(f);

  // Onset curve
  int x0 = 40, y0 = 50, w = 820, h = 220;
  float top = 1e-6;
  for (float v : f) top = max(top, v);
  noFill();
  stroke(BLUE);
  beginShape();
  for (int i = 0; i < HISTORY; i++) vertex(x0 + i * w / (float) HISTORY, y0 + h - f[i] / top * h);
  endShape();
  stroke(ORANGE);
  beginShape();
  for (int i = 0; i < HISTORY; i++) vertex(x0 + i * w / (float) HISTORY, y0 + h - min(th[i] / top, 1) * h);
  endShape();
  noStroke();
  fill(ORANGE);
  for (int i = 0; i < HISTORY; i++) if (on[i]) circle(x0 + i * w / (float) HISTORY, y0 + h - f[i] / top * h, 8);
  fill(INK);
  text("Spectral flux (last 6 s), threshold and onsets", x0, y0 - 15);
  fill(frameCount - flashFrame < 5 ? ORANGE : 230);
  circle(840, 30, 26);

  // Tempo
  int ty = 330, th2 = 160;
  fill(INK);
  text("Autocorrelation of the flux against tempo", x0, ty - 15);
  stroke(220);
  line(x0, ty + th2, x0 + w, ty + th2);
  if (ac.length > 0) {
    noFill();
    stroke(TEAL);
    beginShape();
    for (int b = 60; b <= 200; b++) vertex(map(b, 60, 200, x0, x0 + w), ty + th2 - max(ac[b], 0) * th2);
    endShape();
    fill(GREY);
    for (int b = 60; b <= 200; b += 20) text(b, map(b, 60, 200, x0, x0 + w) - 10, ty + th2 + 18);
    if (tempo > 0) {
      fill(ORANGE);
      noStroke();
      circle(map(tempo, 60, 200, x0, x0 + w), ty + th2 - max(ac[round(tempo)], 0) * th2, 10);
    }
  }
  fill(INK);
  textSize(20);
  text(tempo > 0 ? "Tempo ≈ " + round(tempo) + " BPM" : "Tempo: play something rhythmic", x0, 540);
  textSize(13);
  fill(GREY);
  text(status + "    threshold +" + nf(delta, 1, 1) + " (Up/Down)", 380, 540);
}

void keyPressed() {
  if (keyCode == UP) delta += 0.2;
  if (keyCode == DOWN) delta = max(0.2, delta - 0.2);
}

// Autocorrelation of the mean-removed flux, for every whole tempo from 60 to 200 BPM
void estimateTempo(float[] f) {
  float m = 0;
  for (float v : f) m += v / f.length;
  float[] z = new float[f.length];
  float energy = 0;
  for (int i = 0; i < f.length; i++) {
    z[i] = f[i] - m;
    energy += z[i] * z[i];
  }
  if (energy <= 0) return;
  ac = new float[201];
  float best = 0;
  tempo = 0;
  for (int b = 60; b <= 200; b++) {
    float lag = 60.0 / b / HOP_S;
    int l = floor(lag);
    float frac = lag - l, s = 0;
    for (int i = l + 1; i < z.length; i++) s += z[i] * (z[i - l] * (1 - frac) + z[i - l - 1] * frac);
    ac[b] = s / energy;
    if (ac[b] > best) {
      best = ac[b];
      tempo = b;
    }
  }
  if (best < 0.1) tempo = 0;
}

// ---- Microphone: spectral flux in a thread of its own ----

class MicIn extends Thread {
  MicIn() {
    setDaemon(true);
  }
  public void run() {
    try {
      AudioFormat format = new AudioFormat(RATE, 16, 1, true, false);
      TargetDataLine line = AudioSystem.getTargetDataLine(format);
      line.open(format, 8192);
      line.start();
      status = "Listening";
      byte[] buf = new byte[HOP * 2];
      float[] frame = new float[N_FFT], window = new float[N_FFT], prev = new float[N_FFT / 2];
      for (int i = 0; i < N_FFT; i++) window[i] = 0.5 - 0.5 * cos(TWO_PI * i / (N_FFT - 1));
      float[] re = new float[N_FFT], im = new float[N_FFT];
      while (true) {
        int got = 0;
        while (got < buf.length) got += line.read(buf, got, buf.length - got);
        arrayCopy(frame, HOP, frame, 0, N_FFT - HOP);
        for (int i = 0; i < HOP; i++) frame[N_FFT - HOP + i] = ((buf[2 * i + 1] << 8) | (buf[2 * i] & 0xff)) / 32768.0;
        for (int i = 0; i < N_FFT; i++) {
          re[i] = frame[i] * window[i];
          im[i] = 0;
        }
        fft(re, im);
        float sum = 0;
        for (int k = 1; k < N_FFT / 2; k++) {
          float mag = log(1 + 100 * sqrt(re[k] * re[k] + im[k] * im[k]));
          sum += max(0, mag - prev[k]);
          prev[k] = mag;
        }
        addFlux(sum);
      }
    } catch (Exception e) {
      status = "No microphone: " + e.getMessage();
    }
  }
}

// Store the new flux value, compute the threshold (local median + delta) and mark a peak 1 frame back
void addFlux(float v) {
  synchronized (flux) {
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
    int c = HISTORY - 2;                   // the previous value: is it a peak?
    if (flux[c] > thresh[c] && flux[c] >= flux[c - 1] && flux[c] > flux[c + 1] && written - lastOnsetFrame > 4) {
      onset[c] = true;
      lastOnsetFrame = written;
      flashFrame = frameCount;
    }
  }
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
