// Kuramoto model: N oscillators, each with its own natural frequency, pulled towards the others.
// Left: the oscillators on a circle, with the order parameter r as an arrow from the centre.
// Right: the waves sin(theta) of some oscillators, and sliders for the particles, the waves and K.
// Space shows or hides the r arrow and picks new random particles for the waves.
// Each wave drawn beeps when its particle passes theta = 0: in sync, the beeps fall together.
// Beeps are scheduled ahead to cover the audio delay. With Bluetooth, raise the Delay button
// until the beep falls on the flash at theta = 0.
// Sound uses Java's own audio, so no library is needed.

import javax.sound.sampled.*;

int N = 30;
float K = 1.5;
float[] theta = new float[0], omega = new float[0];

int HISTORY = 400;                       // one sample per frame: about 6.7 s of waves
int MAX_WAVES = 20;
int waves = 2;                           // waves drawn
int samples = 0;
int[] order = new int[0];                // all the particles; the first ones are the waves drawn
boolean showR = true;
boolean soundOn = true, audioOK = true;
float[] SPEEDS = {0.25, 0.5, 1, 2, 4};
int speedIndex = 2;
int[] DELAYS = {0, 50, 100, 150, 200, 250, 300, 400};   // ms of output delay: 0 for speakers, more for Bluetooth
int delayIndex = 0;
int BX1 = 30, BX2 = 175, BX3 = 320, BY = 522, BW = 135, BH = 28;   // the buttons

int CX = 240, CY = 290, RADIUS = 190;    // the circle
int PX = 500, PY = 70, PW = 400, PH = 240;   // the wave plot
int SY_N = 360, SY_W = 425, SY_K = 490;  // the sliders, as wide as the plot
int MIN_N = 2, MAX_N = 200;
float[][] wave = new float[MAX_N][HISTORY];   // kept for all, so a new selection shows its past
int[] born = new int[MAX_N];             // when each oscillator appeared
boolean[] scheduled = new boolean[MAX_N];   // its beep for this turn is already planned
int[] flash = new int[MAX_N];            // the frame it last passed theta = 0
float MAX_K = 3, KC = 0.8;               // Kc is about 1.6 times the spread of omega, here 0.5
int active = 0;                          // the slider being dragged: 1 particles, 2 K, 3 waves

color ORANGE = #E36C0A, GREY = #AAAAAA, INK = #262626, GOLD = #FFC933;
color[] COLORS = {#2E75B6, #E36C0A, #1F9E89, #C2185B};   // the first waves; the rest are grey

void setup() {
  size(940, 560);
  setN(N);
  new AudioOut().start();
}

void setN(int n) {
  int old = theta.length;
  theta = expand(theta, n);
  omega = expand(omega, n);
  for (int i = old; i < n; i++) {
    theta[i] = random(TWO_PI);
    omega[i] = 2 + 0.5 * randomGaussian();   // rad/s
    born[i] = samples;
    scheduled[i] = false;
    flash[i] = -100;
  }
  // Keep the selection: drop the particles removed, add the new ones at the end
  int[] next = new int[n];
  int m = 0;
  for (int i : order) if (i < n) next[m++] = i;
  for (int i = old; i < n; i++) next[m++] = i;
  order = next;
  N = n;
}

void shuffleWaves() {
  for (int i = N - 1; i > 0; i--) {
    int j = int(random(i + 1));
    int t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
}

void draw() {
  float dt = SPEEDS[speedIndex] / 60.0;
  float[] speed = new float[N];
  for (int i = 0; i < N; i++) {
    float sum = 0;
    for (int j = 0; j < N; j++)
      sum += sin(theta[j] - theta[i]);
    speed[i] = omega[i] + K / N * sum;
  }
  for (int i = 0; i < N; i++) {
    theta[i] += speed[i] * dt;
    if (theta[i] >= TWO_PI) {     // keep theta in one turn, and note who passed theta = 0
      theta[i] -= TWO_PI;
      scheduled[i] = false;
      flash[i] = frameCount;
    } else if (theta[i] < 0) {
      theta[i] += TWO_PI;
      scheduled[i] = false;
    }
  }
  if (soundOn && audio != null) {
    // Plan each beep just early enough to sound when the particle reaches theta = 0
    float ahead = audio.getBufferSize() / 2.0 / RATE + DELAYS[delayIndex] / 1000.0 + 0.05;
    for (int p = 0; p < min(waves, N); p++) {
      int i = order[p];
      float rate = speed[i] * dt * frameRate;   // radians per real second
      if (!scheduled[i] && rate > 0) {
        float t = (TWO_PI - theta[i]) / rate;
        if (t < ahead) {
          beep(p, min(waves, N), t);
          scheduled[i] = true;
        }
      }
    }
  }

  for (int i = 0; i < N; i++) wave[i][samples % HISTORY] = sin(theta[i]);
  samples++;

  background(255);
  drawCircle();
  drawWaves();
  drawSlider(SY_N, "Particles: N = " + N, (N - MIN_N) / float(MAX_N - MIN_N));
  drawSlider(SY_W, "Waves drawn: " + min(waves, N), (waves - 1) / float(MAX_WAVES - 1));
  drawSlider(SY_K, "Coupling: K = " + nf(K, 1, 2), K / MAX_K);
  stroke(ORANGE);
  float kx = PX + PW * KC / MAX_K;
  line(kx, SY_K - 8, kx, SY_K + 8);
  noStroke();
  fill(INK);
  textSize(13);
  textAlign(CENTER, TOP);
  text("Kc ≈ 0.8", kx, SY_K + 12);
  drawButton(BX1, "Speed: " + SPEEDS[speedIndex] + "x");
  drawButton(BX2, !audioOK ? "No sound device" : soundOn ? "Sound: on" : "Sound: off");
  drawButton(BX3, "Delay: " + DELAYS[delayIndex] + " ms");
}

void drawCircle() {
  // r exp(i psi) = (1/N) sum of exp(i theta)
  float re = 0, im = 0;
  for (int i = 0; i < N; i++) {
    re += cos(theta[i]);
    im += sin(theta[i]);
  }
  re /= N;
  im /= N;
  float r = sqrt(re * re + im * im);

  // Axes: x = cos(theta), y = sin(theta), upwards as in the wave plot. The circle is at 1.
  stroke(225);
  strokeWeight(1);
  line(CX - RADIUS, CY, CX + RADIUS, CY);
  line(CX, CY - RADIUS, CX, CY + RADIUS);
  noStroke();
  fill(110);
  textSize(13);
  textAlign(RIGHT, BOTTOM);
  text("cos θ", CX + RADIUS - 8, CY - 2);
  textAlign(LEFT, TOP);
  text("sin θ", CX + 5, CY - RADIUS + 8);
  // theta = 0, where each cycle starts and the beeps are
  stroke(GOLD);
  strokeWeight(7);
  line(CX, CY, CX + RADIUS, CY);
  stroke(INK);
  strokeWeight(3);
  line(CX + RADIUS - 9, CY, CX + RADIUS + 9, CY);
  strokeWeight(1);
  noStroke();
  fill(INK);
  textAlign(LEFT, CENTER);
  text("θ = 0\nstart", CX + RADIUS + 13, CY);
  fill(110);
  textAlign(RIGHT, CENTER);
  text("π", CX - RADIUS - 12, CY);
  textAlign(CENTER, BOTTOM);
  text("π/2", CX, CY - RADIUS - 10);
  textAlign(CENTER, TOP);
  text("3π/2", CX, CY + RADIUS + 3);

  noFill();
  stroke(GREY);
  strokeWeight(1);
  circle(CX, CY, 2 * RADIUS);

  // The oscillators of the coloured waves go on top
  noStroke();
  fill(GREY);
  int coloured = min(min(waves, N), COLORS.length);
  for (int p = coloured; p < N; p++) circle(CX + RADIUS * cos(theta[order[p]]), CY - RADIUS * sin(theta[order[p]]), 11);
  for (int p = coloured - 1; p >= 0; p--) {
    fill(COLORS[p]);
    float dx = CX + RADIUS * cos(theta[order[p]]), dy = CY - RADIUS * sin(theta[order[p]]);
    circle(dx, dy, 18);
    int age = frameCount - flash[order[p]];
    if (age < 12) {      // a ring that opens as the particle passes theta = 0
      noFill();
      stroke(COLORS[p]);
      strokeWeight(3);
      circle(dx, dy, 20 + 3 * age);
      strokeWeight(1);
      noStroke();
    }
  }

  // The r vector: length r times the radius, direction the mean phase
  if (showR) {
    float x = CX + RADIUS * re, y = CY - RADIUS * im;
    stroke(INK);
    strokeWeight(4);
    line(CX, CY, x, y);
    strokeWeight(1);
    noStroke();
    fill(INK);
    circle(CX, CY, 8);
    pushMatrix();
    translate(x, y);
    rotate(atan2(-im, re));
    triangle(8, 0, -8, -7, -8, 7);
    popMatrix();
  }

  fill(INK);
  textSize(22);
  textAlign(CENTER, TOP);
  text("r = " + nf(r, 1, 2), CX, 30);
  fill(110);
  textSize(13);
  text("Space: new wave particles, show or hide the r arrow", CX, CY + RADIUS + 23);
}

void drawWaves() {
  fill(INK);
  noStroke();
  textSize(14);
  textAlign(LEFT, BASELINE);
  int drawn = min(waves, N);
  text("sin(θ) of " + drawn + (drawn == 1 ? " particle" : " particles") + ", last " + nf(HISTORY / 60.0 * SPEEDS[speedIndex], 1, 1) + " s; ω in rad/s", PX, PY - 34);
  textSize(13);
  for (int p = 0; p < min(drawn, COLORS.length); p++) {
    fill(COLORS[p]);
    circle(PX + 5 + p * 100, PY - 14, 10);
    fill(INK);
    text("ω = " + nf(omega[order[p]], 1, 2), PX + 13 + p * 100, PY - 10);
  }

  noFill();
  stroke(225);
  rect(PX, PY, PW, PH);
  stroke(GOLD);                  // sin(theta) = 0: going up through it is the start of a cycle
  strokeWeight(2);
  line(PX, PY + PH / 2, PX + PW, PY + PH / 2);
  strokeWeight(1);

  for (int p = drawn - 1; p >= 0; p--) {
    stroke(p < COLORS.length ? COLORS[p] : GREY);
    strokeWeight(p < COLORS.length ? 2 : 1);
    beginShape();
    int count = min(samples - born[order[p]], HISTORY);
    for (int k = 0; k < count; k++) {
      float v = wave[order[p]][(samples - count + k) % HISTORY];
      vertex(PX + PW - count + k, PY + PH / 2 - v * (PH / 2 - 8));
    }
    endShape();
    if (p < COLORS.length) {     // a dot at each start of a cycle
      noStroke();
      fill(COLORS[p]);
      for (int k = 1; k < count; k++) {
        float a = wave[order[p]][(samples - count + k - 1) % HISTORY], b = wave[order[p]][(samples - count + k) % HISTORY];
        if (a < 0 && b >= 0) circle(PX + PW - count + k, PY + PH / 2, 9);
      }
      noFill();
    }
  }
  strokeWeight(1);
}

void drawSlider(int y, String label, float t) {
  fill(INK);
  noStroke();
  textSize(15);
  textAlign(LEFT, BASELINE);
  text(label, PX, y - 16);
  stroke(GREY);
  strokeWeight(3);
  line(PX, y, PX + PW, y);
  strokeWeight(1);
  noStroke();
  fill(INK);
  circle(PX + PW * t, y, 18);
}

void drawButton(int x, String label) {
  stroke(GREY);
  fill(245);
  rect(x, BY, BW, BH, 6);
  noStroke();
  fill(INK);
  textSize(14);
  textAlign(CENTER, CENTER);
  text(label, x + BW / 2, BY + BH / 2 - 1);
}

boolean inButton(int x) {
  return mouseX >= x && mouseX <= x + BW && mouseY >= BY && mouseY <= BY + BH;
}

void mousePressed() {
  active = 0;
  if (inButton(BX1)) speedIndex = (speedIndex + 1) % SPEEDS.length;
  if (inButton(BX2)) {
    soundOn = !soundOn;
    synchronized (voices) {
      voices.clear();
    }
  }
  if (inButton(BX3)) delayIndex = (delayIndex + 1) % DELAYS.length;
  if (mouseX > PX - 12 && mouseX < PX + PW + 12) {
    if (abs(mouseY - SY_N) < 16) active = 1;
    if (abs(mouseY - SY_K) < 16) active = 2;
    if (abs(mouseY - SY_W) < 16) active = 3;
  }
  mouseDragged();
}

void mouseDragged() {
  float t = constrain((mouseX - PX) / float(PW), 0, 1);
  if (active == 1) setN(round(lerp(MIN_N, MAX_N, t)));
  if (active == 2) K = t * MAX_K;
  if (active == 3) waves = round(lerp(1, MAX_WAVES, t));
}

void mouseReleased() {
  active = 0;
}

void keyPressed() {
  if (key == ' ') {
    showR = !showR;
    shuffleWaves();
  }
}

// ---- Sound: short sine beeps, mixed in a thread of their own ----

int RATE = 44100;
float[] NOTES = {0, 4, 7, 9, 2, 12, 16, 19, 21, 14};   // semitones above C5: a pentatonic scale, so any mix sounds pleasant
ArrayList<Voice> voices = new ArrayList<Voice>();
SourceDataLine audio;
long written = 0;                 // samples sent to the audio device so far

class Voice {
  float freq, amp;
  long start;                     // the sample at which it sounds
  int age = 0;
  Voice(float freq, float amp, long start) {
    this.freq = freq;
    this.amp = amp;
    this.start = start;
  }
}

// A beep to be heard t seconds from now
void beep(int slot, int drawn, float t) {
  float freq = 523.25 * pow(2, NOTES[slot % NOTES.length] / 12);
  long start = audio.getLongFramePosition() + round((t - DELAYS[delayIndex] / 1000.0) * RATE);
  synchronized (voices) {
    if (voices.size() < 40) voices.add(new Voice(freq, 0.45 / sqrt(drawn), start));
  }
}

// Fills a buffer of 16-bit samples: 5 ms attack, then a gentle decay
void mix(byte[] buf) {
  synchronized (voices) {
    for (int n = 0; n < buf.length / 2; n++) {
      double sum = 0;
      for (Voice v : voices) {
        if (written + n < v.start) continue;
        double t = v.age++ / (double) RATE;
        sum += v.amp * Math.min(t / 0.005, 1) * Math.exp(-t / 0.08) * Math.sin(TWO_PI * v.freq * t);
      }
      short out = (short) (32000 * Math.tanh(sum));   // tanh: many beeps together never clip
      buf[2 * n] = (byte) out;
      buf[2 * n + 1] = (byte) (out >> 8);
    }
    written += buf.length / 2;
    for (int i = voices.size() - 1; i >= 0; i--) {
      if (voices.get(i).age > RATE / 2) voices.remove(i);
    }
  }
}

class AudioOut extends Thread {
  AudioOut() {
    setDaemon(true);
  }
  public void run() {
    try {
      AudioFormat format = new AudioFormat(RATE, 16, 1, true, false);
      SourceDataLine line = AudioSystem.getSourceDataLine(format);
      line.open(format, 4096);
      line.start();
      audio = line;
      byte[] buf = new byte[1024];
      while (true) {
        mix(buf);
        audio.write(buf, 0, buf.length);
      }
    } catch (Exception e) {
      audioOK = false;
      soundOn = false;
    }
  }
}
