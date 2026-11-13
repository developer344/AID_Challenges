// Tapping with a metronome: press Space on each beat.
// Each tap is compared with the nearest beat: the asynchrony (tap minus beat, in ms) is plotted.
// Most people tap a little early (negative mean): anticipation. The SD shows how steady you are.
// Turn the metronome off (M) and keep tapping: your own tempo is estimated from the intervals between taps.
// Keys: Space taps, Up/Down change the tempo, M turns the metronome on and off,
// D adds output delay (raise it with Bluetooth headphones until the beep falls on the flash), C clears.
// Sound uses Java's own audio, so no library is needed.

import javax.sound.sampled.*;

int RATE = 44100;
float bpm = 100;
boolean metronome = true, audioOK = true;
int[] DELAYS = {0, 50, 100, 150, 200, 250, 300};
int delayIndex = 0;
long nextBeat = -1;                       // sample at which the next beat sounds
ArrayList<Long> beats = new ArrayList<Long>();
ArrayList<Float> asyncs = new ArrayList<Float>();
ArrayList<Long> taps = new ArrayList<Long>();
int flashBeat = -1000, flashTap = -1000;

color BLUE = #2E75B6, ORANGE = #E36C0A, INK = #262626, GREY = #AAAAAA;

void setup() {
  size(900, 520);
  textFont(createFont("Arial", 14));
  new AudioOut().start();
}

// The audio clock in samples (or a clock from millis if there is no audio)
long now() {
  return audio != null ? audio.getLongFramePosition() : (long) (millis() / 1000.0 * RATE);
}

void draw() {
  if (audio == null && audioOK && millis() < 3000) return;   // wait for the audio clock to start
  long t = now();
  long period = round(60 / bpm * RATE);
  if (nextBeat < 0) nextBeat = t + RATE / 2;
  // Schedule beats 150 ms ahead so the sound is never late
  while (nextBeat < t + 0.15 * RATE) {
    if (metronome) {
      long heard = nextBeat + DELAYS[delayIndex] * RATE / 1000;
      synchronized (beats) {
        beats.add(heard);
        if (beats.size() > 50) beats.remove(0);
      }
      if (audio != null) synchronized (voices) {
        voices.add(new Voice(880, 0.5, nextBeat));
      }
    }
    nextBeat += period;
  }
  // Flash when a beat is heard
  synchronized (beats) {
    for (long b : beats) if (abs(t - b) < RATE / 60) flashBeat = frameCount;
  }

  background(255);
  noStroke();
  fill(frameCount - flashBeat < 6 ? ORANGE : 235);
  circle(120, 130, 120);
  fill(frameCount - flashTap < 6 ? BLUE : 235);
  circle(120, 290, 80);
  fill(INK);
  textSize(14);
  textAlign(CENTER);
  text(metronome ? "Beat" : "Metronome off", 120, 205);
  text("Your tap", 120, 350);
  textAlign(LEFT);

  // Asynchronies
  int x0 = 260, y0 = 60, w = 600, h = 300;
  stroke(220);
  line(x0, y0 + h / 2, x0 + w, y0 + h / 2);
  fill(GREY);
  textSize(12);
  text("+150 ms (late)", x0 + w - 85, y0 - 5);
  text("−150 ms (early)", x0 + w - 90, y0 + h + 15);
  text("0", x0 - 15, y0 + h / 2 + 4);
  int n = asyncs.size(), first = max(0, n - 40);
  noStroke();
  fill(BLUE);
  for (int i = first; i < n; i++) {
    float x = x0 + (i - first + 0.5) * w / 40.0, y = y0 + h / 2 - constrain(asyncs.get(i), -150, 150) / 150 * h / 2;   // late = up
    circle(x, y, 9);
  }

  // Statistics
  fill(INK);
  textSize(16);
  text("Metronome: " + round(bpm) + " BPM" + (metronome ? "" : " (off)") + "    output delay: " + DELAYS[delayIndex] + " ms", 260, 410);
  if (n >= 4) {
    float m = 0, s = 0;
    int k = min(n, 20);
    for (int i = n - k; i < n; i++) m += asyncs.get(i) / k;
    for (int i = n - k; i < n; i++) s += sq(asyncs.get(i) - m) / (k - 1);
    text("Last " + k + " taps: mean asynchrony " + round(m) + " ms, SD " + round(sqrt(s)) + " ms" + (m < 0 ? "  (early: you anticipate)" : ""), 260, 440);
  }
  if (taps.size() >= 4) text("Your tempo from the last taps: " + round(tapTempo()) + " BPM", 260, 470);
  fill(GREY);
  textSize(13);
  text("Space: tap   Up/Down: tempo   M: metronome on/off   D: output delay   C: clear" + (audioOK ? "" : "   (no audio device)"), 20, 505);
}

void keyPressed() {
  if (key == ' ') {
    long t = now();
    flashTap = frameCount;
    taps.add(t);
    if (taps.size() > 30) taps.remove(0);
    if (metronome) {
      long nearest = -1;
      synchronized (beats) {
        for (long b : beats) if (nearest < 0 || abs(b - t) < abs(nearest - t)) nearest = b;
      }
      float ms = (t - nearest) * 1000.0 / RATE;
      if (abs(ms) < 30000 / bpm) asyncs.add(ms);     // only within half a beat
    }
  }
  if (keyCode == UP) bpm = min(200, bpm + 5);
  if (keyCode == DOWN) bpm = max(40, bpm - 5);
  if (key == 'm' || key == 'M') metronome = !metronome;
  if (key == 'd' || key == 'D') delayIndex = (delayIndex + 1) % DELAYS.length;
  if (key == 'c' || key == 'C') {
    asyncs.clear();
    taps.clear();
  }
}

// Tempo from the median interval between the last taps
float tapTempo() {
  int k = min(taps.size() - 1, 12);
  float[] iti = new float[k];
  for (int i = 0; i < k; i++) iti[i] = (taps.get(taps.size() - 1 - i) - taps.get(taps.size() - 2 - i)) / (float) RATE;
  iti = sort(iti);
  return 60 / iti[k / 2];
}

// ---- Sound: short sine beeps, mixed in a thread of their own ----

ArrayList<Voice> voices = new ArrayList<Voice>();
SourceDataLine audio;
long written = 0;                         // samples sent to the audio device so far

class Voice {
  float freq, amp;
  long start;                             // the sample at which it sounds
  int age = 0;
  Voice(float freq, float amp, long start) {
    this.freq = freq;
    this.amp = amp;
    this.start = start;
  }
}

void mix(byte[] buf) {
  synchronized (voices) {
    for (int n = 0; n < buf.length / 2; n++) {
      double sum = 0;
      for (Voice v : voices) {
        if (written + n < v.start) continue;
        double t = v.age++ / (double) RATE;
        sum += v.amp * Math.min(t / 0.002, 1) * Math.exp(-t / 0.04) * Math.sin(TWO_PI * v.freq * t);
      }
      short out = (short) (32000 * Math.tanh(sum));
      buf[2 * n] = (byte) out;
      buf[2 * n + 1] = (byte) (out >> 8);
    }
    written += buf.length / 2;
    for (int i = voices.size() - 1; i >= 0; i--) if (voices.get(i).age > RATE / 4) voices.remove(i);
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
    }
  }
}
