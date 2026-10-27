/*
  Visual / auditory reaction time: StandardFirmata driven with raw Serial.write
  ------------------------------------------------------------------------------
  Arduino: StandardFirmata uploaded (File > Examples > Firmata > StandardFirmata).

  Wiring:
    Button  : D2 -> GND (internal pull-up, pressed = HIGH in this configuration)
    LED     : D4 -> 220 ohm -> LED anode, cathode -> GND
    Buzzer  : D6 -> active buzzer (+), buzzer (-) -> GND (beeps when pin is HIGH)

  Keys:  SPACE = start | L = test LED | B = test buzzer (before starting)
*/

import processing.serial.*;

// ---------- settings ----------
String participant = "P01";
int TRIALS_PER_TYPE   = 10;       // 10 visual, 10 auditory
int FOREPERIOD_MIN    = 1000;     // ms, random wait before the stimulus
int FOREPERIOD_MAX    = 4000;
int RESPONSE_WINDOW   = 2000;     // ms, max time to respond
int ITI               = 800;      // ms, pause between trials
int SERIAL_INDEX      = 3;        // index of the Arduino in the printed port list
int BAUD              = 57600;    // must match Firmata.begin(...) in StandardFirmata

final int BTN = 2, LED = 4, BUZ = 6;

// ---------- Firmata protocol bytes ----------
final int SET_PIN_MODE     = 0xF4;
final int SET_DIGITAL_PIN  = 0xF5;
final int REPORT_DIGITAL   = 0xD0;
final int DIGITAL_MESSAGE  = 0x90;
final int MODE_OUTPUT      = 1;
final int MODE_PULLUP      = 0x0B;

// ---------- state ----------
Serial port;
ArrayList<Character> trials = new ArrayList<Character>();
Table table;
FloatList visualRTs   = new FloatList();   // valid reaction times (ms), kept in memory for the statistics
FloatList auditoryRTs = new FloatList();

final int IDLE = 0, PAUSE = 1, FORE = 2, STIM = 3, DONE = 4;
int state = IDLE;
int idx = 0;
long stateStartMs;
int foreperiodMs;
long tStim;                          // nanoTime at stimulus onset
volatile long lastPressNs = 0;       // nanoTime when the press arrived
volatile int btnLevel = -1;          // -1 = no data, 1 = pressed, 0 = released
String status = "Press SPACE to start";
String summary = "";
boolean ledTest = false, buzTest = false;

// parser state for incoming Firmata messages
int cmd = 0, parsePort = 0, parseCount = 0, lsb = 0;

void setup() {
  size(600, 360);
  println(Serial.list());
  port = new Serial(this, Serial.list()[SERIAL_INDEX], BAUD);
  delay(2500);                       // Arduino resets when the port opens

  setPinMode(BTN, MODE_PULLUP);
  setPinMode(LED, MODE_OUTPUT);
  setPinMode(BUZ, MODE_OUTPUT);      // Active buzzer: plain digital on/off

  port.write(REPORT_DIGITAL | 0);
  port.write(1);
  stimulusOff();

  table = new Table();
  table.addColumn("participant");
  table.addColumn("trial");
  table.addColumn("stimulus");
  table.addColumn("foreperiod_ms");
  table.addColumn("rt_ms");
  table.addColumn("outcome");

  for (int i = 0; i < TRIALS_PER_TYPE; i++) { trials.add('V'); trials.add('A'); }
  java.util.Collections.shuffle(trials);   // randomized order

  textSize(20);
}

// ---------- low-level Firmata writes ----------
void setPinMode(int pin, int mode) {
  port.write(SET_PIN_MODE);
  port.write(pin);
  port.write(mode);
}

void setPin(int pin, int value) {
  port.write(SET_DIGITAL_PIN);
  port.write(pin);
  port.write(value);
}

// ---------- incoming data ----------
void serialEvent(Serial p) {
  long t = System.nanoTime();
  while (p.available() > 0) {
    int b = p.read();
    if (b >= 0x80) {
      if ((b & 0xF0) == DIGITAL_MESSAGE) { cmd = DIGITAL_MESSAGE; parsePort = b & 0x0F; parseCount = 0; }
      else cmd = 0;
    } else if (cmd == DIGITAL_MESSAGE) {
      if (parseCount == 0) { lsb = b; parseCount = 1; }
      else {
        parseCount = 0;
        if (parsePort == 0) {
          int level = (lsb >> BTN) & 1;
          if (btnLevel == 0 && level == 1) lastPressNs = t;   // Button pressed
          btnLevel = level;
        }
      }
    }
  }
}

void draw() {
  background(30);
  fill(255);
  textAlign(CENTER, CENTER);
  text(status, width/2, height/2 - 40);

  // Live hardware check
  textSize(14);
  String b = (btnLevel == -1) ? "NO DATA from Arduino" : (btnLevel == 1 ? "PRESSED" : "released");
  text("Button D2: " + b + "   |   test keys: L = LED, B = buzzer", width/2, height - 25);
  textSize(20);

  if (state != DONE) {
    text("Trial " + min(idx + 1, trials.size()) + " / " + trials.size(), width/2, height/2);
  } else {
    textSize(16);
    text(summary, width/2, height/2 + 60);
    textSize(20);
  }

  long now = millis();

  if (state == PAUSE && now - stateStartMs > ITI) {
    foreperiodMs = int(random(FOREPERIOD_MIN, FOREPERIOD_MAX));
    stateStartMs = now;
    lastPressNs = 0;
    state = FORE;
    status = "Get ready...";
  }
  else if (state == FORE) {
    if (lastPressNs != 0) {
      logTrial("", "false_start");
      status = "Too early!";
    } else if (now - stateStartMs >= foreperiodMs) {
      stimulusOn(trials.get(idx));
      tStim = System.nanoTime();
      stateStartMs = now;
      state = STIM;
    }
  }
  else if (state == STIM) {
    if (lastPressNs > tStim) {
      float rt = (lastPressNs - tStim) / 1000000.0;
      stimulusOff();

      // keep the number in memory for the statistics
      if (trials.get(idx) == 'V') visualRTs.append(rt);
      else                        auditoryRTs.append(rt);

      // always write the CSV value with a decimal point, whatever the computer's language setting
      logTrial(String.format(java.util.Locale.US, "%.2f", rt), "ok");
      status = nf(rt, 0, 1) + " ms";
    } else if (now - stateStartMs > RESPONSE_WINDOW) {
      stimulusOff();
      logTrial("", "timeout");
      status = "Too slow";
    }
  }
}

void stimulusOn(char type) {
  if (type == 'V') {
    setPin(LED, 1);
  } else {
    setPin(BUZ, 1);
  }
}

void stimulusOff() {
  setPin(LED, 0);
  setPin(BUZ, 0);
}

void logTrial(String rt, String outcome) {
  TableRow row = table.addRow();
  row.setString("participant", participant);
  row.setInt("trial", idx + 1);
  row.setString("stimulus", trials.get(idx) == 'V' ? "visual" : "auditory");
  row.setInt("foreperiod_ms", foreperiodMs);
  row.setString("rt_ms", rt);
  row.setString("outcome", outcome);

  idx++;
  stateStartMs = millis();
  state = PAUSE;

  if (idx >= trials.size()) finish();
}

void finish() {
  state = DONE;
  stimulusOff();
  String file = "data/" + participant + "_reaction_times_" + year() + nf(month(), 2) + nf(day(), 2)
              + "_" + nf(hour(), 2) + nf(minute(), 2) + ".csv";
  saveTable(table, file);
  summary = "Visual:   " + describe("visual", visualRTs) + "\nAuditory: " + describe("auditory", auditoryRTs);
  println(summary);
  status = "Done! Data saved.";
}

// Median and mean of the valid reaction times; errors = false starts + timeouts
String describe(String stim, FloatList rts) {
  int bad = 0;
  for (TableRow r : table.rows()) {
    if (r.getString("stimulus").equals(stim) && !r.getString("outcome").equals("ok")) bad++;
  }
  if (rts.size() == 0) return "no valid trials, errors=" + bad;

  FloatList sorted = rts.copy();
  sorted.sort();
  int n = sorted.size();
  float median = (n % 2 == 1) ? sorted.get(n/2) : (sorted.get(n/2 - 1) + sorted.get(n/2)) / 2.0;
  float mean = rts.sum() / n;
  return "median " + nf(median, 0, 1) + " ms, mean " + nf(mean, 0, 1) + " ms, n=" + n + ", errors=" + bad;
}

void keyPressed() {
  // Start the experiment on SPACE press
  if (key == ' ' && state == IDLE) {
    stimulusOff();
    state = PAUSE;
    stateStartMs = millis();
  }
}

void keyReleased() {
  // Use keyReleased for testing to prevent auto-repeat stuttering
  if (state == IDLE) {
    if (key == 'l' || key == 'L') {
      ledTest = !ledTest;
      setPin(LED, ledTest ? 1 : 0);
    }
    if (key == 'b' || key == 'B') {
      buzTest = !buzTest;
      setPin(BUZ, buzTest ? 1 : 0);
    }
  }
}
