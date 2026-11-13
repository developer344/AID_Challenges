// Tap_Server: up to 30 people tap together over the network, and this sketch shows how synchronised they are.
// Each player runs Tap_Client (or sends OSC from any program) and taps with Space, any key or the mouse.
// Left: every player tapping is a dot on a circle at its phase. Each tap is phase 0 (the top); between taps the phase
// grows, clockwise, at that player's own tempo. The arrow is the average of all the dots: its length is Kuramoto's
// order parameter R (1: everyone taps together; near 0: no common beat) and its direction the group's phase.
// The spread is the same thing in milliseconds: how far apart the taps are, typically.
// Right: every tap on a time line (orange lines: the moments the group's phase passes 0, the group's own beat),
// and R over the last minute.
// Taps are timed on each client's clock and put on the server's clock: the server measures each clock's offset from
// the round trip of pings, like NTP, so the delay of the network does not blur the taps.
// M: a metronome (Up/Down: its tempo); then also R and the mean asynchrony to the beat, as in Tap_Asynchrony.
// S: 10 simulated tappers (no network needed), each at its own tempo and pulled towards the group's phase with a
// coupling K, as in Kuramoto's model (Left/Right: K; at 0 each keeps its own tempo). They also follow real players.
// Space: tap yourself. C: clear.
// Network: UDP port 12000 (OSC, no library needed); the server also says where it is on port 12001 every second, so
// clients on the same network find it by themselves. Some networks (eduroam, guest Wi-Fi) do not let laptops talk to
// each other: use a phone's hotspot or a small router.
//   to the server:   /tap/join id name          every 2 s (it also tells the server the player is still there)
//                    /tap/tap id time           time on the client's clock in ms (without it, the arrival time counts)
//                    /tap/pong id serverTime clientTime        /tap/leave id
//   to a client:     /tap/welcome number max    /tap/full    /tap/ping serverTime
//                    /tap/state R psi tapping bpm you number roundTrip      ten times a second (angles in degrees)
//   to everyone:     /tap/server port           broadcast every second, to port 12001

import javax.sound.sampled.*;
import java.util.*;

int PORT = 12000, FIND_PORT = 12001, MAX = 30;
Net net;
String netError = "";
Player[] players = new Player[MAX + 1];        // by number, 1 to 30
HashMap<Integer, Player> byId = new HashMap<Integer, Player>();
ArrayList<String> myAddresses = new ArrayList<String>();
ArrayList<InetAddress> broadcasts = new ArrayList<InetAddress>();

float R = 0, psi = 0, prevPsi = 0, groupPeriod = 600;
int tapping = 0;
float[] history = new float[600];               // R ten times a second, the last minute (−1: fewer than 2 tapping)
int historyN = 0;
ArrayList<Double> groupBeats = new ArrayList<Double>();    // when the group's phase passed 0
double lastState = 0, lastPing = 0, lastFind = -100000, lastAddresses = -100000, lastHistory = 0, lastFrame = 0;

boolean metronome = false;
float bpm = 100;
double beat0 = 0;                               // the time of the first beat
long nextBeat = 0;                              // the number of the next beat to schedule
ArrayList<Double> beats = new ArrayList<Double>();

float K = 0.2;                                  // the coupling of the simulated tappers
boolean spaceDown = false;
String message = "";
double messageTime = -100000;

color INK = #262626, GREY = #8C8C8C, LIGHT = #E6E6E6, BLUE = #2E75B6, ORANGE = #E36C0A;

void setup() {
  size(1280, 720);
  surface.setTitle("Tap_Server");
  textFont(createFont("Arial", 14));
  try {
    net = new Net(PORT);
  }
  catch (Exception e) {
    netError = "Port " + PORT + " is taken (another Tap_Server?): no network; S still gives simulated tappers.";
  }
  new AudioOut().start();
}

void draw() {
  double now = clock();
  float dt = (float) Math.min(now - lastFrame, 50);
  lastFrame = now;
  receive();
  scheduleBeats(now);
  simulate(now, dt);
  group(now);
  network(now);

  background(255);
  drawHeader();
  drawCircle(now);
  drawRaster(now);
  drawHistory();
  drawInfo(now);
}

// ---- The players ----

class Player {
  int id, number;
  String name;
  boolean simulated;
  InetSocketAddress address;                    // where to answer; null for the simulated ones and for Space
  double seen;                                  // when we last heard from it
  ArrayList<Double> taps = new ArrayList<Double>();       // tap times on our clock (ms), the last 40
  double flash = -100000;
  ArrayList<Float> asyncs = new ArrayList<Float>();       // with the metronome: tap minus the nearest beat (ms)
  double[] roundTrips = new double[20], offsets = new double[20];   // from the last 20 pings
  int pings = 0;
  float theta, omega;                           // simulated: its phase (rad) and its own speed (rad per ms)

  Player(int id, String name, int number) {
    this.id = id;
    this.name = name;
    this.number = number;
    seen = clock();
  }

  void tap(double t) {
    int k = taps.size();
    while (k > 0 && taps.get(k - 1) > t) k--;  // in time order, even if packets arrive out of order
    taps.add(k, t);
    if (taps.size() > 40) taps.remove(0);
    flash = clock();
    if (metronome) {
      double a = asynchrony(t);
      if (Math.abs(a) < beatPeriod() / 2) {
        asyncs.add((float) a);
        if (asyncs.size() > 20) asyncs.remove(0);
      }
    }
  }

  double last() {
    return taps.get(taps.size() - 1);
  }

  float period() {                              // the median of the last intervals between taps (ms)
    int k = min(taps.size() - 1, 6);
    if (k < 1) return 0;
    float[] iti = new float[k];
    for (int i = 0; i < k; i++) iti[i] = (float) (taps.get(taps.size() - 1 - i) - taps.get(taps.size() - 2 - i));
    iti = sort(iti);
    return iti[k / 2];
  }

  boolean tapping(double t) {                   // at least 3 taps, a sensible tempo, and not stopped
    float p = period();
    return taps.size() >= 3 && p > 150 && p < 2000 && t - last() < 2 * p;
  }

  float phase(double t) {                       // 0 at a tap, growing at its own tempo until the next
    return wrap(TWO_PI * (float) (t - last()) / period());
  }

  // A ping came back: its round trip, and the client's clock minus ours (assuming the way there and back take as long)
  void pong(double sent, double theirs, double now) {
    roundTrips[pings % 20] = now - sent;
    offsets[pings % 20] = theirs - (sent + now) / 2;
    pings++;
  }

  int best() {                                  // the ping with the shortest round trip: the least delayed
    int b = -1;
    for (int i = 0; i < min(pings, 20); i++) if (b < 0 || roundTrips[i] < roundTrips[b]) b = i;
    return b;
  }

  double offset() {
    return pings > 0 ? offsets[best()] : 0;
  }

  float roundTrip() {
    return pings > 0 ? (float) roundTrips[best()] : -1;
  }
}

Player join(int id, String name) {
  Player p = byId.get(id);
  if (p != null) {
    p.name = name;
    return p;
  }
  for (int k = 1; k <= MAX; k++) if (players[k] == null) {
    p = new Player(id, name, k);
    players[k] = p;
    byId.put(id, p);
    return p;
  }
  return null;                                  // full
}

void remove(Player p) {
  players[p.number] = null;
  byId.remove(p.id);
}

// ---- The network ----

boolean answer(Net n, Msg m) {                  // the server answers everything in draw()
  return false;
}

void receive() {
  if (net == null) return;
  Msg m;
  while ((m = net.inbox.poll()) != null) {
    if (m.is("/tap/join", 2)) {
      Player p = join(m.i(0), m.s(1));
      if (p == null) {
        net.send(new Msg("/tap/full", MAX), m.from);
        continue;
      }
      if (p.address == null) say(p.name + " joined as #" + p.number);
      p.address = m.from;
      p.seen = m.received;
      net.send(new Msg("/tap/welcome", p.number, MAX), m.from);
    } else if (m.is("/tap/tap", 1)) {
      Player p = byId.get(m.i(0));
      if (p == null) {                          // a program that taps without joining first
        p = join(m.i(0), "Player " + m.i(0));
        if (p == null) continue;
        p.address = m.from;
        say(p.name + " joined as #" + p.number);
      }
      p.seen = m.received;
      double t = m.received;
      if (m.args.length >= 2 && p.pings > 0) {
        double c = m.d(1) - p.offset();         // the client's time of the tap, on our clock
        if (Math.abs(c - m.received) < 1000) t = Math.min(c, m.received);
      }
      p.tap(t);
    } else if (m.is("/tap/pong", 3)) {
      Player p = byId.get(m.i(0));
      if (p != null) {
        p.pong(m.d(1), m.d(2), m.received);
        p.seen = m.received;
      }
    } else if (m.is("/tap/leave", 1)) {
      Player p = byId.get(m.i(0));
      if (p != null) {
        say(p.name + " left");
        remove(p);
      }
    }
  }
}

void network(double now) {
  if (net == null) return;
  if (now - lastAddresses > 5000) {
    findAddresses();
    lastAddresses = now;
  }
  if (now - lastFind > 1000) {                  // tell clients looking for the server where it is
    lastFind = now;
    for (InetAddress b : broadcasts) net.send(new Msg("/tap/server", PORT), new InetSocketAddress(b, FIND_PORT));
  }
  boolean ping = now - lastPing > 500, state = now - lastState > 100;
  if (ping) lastPing = now;
  if (state) lastState = now;
  for (Player p : players) {
    if (p == null || p.address == null) continue;
    if (now - p.seen > 10000) {
      say(p.name + " is gone");
      remove(p);
      continue;
    }
    if (ping) net.send(new Msg("/tap/ping", now), p.address);
    if (state) {
      float you = tapping >= 2 && p.tapping(now) ? degrees(wrap(p.phase(now) - psi + PI) - PI) : -1000;
      net.send(new Msg("/tap/state", R, degrees(psi), tapping, 60000 / groupPeriod, you, p.number, p.roundTrip()), p.address);
    }
  }
}

void findAddresses() {                          // this computer's addresses, and where to broadcast
  ArrayList<String> mine = new ArrayList<String>();
  ArrayList<InetAddress> casts = new ArrayList<InetAddress>();
  try {
    casts.add(InetAddress.getByName("127.0.0.1"));     // clients on this computer
    for (NetworkInterface ni : Collections.list(NetworkInterface.getNetworkInterfaces())) {
      if (!ni.isUp() || ni.isLoopback()) continue;
      for (InterfaceAddress ia : ni.getInterfaceAddresses()) {
        if (!(ia.getAddress() instanceof Inet4Address)) continue;
        mine.add(ia.getAddress().getHostAddress());
        if (ia.getBroadcast() != null) casts.add(ia.getBroadcast());
      }
    }
  }
  catch (Exception e) {
  }
  myAddresses = mine;
  broadcasts = casts;
}

// ---- The group: Kuramoto's order parameter from everyone's phase ----

void group(double now) {
  float c = 0, s = 0;
  FloatList periods = new FloatList();
  for (Player p : players) if (p != null && p.tapping(now)) {
    float ph = p.phase(now);
    c += cos(ph);
    s += sin(ph);
    periods.append(p.period());
  }
  int n = periods.size();
  tapping = n;
  R = n >= 2 ? sqrt(c * c + s * s) / n : 0;
  psi = wrap(atan2(s, c));
  if (n > 0) {
    periods.sort();
    groupPeriod = periods.get(n / 2);
  }
  if (n >= 2 && R > 0.3 && prevPsi > 1.5 * PI && psi < 0.5 * PI) groupBeats.add(now);   // the group's beat
  prevPsi = psi;
  while (groupBeats.size() > 0 && now - groupBeats.get(0) > 9000) groupBeats.remove(0);
  if (now - lastHistory >= 100) {
    lastHistory = now;
    history[historyN % history.length] = n >= 2 ? R : -1;
    historyN++;
  }
}

float spread() {                                // the circular standard deviation, in ms
  return R >= 0.999 ? 0 : sqrt(-2 * log(R)) / TWO_PI * groupPeriod;
}

// ---- Simulated tappers: Kuramoto oscillators that tap at phase 0 ----

void toggleSimulated() {
  boolean removed = false;
  for (Player p : players) if (p != null && p.simulated) {
    remove(p);
    removed = true;
  }
  if (removed) {
    say("Simulated tappers removed");
    return;
  }
  int added = 0;
  for (int k = 0; k < 10; k++) {
    Player p = join(-1 - k, "Simulated " + (k + 1));    // negative ids: never a client's
    if (p == null) break;
    p.simulated = true;
    p.omega = TWO_PI / (600 * (1 + 0.05 * randomGaussian()));   // about 100 BPM, each a little different
    p.theta = random(TWO_PI);
    added++;
  }
  say(added + " simulated tappers, each at its own tempo, pulled towards the group by K (Left/Right)");
}

void simulate(double now, float dt) {
  for (Player p : players) {
    if (p == null || !p.simulated) continue;
    float target = metronome ? beatPhase(now) : psi;    // the metronome, or the group
    float pull = metronome ? 1 : (tapping >= 2 ? R : 0);
    p.theta += dt * (p.omega + K * (TWO_PI / 600) * pull * sin(target - p.theta));
    if (p.theta >= TWO_PI) {
      p.theta -= TWO_PI;
      p.tap(now + 10 * randomGaussian());     // a tap, with 10 ms of motor noise
    }
  }
}

// ---- The metronome ----

float beatPeriod() {
  return 60000 / bpm;
}

float beatPhase(double t) {
  return wrap(TWO_PI * (float) ((t - beat0) / beatPeriod()));
}

double asynchrony(double t) {                   // tap minus the nearest beat (ms); NaN without beats
  double a = Double.NaN;
  for (double b : beats) if (Double.isNaN(a) || Math.abs(t - b) < Math.abs(a)) a = t - b;
  return a;
}

void startBeats(double from) {
  beat0 = from;
  nextBeat = 0;
}

void scheduleBeats(double now) {
  if (!metronome) return;
  while (beat0 + nextBeat * beatPeriod() < now + 150) {  // 150 ms ahead, so the sound is never late
    double b = beat0 + nextBeat * beatPeriod();
    beats.add(b);
    if (audio != null) synchronized (voices) {
      voices.add(new Voice(880, 0.5, audio.getLongFramePosition() + Math.round((b - now) * RATE / 1000)));
    }
    nextBeat++;
  }
  while (beats.size() > 0 && now - beats.get(0) > 9000) beats.remove(0);
}

float[] toBeat(double now) {                    // R of the taps against the beat, mean asynchrony (ms), taps
  float c = 0, s = 0, sum = 0;
  int n = 0;
  for (Player p : players) if (p != null) for (double t : p.taps) {
    double a = asynchrony(t);
    if (now - t < 10000 && Math.abs(a) < beatPeriod() / 2) {
      float th = TWO_PI * (float) a / beatPeriod();
      c += cos(th);
      s += sin(th);
      sum += a;
      n++;
    }
  }
  return new float[] {n > 0 ? sqrt(c * c + s * s) / n : 0, n > 0 ? sum / n : 0, n};
}

// ---- Keys ----

void keyPressed(KeyEvent e) {
  double now = clock();
  if (key == ' ') {
    if (spaceDown) return;                      // a key held down repeats: one tap only
    spaceDown = true;
    long age = System.currentTimeMillis() - e.getMillis();   // when the key was really pressed
    Player p = byId.get(0);
    if (p == null) p = join(0, "This computer");
    if (p != null) p.tap(now - (age >= 0 && age < 150 ? age : 0));
  }
  if (key == 'm' || key == 'M') {
    metronome = !metronome;
    beats.clear();
    if (metronome) startBeats(now + 300);
    for (Player p : players) if (p != null) p.asyncs.clear();
  }
  if (key == CODED && (keyCode == UP || keyCode == DOWN)) {
    double last = beats.size() > 0 ? beats.get(beats.size() - 1) : now;
    bpm = constrain(bpm + (keyCode == UP ? 5 : -5), 40, 200);
    if (metronome) startBeats(last + beatPeriod());
  }
  if (key == CODED && keyCode == RIGHT) K = min(1, K + 0.05);
  if (key == CODED && keyCode == LEFT) K = max(0, K - 0.05);
  if (key == 's' || key == 'S') toggleSimulated();
  if (key == 'c' || key == 'C') {
    for (Player p : players) if (p != null) {
      p.taps.clear();
      p.asyncs.clear();
    }
    historyN = 0;
    groupBeats.clear();
  }
}

void keyReleased() {
  if (key == ' ') spaceDown = false;
}

void say(String s) {
  message = s;
  messageTime = clock();
}

// ---- Drawing ----

void drawHeader() {
  textAlign(LEFT, BASELINE);
  fill(INK);
  textSize(24);
  text("Tap together", 20, 38);
  textSize(17);
  if (netError.length() > 0) {
    fill(ORANGE);
    text(netError, 190, 37);
  } else {
    text("Server: " + (myAddresses.isEmpty() ? "no network" : String.join("  or  ", myAddresses)) + "   (port " + PORT + ")", 190, 37);
    fill(GREY);
    textSize(13);
    text("Players run Tap_Client: on the same network it finds this server by itself; if not, type this address.", 190, 57);
  }
}

void drawCircle(double now) {
  float cx = 270, cy = 320, r = 200;
  noFill();
  stroke(LIGHT);
  strokeWeight(2);
  circle(cx, cy, 2 * r);
  stroke(GREY);
  line(cx, cy - r - 10, cx, cy - r + 10);
  noStroke();
  fill(GREY);
  textSize(12);
  textAlign(CENTER);
  text("a tap: phase 0  ·  time runs clockwise", cx, cy - r - 16);
  if (metronome) for (double b : beats) if (now >= b && now - b < 120) {    // the metronome's beat
    noFill();
    stroke(ORANGE);
    strokeWeight(6);
    circle(cx, cy - r, 40);
  }
  for (Player p : players) {
    if (p == null || !p.tapping(now)) continue;
    float a = p.phase(now), x = cx + r * sin(a), y = cy - r * cos(a);
    noStroke();
    fill(playerColour(p.number));
    circle(x, y, now - p.flash < 150 ? 32 : 22);
    fill(255);
    textSize(11);
    textAlign(CENTER, CENTER);
    text(p.number, x, y - 1);
  }
  if (tapping >= 2) {                           // the group's arrow: its length is R
    stroke(INK);
    strokeWeight(5);
    arrow(cx, cy, cx + R * r * sin(psi), cy - R * r * cos(psi));
    noStroke();
    fill(INK);
    circle(cx, cy, 10);
  }
  textAlign(CENTER, BASELINE);
  fill(INK);
  textSize(32);
  text(tapping >= 2 ? "R = " + nf(R, 1, 2) : "R: at least 2 people tapping", cx, cy + r + 54);
  textSize(16);
  fill(GREY);
  if (tapping >= 2) text(tapping + " tapping  ·  " + round(60000 / groupPeriod) + " BPM  ·  spread ±" + round(spread()) + " ms", cx, cy + r + 80);
  textAlign(LEFT, BASELINE);
}

void arrow(float x0, float y0, float x1, float y1) {
  line(x0, y0, x1, y1);
  pushMatrix();
  translate(x1, y1);
  rotate(atan2(y1 - y0, x1 - x0));
  noStroke();
  fill(INK);
  triangle(4, 0, -14, -9, -14, 9);
  popMatrix();
}

void drawRaster(double now) {
  int lx = 520, x0 = 680, x1 = 1255, y0 = 100, y1 = 470;
  float span = 8000;
  ArrayList<Player> rows = new ArrayList<Player>();
  for (Player p : players) if (p != null) rows.add(p);
  float h = rows.isEmpty() ? 0 : min(26, (y1 - y0) / (float) rows.size());
  fill(INK);
  textSize(16);
  text("Every tap, the last 8 seconds", lx, 76);
  fill(GREY);
  textSize(11);
  textAlign(RIGHT);
  text("BPM", x0 - 8, 95);
  textAlign(LEFT);
  float bottom = y0 + h * rows.size();
  strokeWeight(1);
  stroke(215);
  for (double b : beats) if (b <= now && now - b < span) {
    float x = x1 - (float) (now - b) / span * (x1 - x0);
    line(x, y0, x, bottom);
  }
  stroke(ORANGE, 110);
  strokeWeight(2);
  for (double b : groupBeats) if (now - b < span) {
    float x = x1 - (float) (now - b) / span * (x1 - x0);
    line(x, y0, x, bottom);
  }
  for (int i = 0; i < rows.size(); i++) {
    Player p = rows.get(i);
    float y = y0 + i * h;
    color c = playerColour(p.number);
    if (i % 2 == 0) {
      noStroke();
      fill(0, 10);
      rect(x0, y, x1 - x0, h);
    }
    textSize(constrain(h * 0.6, 9, 14));
    textAlign(LEFT, CENTER);
    fill(c);
    text("#" + p.number + " " + (p.name.length() > 14 ? p.name.substring(0, 14) : p.name), lx, y + h / 2);
    textAlign(RIGHT, CENTER);
    fill(GREY);
    if (p.taps.size() >= 3) text(round(60000 / p.period()), x0 - 8, y + h / 2);
    stroke(c);
    strokeWeight(3);
    for (double t : p.taps) if (t <= now && now - t < span) {
      float x = x1 - (float) (now - t) / span * (x1 - x0);
      line(x, y + 2, x, y + h - 2);
    }
  }
  textAlign(CENTER, BASELINE);
  fill(GREY);
  textSize(11);
  for (int sec = 0; sec <= 8; sec += 2) text(sec == 0 ? "now" : "−" + sec + " s", x1 - sec / 8.0 * (x1 - x0), y1 + 14);
  textAlign(LEFT, BASELINE);
}

void drawHistory() {
  int x0 = 680, x1 = 1255, y0 = 525, y1 = 645;
  fill(INK);
  textSize(16);
  text("R over the last minute", 520, y0 - 12);
  textSize(11);
  textAlign(RIGHT, CENTER);
  for (int k = 0; k <= 2; k++) {
    float y = y1 - k / 2.0 * (y1 - y0);
    stroke(LIGHT);
    strokeWeight(1);
    line(x0, y, x1, y);
    fill(GREY);
    text(k == 0 ? "0" : k == 1 ? "0.5" : "1", x0 - 8, y);
  }
  textAlign(CENTER, BASELINE);
  text("−60 s", x0, y1 + 16);
  text("now", x1, y1 + 16);
  textAlign(LEFT, BASELINE);
  int n = min(historyN, history.length);
  stroke(BLUE);
  strokeWeight(2.5);
  noFill();
  boolean drawing = false;
  for (int k = 0; k < n; k++) {
    float v = history[(historyN - n + k) % history.length];
    float x = x1 - (n - 1 - k) / (float) (history.length - 1) * (x1 - x0), y = y1 - v * (y1 - y0);
    if (v < 0) {
      if (drawing) endShape();
      drawing = false;
      continue;
    }
    if (!drawing) beginShape();
    drawing = true;
    vertex(x, y);
  }
  if (drawing) endShape();
}

void drawInfo(double now) {
  float y = 640;
  textSize(15);
  if (metronome) {
    float[] b = toBeat(now);
    fill(INK);
    text("Metronome " + round(bpm) + " BPM" + (b[2] >= 4 ? ":  to the beat, R = " + nf(b[0], 1, 2) + ", mean asynchrony " + round(b[1]) + " ms" : ""), 20, y);
    y += 22;
  }
  boolean sims = false;
  for (Player p : players) if (p != null && p.simulated) sims = true;
  if (sims) {
    fill(INK);
    text("Simulated tappers: coupling K = " + nf(K, 1, 2) + "  (Left/Right; 0: each at its own tempo)", 20, y);
    y += 22;
  }
  if (now - messageTime < 4000) {
    fill(BLUE);
    text(message, 20, y);
  }
  fill(GREY);
  textSize(13);
  text("Space: tap yourself    M: metronome    Up/Down: its tempo    S: 10 simulated tappers    Left/Right: their coupling K    C: clear", 20, height - 12);
}

// ---- Sound for the metronome: short beeps, mixed in a thread of their own (as in Tap_Asynchrony) ----

int RATE = 44100;
ArrayList<Voice> voices = new ArrayList<Voice>();
SourceDataLine audio;
long written = 0;                               // samples sent to the audio device so far

class Voice {
  float freq, amp;
  long start;                                   // the sample at which it sounds
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
    }
    catch (Exception e) {
    }
  }
}
