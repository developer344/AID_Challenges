// Tap_Client: tap together with the whole class; Tap_Server shows how synchronised you all are.
// A tap: Space, any key, or a click (or a touch, on a touch screen) anywhere but the two fields.
// Each tap is sent by OSC with its time on this computer's clock. The server measures this clock against its own,
// from the round trip of pings (like NTP), so the delay of the network does not change when your tap counts.
// Type your name and, if the server is not found by itself, its address (at the top of the server's window): click a
// field, type, Enter. Both are remembered for next time (settings.txt, next to this sketch).
// Right: you and the group, as the server sees you. The arrow is the group's phase, always drawn up; its length is R
// (1 when everyone taps together). Your dot is your phase against the group's: at the top you tap with the group,
// to the right you are ahead (early), to the left behind (late).
// Network: messages to the server on UDP port 12000; the server says where it is on port 12001 (see Tap_Server).

int PORT = 12000, FIND_PORT = 12001;
Net net;
int id = 1 + new java.util.Random().nextInt(2000000000);   // who we are, for the server
String name = "", server = "", note = "";
InetSocketAddress serverAddress;
int editing = -1;                               // the field being typed in: 0 the name, 1 the server
int number = 0;                                 // our number at the server, 1 to 30
boolean full = false, keyDown = false;
double lastHeard = -100000, lastJoin = -100000, flash = -100000, chosen = -100000;
float R = 0, groupBpm = 0, you = -1000, roundTrip = -1;
int tapping = 0, seq = 0;
ArrayList<Double> taps = new ArrayList<Double>();
int[][] FIELDS = {{80, 16, 180, 32}, {350, 16, 230, 32}};    // x, y, width, height

color INK = #262626, GREY = #8C8C8C, LIGHT = #E6E6E6, BLUE = #2E75B6, ORANGE = #E36C0A, GREEN = #3A8B2E;

void setup() {
  size(600, 440);
  textFont(createFont("Arial", 14));
  if (new File(sketchPath("settings.txt")).exists()) {
    String[] saved = loadStrings("settings.txt");
    if (saved.length >= 2) {
      name = saved[0];
      server = saved[1];
    }
  }
  if (name.length() == 0) name = "Player " + (1 + id % 99);
  surface.setTitle("Tap_Client: " + name);
  try {
    net = new Net(0);                           // any free port: the server answers to it
  }
  catch (Exception e) {
    note = "No network: " + e.getMessage();
  }
  new Finder().start();
  setServer(server, PORT);
}

void draw() {
  double now = clock();
  receive();
  if (net != null && serverAddress != null && now - lastJoin > 2000) {     // join, and keep saying we are here
    net.send(new Msg("/tap/join", id, name), serverAddress);
    lastJoin = now;
  }
  background(255);
  drawFields();
  drawStatus(now);
  drawTap(now);
  drawGroup(now);
  fill(GREY);
  textSize(12);
  text("Tap: Space, any key or a click.   Name and server: click the field, type, Enter.", 20, height - 12);
}

// ---- Tapping ----

void tap(double t) {
  taps.add(t);
  if (taps.size() > 30) taps.remove(0);
  flash = clock();
  seq++;
  if (net != null && serverAddress != null) net.send(new Msg("/tap/tap", id, t, seq), serverAddress);
}

double eventTime(processing.event.Event e) {    // when the key or the button was really pressed, on clock()
  long age = System.currentTimeMillis() - e.getMillis();
  return clock() - (age >= 0 && age < 150 ? age : 0);
}

float myPeriod() {                              // the median of the last intervals between our taps (ms)
  int k = min(taps.size() - 1, 6);
  float[] iti = new float[k];
  for (int i = 0; i < k; i++) iti[i] = (float) (taps.get(taps.size() - 1 - i) - taps.get(taps.size() - 2 - i));
  iti = sort(iti);
  return iti[k / 2];
}

void keyPressed(KeyEvent e) {
  if (editing >= 0) {
    edit();
    return;
  }
  if (keyDown) return;                          // a key held down repeats: one tap only
  keyDown = true;
  tap(eventTime(e));
}

void keyReleased() {
  keyDown = false;
}

void mousePressed(MouseEvent e) {
  for (int f = 0; f < 2; f++) {
    int[] r = FIELDS[f];
    if (mouseX >= r[0] && mouseX <= r[0] + r[2] && mouseY >= r[1] && mouseY <= r[1] + r[3]) {
      if (editing >= 0 && editing != f) finishEditing();
      editing = f;
      return;
    }
  }
  if (editing >= 0) {                           // a click outside the fields ends the typing (it is not a tap)
    finishEditing();
    return;
  }
  tap(eventTime(e));
}

// ---- The two fields ----

void edit() {
  String s = editing == 0 ? name : server;
  if (key == ENTER || key == RETURN || key == TAB) {
    finishEditing();
    return;
  }
  if (key == ESC) {
    key = 0;                                    // ESC would close the sketch
    finishEditing();
    return;
  }
  if (key == BACKSPACE) s = s.substring(0, max(0, s.length() - 1));
  else if (key != CODED && key >= 32 && s.length() < 24) s += key;
  if (editing == 0) name = s;
  else server = s;
}

void finishEditing() {
  if (editing == 0) {
    name = name.trim().length() > 0 ? name.trim() : "Player";
    surface.setTitle("Tap_Client: " + name);
    saveSettings();
    lastJoin = -100000;                         // tell the server the new name at once
  }
  if (editing == 1) setServer(server, PORT);
  editing = -1;
}

void setServer(String s, int port) {
  server = s.trim();
  serverAddress = null;
  number = 0;
  full = false;
  lastHeard = -100000;
  note = "";
  if (server.length() > 0) {
    try {
      serverAddress = new InetSocketAddress(InetAddress.getByName(server), port);
    }
    catch (Exception e) {
      note = "Unknown address: " + server;
    }
  }
  lastJoin = -100000;
  saveSettings();
}

void saveSettings() {
  saveStrings("settings.txt", new String[] {name, server});
}

// ---- The network ----

boolean answer(Net n, Msg m) {                  // pings are answered at once, in the network's thread
  if (m.is("/tap/ping", 1)) {
    n.send(new Msg("/tap/pong", id, m.d(0), m.received), m.from);
    return true;
  }
  return false;
}

void receive() {
  if (net == null) return;
  Msg m;
  while ((m = net.inbox.poll()) != null) {
    if (m.is("/tap/server", 1)) {               // a server saying where it is
      boolean lost = serverAddress == null || clock() - lastHeard > 3000;
      String ip = m.from.getAddress().getHostAddress();
      if (lost && editing != 1 && clock() - chosen > 2500 && !(ip.equals(server) && serverAddress != null)) {
        setServer(ip, m.i(0));                  // try it; another one only if it does not answer within 2.5 s
        chosen = clock();
      }
    } else if (m.is("/tap/welcome", 1)) {
      number = m.i(0);
      full = false;
      lastHeard = m.received;
    } else if (m.is("/tap/full", 0)) {
      full = true;
      lastHeard = m.received;
    } else if (m.is("/tap/state", 7)) {
      R = (float) m.d(0);
      tapping = m.i(2);
      groupBpm = (float) m.d(3);
      you = (float) m.d(4);
      number = m.i(5);
      roundTrip = (float) m.d(6);
      lastHeard = m.received;
    }
  }
}

class Finder extends Thread {                   // listens for servers saying where they are (port 12001)
  Finder() {
    setDaemon(true);
  }
  public void run() {
    try {
      DatagramSocket s = new DatagramSocket(null);
      s.setReuseAddress(true);                  // several clients on one computer can all listen
      s.bind(new InetSocketAddress(FIND_PORT));
      byte[] buf = new byte[512];
      while (true) {
        DatagramPacket p = new DatagramPacket(buf, buf.length);
        s.receive(p);
        Msg m = decode(p.getData(), p.getLength());
        if (m != null && m.is("/tap/server", 1) && net != null) {
          m.from = (InetSocketAddress) p.getSocketAddress();
          m.received = clock();
          net.inbox.add(m);
        }
      }
    }
    catch (Exception e) {
    }
  }
}

void exit() {
  if (net != null && serverAddress != null) net.send(new Msg("/tap/leave", id), serverAddress);
  super.exit();
}

// ---- Drawing ----

void drawFields() {
  textSize(14);
  textAlign(LEFT, CENTER);
  fill(INK);
  text("Name", 20, 32);
  text("Server", 290, 32);
  for (int f = 0; f < 2; f++) {
    int[] r = FIELDS[f];
    stroke(editing == f ? BLUE : 200);
    strokeWeight(editing == f ? 2 : 1);
    fill(editing == f ? 255 : 248);
    rect(r[0], r[1], r[2], r[3], 4);
    String s = f == 0 ? name : server;
    if (f == 1 && s.length() == 0 && editing != 1) {
      fill(GREY);
      text("found by itself, or type it", r[0] + 8, r[1] + r[3] / 2);
    } else {
      fill(INK);
      text(s + (editing == f && frameCount / 30 % 2 == 0 ? "|" : ""), r[0] + 8, r[1] + r[3] / 2);
    }
  }
  textAlign(LEFT, BASELINE);
}

void drawStatus(double now) {
  String s;
  color c;
  if (net == null) {
    s = note;
    c = ORANGE;
  } else if (serverAddress == null) {
    s = note.length() > 0 ? note : "Looking for the server on this network... or type its address above.";
    c = note.length() > 0 ? ORANGE : GREY;
  } else if (full) {
    s = "The server is full.";
    c = ORANGE;
  } else if (now - lastHeard < 3000) {
    s = "Connected to " + server + ": you are #" + number + (roundTrip >= 0 ? "   (round trip " + round(roundTrip) + " ms)" : "");
    c = GREEN;
  } else {
    s = "No answer from " + server + " yet. Is Tap_Server running, on the same network?";
    c = ORANGE;
  }
  textSize(14);
  fill(c);
  text(s, 20, 76);
}

void drawTap(double now) {
  float x = 160, y = 245, r = 110;
  boolean just = now - flash < 120;
  noStroke();
  fill(just ? BLUE : 236);
  circle(x, y, 2 * r);
  textAlign(CENTER, CENTER);
  fill(just ? 255 : INK);
  textSize(36);
  text("TAP", x, y - 8);
  textSize(12);
  fill(just ? 255 : GREY);
  text("Space, any key or a click", x, y + 26);
  textAlign(CENTER, BASELINE);
  textSize(15);
  fill(INK);
  if (taps.size() >= 4) text("Your tempo: " + round(60000 / myPeriod()) + " BPM", x, y + r + 34);
  textAlign(LEFT, BASELINE);
}

void drawGroup(double now) {
  float x = 445, y = 235, r = 95;
  boolean live = serverAddress != null && !full && now - lastHeard < 3000;
  noFill();
  stroke(LIGHT);
  strokeWeight(2);
  circle(x, y, 2 * r);
  stroke(GREY);
  line(x, y - r - 8, x, y - r + 8);
  noStroke();
  fill(GREY);
  textSize(11);
  textAlign(CENTER);
  text("with the group", x, y - r - 14);
  text("ahead (early)", x + r * 0.78 + 26, y - r * 0.78 - 2);
  text("behind (late)", x - r * 0.78 - 26, y - r * 0.78 - 2);
  if (live && tapping >= 2) {
    stroke(INK);                                // the group: always up, its length is R
    strokeWeight(4);
    line(x, y, x, y - R * r);
    noStroke();
    fill(INK);
    triangle(x, y - R * r - 4, x - 8, y - R * r + 12, x + 8, y - R * r + 12);
    circle(x, y, 8);
    if (you > -999) {                           // you, against the group
      float a = radians(you), dx = x + r * sin(a), dy = y - r * cos(a);
      fill(playerColour(number));
      circle(dx, dy, now - flash < 120 ? 30 : 22);
      fill(255);
      textAlign(CENTER, CENTER);
      text(number, dx, dy - 1);
    }
  }
  textAlign(CENTER, BASELINE);
  textSize(15);
  fill(INK);
  if (live) {
    text(tapping >= 2 ? "Group: R = " + nf(R, 1, 2) + "  ·  " + tapping + " tapping  ·  " + round(groupBpm) + " BPM" : "Waiting for at least 2 people to tap", x, y + r + 34);
    if (tapping >= 2 && you > -999) {
      float ms = -you / 360 * 60000 / groupBpm;        // ahead of the group = early = negative
      text("You: " + (ms > 0 ? "+" : "") + round(ms) + " ms " + (ms < 0 ? "(early)" : "(late)"), x, y + r + 56);
    }
  }
  textAlign(LEFT, BASELINE);
}
