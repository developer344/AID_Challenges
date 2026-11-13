// Receives YOLO detections from yolo_osc.py and draws them: a box, a name and a confidence per object.
// Start yolo_osc.py first (see the comments at its top), then this sketch.
// Use the boxes in your interface: count people, follow a prop, trigger a sound when an object appears.
// Needs the oscP5 library (Sketch → Import Library → Manage Libraries → oscP5).

import oscP5.*;

OscP5 osc;
ArrayList<Detection> incoming = new ArrayList<Detection>(), shown = new ArrayList<Detection>();
int lastFrame = -10000;

class Detection {
  String name;
  float conf, x1, y1, x2, y2;              // corners from 0 to 1 of the camera image
}

void setup() {
  size(800, 600);
  textFont(createFont("Arial", 16));
  osc = new OscP5(this, 12000);
}

void draw() {
  background(255);
  ArrayList<Detection> boxes;
  synchronized (this) {
    boxes = new ArrayList<Detection>(shown);
  }
  for (Detection d : boxes) {
    noFill();
    stroke(#2E75B6);
    strokeWeight(3);
    rect(d.x1 * width, d.y1 * height, (d.x2 - d.x1) * width, (d.y2 - d.y1) * height);
    noStroke();
    fill(#2E75B6);
    text(d.name + " " + nf(d.conf, 1, 2), d.x1 * width + 4, d.y1 * height - 6);
  }
  fill(#262626);
  text(millis() - lastFrame > 2000 ? "Waiting for yolo_osc.py…" : boxes.size() + " objects", 10, 25);
}

void oscEvent(OscMessage m) {
  if (m.checkAddrPattern("/yolo/frame")) {
    synchronized (this) {
      shown = incoming;                    // the previous frame is complete
      incoming = new ArrayList<Detection>();
    }
    lastFrame = millis();
  } else if (m.checkAddrPattern("/yolo/box")) {
    Detection d = new Detection();
    d.name = m.get(0).stringValue();
    d.conf = m.get(1).floatValue();
    d.x1 = m.get(2).floatValue();
    d.y1 = m.get(3).floatValue();
    d.x2 = m.get(4).floatValue();
    d.y2 = m.get(5).floatValue();
    synchronized (this) {
      incoming.add(d);
    }
  }
}
