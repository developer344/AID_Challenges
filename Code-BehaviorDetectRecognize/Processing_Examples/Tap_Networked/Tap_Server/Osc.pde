// ---- OSC over UDP with no library (the same tab is in Tap_Server and Tap_Client) ----
// An OSC message is an address ("/tap/tap"), a list of type tags (",id": an int and a double) and the values,
// each piece padded with zeros to a multiple of 4 bytes, numbers big-endian.

import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentLinkedQueue;

long T0 = System.nanoTime();

// This computer's clock, in milliseconds since the sketch started (with fractions of a millisecond)
double clock() {
  return (System.nanoTime() - T0) / (double) 1000000;
}

class Msg {
  String address;
  Object[] args;
  InetSocketAddress from;               // who sent it
  double received;                      // when it arrived, on clock()

  Msg(String address, Object... args) {
    this.address = address;
    this.args = args;
  }
  boolean is(String a, int n) {         // this address, with at least n values
    return address.equals(a) && args.length >= n;
  }
  int i(int k) {
    return ((Number) args[k]).intValue();
  }
  double d(int k) {
    return ((Number) args[k]).doubleValue();
  }
  String s(int k) {
    return args[k].toString();
  }
}

byte[] encode(Msg m) {
  ByteBuffer b = ByteBuffer.allocate(1024);
  putString(b, m.address);
  String tags = ",";
  for (Object a : m.args) tags += a instanceof Integer ? "i" : a instanceof Float ? "f" : a instanceof Double ? "d" : "s";
  putString(b, tags);
  for (Object a : m.args) {
    if (a instanceof Integer) b.putInt((Integer) a);
    else if (a instanceof Float) b.putFloat((Float) a);
    else if (a instanceof Double) b.putDouble((Double) a);
    else putString(b, a.toString());
  }
  return java.util.Arrays.copyOf(b.array(), b.position());
}

void putString(ByteBuffer b, String s) {
  byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
  b.put(bytes);
  for (int k = bytes.length % 4; k < 4; k++) b.put((byte) 0);     // 1 to 4 zeros
}

String getString(ByteBuffer b) {
  int start = b.position();
  while (b.get() != 0);
  String s = new String(b.array(), start, b.position() - 1 - start, StandardCharsets.UTF_8);
  b.position((b.position() + 3) & ~3);                               // to the next multiple of 4
  return s;
}

// A message from the bytes of a packet, or null if it is not one we understand
Msg decode(byte[] data, int length) {
  try {
    ByteBuffer b = ByteBuffer.wrap(data, 0, length);
    String address = getString(b);
    if (!address.startsWith("/")) return null;
    String tags = b.hasRemaining() ? getString(b) : ",";
    if (!tags.startsWith(",")) return null;
    Object[] args = new Object[tags.length() - 1];
    for (int k = 1; k < tags.length(); k++) {
      char t = tags.charAt(k);
      if (t == 'i') args[k - 1] = b.getInt();
      else if (t == 'f') args[k - 1] = b.getFloat();
      else if (t == 'd') args[k - 1] = b.getDouble();
      else if (t == 'h') args[k - 1] = b.getLong();
      else if (t == 's') args[k - 1] = getString(b);
      else return null;
    }
    return new Msg(address, args);
  }
  catch (Exception e) {
    return null;                        // too short, or not OSC
  }
}

// A UDP socket that receives in a thread of its own; messages wait in the inbox for draw().
// Messages that need an answer at once (a ping) can be answered in the thread: see answer() in the sketch.
class Net extends Thread {
  DatagramSocket socket;
  ConcurrentLinkedQueue<Msg> inbox = new ConcurrentLinkedQueue<Msg>();

  Net(int port) throws SocketException {
    socket = port > 0 ? new DatagramSocket(port) : new DatagramSocket();
    socket.setBroadcast(true);
    setDaemon(true);
    start();
  }
  public void run() {
    byte[] buf = new byte[2048];
    while (!socket.isClosed()) {
      try {
        DatagramPacket p = new DatagramPacket(buf, buf.length);
        socket.receive(p);
        Msg m = decode(p.getData(), p.getLength());
        if (m == null) continue;
        m.received = clock();
        m.from = (InetSocketAddress) p.getSocketAddress();
        if (!answer(this, m)) inbox.add(m);
      }
      catch (Exception e) {
      }
    }
  }
  void send(Msg m, SocketAddress to) {
    try {
      byte[] data = encode(m);
      socket.send(new DatagramPacket(data, data.length, to));
    }
    catch (Exception e) {
    }
  }
}

// The colour of player number k (1 to 30): golden-angle hues, so neighbours differ
color playerColour(int k) {
  colorMode(HSB, 360, 100, 100);
  color c = color((k * 137.5) % 360, 75, 80);
  colorMode(RGB, 255);
  return c;
}

float wrap(float a) {                   // an angle into 0 .. 2π
  a %= TWO_PI;
  return a < 0 ? a + TWO_PI : a;
}
