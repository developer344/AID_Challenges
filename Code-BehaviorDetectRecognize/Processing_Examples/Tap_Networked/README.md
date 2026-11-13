# Tap_Networked: the whole class taps together

Two Processing sketches: `Tap_Server` on the computer at the projector, and `Tap_Client` on every player's laptop
(up to 30). Each player taps with Space, any key or a click; the server shows how synchronised the group is, with
the measures of the coordination lecture (Kuramoto's order parameter R, the phase, the asynchrony to a beat).
No library is needed: the sketches send OSC over UDP with Java's own networking.

## Setting up

1. All computers on one local network where laptops can reach each other. University networks such as eduroam
   usually block this: use a phone's hotspot or a small Wi-Fi router.
2. Run `Tap_Server`. The top of its window shows its address, for example `192.168.1.23`.
3. Everyone runs `Tap_Client`. It finds the server by itself within a second or two (the server announces itself on
   the network); if not, click the Server field, type the address and press Enter. Type a name too. Both are
   remembered for next time.
4. If macOS asks whether Java may accept incoming network connections, allow it (on the server, at least).

Ports: UDP 12000 for the messages and 12001 for finding the server. If they are taken, change `PORT` and
`FIND_PORT` at the top of both sketches.

## What the server shows

| On the screen | How it is computed |
|---|---|
| A dot per player on the circle | The player's phase: each tap is phase 0 (the top); between taps it grows, clockwise, at the player's own tempo (the median of their last 6 intervals). A player counts as tapping after 3 taps, until they pause for two of their intervals |
| The arrow, and **R** | The average of e^{iφ} over all players tapping: Kuramoto's order parameter. Its length R is 1 when everyone taps together; 30 people at unrelated tempos give about 0.2 (1/√N) by chance. Its direction is the group's phase |
| Spread (ms) | The same in milliseconds: the circular standard deviation, √(−2 ln R) / 2π × the group's period |
| Every tap, the last 8 seconds | One row per player, with their tempo. Orange lines: the moments the group's phase passes 0, the group's own beat. Grey lines: the metronome |
| R over the last minute | R ten times a second |
| With the metronome (**M**, Up/Down: tempo) | R and the mean asynchrony (tap minus beat) of all taps of the last 10 s against the beat, as in `Tap_Asynchrony` |
| Simulated tappers (**S**) | 10 Kuramoto oscillators: each at its own tempo (about 100 BPM ± 5%), with 10 ms of motor noise, pulled towards the group's phase (or the metronome) with a coupling K (Left/Right). At K = 0 they drift apart; from about 0.1 they lock together. They also follow the real players, so a demo works with a few people or none |

Space on the server makes it a player too ("This computer"). C clears.

## Timing over a network

A Wi-Fi network delays every packet by a few milliseconds to more than a hundred, differently each time: as much as
the differences we want to measure. So each client stamps every tap with its own clock (the moment the key or the
button was pressed), and the server converts that time to its own clock. To know how the two clocks differ, the
server pings each client twice a second, and the client answers at once with its own time. If the way there and the
way back took as long, the client read its clock at the middle of the round trip; the server keeps the ping with the
shortest round trip of the last 20, the least delayed one. This is how NTP sets computer clocks. The client shows
its round trip.

## Messages (OSC over UDP)

| Message | From → to | When |
|---|---|---|
| `/tap/join id name` | client → server | at the start and every 2 s (it also says the player is still there; after 10 s of silence a player is removed) |
| `/tap/tap id time` | client → server | a tap; `time` on the client's clock in ms (a double). Without it, the arrival time counts |
| `/tap/pong id serverTime clientTime` | client → server | the answer to a ping, at once |
| `/tap/leave id` | client → server | when the client closes |
| `/tap/welcome number max` / `/tap/full max` | server → client | the player's number (1 to 30), or no room |
| `/tap/ping serverTime` | server → client | twice a second |
| `/tap/state R psi tapping bpm you number roundTrip` | server → client | ten times a second: the group (psi in degrees), and the player's phase against the group's (`you`, degrees; −1000 when not tapping) |
| `/tap/server port` | server → everyone, port 12001 | every second, broadcast: where the server is |

Any other program (Max, Pure Data, Python, a phone OSC app) can tap too: send `/tap/tap id` to port 12000.

## Checked

Both sketches compile with Processing 4 and were run on one computer, on other ports: the server, a real
`Tap_Client` with automatic taps, and 25 fake players from a Python script, each with its clock off by up to ±5 s
and network delays of 5 to 150 ms each way. The client found the server by itself. Tapping together (each fake with
its own bias and jitter, 19 ms in all), the server gave R = 0.95 to 0.98 and a spread of 18 to 30 ms: the network's
delays, which alone would give about 46 ms, did not blur the taps. Tapping at unrelated tempos, R fell to 0.12 to
0.19. The simulated tappers drifted apart at K = 0 (R between 0.1 and 0.4) and locked at K = 0.3 (R = 0.97), and
then followed the metronome (R to the beat 0.97). With 31 players the server answered "full", as it should.
Not yet with 30 laptops on a real Wi-Fi network.
