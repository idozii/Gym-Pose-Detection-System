# W8 - Private multi-window browser robustness and mobile QA

## Scope
W8 extends W7's **recent 120-frame** diagnostics with an optional
**longer-session observational envelope**. It never measures model accuracy
or official dataset TEST. It does not turn a desktop benchmark into a
mobile-device performance claim. No new MediaPipe calls are made.

`web/src/stability_audit.js` receives only the already observed pose-call
wall-clock end and synchronous pose-call duration. It aggregates
**non-overlapping blocks of exactly 120 processed frames**, retaining a
maximum of 16 completed blocks, plus a current incomplete block of <=119
latencies and a monotonic session gap counter. Old completed blocks are
discarded; it does not retain raw frame pixels, landmarks, file names,
device/browser identifiers, URLs, full frame timestamps or history.

After **240 analyzed frames**, a manual **Capture W8 stability summary**
button makes a read-only, selectable JSON on the same browser page:
- Number of analyzed frames, total and retained completed windows
- Range of per-window median latency and per-window P95 latency
- Range of per-window analyzed-frame wall-clock throughput
- Gaps >750 ms over entire current session, and in retained windows
- Source TYPE (camera/video) and exercise TYPE only

The report is a **min/max envelope of 120-frame window statistics**, not
an overall session-wide P95, not camera sensor FPS, and not proof of zero
frame drops. A pause at the border between windows can affect the
session gap counter without reducing either within-window throughput.
If timer quantization produces zero elapsed wall time in a window,
throughput is undefined and W8 refuses to export a misleading rate.

The W8 accumulator and exported text reset on Stop, Start, source swap,
local-video replay/clear, exercise change and Reset set. Existing W7
snapshot behavior is unchanged. No storage, cookies, network telemetry
or automatic downloading. Remote inference consent remains separate.

## Real hardware acceptance protocol

Use only a camera session or an MP4/WebM video you own, not the publisher
TRAIN/VALID/TEST clips. When testing a mobile camera, use an **HTTPS**
deployment; a local `http://192.168.*` URL does not provide a secure
camera context on most mobile browsers.

1. Ubuntu Chrome: run 60+ seconds of Squat, OHP and Barbell Row; capture
   both W7 and W8 JSON without uploading them automatically. Repeat
   three independent Start/Stop sessions. Observe UI and memory behavior
   manually; exported metrics do not establish memory safety.
2. Ubuntu Firefox: camera permission, denial/error, consecutive starts,
   Reset set, frontend at 320 px and the same aggregate captures.
3. Local owned MP4/WebM: Play/Pause, Replay from start and Clear; each
   lifecycle reset must remove prior W8 text/aggregate. No remote RF
   calls permitted for local video.
4. Android Chrome HTTPS and iPhone Safari HTTPS: opt-in permission,
   front/rear camera switch, background tab revocation, resize/rotation,
   warm and cold start, 240+ frames, capture and reset. Avoid claiming
   support until tested on each specific browser/device combination.
5. Negative paths: interrupted camera, denied permission, closed tab,
   missing camera, codec failure and hidden/visible transitions.
   Confirm the UI reports the error, camera resources are revoked and
   private report content is cleared.

Record a separate **manually written** outcome table with platform and
browser *families* (and versions only if willingly supplied), source TYPE,
exercise TYPE, 3 repeated-session pass/fail outcomes and qualitative
permission/layout observations. Do not export user agent strings, model
files, network traces with sensitive URLs or video screenshots. Keep
any consented local JSON as private research notes, not public
independent-device performance statistics.

## Validation and acceptance gates

- `npm test --prefix web`: bounded window eviction, summary math,
  insufficient-sample and malformed-data refusal, lifecycle wiring, no
  persistence/telemetry static regression checks
- `npm run build --prefix web`: frontend build
- GitHub CI: Python 3.10/3.11 + web green
- **Human hardware acceptance remains open** until actual Chrome,
  Firefox, Android and iPhone checks are performed. CI generates no
  real-device speed samples.

Owner-reported W7 evidence already observed:
Squat camera 511 processed frames, median 19.5 ms, P95 21.8 ms,
recent processed throughput 43.28 fps, zero recent gaps; OHP camera
3805 processed frames, median 19.8 ms, P95 26.65 ms,
throughput 40.78 fps, zero recent gaps. These are two separate
120-frame rolling snapshots, NOT W8 multi-window measurements. The
owner reported that Barbell Row camera ran without errors but did
not provide a Barbell Row performance JSON. Do not assign it latency
figures or extrapolate to mobile hardware.

W8 is an engineering observational study, not fitness advice,
a guaranteed 30/60-FPS rate, CPU/GPU/energy measurement,
memory/leak proof, media decode FPS, clinical safety or TEST accuracy.
