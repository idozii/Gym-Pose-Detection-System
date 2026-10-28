# W7 - Browser Performance, Privacy, and Real-Device Acceptance

## Implementation

W7 adds local observational CPU performance telemetry to the existing
W1-W6 movement studio without extra model calls, backend requests,
camera permissions or persistent storage.

The new web/src/runtime_health.js maintains at most 120 pose
processing durations and wall-clock timestamps from already
processed video frames, plus aggregate frame and gap counts.

- Pose median and P95 measure only elapsed milliseconds of the
  synchronous MediaPipe detectForVideo call. They do not include
  video decoding, browser paint, model download or optional RF API.
- Processed throughput measures analyzed frames per wall-clock
  second, NOT camera sensor FPS or source video nominal FPS.
- Observed gaps >750ms count large intervals between processed
  frames. A pause, decode stall, background tab or slow CPU
  can all cause them. It is NOT a count of dropped video frames.
- Status flags are descriptive UI hints, not exercise safety,
  camera quality grades or device compatibility promises.

W7 metrics reset on stop/start, source switch and file replay.
No video frames, landmarks, filenames, browser device fingerprints
or individual GT event timestamps are logged or uploaded.
The existing opt-in webcam RF inference remains unchanged.

## W7b opt-in local QA snapshot

W7 now includes **Capture local QA summary** below the live
median/P95/throughput cards. The button is disabled until >=30
processed frames in the current bounded window. Clicking it
copies no data to a server; it displays a readonly JSON text
box which the owner can select and copy by choice. It includes
only source TYPE (camera/video), selected exercise type, count
of sampled/total analyzed frames, rounded local pose latency,
processed throughput, and count of gaps within the **recent**
sample window. It does NOT record browser/OS fingerprint,
IP address, filenames, personal information, raw videos,
joint positions, individual timestamps, GT or user accounts.

Stop, replay, set reset and exercise changes clear the
previous QA text and reset the sampled timings. Startup
stalls no longer permanently affect subsequent samples:
the gap diagnostic reflects the current rolling window.
CI tests simulate the calculations and privacy guard;
they do not generate real hardware performance evidence.

Suggested owner-side on-device acceptance for each test:
1. Start local-only camera or a short video **you own**.
2. Select Squat, let at least 30 frames process, ideally
   a full uninterrupted 60-second observation.
3. Click "Capture local QA summary" and select/copy only the
   JSON aggregate. Do not share browser console URLs, personal
   screenshots or sensitive local videos.
4. Repeat on Ubuntu Chrome and Firefox, then on a mobile
   phone via trusted HTTPS. Record layout/permission outcomes
   manually beside the JSON if appropriate.
5. Assess actual behavior before claiming device compatibility.
   A successful benchmark is not an assessment of rep accuracy,
   clinical safety or MediaPipe Python/browser parity.

## Acceptance matrix (real devices)

CI covers calculation arithmetic, latency percentiles, bounded
buffers, large gaps, lifecycle wiring, no storage/API and build.
Human real-device checks remain REQUIRED:

| Case | Ubuntu Chrome | Ubuntu Firefox | iPhone Safari HTTPS | Android Chrome HTTPS |
| --- | --- | --- | --- | --- |
| Camera opt-in, stop and front/rear | Manual | Manual | Manual | Manual |
| Permission-denied, missing-camera help | Manual | Manual | Manual | Manual |
| Hide tab during camera initialization | Manual | Manual | Manual | Manual |
| Restart after unplugged/ended track | Manual | Manual | Manual | Manual |
| M34/M45 Squat cycles and W3 dashboard | Manual | Manual | Manual | Manual |
| W7 median/P95, throughput, gap display | Manual | Manual | Manual | Manual |
| Stop and source switch clear W7 | Manual | Manual | Manual | Manual |
| Local MP4 play/pause/replay/clear | Manual | Manual | Manual | Manual |
| Local video sends no RF API requests | Manual | Manual | Manual | Manual |
| 320px layout and accessible controls | Manual | Manual | Manual | Manual |
| Built website and research HTML routes | CI | CI | Manual | Manual |

No measured device statistics are created by CI. Do not claim
the app holds 30 FPS or optimized overall memory without real data.

A Vite LAN host at http://192.168.x.x:5173 is usually NOT a secure
context for mobile getUserMedia. Test phone cameras via an actual
trusted HTTPS deployment. LAN binding does not provide TLS.

To benchmark, reuse an owned non-TEST 30-90 second local clip
and one 60-second camera session on each device. Record only
OS/browser version, device class, source type, W7 median/P95,
processed throughput, observed gaps, permission outcome and
layout pass/fail. Do not export other people's identifiable
frames or landmarks.

## Commands

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
npm test --prefix web
npm run build --prefix web
cd web
npm run dev
~~~

W7 initially shows blank metrics until actual pose frames run.
The bounded W7 sampling window does not grow with long sessions,
but this does not establish full app or MediaPipe memory safety.

## Production release gate

- Check actual HTTPS camera on desktop/mobile.
- Verify actual Vercel production SHA, quota and deployment state.
- Confirm user-initiated webcam, revocation and local video privacy.
- Ensure HTTPS CDN/WASM/model downloads work; avoid untrusted proxies.
- Publish no speculative speed or phone-support metrics.

Original frozen M34/M39/M45 and all 17 M48 hashes are unchanged.
