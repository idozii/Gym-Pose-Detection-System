# W5 - Research Evidence & Methodology Page

## Purpose

The new static second page, `web/research.html`, is a
portfolio-oriented **research evidence logbook**, NOT a
live model performance monitor. It is accessible from
the existing W1-W4 movement studio and from its own
relative URL on both Vercel root and the existing
GitHub Pages project subpath.

W5 is deliberately separate from the camera UI:
`web/src/research_page.js` imports only a reviewed,
static frozen evidence module and styles. It never
requests camera permission, loads selected videos,
accesses owner-local archived GT or sends research
inference API calls.

## Fixed scientific scope

The owner ran M46/M47 with **frozen M39/M45 detector
code** and original source-file SHA256 selection. The
resulting TRAIN-only M48 evidence freeze covers:

- **Development TRAIN**: original first 20 source
  hash ranks; 312 published annotated Squat cycles;
  the relative-excursion rule was developed using
  this cohort. NOT independent validation.
- **Replication TRAIN**: original hash ranks 21-40;
  20 new filename-disjoint clips, 209 annotated
  Squat cycles; NOT independently subject/session/
  camera-disjoint.
- **200ms primary** and **400ms shared tracker
  subsample**. The 400ms condition retains every
  other sample from the same 200ms offline Python
  MediaPipe Solutions tracker. It is NOT separate
  end-to-end media inference.
- M39 is the frozen low-to-high baseline; M45 is
  original relative excursion; neither is the
  original browser M34 deployed counter.

| Cohort / cadence | Method | TP / FP / FN | Micro F1 | Absolute count MAE / video |
| --- | --- | --- | ---: | ---: |
| Development / 200ms | M39 | 149 / 11 / 163 | 0.63136 | 7.90 |
| Development / 200ms | M45 | 187 / 15 / 125 | 0.72763 | 5.80 |
| Development / 400ms | M39 | 118 / 3 / 194 | 0.54503 | 9.55 |
| Development / 400ms | M45 | 155 / 8 / 157 | 0.65263 | 7.45 |
| Replication / 200ms | M39 | 117 / 7 / 92 | 0.70270 | 4.45 |
| Replication / 200ms | M45 | 129 / 8 / 80 | 0.74566 | 3.80 |
| Replication / 400ms | M39 | 94 / 6 / 115 | 0.60841 | 5.45 |
| Replication / 400ms | M45 | 121 / 6 / 88 | 0.72024 | 4.20 |

The webpage **computes** micro Event F1 as
2*TP/(2*TP + FP + FN) from the frozen totals; it does
not read stored or fabricated browser F1. It displays
the originally reported absolute count MAE separately.
This is not normalized MAE as defined in TransRAC and
must never be compared directly to literature normalized
metrics without a matching metric/data/protocol.

### M47 replication-only paired uncertainty

The only selectable video-bootstrap intervals are for
the filename-disjoint **TRAIN replication cohort**:
5,000 whole-video paired bootstrap draws, seed 2026.

- 200ms M45-M39 ΔF1 +0.042962; 95% percentile
  [0.001847, 0.084923].
- 200ms Δcount MAE -0.65, [-1.25, -0.10] reps;
  ΔFP/video +0.05 [-0.20, +0.30].
- 400ms ΔF1 +0.111824,
  [0.031327, 0.190676].
- 400ms Δcount MAE -1.25, [-2.45, -0.20] reps;
  ΔFP/video 0 [-0.20, +0.25].

The original 20 development video results have no
corresponding M47 replication bootstrap; W5 therefore
**hides** the bootstrap there, rather than implying
the interval is applicable across cohorts.

Clip resampling neither increases the number of
independent human participants nor guarantees valid
population inference. These are exploratory TRAIN
uncertainty intervals, not confirmatory TEST results.

### M50-M52 source integrity summaries

The owner verified 40 Squat TRAIN clips, 521 GT events:
- M50: zero frame/decode/bound gross timing flags.
- M51: 40/40 previously extracted local TRAIN MP4s
  byte-for-byte matched the same owner's TAR.GZ
  archive members.
- M52: zero descriptive PTS anomalies against
  the original frame-index/OpenCV FPS video clock,
  with largest observed drift around 0.0005ms.
- M48/M49: frozen evidence plus 17 research
  source paths checked against commits and reports.

The checks do not authenticate the publisher source,
independent participant/session identity, correct
manual cycle annotation endpoints or licensing.
M50-M52 inspect TRAIN ONLY. TEST remained untouched.

## Design, routing and source paths

Files created:
- `web/research.html`: semantic page with
  accessible cohort/cadence dropdowns, score bars,
  precise labels, limitations and cited repo links.
- `web/src/research_evidence.js`: strictly
  typed, static owner-confirmed aggregate snapshot,
  pure F1 calculation and TRAIN selector.
- `web/src/research_page.js`: updates only
  existing DOM text and width of score bars.
- `web/src/research.css`: responsive layout,
  semantic bars and reduced-motion support.
- `web/vite.config.js`: multipage Vite build of
  both `index.html` and `research.html`.
- `web/tests/research_evidence.test.js`: exact
  M46/M47 outcome/interval checks, zero TEST claim,
  privacy and Vite route coverage.

Existing `web/src/main.js` and `style.css`
gain a prominent cross-link and research teaser
to this separate page. The original W1-W4
camera/video code, cycle heuristics and 17
M48-frozen research files remain untouched.

The page is **static public reporting**, not a
live-stream accuracy scoring page or a means
to upload private GT or raw media.

## Ubuntu QA

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
cd web
npm install
npm test
npm run build
test -f dist/index.html && test -f dist/research.html
npm run dev
~~~

Visit:
- `http://localhost:5173/`: verify the W1-W4
  camera/local-video UI still works unchanged.
- `http://localhost:5173/research.html`:
  verify the research page and the link back to studio.

Manual W5 acceptance:
1. Start in replication / 200ms:
   M39 F1 0.703, M45 F1 0.746,
   count MAE 4.45 and 3.80,
   F1 delta +0.043, MAE delta -0.65.
2. Change replication to 400ms:
   0.608 / 0.720 and CI [0.031,0.191].
3. Change to development / 200ms:
   0.631 / 0.728 and **bootstrap hidden**.
4. Change development to 400ms:
   0.545 / 0.653, GT 312, bootstrap still hidden.
5. Check that 400ms is explicitly a paired
   shared-tracker subsample and not an
   independent runtime/algorithm ablation.
6. Check headings, contrast, keyboard
   navigation and focus on mobile (~320px).
7. Confirm no paper-style claims about a
   conference acceptance, publisher TEST F1,
   participant independence, proven safe squat
   form or 6-backend accuracy.
8. Follow M45/M46/M47/M48/M50/M51/M52
   source links; no private file paths,
   real MP4 bytes, per-video data or
   human identifiers should be published.
9. Confirm the static `research.html` is
   present in built `web/dist` and works
   under both Vercel root and GitHub Pages
   `/Gym-Pose-Detection-System/` base path.

CI does not prove production deployment.
Any Vercel quota/security limits must be
resolved independently of W5 build results.

## Research priorities NOT implemented by W5

- M53 normalized count MAE and off-by-one
  (OBO) require genuine per-video scores,
  not derivations from aggregate TP/FP/FN.
- Publisher source origin/licensing and
  exact annotation-cycle semantics are open.
- Valid zero-rep negative controls and
  subject/session independence remain open.
- Python MediaPipe Solutions vs browser Tasks
  Vision end-to-end parity is not established.
- Frozen prospective, authorized official
  publisher TEST has NOT been run.

Do not relabel TRAIN cohort replication as
TEST or as participant-independent evaluation.
Do not change the original M39/M45 rules in
order to improve any displayed W5 metric.
