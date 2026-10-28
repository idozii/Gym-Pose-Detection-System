import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  EVIDENCE, RESEARCH_EVIDENCE_VERSION, eventMetrics,
  selectFrozenEvidence, formatFixed, OWNER_M53_W6, selectOwnerRunM53W6
} from '../src/research_evidence.js';

const read = (path) =>
  readFileSync(new URL(path, import.meta.url), 'utf8');

test('W5 reconciles all 4 original M46/M45 TRAIN baseline conditions and GT totals', () => {
  const cases = [
    ['development',200,312,[149,11,163,7.9],[187,15,125,5.8],.63136,.72763],
    ['development',400,312,[118,3,194,9.55],[155,8,157,7.45],.54503,.65263],
    ['replication',200,209,[117,7,92,4.45],[129,8,80,3.8],.70270,.74566],
    ['replication',400,209,[94,6,115,5.45],[121,6,88,4.2],.60841,.72024]
  ];
  assert.equal(RESEARCH_EVIDENCE_VERSION,'w5-frozen-train-evidence-v1');
  for (const [name,ms,gt,m39,m45,f39,f45] of cases) {
    const report=selectFrozenEvidence(name,ms);
    assert.equal(report.cohort.clips,20);
    assert.equal(report.cohort.groundTruthReps,gt);
    assert.deepEqual(
      [report.m39.tp,report.m39.fp,report.m39.fn,report.m39.mae],m39
    );
    assert.deepEqual(
      [report.m45.tp,report.m45.fp,report.m45.fn,report.m45.mae],m45
    );
    assert.ok(Math.abs(report.m39.f1-f39)<.00001);
    assert.ok(Math.abs(report.m45.f1-f45)<.00001);
    assert.equal(report.m39.groundTruthEvents,gt);
    assert.equal(report.m45.groundTruthEvents,gt);
    assert.equal(report.deltaMAE,report.m45.mae-report.m39.mae);
    assert.ok(report.deltaF1>0);
    assert.ok(report.deltaMAE<0);
    assert.ok(Object.isFrozen(report.cohort));
  }
  assert.equal(EVIDENCE.cohorts.development.groundTruthReps+
    EVIDENCE.cohorts.replication.groundTruthReps,521);
});

test('W5 20-clip frozen replication video-bootstrap is ONLY attached to replication', () => {
  assert.equal(EVIDENCE.bootstrap.draws,5000);
  assert.equal(EVIDENCE.bootstrap.seed,2026);
  assert.equal(selectFrozenEvidence('development',200).bootstrap,null);
  assert.equal(selectFrozenEvidence('development',400).bootstrap,null);
  for (const [ms,df1,lo1,hi1,dmae,lomae,himae,fp,lofp,hifp] of [
    [200,.042962,.001847,.084923,-.65,-1.25,-.10,.05,-.20,.30],
    [400,.111824,.031327,.190676,-1.25,-2.45,-.20,0,-.20,.25]
  ]) {
    const report=selectFrozenEvidence('replication',ms);
    const boot=report.bootstrap;
    assert.deepEqual([
      boot.deltaF1.value,boot.deltaF1.lower,boot.deltaF1.upper,
      boot.deltaMAE.value,boot.deltaMAE.lower,boot.deltaMAE.upper,
      boot.deltaFalsePositivesPerVideo.value,
      boot.deltaFalsePositivesPerVideo.lower,
      boot.deltaFalsePositivesPerVideo.upper
    ], [df1,lo1,hi1,dmae,lomae,himae,fp,lofp,hifp]);
    assert.ok(Math.abs(report.deltaF1-df1)<.000001);
    assert.ok(Math.abs(report.deltaMAE-dmae)<1e-8);
    assert.ok(boot.deltaFalsePositivesPerVideo.lower<0);
    assert.ok(boot.deltaFalsePositivesPerVideo.upper>0);
  }
});

test('W5 micro F1 formula is exact, not rounded lookup or normalized count MAE', () => {
  const stats=eventMetrics({tp:129,fp:8,fn:80,mae:3.8});
  assert.equal(stats.f1,258/346);
  assert.equal(stats.precision,129/137);
  assert.equal(stats.recall,129/209);
  assert.deepEqual(eventMetrics({tp:0,fp:0,fn:12,mae:0}),{
    tp:0,fp:0,fn:12,mae:0,precision:0,recall:0,
    f1:0,groundTruthEvents:12
  });
  assert.throws(()=>eventMetrics({tp:-1,fp:2,fn:4,mae:1}),/nonnegative/);
  assert.throws(()=>eventMetrics({tp:1.4,fp:2,fn:4,mae:1}),/nonnegative/);
  assert.throws(()=>eventMetrics({tp:1,fp:2,fn:4,mae:NaN}),/finite/);
  assert.equal(formatFixed(stats.f1),'0.746');
  assert.equal(formatFixed(-.65,2),'-0.65');
});

test('W5 never allows TEST, arbitrary sample sizes or live web camera scores', () => {
  for(const [id,ms] of [
    ['test',200], ['validation',200], ['replication',100],
    ['replication',800], ['unknown',400], ['development',NaN]
  ]) assert.throws(()=>selectFrozenEvidence(id,ms),/Only frozen/);
  const integrity=EVIDENCE.trainIntegrity;
  assert.deepEqual({
    clips:integrity.annotatedClips,
    reps:integrity.groundTruthReps,
    m50:integrity.m50TimingOrDecodeFlags,
    m51:integrity.m51TarByteIdenticalClips,
    m52:integrity.m52DescriptivePtsFlags,
    test:integrity.officialTestRun,
    publisher:integrity.publisherOriginIndependentlyVerified,
    identity:integrity.participantIdentityVerified,
    browser:integrity.remoteModelParityVerified
  }, {
    clips:40,reps:521,m50:0,m51:40,m52:0,test:false,publisher:false,
    identity:false,browser:false
  });
  assert.equal(integrity.m52MaxApproxClockDriftMs,.0005);
});

test('W5 research is a separate, Vite-built accessible HTML page', () => {
  const html=read('../research.html');
  const config=read('../vite.config.js');
  const script=read('../src/research_page.js');
  const css=read('../src/research.css');
  const studio=read('../src/main.js');
  for(const id of [
    'research-title','researchCohort','researchCadence',
    'researchM39F1','researchM45F1','researchM39MAE','researchM45MAE',
    'researchM39Bar','researchM45Bar','researchDeltaF1','researchDeltaMAE',
    'researchBootstrap','researchBootstrapUnavailable','researchCIEvent',
    'researchCICount','researchCIFP','researchGT','integrity','method',
    'limitations','reproduce','m53','m53OwnerResults','m53OwnerUnavailable',
    'm53NormalizedM39','m53NormalizedM45','m53NormalizedDelta',
    'm53OboM39','m53OboM45','m53OboDelta',
    'm53ExactM39','m53ExactM45','m53ExactDelta',
    'w6DirectMismatches','w6GatedMismatches','w6DroppedFrames',
    'm53CINorm','m53CINormStatus',
    'm53CIObo','m53CIOboStatus','m53CIExact','m53CIExactStatus',
    'm53CIAbs','m53CIAbsStatus'
  ]) assert.match(html,new RegExp('id="'+id+'"'));
  assert.match(html,/aria-live="polite"/);
  assert.match(html,/aria-current="page"/);
  assert.match(html,/for="researchCohort"/);
  assert.match(html,/for="researchCadence"/);
  assert.match(html,/rel="noopener noreferrer"/);
  assert.match(html,/href="\.\/"/);
  assert.match(html,/src="\/src\/research_page\.js"/);
  assert.match(config,/research: fileURLToPath\(new URL\('\.\/research\.html'/);
  assert.match(config,/studio: fileURLToPath\(new URL\('\.\/index\.html'/);
  assert.match(studio,/href="\$\{import\.meta\.env\.BASE_URL\}research\.html"/);
  assert.match(script,/selectFrozenEvidence\(/);
  assert.match(script,/\.addEventListener\('change', renderFrozenResults\)/);
  assert.match(script,/\.researchBootstrap\.hidden = bootstrap === null/);
  assert.match(script,/\.researchBootstrapUnavailable\.hidden = bootstrap !== null/);
  assert.match(css,/prefers-reduced-motion/);
  assert.match(css,/@media \(max-width: 460px\)/);
});

test('W5 separates offline TRAIN research from browser inference and permits no raw-data access', () => {
  const html=read('../research.html');
  const script=read('../src/research_page.js');
  const evidence=read('../src/research_evidence.js');
  assert.match(html,/TRAIN evidence only/i);
  assert.match(html,/No official TEST results/i);
  assert.match(html,/No live camera GT is available/i);
  assert.match(html,/New M53 bootstrap confidence intervals and official TEST results are not claimed/i);
  assert.match(html,/full private JSON report was not independently inspected/i);
  assert.match(html,/same-input/i);
  assert.match(html,/source filenames/i);
  assert.match(html,/MediaPipe Solutions/);
  assert.match(html,/browser Tasks Vision/);
  assert.match(html,/No zero-rep/i);
  assert.match(html,/license/i);
  assert.match(html,/human/i);
  assert.match(html,/not TransRAC normalized MAE/);
  for(const source of [script,evidence]) {
    assert.doesNotMatch(source,/\bfetch\s*\(|XMLHttpRequest|getUserMedia|localStorage|sessionStorage|indexedDB|MediaRecorder|PoseLandmarker|ResearchInferenceClient/);
  }
  assert.doesNotMatch(html,/<video\b|<input\s+type="file"/);
  for(const name of ['m46','m47','m48','m50','m51','m52','m45Rule','reproduction']) {
    assert.match(html,new RegExp('docs/|m45_compare_train_relative'));
    assert.match(EVIDENCE.sourceDocuments[name],/^https:\/\/github\.com\/idozii\//);
  }
});


test('W5 only publishes owner-observed M53/W6 TRAIN replication point estimates', () => {
  const declared = OWNER_M53_W6;
  assert.equal(declared.sourceClipCount,20);
  assert.equal(declared.groundTruthReps,209);
  assert.equal(declared.sourceCohort,'replication');
  assert.equal(declared.publisherTestEvaluated,false);
  assert.equal(declared.actualBrowserPoseModelParityVerified,false);
  assert.equal(declared.newM53BootstrapIntervalsIndependentlyReviewed,false);
  assert.equal(declared.ownerSharedM53BootstrapIntervals,true);
  assert.equal(declared.pairedVideoBootstrap.draws,5000);
  assert.equal(declared.pairedVideoBootstrap.seed,2026);
  assert.match(declared.evidenceStatus,/owner-reported/);
  for(const [ms,baseN,newN,obo1,obo2,exact1,exact2,drops] of [
    [200,.4940229816623005,.40997374658365365,.4,.5,.2,.3,2],
    [400,.5397661149480034,.48384441023372904,.35,.45,.15,.2,0]
  ]) {
    const data = selectOwnerRunM53W6('replication',ms);
    assert.equal(data.low_to_high.normalizedMAE,baseN);
    assert.equal(data.relative_excursion.normalizedMAE,newN);
    assert.equal(data.low_to_high.obo,obo1);
    assert.equal(data.relative_excursion.obo,obo2);
    assert.equal(data.low_to_high.exact,exact1);
    assert.equal(data.relative_excursion.exact,exact2);
    assert.ok(data.deltaNormalizedMAE<0);
    assert.equal(data.deltaOBO,obo2-obo1);
    assert.equal(data.deltaExact,exact2-exact1);
    assert.equal(data.directEventDisagreements,0);
    assert.equal(data.gatedEventDisagreements,0);
    assert.equal(data.framesDroppedByGate,drops);
    assert.equal(data.low_to_high.absoluteMAE,
      EVIDENCE.cohorts.replication.conditions[ms].m39.mae);
    assert.equal(data.relative_excursion.absoluteMAE,
      EVIDENCE.cohorts.replication.conditions[ms].m45.mae);
    assert.ok(Object.isFrozen(declared.conditions[ms]));
  }
  assert.equal(selectOwnerRunM53W6('development',200),null);
  assert.equal(selectOwnerRunM53W6('development',400),null);
  for(const pair of [['test',200],['validation',200],['replication',500]]) {
    assert.throws(()=>selectOwnerRunM53W6(...pair),/TRAIN cohorts/);
  }
});

test('W5 M53/W6 selector never leaks results onto original development cohort',()=>{
  const script=read('../src/research_page.js');
  const html=read('../research.html');
  assert.match(script,/selectOwnerRunM53W6\(/);
  assert.match(script,/ui\.m53OwnerResults\.hidden = newEvidence === null/);
  assert.match(script,/ui\.m53OwnerUnavailable\.hidden = newEvidence !== null/);
  assert.match(script,/ui\.w6DirectMismatches\.textContent/);
  assert.match(script,/ui\.w6GatedMismatches\.textContent/);
  assert.match(script,/ui\.w6DroppedFrames\.textContent/);
  assert.match(html,/owner's 20-clip TRAIN replication replay/i);
  assert.match(html,/Paired-video 95% bootstrap intervals below come from the owner's local console output/i);
  assert.match(html,/original M45 comparator and the website's M45 algorithm/i);
  assert.doesNotMatch(html,/official TEST MAE:\s*\d/i);
});

test('W5 M53 bootstrap 5000 paired draws match exactly owner console output', () => {
  const expected = {
    200: {
      normalizedMAE: [-0.08404923507864687,-0.1872277661064424,0.015238095238095165],
      obo: [0.09999999999999998,-0.09999999999999998,0.30000000000000004],
      exact: [0.09999999999999998,0,0.25],
      absoluteMAE: [-0.6500000000000004,-1.25,-0.10000000000000009]
    },
    400: {
      normalizedMAE: [-0.05592170471427438,-0.17944823455698064,0.07934227324533022],
      obo: [0.10000000000000003,-0.09999999999999998,0.30000000000000004],
      exact: [0.05000000000000002,-0.1,0.2],
      absoluteMAE: [-1.25,-2.45,-0.20000000000000018]
    }
  };
  for(const [ms, conditions] of Object.entries(expected)){
    const evidence = selectOwnerRunM53W6('replication',Number(ms));
    const reported = OWNER_M53_W6.pairedVideoBootstrap[ms];
    assert.equal(evidence.bootstrap,reported);
    for(const [metric,[delta,lower,upper]] of Object.entries(conditions)){
      assert.deepEqual(reported[metric],{delta,lower,upper});
      assert.ok(lower <= delta && delta <= upper);
      if(metric==='absoluteMAE')assert.ok(upper<0);
      if(metric==='normalizedMAE'||metric==='obo')
        assert.ok(lower<0&&upper>0);
    }
    if(ms==='200')assert.equal(reported.exact.lower,0);
    else assert.ok(reported.exact.lower<0&&reported.exact.upper>0);
  }
  assert.equal(selectOwnerRunM53W6('development',200),null);
  assert.equal(selectOwnerRunM53W6('development',400),null);
});

test('W5 M53 CI renderer includes zero classification and does not report TEST confidence',()=>{
  const html=read('../research.html');
  const page=read('../src/research_page.js');
  assert.match(page,/formatM53Bootstrap/);
  assert.match(page,/function relationToZero/);
  assert.match(page,/ui\[name\+'Status'\]\.textContent = relationToZero\(ci\)/);
  assert.match(html,/The normalized MAE and OBO intervals include zero/);
  assert.match(html,/exact-count touches zero at 200 ms/);
  assert.match(html,/not adjusted for multiple comparisons or subject\/session overlap/);
  assert.match(html,/owner's local console output/);
  assert.doesNotMatch(html,/normalized MAE improvements? (are|is) statistically significant/i);
});
