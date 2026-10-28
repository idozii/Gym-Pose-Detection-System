import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  EVIDENCE, RESEARCH_EVIDENCE_VERSION, eventMetrics,
  selectFrozenEvidence, formatFixed
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
    'limitations','reproduce'
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
  assert.match(html,/M53 normalized MAE, OBO and official TEST results are not claimed/i);
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
