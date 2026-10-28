/** W5 self-contained TRAIN-only evidence visualization.
 * Do not import webcam/media trackers, API clients or user media here.
 */
import { selectFrozenEvidence, selectOwnerRunM53W6, formatFixed } from './research_evidence.js';
import './style.css';
import './research.css';

const ids = [
  'researchCohort','researchCadence','conditionNote',
  'researchM39F1','researchM45F1','researchM39MAE','researchM45MAE',
  'researchM39TP','researchM45TP','researchM39Bar','researchM45Bar',
  'researchDeltaF1','researchDeltaMAE','researchGT','researchCohortDescription',
  'researchCIEvent','researchCICount','researchCIFP',
  'researchBootstrap','researchBootstrapUnavailable',
  'm53OwnerResults','m53OwnerUnavailable','m53CadenceLabel',
  'm53NormalizedM39','m53NormalizedM45','m53NormalizedDelta',
  'm53OboM39','m53OboM45','m53OboDelta',
  'm53ExactM39','m53ExactM45','m53ExactDelta',
  'w6DirectMismatches','w6GatedMismatches','w6DroppedFrames',
  'm53CINorm','m53CINormStatus','m53CIObo','m53CIOboStatus',
  'm53CIExact','m53CIExactStatus','m53CIAbs','m53CIAbsStatus'
];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
if (ids.some((id) => !ui[id])) {
  throw new Error('W5 required research evidence element is missing');
}

function signed(value, digits) {
  return (value > 0 ? '+' : '') + formatFixed(value, digits);
}

function interval(item, digits) {
  return '[' + signed(item.lower, digits) + ', ' +
    signed(item.upper, digits) + ']';
}

function formatM53Bootstrap(ci, digits, units) {
  const multiplier = units === 'pp' ? 100 : 1;
  const suffix = units === 'pp' ? ' pp' : units === 'reps' ? ' reps' : '';
  return '[' + signed(ci.lower*multiplier,digits) + ', ' +
    signed(ci.upper*multiplier,digits) + ']' + suffix;
}

function relationToZero(ci) {
  if (ci.lower < 0 && ci.upper > 0) return 'Includes zero';
  if (ci.lower === 0 || ci.upper === 0) return 'Touches zero';
  return 'Excludes zero';
}

export function renderFrozenResults() {
  const selected = selectFrozenEvidence(
    ui.researchCohort.value, ui.researchCadence.value
  );
  const {cohort,m39,m45,deltaF1,deltaMAE,bootstrap} = selected;
  ui.conditionNote.textContent =
    cohort.title + ' · ' + cohort.clips + ' original source videos · ' +
    selected.samplingMs + ' ms nominal sampling · ' + selected.condition;
  ui.researchM39F1.textContent = formatFixed(m39.f1);
  ui.researchM45F1.textContent = formatFixed(m45.f1);
  ui.researchM39MAE.textContent = formatFixed(m39.mae,2) + ' reps';
  ui.researchM45MAE.textContent = formatFixed(m45.mae,2) + ' reps';
  ui.researchM39TP.textContent = [m39.tp,m39.fp,m39.fn].join(' / ');
  ui.researchM45TP.textContent = [m45.tp,m45.fp,m45.fn].join(' / ');
  ui.researchM39Bar.style.width = (m39.f1 * 100).toFixed(2) + '%';
  ui.researchM45Bar.style.width = (m45.f1 * 100).toFixed(2) + '%';
  ui.researchM39Bar.parentElement.setAttribute(
    'aria-label','M39 event F1 ' + formatFixed(m39.f1) + ' out of 1');
  ui.researchM45Bar.parentElement.setAttribute(
    'aria-label','M45 event F1 ' + formatFixed(m45.f1) + ' out of 1');
  ui.researchDeltaF1.textContent = signed(deltaF1,3);
  ui.researchDeltaMAE.textContent = signed(deltaMAE,2) + ' reps';
  ui.researchGT.textContent = String(cohort.groundTruthReps);
  ui.researchCohortDescription.textContent =
    cohort.sampleDescription + ' · ' + cohort.independence;
  ui.researchBootstrap.hidden = bootstrap === null;
  ui.researchBootstrapUnavailable.hidden = bootstrap !== null;
  if (bootstrap) {
    ui.researchCIEvent.textContent = interval(bootstrap.deltaF1,3);
    ui.researchCICount.textContent = interval(bootstrap.deltaMAE,2) + ' reps';
    ui.researchCIFP.textContent = interval(bootstrap.deltaFalsePositivesPerVideo,2);
  } else {
    ui.researchCIEvent.textContent = 'Not applicable';
    ui.researchCICount.textContent = 'Not applicable';
    ui.researchCIFP.textContent = 'Not applicable';
  }
  // Owner-run M53/W6 report covers only the frozen replication TRAIN
  // cohort and the same offline pose inputs. No development/TEST
  // counts, confidence intervals or browser-estimator parity invented.
  const newEvidence = selectOwnerRunM53W6(
    ui.researchCohort.value, ui.researchCadence.value
  );
  ui.m53OwnerResults.hidden = newEvidence === null;
  ui.m53OwnerUnavailable.hidden = newEvidence !== null;
  if (newEvidence !== null) {
    const a = newEvidence.low_to_high;
    const b = newEvidence.relative_excursion;
    ui.m53CadenceLabel.textContent =
      'Owner-run replication TRAIN / ' + selected.samplingMs + ' ms';
    ui.m53NormalizedM39.textContent = formatFixed(a.normalizedMAE,4);
    ui.m53NormalizedM45.textContent = formatFixed(b.normalizedMAE,4);
    ui.m53NormalizedDelta.textContent = signed(
      newEvidence.deltaNormalizedMAE,4
    );
    ui.m53OboM39.textContent = formatFixed(a.obo*100,0)+'%';
    ui.m53OboM45.textContent = formatFixed(b.obo*100,0)+'%';
    ui.m53OboDelta.textContent =
      signed(newEvidence.deltaOBO*100,0)+' pp';
    ui.m53ExactM39.textContent = formatFixed(a.exact*100,0)+'%';
    ui.m53ExactM45.textContent = formatFixed(b.exact*100,0)+'%';
    ui.m53ExactDelta.textContent =
      signed(newEvidence.deltaExact*100,0)+' pp';
    ui.w6DirectMismatches.textContent =
      String(newEvidence.directEventDisagreements);
    ui.w6GatedMismatches.textContent =
      String(newEvidence.gatedEventDisagreements);
    ui.w6DroppedFrames.textContent =
      String(newEvidence.framesDroppedByGate);
    for (const [name, field, digits, units] of [
      ['m53CINorm','normalizedMAE',4,''],
      ['m53CIObo','obo',0,'pp'],
      ['m53CIExact','exact',0,'pp'],
      ['m53CIAbs','absoluteMAE',2,'reps']
    ]) {
      const ci = newEvidence.bootstrap[field];
      ui[name].textContent = formatM53Bootstrap(ci,digits,units);
      ui[name+'Status'].textContent = relationToZero(ci);
    }
  }
}

ui.researchCohort.addEventListener('change', renderFrozenResults);
ui.researchCadence.addEventListener('change', renderFrozenResults);
renderFrozenResults();
