/** W5 self-contained TRAIN-only evidence visualization.
 * Do not import webcam/media trackers, API clients or user media here.
 */
import { selectFrozenEvidence, formatFixed } from './research_evidence.js';
import './style.css';
import './research.css';

const ids = [
  'researchCohort','researchCadence','conditionNote',
  'researchM39F1','researchM45F1','researchM39MAE','researchM45MAE',
  'researchM39TP','researchM45TP','researchM39Bar','researchM45Bar',
  'researchDeltaF1','researchDeltaMAE','researchGT','researchCohortDescription',
  'researchCIEvent','researchCICount','researchCIFP',
  'researchBootstrap','researchBootstrapUnavailable'
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
}

ui.researchCohort.addEventListener('change', renderFrozenResults);
ui.researchCadence.addEventListener('change', renderFrozenResults);
renderFrozenResults();
