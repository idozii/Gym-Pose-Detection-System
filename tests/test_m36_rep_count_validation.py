"""Synthetic M36 rep-count validation tests. No personal video or user sessions."""
from __future__ import annotations

import csv
import json

import pytest

from scripts import m36_rep_count_validation as m36


def synthetic_csv(tmp_path):
    rows = []
    for person in range(4):
        for exercise in m36.EXERCISES:
            for view in m36.VIEWS:
                manual = 8 if person != 0 else 0
                predicted = manual + (1 if person == 1 else -1 if person == 2 else 0)
                predicted = max(0, predicted)
                rows.append({
                    "session_id": f"session_p{person}_{exercise}_{view}",
                    "participant_id": f"person_p{person:05d}",
                    "exercise": exercise, "view": view,
                    "human_rep_count": str(manual),
                    "website_cycle_count": str(predicted),
                    "tracking": "stable",
                    "consent_confirmed": "yes",
                    "protocol_version": m36.PROTOCOL
                })
    # An inability to track is an explicit abstention, NOT a false zero.
    rows[0]["tracking"] = "insufficient"
    rows[0]["website_cycle_count"] = ""
    path = tmp_path / "private_m36.csv"
    rewrite(path, rows)
    return path, rows


def rewrite(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=m36.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_m36_template_is_private_and_no_overwrite(tmp_path):
    output = tmp_path / "private" / "manual_session_counts.csv"
    details = m36.create_template(output)
    assert details["rows"] == 0
    with output.open(newline="") as handle:
        assert tuple(next(csv.reader(handle))) == m36.COLUMNS
    with pytest.raises(FileExistsError, match="overwrit"):
        m36.create_template(output)
    with pytest.raises(ValueError, match="No real human-annotated"):
        m36.read_sessions(output)


def test_m36_count_error_coverage_by_person_exercise_and_view(tmp_path):
    path, _ = synthetic_csv(tmp_path)
    records = m36.read_sessions(path)
    report = m36.report_sessions(records)
    assert report["all_consented_sessions"] == 36
    assert report["distinct_pseudonymous_people"] == 4
    assert report["scorable_sessions"] == 35
    assert report["abstained_sessions"] == 1
    assert report["counting_coverage"] == 35 / 36
    assert report["by_tracking"]["insufficient"] == 1
    assert report["by_exercise"]["squat"]["sessions"] == 12
    assert report["by_exercise"]["squat"]["abstained"] == 1
    assert report["by_exercise_and_view"]["squat/front"]["sessions"] == 4
    summary = report["overall"]
    assert summary["sessions"] == 35
    assert summary["overcount_sessions"] == 9
    assert summary["undercount_sessions"] == 9
    assert summary["equal_count_sessions"] == 17
    assert summary["mean_absolute_count_error"] == 18 / 35
    assert summary["exact_session_count_fraction"] == 17 / 35
    assert summary["within_one_cycle_fraction"] == 1.0
    assert summary["mean_signed_count_error"] == 0.
    assert summary["observed_human_reps"] == 27 * 8
    assert summary["website_observed_cycles"] == 27 * 8
    assert report["bootstrap"]["resamples"] == 1000
    assert report["rep_level_precision_recall_f1_available"] is False
    assert report["evaluation_on_new_independent_participants_verified"] is False
    assert report["original_M20_VAL_or_TEST_accessed"] is False
    dump = json.dumps(report)
    assert "person_p00000" not in dump
    assert "session_p" not in dump
    assert "private_m36" not in dump


def test_m36_repeatable_subject_cluster_bootstrap(tmp_path):
    path, _ = synthetic_csv(tmp_path)
    records = m36.read_sessions(path)
    scored = [record for record in records if record["observed"] is not None]
    first = m36._bootstrap_by_person(scored)
    second = m36._bootstrap_by_person(scored)
    assert first == second
    mae = first["mae"]
    assert mae["lower_95"] <= mae["median"] <= mae["upper_95"]
    assert mae["lower_95"] >= 0
    assert m36._bootstrap_by_person(scored[:2]) is None
    with pytest.raises(ValueError, match="100..5000"):
        m36._bootstrap_by_person(scored, draws=10)


@pytest.mark.parametrize("edit,error", [
    ({"consent_confirmed": "no"}, "consent"),
    ({"exercise": "pushup"}, "exercise"),
    ({"view": "birds_eye"}, "camera view"),
    ({"human_rep_count": "-1"}, "human_rep_count"),
    ({"human_rep_count": "101"}, "human_rep_count"),
    ({"website_cycle_count": "7.4"}, "website_cycle_count"),
    ({"tracking": "invisible"}, "Tracking"),
    ({"protocol_version": "wrong"}, "protocol"),
    ({"participant_id": "Alice Smith"}, "pseudonymous"),
    ({"session_id": "session_abc "}, "padded")
])
def test_m36_invalid_rows_fail_closed(tmp_path, edit, error):
    path, rows = synthetic_csv(tmp_path)
    rows[1].update(edit)
    rewrite(path, rows)
    with pytest.raises(ValueError, match=error):
        m36.read_sessions(path)


def test_m36_insufficient_tracking_abstains_and_rejects_nonblank_count(tmp_path):
    path, rows = synthetic_csv(tmp_path)
    rows[0]["website_cycle_count"] = "0"
    rewrite(path, rows)
    with pytest.raises(ValueError, match="abstain"):
        m36.read_sessions(path)
    rows[0]["website_cycle_count"] = ""
    rows[1]["website_cycle_count"] = ""
    rewrite(path, rows)
    with pytest.raises(ValueError, match="website_cycle_count"):
        m36.read_sessions(path)


def test_m36_duplicate_sessions_and_schema_mismatch(tmp_path):
    path, rows = synthetic_csv(tmp_path)
    rows[1]["session_id"] = rows[0]["session_id"]
    rewrite(path, rows)
    with pytest.raises(ValueError, match="Duplicate session_id"):
        m36.read_sessions(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("person,exercise,manual,auto\n")
    with pytest.raises(ValueError, match="source schema"):
        m36.read_sessions(path)


def test_m36_single_person_is_small_pilot_no_interval(tmp_path):
    path, rows = synthetic_csv(tmp_path)
    rewrite(path, rows[:9])
    result = m36.report_sessions(m36.read_sessions(path))
    assert result["distinct_pseudonymous_people"] == 1
    assert result["bootstrap"] is None
    assert result["status"] == "descriptive_pilot_only"
    assert result["by_exercise"]["ohp"]["scorable"] is not None
    assert result["by_exercise"]["barbell_row"]["scorable"] is not None


def test_m36_all_abstentions_report_coverage_no_false_zero(tmp_path):
    path, rows = synthetic_csv(tmp_path)
    for row in rows:
        row["tracking"] = "insufficient"
        row["website_cycle_count"] = ""
    rewrite(path, rows)
    report = m36.report_sessions(m36.read_sessions(path))
    assert report["scorable_sessions"] == 0
    assert report["abstained_sessions"] == 36
    assert report["counting_coverage"] == 0.
    assert report["overall"] is None
    assert report["bootstrap"] is None


def test_m36_cli_audit_aggregates_without_leaking_private_rows(tmp_path):
    path, _ = synthetic_csv(tmp_path)
    destination = tmp_path / "report.json"
    args = ["audit", "--csv", str(path), "--output", str(destination)]
    assert m36.main(args) == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["scorable_sessions"] == 35
    assert m36.main(args) == 1  # no accidental overwrite
    text = destination.read_text(encoding="utf-8")
    assert "person_p" not in text
    assert "session_p" not in text
    assert "participant_id" not in text
    assert "session_id" not in text
