from dataclasses import replace
import json
from pathlib import Path

import pytest

from sil.field_evidence import (
    EvidenceError,
    FieldEvent,
    FieldLedger,
    build_report,
    load_events,
)


def base_event(**overrides):
    data = {
        "event_id": "EX-001",
        "event_type": "HELP",
        "status": "CLOSED",
        "observer": "observer-a",
        "raw_observation": [
            "Participant A asked for a tissue.",
            "Observer provided it.",
        ],
        "raw_language": ["Participant A asked for a tissue."],
        "participants": ["participant-a", "observer-a"],
        "need_problem": "A tissue was needed.",
        "offer_action": ["Observer provided the tissue."],
        "commitment": {
            "state": "not_observed",
            "detail": "Response was immediate.",
        },
        "fulfillment": {"state": "complete", "detail": "Tissue was received."},
        "acknowledgment": {
            "state": "explicit",
            "detail": "Participant A said thanks.",
        },
        "closure": {"state": "closed", "detail": "Immediate need closed."},
        "outcome": ["Immediate need satisfied."],
        "evidence": ["Synthetic fixture."],
        "unresolved": [],
        "provisional_interpretation": ["Immediate HELP closure."],
        "preserve_candidate": False,
    }
    data.update(overrides)
    return FieldEvent.from_dict(data)


def test_interpretation_is_not_part_of_observation_digest():
    event = base_event()
    changed = replace(event, provisional_interpretation=("Different analysis.",))
    assert event.observation_digest() == changed.observation_digest()
    assert event.record_digest() != changed.record_digest()


def test_set_root_is_order_independent_but_chain_root_is_not():
    first = base_event()
    second = base_event(
        event_id="EX-002",
        status="OPEN",
        closure={"state": "open", "detail": "Still pending."},
        fulfillment={"state": "not_complete", "detail": "Not fulfilled."},
        unresolved=["Need remains open."],
    )
    a, b = FieldLedger(), FieldLedger()
    a.append(first)
    a.append(second)
    b.append(second)
    b.append(first)
    assert a.observation_set_root() == b.observation_set_root()
    assert a.chain_root != b.chain_root
    assert a.verify() and b.verify()


def test_duplicate_event_ids_fail_closed():
    ledger = FieldLedger()
    ledger.append(base_event())
    with pytest.raises(EvidenceError, match="duplicate event_id"):
        ledger.append(base_event())


def test_closed_status_cannot_claim_open_closure():
    with pytest.raises(EvidenceError, match="status CLOSED"):
        base_event(closure={"state": "open", "detail": "Contradiction."})


def test_report_keeps_closed_event_with_unresolved_followup_open():
    event = base_event(unresolved=["Long-term outcome not observed."])
    report = build_report([event], source="fixture")
    assert report["verified"] is True
    assert report["open_loops"] == ["EX-001"]
    assert report["event_count"] == 1


def test_load_jsonl(tmp_path: Path):
    raw = base_event().record_payload()
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    loaded = load_events(path)
    assert len(loaded) == 1
    assert loaded[0].event_id == "EX-001"
