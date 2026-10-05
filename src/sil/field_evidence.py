"""Evidence-first bridge from human field observations into SIL.

The module intentionally does not infer truth, causality, competence, or authority.
It preserves a strict boundary between the observed layer and provisional
interpretation, then emits deterministic content hashes and compact audit reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "sil-field-event/1"
REPORT_SCHEMA = "sil-field-evidence-report/1"
GENESIS_DIGEST = "0" * 64

EVENT_TYPES = frozenset({"HELP", "IMPROVE", "PRESERVE", "UNKNOWN"})
EVENT_STATUSES = frozenset({"OPEN", "CLOSED", "PARTIAL", "FAILED", "EXPIRED", "REOPENED"})
STAGE_STATES = {
    "commitment": frozenset({"observed", "not_observed", "unknown"}),
    "fulfillment": frozenset({"complete", "partial", "not_complete", "unknown"}),
    "acknowledgment": frozenset({"explicit", "not_observed", "unknown"}),
    "closure": frozenset({"closed", "open", "partial", "unknown"}),
}
_ALLOWED_KEYS = frozenset(
    {
        "schema",
        "event_id",
        "event_type",
        "status",
        "observer",
        "observed_at",
        "community",
        "raw_observation",
        "raw_language",
        "participants",
        "need_problem",
        "offer_action",
        "commitment",
        "fulfillment",
        "acknowledgment",
        "closure",
        "outcome",
        "evidence",
        "unresolved",
        "provisional_interpretation",
        "preserve_candidate",
        "notes",
    }
)


class EvidenceError(ValueError):
    """Raised when an event violates the field-evidence contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: str, value: Any) -> str:
    return sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value)).hexdigest()


def _strings(name: str, value: Any, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        value = []
    if not isinstance(value, (list, tuple)):
        raise EvidenceError(f"{name} must be an array of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise EvidenceError(f"{name} cannot contain empty strings")
    if required and not result:
        raise EvidenceError(f"{name} must contain at least one item")
    return result


def _text(name: str, value: Any, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise EvidenceError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise EvidenceError(f"{name} is required")
    return value


@dataclass(frozen=True)
class StageObservation:
    """Explicitly observed state for one lifecycle stage."""

    state: str = "unknown"
    detail: str = ""

    @classmethod
    def from_value(cls, name: str, value: Any) -> "StageObservation":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise EvidenceError(f"{name} must be an object with state/detail")
        unknown = set(value) - {"state", "detail"}
        if unknown:
            raise EvidenceError(f"{name} has unsupported fields: {sorted(unknown)}")
        state = _text(f"{name}.state", value.get("state", "unknown"), required=True).lower()
        if state not in STAGE_STATES[name]:
            allowed = ", ".join(sorted(STAGE_STATES[name]))
            raise EvidenceError(f"{name}.state must be one of: {allowed}")
        detail = _text(f"{name}.detail", value.get("detail", ""))
        return cls(state=state, detail=detail)

    def as_dict(self) -> dict[str, str]:
        return {"state": self.state, "detail": self.detail}


@dataclass(frozen=True)
class FieldEvent:
    """One immutable field record with interpretation kept in a separate layer."""

    event_id: str
    event_type: str
    status: str
    observer: str
    raw_observation: tuple[str, ...]
    evidence: tuple[str, ...]
    observed_at: str = ""
    community: str = ""
    raw_language: tuple[str, ...] = field(default_factory=tuple)
    participants: tuple[str, ...] = field(default_factory=tuple)
    need_problem: str = ""
    offer_action: tuple[str, ...] = field(default_factory=tuple)
    commitment: StageObservation = field(default_factory=StageObservation)
    fulfillment: StageObservation = field(default_factory=StageObservation)
    acknowledgment: StageObservation = field(default_factory=StageObservation)
    closure: StageObservation = field(default_factory=StageObservation)
    outcome: tuple[str, ...] = field(default_factory=tuple)
    unresolved: tuple[str, ...] = field(default_factory=tuple)
    provisional_interpretation: tuple[str, ...] = field(default_factory=tuple)
    preserve_candidate: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema: str = SCHEMA

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldEvent":
        if not isinstance(value, dict):
            raise EvidenceError("event must be a JSON object")
        unknown = set(value) - _ALLOWED_KEYS
        if unknown:
            raise EvidenceError(f"unsupported event fields: {sorted(unknown)}")

        schema = _text("schema", value.get("schema", SCHEMA), required=True)
        if schema != SCHEMA:
            raise EvidenceError(f"unsupported schema: {schema}")

        event_type = _text("event_type", value.get("event_type"), required=True).upper()
        status = _text("status", value.get("status"), required=True).upper()
        if event_type not in EVENT_TYPES:
            allowed = ", ".join(sorted(EVENT_TYPES))
            raise EvidenceError(f"event_type must be one of: {allowed}")
        if status not in EVENT_STATUSES:
            allowed = ", ".join(sorted(EVENT_STATUSES))
            raise EvidenceError(f"status must be one of: {allowed}")

        preserve_candidate = value.get("preserve_candidate", False)
        if not isinstance(preserve_candidate, bool):
            raise EvidenceError("preserve_candidate must be boolean")

        event = cls(
            event_id=_text("event_id", value.get("event_id"), required=True),
            event_type=event_type,
            status=status,
            observer=_text("observer", value.get("observer"), required=True),
            raw_observation=_strings(
                "raw_observation", value.get("raw_observation"), required=True
            ),
            evidence=_strings("evidence", value.get("evidence"), required=True),
            observed_at=_text("observed_at", value.get("observed_at", "")),
            community=_text("community", value.get("community", "")),
            raw_language=_strings("raw_language", value.get("raw_language")),
            participants=_strings("participants", value.get("participants")),
            need_problem=_text("need_problem", value.get("need_problem", "")),
            offer_action=_strings("offer_action", value.get("offer_action")),
            commitment=StageObservation.from_value("commitment", value.get("commitment")),
            fulfillment=StageObservation.from_value("fulfillment", value.get("fulfillment")),
            acknowledgment=StageObservation.from_value(
                "acknowledgment", value.get("acknowledgment")
            ),
            closure=StageObservation.from_value("closure", value.get("closure")),
            outcome=_strings("outcome", value.get("outcome")),
            unresolved=_strings("unresolved", value.get("unresolved")),
            provisional_interpretation=_strings(
                "provisional_interpretation", value.get("provisional_interpretation")
            ),
            preserve_candidate=preserve_candidate,
            notes=_strings("notes", value.get("notes")),
            schema=schema,
        )
        event._validate_cross_fields()
        return event

    def _validate_cross_fields(self) -> None:
        if self.status == "CLOSED" and self.closure.state == "open":
            raise EvidenceError("status CLOSED cannot have closure.state=open")
        if self.status == "OPEN" and self.closure.state == "closed":
            raise EvidenceError("status OPEN cannot have closure.state=closed")

    def observation_payload(self) -> dict[str, Any]:
        """Return the factual/field layer only.

        Provisional interpretation, preserve candidacy and analyst notes are
        deliberately excluded. A later analytical change therefore cannot alter
        the identity of the observation layer.
        """
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "status": self.status,
            "observer": self.observer,
            "observed_at": self.observed_at,
            "community": self.community,
            "raw_observation": list(self.raw_observation),
            "raw_language": list(self.raw_language),
            "participants": list(self.participants),
            "need_problem": self.need_problem,
            "offer_action": list(self.offer_action),
            "commitment": self.commitment.as_dict(),
            "fulfillment": self.fulfillment.as_dict(),
            "acknowledgment": self.acknowledgment.as_dict(),
            "closure": self.closure.as_dict(),
            "outcome": list(self.outcome),
            "evidence": list(self.evidence),
            "unresolved": list(self.unresolved),
        }

    def record_payload(self) -> dict[str, Any]:
        payload = self.observation_payload()
        payload.update(
            {
                "provisional_interpretation": list(self.provisional_interpretation),
                "preserve_candidate": self.preserve_candidate,
                "notes": list(self.notes),
            }
        )
        return payload

    def observation_digest(self) -> str:
        return _digest("sil-field-observation/1", self.observation_payload())

    def record_digest(self) -> str:
        return _digest("sil-field-record/1", self.record_payload())


@dataclass(frozen=True)
class LedgerEntry:
    sequence: int
    event_id: str
    observation_digest: str
    record_digest: str
    previous_chain_digest: str
    chain_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "observation_digest": self.observation_digest,
            "record_digest": self.record_digest,
            "previous_chain_digest": self.previous_chain_digest,
            "chain_digest": self.chain_digest,
        }


class FieldLedger:
    """Append-only audit view over field events.

    ``chain_root`` preserves ingestion order. ``observation_set_root`` ignores
    arrival order and is suitable for checking whether two observers hold the
    same observation set without pretending that the events are true or complete.
    """

    def __init__(self) -> None:
        self._events: list[FieldEvent] = []
        self._entries: list[LedgerEntry] = []
        self._ids: set[str] = set()

    @property
    def events(self) -> tuple[FieldEvent, ...]:
        return tuple(self._events)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def chain_root(self) -> str:
        return self._entries[-1].chain_digest if self._entries else GENESIS_DIGEST

    def append(self, event: FieldEvent) -> LedgerEntry:
        if event.event_id in self._ids:
            raise EvidenceError(f"duplicate event_id: {event.event_id}")
        previous = self.chain_root
        sequence = len(self._entries) + 1
        observation_digest = event.observation_digest()
        record_digest = event.record_digest()
        chain_digest = _digest(
            "sil-field-chain/1",
            {
                "sequence": sequence,
                "event_id": event.event_id,
                "record_digest": record_digest,
                "previous_chain_digest": previous,
            },
        )
        entry = LedgerEntry(
            sequence=sequence,
            event_id=event.event_id,
            observation_digest=observation_digest,
            record_digest=record_digest,
            previous_chain_digest=previous,
            chain_digest=chain_digest,
        )
        self._events.append(event)
        self._entries.append(entry)
        self._ids.add(event.event_id)
        return entry

    def observation_set_root(self) -> str:
        members = [
            {"event_id": event.event_id, "observation_digest": event.observation_digest()}
            for event in self._events
        ]
        members.sort(key=lambda item: (item["event_id"], item["observation_digest"]))
        return _digest("sil-field-observation-set/1", members)

    def verify(self) -> bool:
        previous = GENESIS_DIGEST
        if len(self._events) != len(self._entries):
            return False
        seen: set[str] = set()
        for sequence, (event, entry) in enumerate(
            zip(self._events, self._entries), start=1
        ):
            if event.event_id in seen or entry.sequence != sequence:
                return False
            if entry.event_id != event.event_id:
                return False
            if entry.previous_chain_digest != previous:
                return False
            if entry.observation_digest != event.observation_digest():
                return False
            if entry.record_digest != event.record_digest():
                return False
            expected = _digest(
                "sil-field-chain/1",
                {
                    "sequence": sequence,
                    "event_id": event.event_id,
                    "record_digest": entry.record_digest,
                    "previous_chain_digest": previous,
                },
            )
            if entry.chain_digest != expected:
                return False
            seen.add(event.event_id)
            previous = expected
        return True


def load_events(path: Path) -> list[FieldEvent]:
    """Load either a JSON array or newline-delimited JSON field records."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise EvidenceError("JSON document must be an array of event objects")
    else:
        raw = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"invalid JSONL at line {line_no}: {exc.msg}"
                ) from exc
    return [FieldEvent.from_dict(item) for item in raw]


def build_report(events: Iterable[FieldEvent], *, source: str = "") -> dict[str, Any]:
    """Create a deterministic audit summary without semantic scoring."""
    ledger = FieldLedger()
    type_counts = {name: 0 for name in sorted(EVENT_TYPES)}
    status_counts = {name: 0 for name in sorted(EVENT_STATUSES)}
    open_loops: list[str] = []
    preserve_candidates: list[str] = []

    for event in events:
        ledger.append(event)
        type_counts[event.event_type] += 1
        status_counts[event.status] += 1
        if event.status != "CLOSED" or event.unresolved:
            open_loops.append(event.event_id)
        if event.preserve_candidate:
            preserve_candidates.append(event.event_id)

    return {
        "schema": REPORT_SCHEMA,
        "source": source,
        "event_count": len(ledger.events),
        "verified": ledger.verify(),
        "chain_root": ledger.chain_root,
        "observation_set_root": ledger.observation_set_root(),
        "type_counts": type_counts,
        "status_counts": status_counts,
        "open_loops": open_loops,
        "preserve_candidates": preserve_candidates,
        "entries": [entry.as_dict() for entry in ledger.entries],
    }
