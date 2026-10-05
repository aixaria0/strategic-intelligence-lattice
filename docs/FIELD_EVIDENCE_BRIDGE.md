# Field Evidence Bridge v0.8

## Why this exists

SIL began as a synthetic decision-research lab. The Intelligence Lattice plan later identified a missing boundary between **human observation** and **machine interpretation**: if real-world events are ever allowed into a collective-intelligence system, the system must not silently rewrite observations to fit a model.

`src/sil/field_evidence.py` is the first small bridge across that boundary. It does **not** make SIL a real-world decision authority. It provides a strict, machine-inspectable envelope for field observations and an auditable way to summarize them without converting them into truth scores.

The design principle is:

> observation first; interpretation remains provisional.

## Epistemic separation

Each `sil-field-event/1` record has two layers.

### Observation layer

Included in the stable `observation_digest`:

- event id/type/status and observer
- raw observation
- original/raw language
- participants or pseudonyms
- need/problem and observed response/action
- explicit lifecycle states for commitment, fulfillment, acknowledgment and closure
- observed outcome
- evidence/provenance text
- unresolved items

### Analytical layer

Excluded from `observation_digest`, but included in the full immutable `record_digest`:

- provisional interpretation
- `preserve_candidate`
- analyst notes

Changing an interpretation therefore cannot change the identity of the underlying observation. Once a full record is appended to a ledger, however, that record is immutable: a revised analysis should be emitted as a later record/version rather than silently replacing history.

## No semantic inference

The bridge intentionally does not infer lifecycle state from free text.

For example, the system will not convert "thanks" into objective validation, or an answer into successful fulfillment. The recorder explicitly declares the state observed for each stage:

```json
{
  "commitment": {"state": "not_observed", "detail": "Response was immediate."},
  "fulfillment": {"state": "complete", "detail": "Requested object was received."},
  "acknowledgment": {"state": "explicit", "detail": "Requester said thanks."},
  "closure": {"state": "closed", "detail": "Immediate need ended."}
}
```

Allowed event types are `HELP`, `IMPROVE`, `PRESERVE`, and `UNKNOWN`. Allowed statuses are `OPEN`, `CLOSED`, `PARTIAL`, `FAILED`, `EXPIRED`, and `REOPENED`.

A `CLOSED` event may still contain `unresolved` follow-up questions. This is deliberate: consultation can be closed while the downstream outcome remains unknown.

## Two roots, two meanings

The report exposes two different hashes:

- `chain_root` — order-sensitive append history. Reordering the same events changes it.
- `observation_set_root` — order-independent root over `(event_id, observation_digest)` pairs. Two replicas with the same complete observation set obtain the same set root regardless of arrival order.

Neither root proves that an observation is true. They prove deterministic identity of the bytes accepted by this implementation.

## Run

A synthetic fixture is provided so CI and users do not require private human data:

```bash
sil field-lab \
  --input examples/field-events.synthetic.jsonl \
  --output field-report.json
```

The command accepts either newline-delimited JSON or a JSON array.

The output includes:

- deterministic event digests
- append-chain root
- order-independent observation-set root
- counts by event type and status
- events with unresolved/open loops
- declared PRESERVE candidates
- ledger self-verification result

## What this does not do

v0.8 does not:

- decide whether an observation is true
- infer causality
- rank people by trust
- turn successful help into governance power
- validate medical, scientific, social or economic claims
- sign observations cryptographically
- replicate them over a network
- deduplicate two semantically similar events with different ids
- execute actions
- ingest private Sweetwater or other participant data into this public repository

Cryptographic actor attribution, durable journals, multi-process replication, challenge/evidence requests and deterministic verifier receipts belong to the separate Intelligence Lattice event-core slice described in `docs/INTELLIGENCE_LATTICE_PHASE0.md`.

## Why this matters to the larger lattice

The distributed lattice needs a clean boundary before scheduling, consensus or adaptive computation can safely operate on human-derived information:

```text
real event
  -> attributable observation
  -> immutable evidence envelope
  -> unresolved/closed lifecycle state
  -> provisional interpretation
  -> later validation or falsification
  -> only then: allocation / coordination / decision procedures
```

That ordering prevents the computation layer from laundering interpretation into observation. It also creates a future interface where experimentally tested SIL allocation algorithms can decide **where to spend verification effort** without being allowed to manufacture the evidence they are prioritizing.

## Privacy and public-repository rule

This repository is public. Do not commit participant names, sensitive health data, private conversations, precise locations, credentials, or raw field logs that were not explicitly cleared for publication.

Use pseudonyms and synthetic fixtures for tests. Real field records should remain in an appropriate private store unless disclosure is intentional and authorized.
