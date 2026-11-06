"""Experiment A: re-split probes by evidence-event holdout within each session.

For each session:
  - Partition evidence events into Seen vs Unseen
  - Train: answerable probes whose evidence ⊆ Seen (minus paraphrase holdouts)
  - eval_paraphrase: held-out probes whose evidence ⊆ Seen
  - eval_unseen: probes whose evidence is disjoint from Seen
  - eval_cross: probes whose evidence intersects both Seen and Unseen

Writes JSONL rows compatible with HypernetDataset (context/prompts/responses).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rpmem.training.sample_store import open_sample_store


def _event_id(event: dict[str, Any], index: int) -> str:
    for key in ("id", "event_id", "eid"):
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"e{index}"


def _probe_evidence(probe: dict[str, Any]) -> set[str]:
    return {str(x) for x in (probe.get("evidence_event_ids") or []) if str(x).strip()}


def _materialize_row(
    session: dict[str, Any],
    probes: list[dict[str, Any]],
    *,
    corpus_split: str,
    seen_events: list[str],
    unseen_events: list[str],
) -> dict[str, Any] | None:
    if not probes:
        return None
    row = {
        key: session[key]
        for key in (
            "session_index",
            "session_id",
            "probe_start",
            "probe_count",
            "source",
            "domain",
            "context",
            "context_tokens",
            "context_token_max",
            "context_token_counts_json",
            "context_hash",
            "simhash",
            "events_json",
            "memory_atoms_json",
            "provenance_json",
            "metadata_json",
        )
        if key in session
    }
    row.update(
        {
            "prompts": [str(probe["prompt"]) for probe in probes],
            "responses": [str(probe.get("reference", "")) for probe in probes],
            "probe_ids": [str(probe["probe_id"]) for probe in probes],
            "probe_types": [str(probe.get("probe_type", "")) for probe in probes],
            "probe_answerable": [bool(probe.get("answerable", False)) for probe in probes],
            "probe_indices": [int(probe["probe_index"]) for probe in probes],
            "probe_evidence_event_ids": [
                list(_probe_evidence(probe)) for probe in probes
            ],
            "corpus_split": corpus_split,
            "seen_events": list(seen_events),
            "unseen_events": list(unseen_events),
        }
    )
    return row


def _assign_buckets(
    probes: list[dict[str, Any]],
    seen: set[str],
) -> dict[str, list[dict[str, Any]]]:
    seen_only: list[dict[str, Any]] = []
    unseen_only: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for probe in probes:
        evidence = _probe_evidence(probe)
        if not evidence or not bool(probe.get("answerable", False)):
            skipped.append(probe)
            continue
        if not str(probe.get("reference", "")).strip():
            skipped.append(probe)
            continue
        if evidence <= seen:
            seen_only.append(probe)
        elif evidence.isdisjoint(seen):
            unseen_only.append(probe)
        else:
            cross.append(probe)
    return {
        "seen_only": seen_only,
        "unseen_only": unseen_only,
        "cross": cross,
        "skipped": skipped,
    }


def _choose_partition(
    evidence_events: list[str],
    probes: list[dict[str, Any]],
    rng: random.Random,
    *,
    unseen_event_frac: float,
    min_train_probes: int,
    min_unseen_probes: int,
    paraphrase_per_session: int,
) -> tuple[set[str], set[str], dict[str, list[dict[str, Any]]]]:
    events = sorted(set(evidence_events))
    if len(events) <= 1:
        seen = set(events)
        buckets = _assign_buckets(probes, seen)
        return seen, set(), buckets

    n_unseen = max(1, min(len(events) - 1, int(round(unseen_event_frac * len(events)))))
    order = events[:]
    rng.shuffle(order)

    best: tuple[set[str], set[str], dict[str, list[dict[str, Any]]]] | None = None
    best_score = -1
    # Rotate through candidate held-out sets.
    for start in range(len(order)):
        rotated = order[start:] + order[:start]
        unseen = set(rotated[:n_unseen])
        seen = set(events) - unseen
        buckets = _assign_buckets(probes, seen)
        n_seen = len(buckets["seen_only"])
        n_unseen_p = len(buckets["unseen_only"])
        can_train = n_seen - min(paraphrase_per_session, max(0, n_seen - 1))
        score = 0
        if can_train >= min_train_probes:
            score += 10
        if n_unseen_p >= min_unseen_probes:
            score += 10
        score += min(can_train, 8) + min(n_unseen_p, 4)
        if score > best_score:
            best_score = score
            best = (seen, unseen, buckets)
        if can_train >= min_train_probes and n_unseen_p >= min_unseen_probes:
            break

    assert best is not None
    return best


def split_session_probes(
    session: dict[str, Any],
    probes: list[dict[str, Any]],
    rng: random.Random,
    *,
    unseen_event_frac: float,
    min_train_probes: int,
    min_unseen_probes: int,
    paraphrase_per_session: int,
) -> dict[str, Any]:
    usable = []
    for probe in probes:
        evidence = _probe_evidence(probe)
        if evidence and bool(probe.get("answerable", False)) and str(
            probe.get("reference", "")
        ).strip():
            usable.append(probe)

    evidence_events = sorted({eid for probe in usable for eid in _probe_evidence(probe)})
    seen, unseen, buckets = _choose_partition(
        evidence_events,
        usable,
        rng,
        unseen_event_frac=unseen_event_frac,
        min_train_probes=min_train_probes,
        min_unseen_probes=min_unseen_probes,
        paraphrase_per_session=paraphrase_per_session,
    )

    seen_only = sorted(buckets["seen_only"], key=lambda p: int(p["probe_index"]))
    paraphrase: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    if not seen_only:
        pass
    elif len(seen_only) == 1:
        # Prefer keeping the only seen-event probe for training.
        train = seen_only
    else:
        n_para = min(paraphrase_per_session, len(seen_only) - min_train_probes)
        n_para = max(0, n_para)
        if n_para <= 0 and len(seen_only) > min_train_probes:
            n_para = min(paraphrase_per_session, len(seen_only) - min_train_probes)
        # Deterministic holdout: shuffle copy with session-local rng stream.
        local = seen_only[:]
        rng.shuffle(local)
        paraphrase = sorted(local[:n_para], key=lambda p: int(p["probe_index"]))
        para_ids = {int(p["probe_index"]) for p in paraphrase}
        train = [p for p in seen_only if int(p["probe_index"]) not in para_ids]

    return {
        "seen_events": sorted(seen),
        "unseen_events": sorted(unseen),
        "train": train,
        "paraphrase": paraphrase,
        "unseen": buckets["unseen_only"],
        "cross": buckets["cross"],
        "skipped": buckets["skipped"],
        "n_evidence_events": len(evidence_events),
    }


def _load_session_probes(
    probe_store,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    start = int(session["probe_start"])
    count = int(session["probe_count"])
    probes = []
    for probe_index in range(start, start + count):
        probe = dict(probe_store[probe_index])
        if int(probe.get("session_index", -1)) != int(session["session_index"]):
            raise ValueError(
                f"probe/session mismatch: session={session['session_index']} "
                f"probe={probe_index}"
            )
        probes.append(probe)
    return probes


def export_event_holdout_split(
    *,
    subset_jsonl: Path,
    corpus_manifest: Path,
    output_dir: Path,
    seed: int = 42,
    unseen_event_frac: float = 0.4,
    min_train_probes: int = 4,
    min_unseen_probes: int = 1,
    paraphrase_per_session: int = 1,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Probe table from the canonical corpus (same underlying sessions/probes).
    corpus = open_sample_store([corpus_manifest])
    corpus_store = corpus.stores[0]
    if not hasattr(corpus_store, "probes"):
        raise TypeError(
            f"expected CanonicalCorpusSampleStore, got {type(corpus_store)!r}"
        )
    probe_store = corpus_store.probes

    paths = {
        "train": output_dir / "train.jsonl",
        "eval_paraphrase": output_dir / "eval_paraphrase.jsonl",
        "eval_unseen": output_dir / "eval_unseen.jsonl",
        "eval_cross": output_dir / "eval_cross.jsonl",
        "session_assignments": output_dir / "session_assignments.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}

    stats: Counter[str] = Counter()
    probe_stats: Counter[str] = Counter()
    coverage_train: list[float] = []

    with subset_jsonl.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_sessions is not None and line_idx >= max_sessions:
                break
            if not line.strip():
                continue
            session = json.loads(line)
            session_rng = random.Random(rng.randint(0, 2**31 - 1))
            probes = _load_session_probes(probe_store, session)
            split = split_session_probes(
                session,
                probes,
                session_rng,
                unseen_event_frac=unseen_event_frac,
                min_train_probes=min_train_probes,
                min_unseen_probes=min_unseen_probes,
                paraphrase_per_session=paraphrase_per_session,
            )

            n_ev = max(split["n_evidence_events"], 1)
            coverage_train.append(len(split["seen_events"]) / n_ev)

            assignment = {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "seen_events": split["seen_events"],
                "unseen_events": split["unseen_events"],
                "n_evidence_events": split["n_evidence_events"],
                "n_train": len(split["train"]),
                "n_paraphrase": len(split["paraphrase"]),
                "n_unseen": len(split["unseen"]),
                "n_cross": len(split["cross"]),
                "train_probe_ids": [p["probe_id"] for p in split["train"]],
                "paraphrase_probe_ids": [p["probe_id"] for p in split["paraphrase"]],
                "unseen_probe_ids": [p["probe_id"] for p in split["unseen"]],
                "cross_probe_ids": [p["probe_id"] for p in split["cross"]],
            }
            handles["session_assignments"].write(
                json.dumps(assignment, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

            bucket_rows = [
                ("train", "train", split["train"]),
                ("eval_paraphrase", "eval_paraphrase", split["paraphrase"]),
                ("eval_unseen", "eval_unseen", split["unseen"]),
                ("eval_cross", "eval_cross", split["cross"]),
            ]
            has_train = False
            for file_key, split_name, bucket_probes in bucket_rows:
                row = _materialize_row(
                    session,
                    bucket_probes,
                    corpus_split=split_name,
                    seen_events=split["seen_events"],
                    unseen_events=split["unseen_events"],
                )
                if row is None:
                    continue
                handles[file_key].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                probe_stats[file_key] += len(bucket_probes)
                if file_key == "train":
                    has_train = True

            stats["sessions"] += 1
            if has_train:
                stats["sessions_with_train"] += 1
            if split["paraphrase"]:
                stats["sessions_with_paraphrase"] += 1
            if split["unseen"]:
                stats["sessions_with_unseen"] += 1
            if split["cross"]:
                stats["sessions_with_cross"] += 1
            if has_train and split["unseen"]:
                stats["sessions_with_train_and_unseen"] += 1
            if has_train and split["paraphrase"] and split["unseen"]:
                stats["sessions_complete_abc"] += 1

    for handle in handles.values():
        handle.close()
    corpus.close()

    manifest = {
        "format": "rpmem.event_holdout_split_v1",
        "seed": seed,
        "unseen_event_frac": unseen_event_frac,
        "min_train_probes": min_train_probes,
        "min_unseen_probes": min_unseen_probes,
        "paraphrase_per_session": paraphrase_per_session,
        "source_subset": str(subset_jsonl),
        "source_corpus": str(corpus_manifest),
        "files": {name: path.name for name, path in paths.items()},
        "stats": dict(stats),
        "probe_stats": dict(probe_stats),
        "mean_seen_event_frac": (
            sum(coverage_train) / len(coverage_train) if coverage_train else 0.0
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset_jsonl",
        type=Path,
        default=Path("data/rpmem/compiler_20k/train.jsonl"),
    )
    parser.add_argument(
        "--corpus_manifest",
        type=Path,
        default=Path("data/rpmem/compiler/train.corpus.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/rpmem/compiler_20k_event_split"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unseen_event_frac", type=float, default=0.4)
    parser.add_argument("--min_train_probes", type=int, default=4)
    parser.add_argument("--min_unseen_probes", type=int, default=1)
    parser.add_argument("--paraphrase_per_session", type=int, default=1)
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    manifest = export_event_holdout_split(
        subset_jsonl=args.subset_jsonl,
        corpus_manifest=args.corpus_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        unseen_event_frac=args.unseen_event_frac,
        min_train_probes=args.min_train_probes,
        min_unseen_probes=args.min_unseen_probes,
        paraphrase_per_session=args.paraphrase_per_session,
        max_sessions=args.max_sessions,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
