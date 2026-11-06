"""Generate auxiliary Experiment-B evaluation probes with an instruct LLM.

Experiment B supplements the event-holdout split with questions that change
*focus* or *extraction style* relative to the original fact_recall probes.

Generated probes are evaluation-only (never mixed into train.jsonl). Answers
must be natural-language strings grounded in the bound event text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

SYSTEM_PROMPT = """You are creating evaluation questions for a memory/extraction benchmark.
Return ONLY a JSON object. Do not use markdown fences.
All answers must be short natural-language phrases copied or lightly paraphrased
from the provided event text. Never answer with bare event ids like e0 or e1.
"""

TYPE_SPECS = {
    "perspective_shift": {
        "instruction": (
            "Write ONE question about the SAME event(s), but with a DIFFERENT focus "
            "than the blocked questions. Examples of focus shifts: quantity↔reason, "
            "entity↔time, action↔outcome, location↔participant. "
            "The answer must be explicitly supported by the event text."
        ),
        "need_events": 1,
    },
    "aggregate_compare": {
        "instruction": (
            "Write ONE question that requires combining information across the "
            "provided events (compare, summarize, relate cause/effect across events). "
            "The answer must be explicitly supported by those events together."
        ),
        "need_events": 2,
    },
    "unanswerable": {
        "instruction": (
            "Write ONE question that looks related to the event topic but CANNOT be "
            "answered from the event text (ask for a missing attribute/detail). "
            "Set answer to exactly: NOT_IN_CONTEXT"
        ),
        "need_events": 1,
    },
}


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:240]!r}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("decoded JSON is not an object")
    return payload


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower()) if len(tok) > 1}


def _answer_grounded(answer: str, event_texts: list[str], *, min_overlap: float = 0.2) -> bool:
    answer = answer.strip()
    if not answer or re.fullmatch(r"e\d+", answer):
        return False
    if answer.upper() == "NOT_IN_CONTEXT":
        return True
    ans_toks = _tokenize(answer)
    if not ans_toks:
        return False
    corpus = "\n".join(event_texts)
    # Direct containment of a substantial answer fragment.
    if len(answer) >= 8 and answer.lower() in corpus.lower():
        return True
    corp_toks = _tokenize(corpus)
    if not corp_toks:
        return False
    overlap = len(ans_toks & corp_toks) / max(len(ans_toks), 1)
    return overlap >= min_overlap


def _parse_events(events_json: str | list | dict) -> list[dict[str, Any]]:
    if isinstance(events_json, str):
        events = json.loads(events_json)
    else:
        events = events_json
    out = []
    for index, event in enumerate(events):
        eid = str(event.get("id") or event.get("event_id") or f"e{index}")
        out.append(
            {
                "id": eid,
                "role": event.get("role"),
                "type": event.get("type"),
                "content": str(event.get("content") or ""),
            }
        )
    return out


def _build_user_prompt(
    *,
    probe_type: str,
    events: list[dict[str, Any]],
    blocked_questions: list[str],
) -> str:
    spec = TYPE_SPECS[probe_type]
    event_blocks = []
    for event in events:
        event_blocks.append(
            f"[event_id={event['id']} role={event.get('role')} type={event.get('type')}]\n"
            f"{event['content'][:2500]}"
        )
    blocked = "\n".join(f"- {q}" for q in blocked_questions[:12]) or "- (none)"
    return (
        f"Task type: {probe_type}\n"
        f"{spec['instruction']}\n\n"
        f"Events:\n"
        + "\n\n".join(event_blocks)
        + "\n\nBlocked questions (do NOT paraphrase these; choose a different focus):\n"
        + blocked
        + "\n\nReturn JSON with keys:\n"
        '  "prompt": string question,\n'
        '  "answer": string natural-language answer (or NOT_IN_CONTEXT),\n'
        '  "evidence_event_ids": array of event ids used,\n'
        '  "focus": short string describing the new focus/extraction style\n'
    )


class JsonGenerator(Protocol):
    def generate_json(self, user_prompt: str) -> dict[str, Any]:
        ...


class LocalInstructGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.3,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                from rpmem.device import resolve_device

                self.device = resolve_device()
            except Exception:
                self.device = torch.device("cpu")
        print(f"[experiment_b] loading local model {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_json(self, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            output = self.model.generate(**inputs, **kwargs)
        decoded = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return _extract_json_object(decoded)


class OpenAICompatibleGenerator:
    """Chat Completions client for OpenAI-compatible endpoints (e.g. gpt-5.1)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        max_new_tokens: int = 384,
        temperature: float = 0.3,
        retries: int = 5,
        timeout: int = 180,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        print(f"[experiment_b] using API model={model} base_url={base_url}")

    def generate_json(self, user_prompt: str) -> dict[str, Any]:
        # Some OpenAI-compatible gateways reject response_format for gpt-5.x;
        # rely on the system/user prompt + JSON extraction instead.
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                content = str(payload["choices"][0]["message"]["content"])
                return _extract_json_object(content)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 20))
        assert last_error is not None
        raise last_error


def _load_exports_from_shell_script(path: Path) -> dict[str, str]:
    """Parse export FOO='bar' assignments from an existing launcher script."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    found: dict[str, str] = {}
    for match in re.finditer(
        r"""(?:export\s+)?(OPENAI_API_KEY|OPENAI_BASE_URL|LLM_MODEL|EVAL_MODEL)=['\"]([^'\"]+)['\"]""",
        text,
    ):
        found[match.group(1)] = match.group(2)
    return found


def build_generator(args: argparse.Namespace) -> JsonGenerator:
    if args.provider == "openai":
        if args.credentials_script:
            exported = _load_exports_from_shell_script(Path(args.credentials_script))
            for key, value in exported.items():
                os.environ.setdefault(key, value)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = args.base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.xi-ai.cn/v1"
        )
        model = args.model or os.environ.get("LLM_MODEL", "gpt-5.1")
        return OpenAICompatibleGenerator(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    model_path = args.model_path or args.model
    if not model_path:
        raise ValueError("--model_path is required for provider=local")
    return LocalInstructGenerator(
        model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )


def _materialize_eval_row(
    session_row: dict[str, Any],
    probes: list[dict[str, Any]],
    *,
    corpus_split: str,
) -> dict[str, Any]:
    row = {
        key: session_row[key]
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
            "seen_events",
            "unseen_events",
        )
        if key in session_row
    }
    row.update(
        {
            "prompts": [p["prompt"] for p in probes],
            "responses": [p["answer"] for p in probes],
            "probe_ids": [p["probe_id"] for p in probes],
            "probe_types": [p["probe_type"] for p in probes],
            "probe_answerable": [p["answerable"] for p in probes],
            "probe_evidence_event_ids": [p["evidence_event_ids"] for p in probes],
            "probe_focus": [p.get("focus", "") for p in probes],
            "corpus_split": corpus_split,
        }
    )
    return row


def _load_index(split_dir: Path) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    assignments = {}
    with (split_dir / "session_assignments.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assignments[int(row["session_index"])] = row

    train_by_si = {}
    with (split_dir / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            train_by_si[int(row["session_index"])] = row

    # Prefer train row for context; fall back to paraphrase/unseen if needed.
    context_by_si = dict(train_by_si)
    for name in ("eval_paraphrase.jsonl", "eval_unseen.jsonl"):
        path = split_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                si = int(row["session_index"])
                context_by_si.setdefault(si, row)
    return assignments, train_by_si, context_by_si


def _blocked_questions(assignment: dict[str, Any], train_row: dict[str, Any] | None) -> list[str]:
    blocked = []
    if train_row:
        blocked.extend(train_row.get("prompts") or [])
    return blocked


def generate_for_session(
    generator: InstructGenerator,
    *,
    assignment: dict[str, Any],
    session_row: dict[str, Any],
    rng: random.Random,
    types: list[str],
) -> list[dict[str, Any]]:
    events = _parse_events(session_row["events_json"])
    by_id = {e["id"]: e for e in events}
    seen = set(assignment.get("seen_events") or session_row.get("seen_events") or [])
    unseen = set(assignment.get("unseen_events") or session_row.get("unseen_events") or [])
    blocked = _blocked_questions(assignment, session_row)

    created: list[dict[str, Any]] = []
    for probe_type in types:
        need = TYPE_SPECS[probe_type]["need_events"]
        if probe_type == "aggregate_compare":
            # Prefer mixing one unseen + one seen when available.
            pool: list[str] = []
            if unseen and seen:
                pool = [rng.choice(sorted(unseen)), rng.choice(sorted(seen))]
            elif len(unseen) >= 2:
                pool = rng.sample(sorted(unseen), 2)
            elif len(seen) >= 2:
                pool = rng.sample(sorted(seen), 2)
            else:
                continue
        else:
            # Perspective / unanswerable: prioritize unseen events.
            cand = sorted(unseen) if unseen else sorted(seen)
            if len(cand) < need:
                continue
            pool = rng.sample(cand, need)

        selected = [by_id[eid] for eid in pool if eid in by_id and by_id[eid]["content"].strip()]
        if len(selected) < need:
            continue

        user_prompt = _build_user_prompt(
            probe_type=probe_type,
            events=selected,
            blocked_questions=blocked,
        )
        try:
            payload = generator.generate_json(user_prompt)
        except Exception as exc:  # noqa: BLE001
            created.append(
                {
                    "status": "error",
                    "probe_type": probe_type,
                    "error": str(exc),
                    "session_index": assignment["session_index"],
                }
            )
            continue

        prompt = str(payload.get("prompt") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        evidence = [str(x) for x in (payload.get("evidence_event_ids") or [])]
        focus = str(payload.get("focus") or "").strip()
        allowed = {e["id"] for e in selected}
        if not evidence:
            evidence = [e["id"] for e in selected]
        if not prompt or not answer:
            created.append(
                {
                    "status": "reject_empty",
                    "probe_type": probe_type,
                    "session_index": assignment["session_index"],
                    "raw": payload,
                }
            )
            continue
        if any(eid not in allowed for eid in evidence):
            created.append(
                {
                    "status": "reject_evidence",
                    "probe_type": probe_type,
                    "session_index": assignment["session_index"],
                    "raw": payload,
                }
            )
            continue
        event_texts = [by_id[eid]["content"] for eid in evidence if eid in by_id]
        if probe_type == "unanswerable":
            ok = answer.upper() == "NOT_IN_CONTEXT"
        else:
            ok = _answer_grounded(answer, event_texts)
        if not ok:
            created.append(
                {
                    "status": "reject_ungrounded",
                    "probe_type": probe_type,
                    "session_index": assignment["session_index"],
                    "raw": payload,
                }
            )
            continue

        created.append(
            {
                "status": "ok",
                "probe_id": f"b_{probe_type}_{assignment['session_index']}_{len(created)}",
                "probe_type": probe_type,
                "prompt": prompt,
                "answer": answer if probe_type != "unanswerable" else "NOT_IN_CONTEXT",
                "answerable": probe_type != "unanswerable",
                "evidence_event_ids": evidence,
                "focus": focus,
                "session_index": assignment["session_index"],
                "anchor_event_ids": [e["id"] for e in selected],
            }
        )
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=Path("data/rpmem/compiler_20k_event_split"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/rpmem/compiler_20k_event_split/experiment_b"),
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "local"),
        default="openai",
        help="openai = OpenAI-compatible chat API; local = HuggingFace weights",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="API model name (default: $LLM_MODEL or gpt-5.1)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/aistor/sjtu/hpc_stor01/home/sunwenjie/workshop/models/Qwen2.5-7B-Instruct",
        help="Local HF model path when --provider local",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY for provider=openai",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Override OPENAI_BASE_URL for provider=openai",
    )
    parser.add_argument(
        "--credentials_script",
        type=str,
        default=None,
        help=(
            "Existing shell launcher that already contains "
            "OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL exports"
        ),
    )
    parser.add_argument("--max_sessions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["perspective_shift", "aggregate_compare", "unanswerable"],
        choices=list(TYPE_SPECS),
    )
    parser.add_argument("--require_unseen", action="store_true", default=True)
    parser.add_argument("--no_require_unseen", action="store_false", dest="require_unseen")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device for provider=local, e.g. cpu / npu / cuda.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    assignments, train_by_si, context_by_si = _load_index(args.split_dir)
    candidates = []
    for si, assignment in assignments.items():
        if args.require_unseen and int(assignment.get("n_unseen", 0)) <= 0:
            continue
        if si not in context_by_si:
            continue
        if int(assignment.get("n_train", 0)) <= 0 and si not in train_by_si:
            # still ok if context exists
            pass
        candidates.append(si)
    rng.shuffle(candidates)
    selected = candidates[: args.max_sessions]
    model_label = (
        args.model
        or os.environ.get("LLM_MODEL", "gpt-5.1")
        if args.provider == "openai"
        else args.model_path
    )
    print(
        f"selected_sessions={len(selected)} / candidates={len(candidates)} "
        f"types={args.types} provider={args.provider} model={model_label}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "experiment_b_raw.jsonl"
    out_paths = {
        "perspective_shift": args.output_dir / "eval_b_perspective.jsonl",
        "aggregate_compare": args.output_dir / "eval_b_aggregate.jsonl",
        "unanswerable": args.output_dir / "eval_b_unanswerable.jsonl",
    }
    # Resume-friendly: skip sessions already present in raw file.
    done_sessions: set[int] = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_sessions.add(int(row.get("session_index", -1)))
        print(f"resume: {len(done_sessions)} sessions already in raw log")

    generator = build_generator(args)

    stats: Counter[str] = Counter()
    bucket_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {} for name in out_paths
    }
    t0 = time.time()
    with raw_path.open("a", encoding="utf-8") as raw_handle:
        for idx, si in enumerate(selected):
            if si in done_sessions:
                stats["skipped_resume"] += 1
                continue
            assignment = assignments[si]
            session_row = context_by_si[si]
            created = generate_for_session(
                generator,
                assignment=assignment,
                session_row=session_row,
                rng=random.Random(rng.randint(0, 2**31 - 1)),
                types=args.types,
            )
            raw_handle.write(
                json.dumps(
                    {
                        "session_index": si,
                        "session_id": assignment.get("session_id"),
                        "created": created,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            raw_handle.flush()
            stats["sessions_processed"] += 1
            for item in created:
                stats[item.get("status", "unknown")] += 1
                stats[f"type_{item.get('probe_type', 'na')}"] += 1
                if item.get("status") != "ok":
                    continue
                ptype = item["probe_type"]
                bucket_rows[ptype].setdefault(si, []).append(item)
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"[{idx+1}/{len(selected)}] ok={stats['ok']} "
                    f"reject={stats['reject_ungrounded']+stats['reject_evidence']+stats['reject_empty']} "
                    f"err={stats['error']} elapsed_min={elapsed/60:.1f}"
                )

    # Rebuild accepted probes from the full raw log (supports resume).
    bucket_rows = {name: {} for name in out_paths}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                si = int(payload["session_index"])
                for item in payload.get("created") or []:
                    if item.get("status") != "ok":
                        continue
                    ptype = item["probe_type"]
                    if ptype not in bucket_rows:
                        continue
                    bucket_rows[ptype].setdefault(si, []).append(item)

    # Materialize one eval row per session per type.
    for ptype, path in out_paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for si, probes in sorted(bucket_rows[ptype].items()):
                row = _materialize_eval_row(
                    context_by_si[si],
                    probes,
                    corpus_split=f"eval_b_{ptype}",
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats[f"rows_{ptype}"] += 1
                stats[f"probes_{ptype}"] += len(probes)

    manifest = {
        "format": "rpmem.experiment_b_probes_v1",
        "split_dir": str(args.split_dir),
        "provider": args.provider,
        "model": model_label,
        "model_path": args.model_path if args.provider == "local" else None,
        "base_url": (
            args.base_url
            or os.environ.get("OPENAI_BASE_URL", "https://api.xi-ai.cn/v1")
            if args.provider == "openai"
            else None
        ),
        "seed": args.seed,
        "max_sessions": args.max_sessions,
        "types": args.types,
        "require_unseen": args.require_unseen,
        "stats": dict(stats),
        "files": {k: v.name for k, v in out_paths.items()}
        | {"raw": raw_path.name},
    }
    (args.output_dir / "manifest_b.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
