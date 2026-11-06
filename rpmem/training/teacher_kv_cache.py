"""Optional teacher precompute path that reuses a shared context KV prefix.

Default teacher precompute runs a full forward per (prompt, response). When many
pairs share a long common token prefix (the session context inside the chat
template), this module:

  1. prefills that common prefix once with ``use_cache=True``;
  2. clones the resulting past KV for each pair;
  3. runs only the pair-specific suffix.

Correctness requires the shared region to be an *exact token prefix* of each
full teacher sequence (longest common prefix). BPE boundary effects are handled
automatically: if the common prefix is too short, callers should fall back to a
full forward.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import Tensor


def longest_common_prefix_len(sequences: Sequence[Sequence[int]]) -> int:
    """Return the length of the shared leading token prefix across sequences."""

    if not sequences:
        return 0
    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        limit = min(len(prefix), len(sequence))
        cut = 0
        while cut < limit and prefix[cut] == sequence[cut]:
            cut += 1
        prefix = prefix[:cut]
        if not prefix:
            return 0
    return len(prefix)


def clone_past_key_values(past: Any) -> Any:
    """Deep-copy HF past_key_values (tuple legacy cache or DynamicCache)."""

    if past is None:
        return None
    if hasattr(past, "to_legacy_cache") or past.__class__.__name__ == "DynamicCache":
        return deepcopy(past)
    return tuple(tuple(tensor.clone() for tensor in layer) for layer in past)


def _gather_prediction_logits(
    *,
    prefill_logits: Tensor,
    suffix_logits: Tensor,
    prediction_positions: Sequence[int],
    prefix_len: int,
) -> Tensor:
    """Gather next-token logits at absolute prediction positions.

    ``prefill_logits``: [prefix_len, vocab]
    ``suffix_logits``: [suffix_len, vocab]
    ``prediction_positions``: absolute indices into the full sequence where the
    model emits the logit used for the following label token.
    """

    rows: list[Tensor] = []
    for position in prediction_positions:
        if position < 0:
            raise ValueError("prediction positions must be non-negative")
        if position < prefix_len:
            rows.append(prefill_logits[position])
        else:
            local = position - prefix_len
            if local >= suffix_logits.shape[0]:
                raise ValueError(
                    "prediction position falls outside prefill+suffix logits: "
                    f"position={position} prefix_len={prefix_len} "
                    f"suffix_len={suffix_logits.shape[0]}"
                )
            rows.append(suffix_logits[local])
    return torch.stack(rows, dim=0)


def response_logits_with_prefix_kv(
    model,
    *,
    full_input_ids: list[int],
    label_positions: Sequence[int],
    prefix_len: int,
    past_key_values: Any,
    prefill_logits: Tensor,
    device: torch.device,
) -> Tensor:
    """Continue from a cached prefix and return logits at response labels.

    Returns float logits with shape ``[n_labels, vocab]``.
    """

    if prefix_len <= 0 or prefix_len > len(full_input_ids):
        raise ValueError(f"invalid prefix_len={prefix_len} for seq_len={len(full_input_ids)}")
    if not label_positions:
        raise ValueError("label_positions must be non-empty")

    suffix_ids = full_input_ids[prefix_len:]
    prediction_positions = [int(position) - 1 for position in label_positions]
    if any(position < 0 for position in prediction_positions):
        raise ValueError("response labels must have a preceding prediction position")

    if not suffix_ids:
        # Entire sequence lives in the prefix (unusual); read from prefill only.
        return _gather_prediction_logits(
            prefill_logits=prefill_logits,
            suffix_logits=prefill_logits.new_zeros((0, prefill_logits.shape[-1])),
            prediction_positions=prediction_positions,
            prefix_len=prefix_len,
        )

    suffix = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones(
        1, prefix_len + len(suffix_ids), dtype=torch.long, device=device
    )
    position_ids = torch.arange(
        prefix_len, prefix_len + len(suffix_ids), device=device
    ).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(
            input_ids=suffix,
            past_key_values=clone_past_key_values(past_key_values),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    suffix_logits = outputs.logits[0]
    return _gather_prediction_logits(
        prefill_logits=prefill_logits,
        suffix_logits=suffix_logits,
        prediction_positions=prediction_positions,
        prefix_len=prefix_len,
    )


def prefill_shared_prefix(
    model,
    *,
    prefix_ids: list[int],
    device: torch.device,
) -> tuple[Any, Tensor]:
    """Run a prefix prefill and return ``(past_key_values, logits[prefix_len, vocab])``."""

    if not prefix_ids:
        raise ValueError("prefix_ids must be non-empty")
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
    if outputs.past_key_values is None:
        raise RuntimeError("model returned no past_key_values during prefix prefill")
    return outputs.past_key_values, outputs.logits[0]
