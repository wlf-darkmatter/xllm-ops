# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-CPU pytest checks for the DeepSeek-V4.1 A3 QLI INT8 indexer reference.

Formerly the standalone script ``tools/run_qli_dsv41_reference.py``; folded
into the test tree (one ``test_*`` per former ``_check_*``) so the QLI
reference invariants ride the same CI pytest wall as the rest of the v4.1
suite. Inlines the independent CPU reference for the A3 ``quant_lightning_indexer``
and exercises its *certain* pieces -- the INT8 score pipeline, the
``mask_mode=3`` causal boundary (ratio 1/2 incl. residual), the page/block-table
logical mapping, the two-level candidate pool -- and, crucially, the **negative
controls**: a deliberately wrong ratio/residual boundary must change the
selection, a non-discriminating (tie) case must be refused rather than guessed,
and the mode=2 candidate-pool leak (a finite ``NEG_HUGE`` penalty keeping a real
index where the pool cannot fill ``topk``) must diverge from the oracle while the
``score=-inf / index=-1`` semantic pairing agrees with it, and a blind output
index offset must be shown to corrupt the ``-1`` sentinel (``rowOff - 1``) where
the guarded rule preserves it. It also checks the arch22 ``SortAll``
group-reduction capacity the mode=2 candidate sort must pad to: a raw 64-multiple
candidate sort drops whole groups, while the smallest lossless length keeps every
candidate in the first ``candBlocks`` slots. It touches no NPU, no compiled
operator and no other repository.

Reference scope (inlined below)
-------------------------------
Re-derives, from the frozen operator contract, only the *locally checkable*
parts of the A3 ``quant_lightning_indexer`` numerical reference:

* INT8 ``quant_mode=2`` QK: exact INT32 dot, ``/1024`` descale, ReLU in FP32
  then rounded to FP16;
* the FP16 per-head ``weight * q_scale`` product, the FP32 head reduction and
  the FP16 ``k_scale`` multiply;
* the ``mask_mode=3`` right-down-causal visibility boundary for
  ``compress_ratio`` 1 and 2, including the ``cmp_residual`` term;
* the PA_BBND page / block-table logical-row mapping;
* the two-level candidate pool: mode 1 (source) blockwise ``max`` selection and
  mode 2 (consumer) pool-restricted token top-k;
* the AscendC ``SortAll`` group-reduction capacity that bounds the arch22 mode 2
  candidate sort (which element counts keep every ``Sort32`` group).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Named A3 contract constants (re-declared locally; never imported from a repo
# outside this workspace).
# --------------------------------------------------------------------------
# ``quant_mode`` of the A3 INT8 indexer path.
QLI_V3_QUANT_MODE_INT8 = 2
# Index head width of the operator contract.
QLI_V3_INDEX_HEAD_DIM = 128
# PA_BBND index-key head count (the A3 tiling requires exactly one).
QLI_V3_KEY_NUM_HEADS = 1
# Candidate block size of the A3 implementation (tokens per logical block).
QLI_V3_CAND_BLOCK_SIZE = 8
# ``mask_mode`` / AscendC ``SPARSE_MODE_LOWER`` (right-down causal).
QLI_V3_MASK_MODE_RIGHT_DOWN_CAUSAL = 3
# Compression ratios the A3 index rows are defined for.
QLI_V3_SUPPORTED_COMPRESS_RATIOS = (1, 2)
# Two-level candidate modes.
QLI_V3_CAND_MODE_SOURCE = 1
QLI_V3_CAND_MODE_CONSUMER = 2
QLI_V3_CAND_MODE_OFF = 3
# Candidate block-pool bound: ``(0, 2048]`` and a multiple of 64.
QLI_V3_CAND_TOPK_BLOCKS_MAX = 2048
QLI_V3_CAND_TOPK_BLOCKS_ALIGN = 64
# Position top-k bound.
QLI_V3_INDEX_TOPK_MAX = 2048
# The INT32 QK is descaled by exactly this power of two before the FP16 ReLU.
QLI_V3_QK_DOT_DESCALE = 1024.0

_INT32_MAX = 2**31 - 1


class QliA3ReferenceConfigError(ValueError):
    """Raised for a configuration outside the A3 QLI INT8 contract.

    The reference refuses to guess: an unsupported dtype, ratio, quant mode,
    mask mode, candidate mode/size, or an ill-formed pool is rejected here
    rather than silently coerced into a plausible-but-wrong computation.
    """


class QliA3ReferenceTieError(ValueError):
    """Raised when a top-k boundary tie makes the selected *set* non-unique.

    The A3 tie-break / emit order is unverified, so the reference cannot pick a
    side. A boundary tie is a data-quality problem (a non-discriminating case),
    not a reference result, so it fails loudly instead of guessing.
    """


@dataclass(frozen=True)
class QliA3Config:
    """Frozen, self-validating descriptor of one A3 indexer invocation.

    Constructing it validates the whole A3 contract that the reference can
    decide on CPU; any field outside the contract raises
    :class:`QliA3ReferenceConfigError`.
    """

    compress_ratio: int
    topk: int
    candidate_mode: int = QLI_V3_CAND_MODE_OFF
    candidate_topk_blocks: int = QLI_V3_CAND_TOPK_BLOCKS_MAX
    candidate_block_size: int = QLI_V3_CAND_BLOCK_SIZE
    mask_mode: int = QLI_V3_MASK_MODE_RIGHT_DOWN_CAUSAL
    quant_mode: int = QLI_V3_QUANT_MODE_INT8
    head_dim: int = QLI_V3_INDEX_HEAD_DIM
    key_num_heads: int = QLI_V3_KEY_NUM_HEADS

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.quant_mode != QLI_V3_QUANT_MODE_INT8:
            raise QliA3ReferenceConfigError(
                f"A3 supports quant_mode={QLI_V3_QUANT_MODE_INT8} (INT8) only, got {self.quant_mode}"
            )
        if self.head_dim != QLI_V3_INDEX_HEAD_DIM:
            raise QliA3ReferenceConfigError(f"A3 index head_dim is {QLI_V3_INDEX_HEAD_DIM}, got {self.head_dim}")
        if self.key_num_heads != QLI_V3_KEY_NUM_HEADS:
            raise QliA3ReferenceConfigError(
                f"A3 PA_BBND index-key head count is {QLI_V3_KEY_NUM_HEADS}, got {self.key_num_heads}"
            )
        if self.compress_ratio not in QLI_V3_SUPPORTED_COMPRESS_RATIOS:
            raise QliA3ReferenceConfigError(
                f"A3 compress_ratio must be one of {QLI_V3_SUPPORTED_COMPRESS_RATIOS}, got {self.compress_ratio}"
            )
        if self.mask_mode != QLI_V3_MASK_MODE_RIGHT_DOWN_CAUSAL:
            raise QliA3ReferenceConfigError(
                f"this reference implements mask_mode={QLI_V3_MASK_MODE_RIGHT_DOWN_CAUSAL} only, got {self.mask_mode}"
            )
        if not 1 <= self.topk <= QLI_V3_INDEX_TOPK_MAX:
            raise QliA3ReferenceConfigError(f"topk must be in [1, {QLI_V3_INDEX_TOPK_MAX}], got {self.topk}")
        if self.candidate_block_size != QLI_V3_CAND_BLOCK_SIZE:
            raise QliA3ReferenceConfigError(
                f"A3 candidate_block_size is {QLI_V3_CAND_BLOCK_SIZE}, got {self.candidate_block_size}"
            )
        if self.candidate_mode not in (
            QLI_V3_CAND_MODE_SOURCE,
            QLI_V3_CAND_MODE_CONSUMER,
            QLI_V3_CAND_MODE_OFF,
        ):
            raise QliA3ReferenceConfigError(
                f"candidate_mode must be one of "
                f"{{{QLI_V3_CAND_MODE_SOURCE}, {QLI_V3_CAND_MODE_CONSUMER}, {QLI_V3_CAND_MODE_OFF}}}, "
                f"got {self.candidate_mode}"
            )
        if self.candidate_mode != QLI_V3_CAND_MODE_OFF:
            blocks = self.candidate_topk_blocks
            if not 0 < blocks <= QLI_V3_CAND_TOPK_BLOCKS_MAX or blocks % QLI_V3_CAND_TOPK_BLOCKS_ALIGN != 0:
                raise QliA3ReferenceConfigError(
                    f"candidate_topk_blocks must be a positive multiple of {QLI_V3_CAND_TOPK_BLOCKS_ALIGN} "
                    f"<= {QLI_V3_CAND_TOPK_BLOCKS_MAX}, got {blocks}"
                )


def validate_a3_dtypes(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
) -> None:
    """Reject any dtype outside the A3 INT8 quant_mode=2 plane.

    ``query``/``key`` are INT8; ``weights``/``query_scale``/``key_scale`` are
    FP16. Anything else is a misconfiguration, not a case to coerce.
    """
    pairs = (
        ("query", query, torch.int8),
        ("key", key, torch.int8),
        ("weights", weights, torch.float16),
        ("query_scale", query_scale, torch.float16),
        ("key_scale", key_scale, torch.float16),
    )
    for name, tensor, expected in pairs:
        if tensor.dtype != expected:
            raise QliA3ReferenceConfigError(f"'{name}' must be {expected}, got {tensor.dtype}")


# --------------------------------------------------------------------------
# Score pipeline (INT8 dot -> FP16 ReLU -> FP16 weight*scale -> FP32 reduce).
# --------------------------------------------------------------------------
def int8_dot_int32(query_int8: torch.Tensor, key_int8: torch.Tensor) -> torch.Tensor:
    """Exact INT32 ``dot[..., g, t] = sum_d int32(q[..., g, d]) * int32(k[..., t, d])``.

    Mirrors the A3 ``INT32 QK`` accumulator. The product is formed in INT64 and
    range-checked before narrowing, so the INT32 result is provably exact (it
    would raise rather than wrap if a caller fed a shape that overflowed the
    A3 accumulator). Intended for the small, readable CPU cases this reference
    exists for -- not for the 16K/128K production shapes.
    """
    q = query_int8.to(torch.int64).unsqueeze(-2)  # [..., g, 1, d]
    k = key_int8.to(torch.int64).unsqueeze(-3)  # [..., 1, t, d]
    dot = (q * k).sum(dim=-1)  # [..., g, t]
    if dot.numel() and int(dot.abs().max()) > _INT32_MAX:
        raise QliA3ReferenceConfigError("int32 QK accumulation overflowed; the A3 contract does not apply")
    return dot.to(torch.int32)


def rectified_scaled_qk(dot_int32: torch.Tensor) -> torch.Tensor:
    """``a = fp16(relu(fp32(dot) / 1024))`` -- the FP16 rectified, descaled QK.

    The descale by 1024 is an exact power-of-two scaling of the INT32 dot; the
    ReLU is applied in FP32 and only then rounded to FP16, matching the A3
    reference order (INT32 QK / 1024 -> FP16 ReLU).
    """
    scaled = (dot_int32.to(torch.float32) / QLI_V3_QK_DOT_DESCALE).clamp_min(0.0)
    return scaled.to(torch.float16)


def head_weight_scale_product(weights_fp16: torch.Tensor, query_scale_fp16: torch.Tensor) -> torch.Tensor:
    """``u = fp16(w * q_scale)`` -- the FP16 per-head weight-times-scale product.

    Both inputs must already be FP16: a wrong dtype is a misconfiguration that
    :func:`validate_a3_dtypes` rejects, not a case this helper silently coerces.
    """
    return (weights_fp16 * query_scale_fp16).to(torch.float16)


def indexer_scores(
    rectified_fp16: torch.Tensor,
    weight_scale_fp16: torch.Tensor,
    key_scale_fp16: torch.Tensor,
) -> torch.Tensor:
    """``score[..., t] = (sum_h fp32(u_h) * fp32(a_h,t)) * fp32(k_scale_t)`` in FP32.

    ``rectified_fp16`` is ``[..., g, t]`` (one entry per query head per key
    token), ``weight_scale_fp16`` is ``[..., g]`` (one entry per query head) and
    ``key_scale_fp16`` is ``[..., t]``. The head reduction is done in FP32 and
    the per-token ``k_scale`` multiply is applied last, in FP32.
    """
    per_head = weight_scale_fp16.to(torch.float32).unsqueeze(-1) * rectified_fp16.to(torch.float32)
    reduced = per_head.sum(dim=-2)  # sum over the query-head dim
    return reduced * key_scale_fp16.to(torch.float32)


# --------------------------------------------------------------------------
# mask_mode=3 right-down-causal visibility boundary.
# --------------------------------------------------------------------------
def causal_visible_counts(
    compress_ratio: int,
    committed_rows: int,
    residual: int,
    query_length: int,
) -> torch.Tensor:
    """Per-query visible compressed-row counts ``V_i`` for one causal chunk.

    With ``L_orig = committed_rows * ratio + residual`` (the original,
    uncompressed key length), the chunk's ``query_length`` tokens occupy
    original positions ``p_i = L_orig - query_length + i``, and::

        V_i = clamp(floor((p_i + 1) / ratio), 0, committed_rows)

    Token ``i`` may attend compressed rows ``0 <= j < V_i``. ``residual`` is the
    ``cmp_residual_k`` operand and must be 0 for ``ratio == 1`` (the tiling
    rejects a residual tensor there).
    """
    if compress_ratio not in QLI_V3_SUPPORTED_COMPRESS_RATIOS:
        raise QliA3ReferenceConfigError(
            f"compress_ratio must be one of {QLI_V3_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}"
        )
    if compress_ratio == 1 and residual != 0:
        raise QliA3ReferenceConfigError("ratio=1 has no residual term; residual must be 0")
    if not 0 <= residual < compress_ratio:
        raise QliA3ReferenceConfigError(f"residual must be in [0, {compress_ratio}), got {residual}")
    if committed_rows < 0 or query_length < 0:
        raise QliA3ReferenceConfigError("committed_rows and query_length must be non-negative")
    original_length = committed_rows * compress_ratio + residual
    if query_length > original_length:
        raise QliA3ReferenceConfigError(f"query_length {query_length} exceeds the original length {original_length}")
    positions = original_length - query_length + torch.arange(query_length, dtype=torch.int64)
    return causal_visible_counts_from_positions(positions, compress_ratio, committed_rows)


def causal_visible_counts_from_positions(
    positions: torch.Tensor,
    compress_ratio: int,
    committed_rows: int,
) -> torch.Tensor:
    """``V_i = clamp(floor((p_i + 1) / ratio), 0, committed_rows)`` from raw positions.

    This is the per-query visibility the model's indexer masks with
    (``(p + 1) // ratio``); it must agree with :func:`causal_visible_counts`
    when the chunk positions are contiguous.
    """
    if compress_ratio not in QLI_V3_SUPPORTED_COMPRESS_RATIOS:
        raise QliA3ReferenceConfigError(
            f"compress_ratio must be one of {QLI_V3_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}"
        )
    pos = positions.to(torch.int64)
    raw = torch.div(pos + 1, compress_ratio, rounding_mode="floor")
    return raw.clamp(min=0, max=committed_rows)


def causal_visibility_mask(visible: torch.Tensor, width: int) -> torch.Tensor:
    """Boolean ``[rows, width]`` mask, ``True`` where ``j < visible[row]``."""
    if width < 0:
        raise QliA3ReferenceConfigError("width must be non-negative")
    columns = torch.arange(width, dtype=torch.int64).unsqueeze(0)
    return columns < visible.to(torch.int64).unsqueeze(-1)


def apply_causal_mask(score: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Return ``score`` with causally-invisible columns set to ``-inf``.

    ``score`` is ``[rows, width]`` FP32 and ``visible`` is ``[rows]``.
    """
    if score.size(0) != visible.numel():
        raise QliA3ReferenceConfigError(f"score has {score.size(0)} rows but visible has {visible.numel()} entries")
    mask = causal_visibility_mask(visible, score.size(-1))
    return score.masked_fill(~mask, float("-inf"))


# --------------------------------------------------------------------------
# PA_BBND page / block-table logical mapping.
# --------------------------------------------------------------------------
def logical_physical_locations(
    block_table: torch.Tensor,
    num_rows: int,
    page_size: int,
) -> torch.Tensor:
    """Flat physical row index of each logical row under a page block table.

    ``block_table[b]`` is the physical page id holding the ``page_size`` logical
    rows ``[b * page_size, (b + 1) * page_size)``. The returned ``[num_rows]``
    INT64 tensor indexes a ``[num_pages * page_size, ...]`` packed buffer, so a
    caller must gather logical rows in table order rather than reading the
    physical buffer left-to-right. A negative page id inside the addressed
    range is a malformed table and is rejected.
    """
    if page_size <= 0:
        raise QliA3ReferenceConfigError(f"page_size must be positive, got {page_size}")
    if num_rows < 0:
        raise QliA3ReferenceConfigError("num_rows must be non-negative")
    table = block_table.to(torch.int64).reshape(-1)
    pages_needed = (num_rows + page_size - 1) // page_size
    if pages_needed > table.numel():
        raise QliA3ReferenceConfigError(f"block_table holds {table.numel()} pages but {pages_needed} are addressed")
    pages = table[:pages_needed]
    if num_rows and bool((pages < 0).any()):
        raise QliA3ReferenceConfigError("block_table addresses a negative (absent) page")
    rows = torch.arange(num_rows, dtype=torch.int64)
    return pages[rows // page_size] * page_size + (rows % page_size)


def depage_rows(
    paged: torch.Tensor,
    block_table: torch.Tensor,
    num_rows: int,
    page_size: int,
) -> torch.Tensor:
    """Gather logical rows ``0..num_rows-1`` out of a paged ``[pages, page_size, ...]`` tensor.

    A block table that addresses a page id outside the paged tensor, or a
    negative/absent page, is rejected with :class:`QliA3ReferenceConfigError`.
    """
    if paged.dim() < 2:
        raise QliA3ReferenceConfigError("paged tensor must have at least [pages, page_size] dims")
    if paged.size(1) != page_size:
        raise QliA3ReferenceConfigError(f"paged tensor page dim is {paged.size(1)} but page_size is {page_size}")
    flat = paged.reshape(paged.size(0) * page_size, *paged.shape[2:])
    locations = logical_physical_locations(block_table, num_rows, page_size)
    if locations.numel() and int(locations.max()) >= flat.size(0):
        raise QliA3ReferenceConfigError(
            f"block_table addresses a page >= {paged.size(0)}; the paged tensor has only {paged.size(0)} pages"
        )
    return flat.index_select(0, locations)


# --------------------------------------------------------------------------
# Selections (set semantics; boundary ties are refused).
# --------------------------------------------------------------------------
def select_top_index_set(
    values: torch.Tensor,
    eligible: torch.Tensor,
    k: int,
    *,
    pinned: int | None = None,
) -> frozenset[int]:
    """Set of up to ``k`` indices with the largest values among ``eligible``.

    Returns a ``frozenset`` because the A3 emit order and tie-break are
    unverified -- this reference promises *which* indices are selected, not in
    what order. ``pinned`` (if given) is always included. Raises
    :class:`QliA3ReferenceTieError` when the ``k``-th and ``(k + 1)``-th best
    eligible values are equal, i.e. when even the set is not unique, or when an
    eligible value is NaN (which ``argsort`` would rank first and the tie guard
    could not compare).
    """
    if values.dim() != 1:
        raise QliA3ReferenceConfigError("values must be 1-D")
    if eligible.numel() != values.numel():
        raise QliA3ReferenceConfigError("eligible must match values")
    if k < 1:
        raise QliA3ReferenceConfigError(f"k must be >= 1, got {k}")
    working = values.to(torch.float64).clone()
    mask = eligible.to(torch.bool).clone()
    if pinned is not None:
        if not 0 <= pinned < values.numel():
            raise QliA3ReferenceConfigError(f"pinned index {pinned} out of range")
        working[pinned] = float("inf")
        mask[pinned] = True
    candidate_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if candidate_idx.numel() == 0:
        return frozenset()
    candidate_val = working[candidate_idx]
    if bool(torch.isnan(candidate_val).any()):
        raise QliA3ReferenceTieError(
            "an eligible value is NaN; torch.argsort ranks NaN first and the tie guard cannot compare it"
        )
    order = torch.argsort(candidate_val, descending=True, stable=True)
    ranked_idx = candidate_idx[order]
    ranked_val = candidate_val[order]
    selected = ranked_idx[:k]
    if ranked_idx.numel() > k and ranked_val[k - 1] == ranked_val[k]:
        raise QliA3ReferenceTieError(
            f"top-{k} boundary tie between values {float(ranked_val[k - 1])!r} and "
            f"{float(ranked_val[k])!r}; the selected set is not unique"
        )
    return frozenset(int(i) for i in selected.tolist())


def block_max_scores(masked_score: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block maximum of causally-valid token scores.

    ``masked_score`` is ``[..., width]`` FP32 with ``-inf`` at invalid
    positions. Returns ``(block_max [..., num_blocks], block_valid [..., num_blocks])``
    where ``num_blocks = ceil(width / block_size)`` and a block with no valid
    token is ``-inf`` / ``False``. The tail block is ``-inf`` padded, so a
    partial block can never be lifted above a full one by its padding.
    """
    if block_size <= 0:
        raise QliA3ReferenceConfigError(f"block_size must be positive, got {block_size}")
    width = masked_score.size(-1)
    if width == 0:
        return masked_score.new_full((*masked_score.shape[:-1], 0), float("-inf")), torch.zeros(
            (*masked_score.shape[:-1], 0), dtype=torch.bool
        )
    num_blocks = (width + block_size - 1) // block_size
    pad = num_blocks * block_size - width
    padded = F.pad(masked_score, (0, pad), value=float("-inf")) if pad else masked_score
    blocks = padded.unflatten(-1, (num_blocks, block_size))
    block_max = blocks.amax(dim=-1)
    return block_max, torch.isfinite(block_max)


def source_candidate_blocks_per_row(
    masked_score: torch.Tensor,
    visible: torch.Tensor,
    config: QliA3Config,
    *,
    pin_newest_block: bool = True,
) -> tuple[frozenset[int], ...]:
    """mode 1 (source): per-row block-index candidate pool.

    Each block's score is the ``max`` over its causally-valid token scores, then
    the ``min(candidate_topk_blocks, num_blocks)`` best blocks form the pool. The
    block holding the newest visible row (``(V_i - 1) // block_size``) is pinned
    when ``pin_newest_block`` is set, matching the design's "newest committed
    block is retained" rule; pass ``False`` to model a variant without the pin.

    Set semantics only: the pool is returned as a ``frozenset`` per row, never
    as an ordered / ``-1``-padded block-id array.
    """
    if config.candidate_mode != QLI_V3_CAND_MODE_SOURCE:
        raise QliA3ReferenceConfigError("source_candidate_blocks_per_row requires candidate_mode=1")
    block_size = config.candidate_block_size
    block_max, block_valid = block_max_scores(masked_score, block_size)
    if block_max.size(0) != visible.numel():
        raise QliA3ReferenceConfigError(
            f"block_max has {block_max.size(0)} rows but visible has {visible.numel()} entries"
        )
    pools: list[frozenset[int]] = []
    for row in range(block_max.size(0)):
        row_valid = visible[row].item()
        pinned = None
        if pin_newest_block and row_valid > 0:
            pinned = (row_valid - 1) // block_size
        pools.append(
            select_top_index_set(
                block_max[row],
                block_valid[row],
                config.candidate_topk_blocks,
                pinned=pinned,
            )
        )
    return tuple(pools)


def token_topk_per_row(
    masked_score: torch.Tensor,
    visible: torch.Tensor,
    config: QliA3Config,
) -> tuple[frozenset[int], ...]:
    """Per-row position top-k over the causally-visible tokens (mode 3 / source top-k).

    ``masked_score`` is ``[rows, width]`` FP32 with ``-inf`` outside the causal
    boundary. Returns one ``frozenset`` of 0-based compressed-row indices per
    row; ordering, tie-break and ``-1`` padding are out of scope.
    """
    rows = masked_score.size(0)
    if rows != visible.numel():
        raise QliA3ReferenceConfigError(f"score has {rows} rows but visible has {visible.numel()} entries")
    causal = causal_visibility_mask(visible, masked_score.size(-1))
    results: list[frozenset[int]] = []
    for row in range(rows):
        eligible = causal[row] & torch.isfinite(masked_score[row])
        results.append(select_top_index_set(masked_score[row], eligible, config.topk))
    return tuple(results)


def consumer_token_topk_per_row(
    masked_score: torch.Tensor,
    visible: torch.Tensor,
    pool_per_row: tuple[frozenset[int], ...] | list[frozenset[int]],
    config: QliA3Config,
) -> tuple[frozenset[int], ...]:
    """mode 2 (consumer): per-row top-k restricted to the source block pool.

    Keeps only tokens whose ``floor(j / block_size)`` is in that row's pool
    *and* that are causally visible, then runs this layer's own top-k over the
    surviving tokens. The pool is per row and never re-derived here -- a
    consumer must not regenerate the source's pool.
    """
    if config.candidate_mode != QLI_V3_CAND_MODE_CONSUMER:
        raise QliA3ReferenceConfigError("consumer_token_topk_per_row requires candidate_mode=2")
    rows = masked_score.size(0)
    if rows != visible.numel():
        raise QliA3ReferenceConfigError(f"score has {rows} rows but visible has {visible.numel()} entries")
    if len(pool_per_row) != rows:
        raise QliA3ReferenceConfigError(f"pool_per_row has {len(pool_per_row)} rows but score has {rows}")
    block_size = config.candidate_block_size
    width = masked_score.size(-1)
    block_of_token = torch.arange(width, dtype=torch.int64) // block_size
    causal = causal_visibility_mask(visible, width)
    results: list[frozenset[int]] = []
    for row in range(rows):
        pool = {int(b) for b in pool_per_row[row]}
        in_pool = torch.zeros(width, dtype=torch.bool)
        if pool and width:
            in_pool = torch.tensor([int(b) in pool for b in block_of_token.tolist()], dtype=torch.bool)
        eligible = causal[row] & in_pool & torch.isfinite(masked_score[row])
        results.append(select_top_index_set(masked_score[row], eligible, config.topk))
    return tuple(results)


# --------------------------------------------------------------------------
# AscendC SortAll group-reduction capacity (arch22 mode=2 candidate sort).
# --------------------------------------------------------------------------
# ``SortAll`` sorts ``logits_num`` elements as ``logits_num / 32`` ``Sort32`` groups
# of 32, then repeatedly 4-way merges (``MrgSort``) with ``repeatTimes = groups // 4``
# and halves the live group count to ``groups // 4``; the final step covers the
# remaining 1..4 groups via ``validBit``. Whenever the live group count is not a
# multiple of four, the ``groups % 4`` trailing groups are never read by any repeat
# and are silently dropped. So the only element counts that lose no group are
# ``32 * m * 4**k`` with ``m in {1,2,3,4}`` -- within the candidate bound the eight
# 64-multiples 64,128,256,384,512,1024,1536,2048. The arch22 mode=2 candidate sort
# must therefore pad a ``candBlocks`` row (any multiple of 64 in ``(0, 2048]``) up to
# the smallest such length, filling ``(-inf, -1)``.
QLI_V3_SORT_GROUP = 32
QLI_V3_SORT_FANIN = 4


def sortall_lossless_lengths(limit: int = QLI_V3_CAND_TOPK_BLOCKS_MAX) -> tuple[int, ...]:
    """``SortAll`` element counts in ``(0, limit]`` that drop no merge group."""
    if limit < 0:
        raise QliA3ReferenceConfigError(f"limit must be non-negative, got {limit}")
    lengths: set[int] = set()
    for m in range(1, QLI_V3_SORT_FANIN + 1):
        groups = m
        while groups * QLI_V3_SORT_GROUP <= limit:
            lengths.add(groups * QLI_V3_SORT_GROUP)
            groups *= QLI_V3_SORT_FANIN
    return tuple(sorted(lengths))


def sortall_is_lossless(logits_num: int) -> bool:
    """Whether ``SortAll(logits_num)`` keeps every ``Sort32`` group."""
    if logits_num <= 0 or logits_num % QLI_V3_SORT_GROUP != 0:
        return False
    groups = logits_num // QLI_V3_SORT_GROUP
    while groups > QLI_V3_SORT_FANIN:
        if groups % QLI_V3_SORT_FANIN != 0:
            return False
        groups //= QLI_V3_SORT_FANIN
    return True


def sortall_surviving_groups(logits_num: int) -> tuple[int, ...]:
    """Original ``Sort32`` group ids that survive ``SortAll``'s 4-way merge reduction.

    A faithful model of the reduction loop: each step merges the first
    ``floor(groups / 4) * 4`` live groups into ``floor(groups / 4)`` blocks and drops
    the trailing ``groups % 4`` live groups (never read by any repeat).
    """
    if logits_num <= 0 or logits_num % QLI_V3_SORT_GROUP != 0:
        raise QliA3ReferenceConfigError(
            f"logits_num must be a positive multiple of {QLI_V3_SORT_GROUP}, got {logits_num}"
        )
    alive: list[list[int]] = [[g] for g in range(logits_num // QLI_V3_SORT_GROUP)]
    while len(alive) > QLI_V3_SORT_FANIN:
        full = (len(alive) // QLI_V3_SORT_FANIN) * QLI_V3_SORT_FANIN
        alive = [
            [g for block in alive[k : k + QLI_V3_SORT_FANIN] for g in block] for k in range(0, full, QLI_V3_SORT_FANIN)
        ]
    return tuple(sorted(g for block in alive for g in block))


def sortall_align_length(count: int, limit: int = QLI_V3_CAND_TOPK_BLOCKS_MAX) -> int:
    """Smallest lossless ``SortAll`` length ``>= count`` (the candidate-row pad target).

    Explicit set membership: a plain up-alignment is not the capacity rule, and
    neither is the arch22 ``AlignS2`` helper (which aligns >512 to 512 and does not
    land in this set for every input).
    """
    if count <= 0:
        raise QliA3ReferenceConfigError(f"count must be positive, got {count}")
    for candidate in sortall_lossless_lengths(limit):
        if candidate >= count:
            return candidate
    raise QliA3ReferenceConfigError(f"no lossless SortAll length >= {count} within {limit}")


# --------------------------------------------------------------------------
# Output contract helper (A3: return_value=False -> empty BF16 second output).
# --------------------------------------------------------------------------
def assert_a3_values_output_empty(values: torch.Tensor) -> None:
    """Assert the A3 second output is the empty BF16 placeholder.

    A3 uses ``return_value=False``, so ``sparse_values`` must be a BF16 tensor
    with zero elements. This pins that contract; it is *not* a numerical gate
    and says nothing about a non-empty BF16 ``sparse_values`` (that path is a
    separate, unverified A5/optional case).
    """
    if values.dtype != torch.bfloat16:
        raise QliA3ReferenceConfigError(f"A3 values output must be bfloat16, got {values.dtype}")
    if values.numel() != 0:
        raise QliA3ReferenceConfigError(
            f"A3 uses return_value=False; values output must be empty, got numel={values.numel()}"
        )


def test_score_pipeline() -> None:
    torch.manual_seed(2026)
    heads, width, head_dim = 2, 3, QLI_V3_INDEX_HEAD_DIM
    query = torch.randint(-127, 128, (heads, head_dim), dtype=torch.int8)
    key = torch.randint(-127, 128, (width, head_dim), dtype=torch.int8)
    weights = torch.randn(heads, dtype=torch.float32).to(torch.float16)
    query_scale = torch.rand(heads, dtype=torch.float32).to(torch.float16)
    key_scale = torch.rand(width, dtype=torch.float32).to(torch.float16)
    validate_a3_dtypes(query, key, weights, query_scale, key_scale)

    dot = int8_dot_int32(query, key)
    assert dot.dtype == torch.int32 and tuple(dot.shape) == (heads, width)
    # Independent exact-int reference over python ints.
    manual_dot = [
        [sum(int(query[g, d]) * int(key[t, d]) for d in range(head_dim)) for t in range(width)] for g in range(heads)
    ]
    assert dot.tolist() == manual_dot, "int8_dot_int32 disagrees with the python-int dot"

    # Independent FP32 oracle over numpy scalars, threading the exact A3 contract
    # order: fp16 rectified = fp16(fp32(dot)/1024 clamped at 0), fp16 w*q_scale,
    # fp32 per-head product, fp32 head sum, fp32 * k_scale. This replaces the old
    # self-referential expectations (each was the tested function's own expression)
    # and is not seed-lucky: the oracle rounds at the same steps the pipeline does.
    oracle_rectified = np.array(
        [
            [
                np.float16(max(np.float32(manual_dot[g][t]) / np.float32(QLI_V3_QK_DOT_DESCALE), np.float32(0.0)))
                for t in range(width)
            ]
            for g in range(heads)
        ],
        dtype=np.float16,
    )
    oracle_weight_scale = np.array(
        [np.float16(np.float16(weights[g].item()) * np.float16(query_scale[g].item())) for g in range(heads)],
        dtype=np.float16,
    )
    oracle_score = np.empty(width, dtype=np.float32)
    for t in range(width):
        reduced = np.float32(0.0)
        for g in range(heads):
            reduced = np.float32(reduced + np.float32(oracle_weight_scale[g]) * np.float32(oracle_rectified[g, t]))
        oracle_score[t] = np.float32(reduced * np.float32(key_scale[t].item()))

    rectified = rectified_scaled_qk(dot)
    assert rectified.dtype == torch.float16
    assert torch.equal(rectified, torch.from_numpy(oracle_rectified)), (
        "rectified_scaled_qk disagrees with the independent FP32 oracle"
    )

    weight_scale = head_weight_scale_product(weights, query_scale)
    assert torch.equal(weight_scale, torch.from_numpy(oracle_weight_scale)), (
        "head_weight_scale_product disagrees with the independent FP16 oracle"
    )

    score = indexer_scores(rectified, weight_scale, key_scale)
    assert score.dtype == torch.float32 and tuple(score.shape) == (width,)
    for t in range(width):
        assert score[t].item() == float(oracle_score[t]), f"score[{t}] != independent FP32 oracle"

    print("[ok] score pipeline (INT32 dot / 1024 -> FP16 ReLU -> FP16 w*scale -> FP32 reduce) vs numpy oracle")


def test_validate_dtypes_negative() -> None:
    """``validate_a3_dtypes`` must reject a wrong dtype on every plane it checks."""
    query = torch.zeros(1, 4, dtype=torch.int8)
    key = torch.zeros(1, 4, dtype=torch.int8)
    weights = torch.zeros(1, dtype=torch.float16)
    query_scale = torch.zeros(1, dtype=torch.float16)
    key_scale = torch.zeros(1, dtype=torch.float16)
    validate_a3_dtypes(query, key, weights, query_scale, key_scale)

    bad = {
        "query": (query.to(torch.float16), key, weights, query_scale, key_scale),
        "key": (query, key.to(torch.int32), weights, query_scale, key_scale),
        "weights": (query, key, weights.to(torch.float32), query_scale, key_scale),
        "query_scale": (query, key, weights, query_scale.to(torch.float32), key_scale),
        "key_scale": (query, key, weights, query_scale, key_scale.to(torch.float32)),
    }
    for name, (q, k, w, qs, ks) in bad.items():
        try:
            validate_a3_dtypes(q, k, w, qs, ks)
        except QliA3ReferenceConfigError:
            pass
        else:
            raise AssertionError(f"validate_a3_dtypes accepted a bad '{name}' dtype")
    print("[ok] validate_a3_dtypes rejects a wrong dtype on every plane")


def test_causal_boundary() -> None:
    ratio1 = causal_visible_counts(1, committed_rows=5, residual=0, query_length=3)
    assert ratio1.tolist() == [3, 4, 5], f"ratio=1 boundary wrong: {ratio1.tolist()}"

    ratio2_res1 = causal_visible_counts(2, committed_rows=5, residual=1, query_length=4)
    assert ratio2_res1.tolist() == [4, 4, 5, 5], f"ratio=2 residual=1 boundary wrong: {ratio2_res1.tolist()}"

    ratio2_res0 = causal_visible_counts(2, committed_rows=5, residual=0, query_length=4)
    assert ratio2_res0.tolist() == [3, 4, 4, 5], f"ratio=2 residual=0 boundary wrong: {ratio2_res0.tolist()}"

    # The per-query model form ((p + 1) // ratio) must agree with the chunk form.
    positions = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
    assert torch.equal(causal_visible_counts_from_positions(positions, 2, 5), ratio2_res1), (
        "position-based visibility disagrees with the chunk form"
    )

    for bad_kwargs in (
        {"compress_ratio": 1, "residual": 1},
        {"compress_ratio": 4, "residual": 0},
        {"compress_ratio": 2, "residual": 2},
    ):
        try:
            causal_visible_counts(committed_rows=5, query_length=1, **bad_kwargs)
        except QliA3ReferenceConfigError:
            pass
        else:
            raise AssertionError(f"causal_visible_counts accepted {bad_kwargs}")

    # A query chunk longer than the original (uncompressed) key length is rejected:
    # ratio=2, committed_rows=2, residual=0 -> L_orig=4 < query_length=5.
    try:
        causal_visible_counts(2, committed_rows=2, residual=0, query_length=5)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("causal_visible_counts accepted query_length > original_length")
    print("[ok] causal visible boundary (ratio 1/2, residual 0/1) + rejects")


def test_causal_forms_agree() -> None:
    """``causal_visible_counts`` must match ``from_positions`` on a sweep of shapes."""
    combos = [
        (1, 1, 0, 1),
        (1, 5, 0, 3),
        (1, 5, 0, 5),
        (1, 8, 0, 6),
        (2, 1, 0, 1),
        (2, 2, 1, 1),
        (2, 3, 1, 2),
        (2, 5, 1, 4),
        (2, 5, 0, 4),
        (2, 7, 1, 6),
        (2, 9, 0, 9),
    ]
    for ratio, committed, residual, query_length in combos:
        original = committed * ratio + residual
        positions = original - query_length + torch.arange(query_length, dtype=torch.int64)
        from_chunk = causal_visible_counts(ratio, committed, residual, query_length)
        from_positions = causal_visible_counts_from_positions(positions, ratio, committed)
        assert torch.equal(from_chunk, from_positions), (
            f"ratio={ratio} committed={committed} residual={residual} query_length={query_length}: "
            f"{from_chunk.tolist()} != {from_positions.tolist()}"
        )
    print("[ok] causal visible counts agree with the position form over a sweep")


def test_mask_negative_controls() -> None:
    """A wrong ceil / residual-dropping boundary must change the selection."""
    ratio, committed, residual, query_length = 2, 5, 1, 4
    correct = causal_visible_counts(ratio, committed, residual, query_length)
    assert correct.tolist() == [4, 4, 5, 5]

    # construct the deliberately wrong boundaries
    original = committed * ratio + residual
    positions = original - query_length + torch.arange(query_length, dtype=torch.int64)
    ceil_wrong = torch.div(positions + 1 + ratio - 1, ratio, rounding_mode="floor").clamp(max=committed)
    no_residual = causal_visible_counts(ratio, committed, residual=0, query_length=query_length)
    assert not torch.equal(correct, ceil_wrong), "ceil control did not actually differ"
    assert not torch.equal(correct, no_residual), "residual-dropping control did not actually differ"

    # Row 1: correct V=4 (positions 0..3 visible), ceil-wrong V=5 (position 4 leaks in).
    width = committed
    config = QliA3Config(compress_ratio=ratio, topk=2)
    score = torch.full((query_length, width), -1.0)
    score[..., 0] = 1.0
    score[..., 1] = 2.0
    score[..., 4] = 9.0  # only visible under the wrong boundary on row 1

    masked_correct = apply_causal_mask(score, correct)
    masked_wrong = apply_causal_mask(score, ceil_wrong)
    picks_correct = token_topk_per_row(masked_correct, correct, config)
    picks_wrong = token_topk_per_row(masked_wrong, ceil_wrong, config)
    assert picks_correct[1] == frozenset({0, 1}), f"correct picks={picks_correct[1]}"
    assert picks_wrong[1] == frozenset({1, 4}), f"wrong picks={picks_wrong[1]}"
    assert 4 not in picks_correct[1], "the leaked position 4 must be invisible under the correct mask"
    assert picks_correct[1] != picks_wrong[1], "a wrong boundary was not caught by the oracle"
    print("[ok] negative control: wrong ceil/residual boundary changes the selection")


def test_unique_index_selection() -> None:
    config = QliA3Config(compress_ratio=1, topk=2)
    score = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0]])
    visible = torch.tensor([5], dtype=torch.int64)
    picks = token_topk_per_row(apply_causal_mask(score, visible), visible, config)
    assert picks[0] == frozenset({1, 3}), f"unique top-2 wrong: {picks[0]}"

    # A boundary tie must be refused, not guessed.
    tied = torch.tensor([[1.0, 1.0, 1.0]])
    visible3 = torch.tensor([3], dtype=torch.int64)
    try:
        token_topk_per_row(apply_causal_mask(tied, visible3), visible3, QliA3Config(1, 1))
    except QliA3ReferenceTieError:
        pass
    else:
        raise AssertionError("a boundary tie was not refused")

    # Ties strictly below the boundary do not make the set ambiguous.
    below = torch.tensor([[5.0, 1.0, 1.0]])
    picks_below = token_topk_per_row(apply_causal_mask(below, visible3), visible3, QliA3Config(1, 1))
    assert picks_below[0] == frozenset({0})
    print("[ok] unique selection + boundary-tie refusal")


def test_select_top_index_set() -> None:
    """Direct unit tests for the ``select_top_index_set`` primitive."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    eligible = torch.tensor([True, True, True, True])

    # An eligible NaN is refused rather than ranked first by argsort (I1).
    nan_values = torch.tensor([1.0, float("nan"), 3.0, 2.0])
    try:
        select_top_index_set(nan_values, eligible, 2)
    except QliA3ReferenceTieError:
        pass
    else:
        raise AssertionError("select_top_index_set accepted an eligible NaN")

    # A pin is forced into the set even when it is outside the top-k.
    picks = select_top_index_set(values, eligible, 2, pinned=0)
    assert picks == frozenset({0, 3}), f"pinned set wrong: {picks}"

    # A pin outside the value range is rejected.
    try:
        select_top_index_set(values, eligible, 2, pinned=4)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("select_top_index_set accepted an out-of-range pin")

    # k < 1 is rejected.
    try:
        select_top_index_set(values, eligible, 0)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("select_top_index_set accepted k < 1")

    # An eligible/values length mismatch is rejected.
    try:
        select_top_index_set(values, torch.tensor([True, True]), 2)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("select_top_index_set accepted a length mismatch")

    # k >= the eligible count returns every eligible index without a tie error.
    picks_all = select_top_index_set(values, eligible, 4)
    assert picks_all == frozenset({0, 1, 2, 3}), f"full set wrong: {picks_all}"

    # Non-eligible entries are never selected.
    sparse_eligible = torch.tensor([True, False, True, False])
    picks_sparse = select_top_index_set(values, sparse_eligible, 2)
    assert picks_sparse == frozenset({0, 2}), f"sparse set wrong: {picks_sparse}"
    print("[ok] select_top_index_set: NaN refusal, pin, range/k/length guards, full return")


def test_block_max_scores() -> None:
    """``block_max_scores`` edge branches and the -inf tail-padding property."""
    # width == 0 early-return branch.
    empty = torch.empty((2, 0), dtype=torch.float32)
    block_max, block_valid = block_max_scores(empty, 4)
    assert block_max.shape == (2, 0) and block_max.numel() == 0, block_max.shape
    assert block_valid.shape == (2, 0) and block_valid.dtype == torch.bool, block_valid.shape

    # block_size < 1 is rejected.
    try:
        block_max_scores(torch.zeros(4), 0)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("block_max_scores accepted block_size < 1")

    # A partial tail block cannot be lifted by its -inf padding: the full block with
    # finite entries out-ranks the tail, and the tail stays invalid (-inf / False).
    score = torch.tensor([[1.0, 2.0, 3.0, 4.0, float("-inf"), float("-inf")]])
    block_max, block_valid = block_max_scores(score, 4)
    assert block_max.tolist() == [[4.0, float("-inf")]], block_max.tolist()
    assert block_valid.tolist() == [[True, False]], block_valid.tolist()

    # A partial tail block that really holds a valid value reports that value.
    score2 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, float("-inf")]])
    block_max2, block_valid2 = block_max_scores(score2, 4)
    assert block_max2.tolist() == [[4.0, 5.0]], block_max2.tolist()
    assert block_valid2.tolist() == [[True, True]], block_valid2.tolist()
    print("[ok] block_max_scores: width==0 branch, block_size guard, -inf tail padding")


def test_block_table_mapping() -> None:
    page_size = 2
    paged = torch.arange(12, dtype=torch.float32).reshape(6, page_size)  # 3 pages x 2 rows
    block_table = torch.tensor([2, 0, 1], dtype=torch.int32)  # a non-identity permutation
    logical = depage_rows(paged, block_table, num_rows=5, page_size=page_size)
    # logical row j -> page block_table[j // 2], offset j % 2
    assert logical.tolist() == [4.0, 5.0, 0.0, 1.0, 2.0], f"depage wrong: {logical.tolist()}"
    naive = paged.reshape(-1)[:5]
    assert not torch.equal(logical, naive), "a permuted block table must not read physical order"

    for call in (
        lambda: logical_physical_locations(block_table, num_rows=9, page_size=page_size),
        lambda: logical_physical_locations(torch.tensor([-1, 0], dtype=torch.int32), 2, page_size),
    ):
        try:
            call()
        except QliA3ReferenceConfigError:
            pass
        else:
            raise AssertionError("an ill-formed block table was not rejected")

    # A page id >= paged.size(0) must be rejected, not leak a raw torch index error.
    try:
        depage_rows(torch.arange(4, dtype=torch.float32).reshape(2, page_size), torch.tensor([2]), 2, page_size)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("depage_rows accepted an out-of-range page id")
    print("[ok] PA_BBND block-table logical mapping + rejects")


def test_source_pin_and_consumer_pool() -> None:
    block_size = QLI_V3_CAND_BLOCK_SIZE
    num_blocks = 65
    width = num_blocks * block_size  # 520 positions
    visible = torch.full((1,), width, dtype=torch.int64)

    # Blocks 0..63 have distinct, decreasing maxima; the newest block 64 is the
    # weakest, so only the pin can carry it into a 64-slot pool.
    masked = torch.full((1, width), float("-inf"))
    for block in range(64):
        masked[0, block * block_size : (block + 1) * block_size] = float(1000 - block)
    masked[0, 64 * block_size :] = -100.0

    source_cfg = QliA3Config(compress_ratio=1, topk=4, candidate_mode=1, candidate_topk_blocks=64)
    pinned = source_candidate_blocks_per_row(masked, visible, source_cfg, pin_newest_block=True)
    unpinned = source_candidate_blocks_per_row(masked, visible, source_cfg, pin_newest_block=False)
    assert len(pinned[0]) == 64 and len(unpinned[0]) == 64
    assert 64 in pinned[0], "the pinned newest block must be retained"
    assert 64 not in unpinned[0], "without the pin the weakest block must drop out"
    assert unpinned[0] == frozenset(range(64))
    assert pinned[0] == frozenset({*range(63), 64})

    # Consumer: same score, pool restricted to block 0 only -> picks stay < 8.
    consumer_cfg = QliA3Config(compress_ratio=1, topk=4, candidate_mode=2, candidate_topk_blocks=64)
    restricted_score = torch.tensor([[float(10 - j) for j in range(8)] + [-1.0, -1.0, 9.0, -1.0] * 2])
    restricted_visible = torch.full((1,), 16, dtype=torch.int64)
    masked_restricted = apply_causal_mask(restricted_score, restricted_visible)
    consumed = consumer_token_topk_per_row(masked_restricted, restricted_visible, (frozenset({0}),), consumer_cfg)
    assert consumed[0] == frozenset({0, 1, 2, 3}), f"consumer picks={consumed[0]}"
    assert all(j < block_size for j in consumed[0]), f"consumer escaped the pool: {consumed[0]}"
    unrestricted = token_topk_per_row(masked_restricted, restricted_visible, QliA3Config(compress_ratio=1, topk=4))
    assert 10 in unrestricted[0] and 10 not in consumed[0], "pool restriction must exclude position 10"

    # mode mismatch is refused in both directions
    try:
        source_candidate_blocks_per_row(masked, visible, consumer_cfg)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("source helper accepted a consumer config")
    try:
        consumer_token_topk_per_row(masked_restricted, restricted_visible, (frozenset({0}),), source_cfg)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("consumer helper accepted a source config")

    # a pool with the wrong number of rows is refused
    try:
        consumer_token_topk_per_row(
            masked_restricted, restricted_visible, (frozenset({0}), frozenset({1})), consumer_cfg
        )
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("consumer helper accepted a pool length mismatch")
    print("[ok] mode=1 blockwise-max pool (pin) + mode=2 pool-restricted top-k + reverse gating")


# --------------------------------------------------------------------------
# mode=2 candidate-pool negative control (CPU semantic layer only).
# --------------------------------------------------------------------------
# The *former* arch22 mode=2 path penalized an out-of-pool *but reachable* score
# with a *finite* penalty constant (``NEG_HUGE_F32``,
# ``quant_lightning_indexer_v2_dsv41_vector.h``) instead of -inf, while
# keeping the real position index. Because a finite penalty still ranks above
# -inf, such a slot wins a top-k slot whenever the pool cannot fill ``topk`` -- a
# real index leaks where the eager model emits the ``-1`` sentinel
# (``csa2_attention.py``). This constant is part of the negative
# control; it is re-declared locally and never imported from the operator.
#
# ``NEG_HUGE_F32`` is the fp32 bit pattern ``0xF14A3E31`` (approx.
# ``-1.001458508e30``); the value below is a *float64 approximation*, NOT
# bit-identical to that constant. The negative control only relies on the
# property the constant was chosen for -- a finite score small enough to rank
# below every real candidate score -- so the exact bits are irrelevant here.
_OLD_OUT_OF_POOL_SCORE = -1e30


def _golden_emission(valid: frozenset[int], topk: int, offset: int) -> list[int]:
    """Eager emission: valid picks shifted by ``offset``, the rest ``-1`` sentinels."""
    emitted = sorted(i + offset for i in valid)
    emitted.extend([-1] * (topk - len(valid)))
    return sorted(emitted)


def _semantic_layer_emission(
    score: torch.Tensor,
    visible: int,
    pool: frozenset[int],
    block_size: int,
    topk: int,
    offset: int,
) -> list[int]:
    """Independent ``score=-inf / index=-1`` pairing -- the correct semantic layer.

    A slot is eligible as a *real* pick only when it is causally reachable, inside
    the candidate pool and its score is finite. Every other slot -- pool miss,
    unreachable, or non-finite (NaN/inf) -- is paired as ``(score=-inf, index=-1)``,
    so a top-``topk`` by score emits ``index+offset`` for real pairs and the ``-1``
    sentinel otherwise: ``-1`` never receives the offset and a NaN can never
    become a real index. Pure CPU; this is not the NPU kernel.
    """
    pairs: list[tuple[float, int]] = []
    for j in range(int(score.numel())):
        value = float(score[j])
        if j < visible and (j // block_size) in pool and math.isfinite(value):
            pairs.append((value, j))
        else:
            pairs.append((float("-inf"), -1))
    order = sorted(range(len(pairs)), key=lambda j: (-pairs[j][0], j))
    return [(-1 if pairs[j][1] < 0 else pairs[j][1] + offset) for j in order[:topk]]


def _old_pool_degrade_emission(
    score: torch.Tensor,
    visible: int,
    pool: frozenset[int],
    block_size: int,
    topk: int,
    offset: int,
) -> list[int]:
    """The former arch22 mode=2 rule: finite ``NEG_HUGE`` + real index on a pool miss.

    Reproduces the pre-fix contract for finite inputs: an out-of-pool reachable
    score is flattened to a finite ``-1e30`` while its real index is kept; only an
    unreachable (``-inf``) slot sinks to the ``-1`` sentinel. The former rule kept
    a real index for any reachable slot (no finiteness gate), so a reachable NaN
    also stayed in the sort input. A simulation of the old rule, not the device --
    the current kernel instead normalizes every invalid slot to ``(-inf, -1)`` up
    front.
    """
    scores: list[float] = []
    indices: list[int] = []
    for j in range(int(score.numel())):
        value = float(score[j])
        in_pool = (j // block_size) in pool
        if j < visible:
            indices.append(j)
            scores.append(value if in_pool else _OLD_OUT_OF_POOL_SCORE)
        else:
            indices.append(-1)
            scores.append(float("-inf"))
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    return [(-1 if indices[j] < 0 else indices[j] + offset) for j in order[:topk]]


def test_consumer_pool_leak_negative_control() -> None:
    """CPU negative control for the mode=2 candidate-pool leak.

    This models the *semantic layer* on CPU only -- it makes no claim about the
    NPU kernel, its sort order or the compiled operator. The oracle is
    ``consumer_token_topk_per_row`` (this file): it restricts to the
    pool *and* requires a finite score, so its valid set is the golden one and the
    sentinel count is ``topk - |valid|``. We (a) freeze the oracle's valid set and
    sentinel count, (b) show the finite-``NEG_HUGE``+real-index rule FAILS at
    least one control, and (c) show the ``score=-inf / index=-1`` pairing PASSES
    them all.
    """
    block_size = QLI_V3_CAND_BLOCK_SIZE  # 8
    offset = 5
    width = 16

    def oracle_valid(score_row: list[float], visible: int, pool: frozenset[int], topk: int) -> frozenset[int]:
        assert len(score_row) == width
        score = torch.tensor([score_row], dtype=torch.float32)
        vis = torch.tensor([visible], dtype=torch.int64)
        config = QliA3Config(compress_ratio=1, topk=topk, candidate_mode=2, candidate_topk_blocks=64)
        masked = apply_causal_mask(score, vis)
        return consumer_token_topk_per_row(masked, vis, (pool,), config)[0]

    # Control 1: visible ∩ pool = 8 < topk = 10 -> the pool cannot fill topk, so
    # the out-of-pool slots (block 1, still reachable) are picked as fill.
    topk1, pool1 = 10, frozenset({0})
    score1 = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] + [100.0 - i for i in range(8)]
    valid1 = oracle_valid(score1, width, pool1, topk1)
    assert valid1 == frozenset(range(8)), f"S1 oracle valid={sorted(valid1)}"
    assert topk1 - len(valid1) == 2, "S1 sentinel count changed"
    golden1 = _golden_emission(valid1, topk1, offset)
    assert golden1 == sorted([-1, -1, 5, 6, 7, 8, 9, 10, 11, 12]), f"S1 golden={golden1}"
    score1_t = torch.tensor(score1, dtype=torch.float32)
    old1 = sorted(_old_pool_degrade_emission(score1_t, width, pool1, block_size, topk1, offset))
    semantic1 = sorted(_semantic_layer_emission(score1_t, width, pool1, block_size, topk1, offset))
    assert old1 != golden1, "S1: the finite-NEG_HUGE pool leak was not caught"
    assert -1 not in old1, f"S1: the old rule emitted no -1 sentinel: {old1}"
    assert semantic1 == golden1, f"S1 semantic={semantic1} != golden={golden1}"
    assert {o - offset for o in semantic1 if o != -1} == set(valid1), "S1 semantic valid set diverges"

    # Control 2: empty pool -> every reachable slot is a pool miss.
    topk2, pool2 = 4, frozenset()
    score2 = [5.0 - i for i in range(width)]
    valid2 = oracle_valid(score2, width, pool2, topk2)
    assert valid2 == frozenset(), f"S2 oracle valid={sorted(valid2)}"
    assert topk2 - len(valid2) == 4, "S2 sentinel count changed"
    golden2 = _golden_emission(valid2, topk2, offset)
    assert golden2 == [-1, -1, -1, -1], f"S2 golden={golden2}"
    score2_t = torch.tensor(score2, dtype=torch.float32)
    old2 = sorted(_old_pool_degrade_emission(score2_t, width, pool2, block_size, topk2, offset))
    semantic2 = sorted(_semantic_layer_emission(score2_t, width, pool2, block_size, topk2, offset))
    assert old2 != golden2, "S2: the empty-pool leak was not caught"
    assert all(o != -1 for o in old2), f"S2: the old rule emitted sentinels only: {old2}"
    assert semantic2 == golden2, f"S2 semantic={semantic2} != golden={golden2}"

    # Control 3: an in-pool, causally-reachable NaN score. The oracle's finite
    # gate drops it (it is not a valid index) while the old rule -- which has no
    # finiteness gate and only sinks an exact -inf -- keeps its real index 0 in the
    # sort input instead of excluding the slot up front.
    topk3, pool3 = 4, frozenset({0})
    score3 = [float("nan")] + [9.0 - i for i in range(7)] + [100.0 - i for i in range(8)]
    assert score3[0] != float("-inf"), "S3 control is vacuous: the NaN was written as -inf"
    valid3 = oracle_valid(score3, width, pool3, topk3)
    assert valid3 == frozenset({1, 2, 3, 4}), f"S3 oracle valid={sorted(valid3)}"
    assert 0 not in valid3, "S3: the reachable NaN slot leaked into the valid index set"
    golden3 = _golden_emission(valid3, topk3, offset)
    score3_t = torch.tensor(score3, dtype=torch.float32)
    semantic3 = sorted(_semantic_layer_emission(score3_t, width, pool3, block_size, topk3, offset))
    assert semantic3 == golden3, f"S3 semantic={semantic3} != golden={golden3}"
    assert 0 not in {o - offset for o in semantic3 if o != -1}, "S3: a NaN flowed into a real index"
    # The old rule keeps a real index for any reachable slot, with no finiteness
    # gate -- so this NaN slot enters the sort carrying the real index 0, whereas
    # the oracle (and the semantic pairing) refuse the slot up front. Whether the
    # NaN then *sorts out* as index 0 on device is NOT asserted here (device NaN
    # ordering is untested).

    print("[ok] mode=2 pool leak: oracle frozen; old finite-NEG_HUGE rule FAILS controls 1&2; semantic -inf/-1 passes")


# --------------------------------------------------------------------------
# mode=2 output-index offset must never shift the -1 sentinel
# (contract provenance: ``...service_vector_arch22.h``).
# --------------------------------------------------------------------------
# This is a CPU *semantic-layer contract* check, not a transcription of a kernel
# source path: an output-index offset site must be sentinel-aware. Once a
# pool-miss / unreachable slot is normalized to ``(score=-inf, index=-1)``, a
# *blind* ``index + rowOff`` -- the anti-pattern this contract forbids -- turns
# every ``-1`` into ``rowOff - 1``, a plausible in-range index, so the sentinel
# silently leaks exactly like the pool leak. The eager contract shifts only real
# picks and keeps ``-1`` (``_golden_emission``). That symbol is the
# contract's provenance, not a claim about the current kernel's literal emit
# path; no claim is made about the device.
def _offset_all_slots(emitted: list[int], offset: int) -> list[int]:
    """Blind offset (forbidden anti-pattern): shifts every slot, sentinels too."""
    return [e + offset for e in emitted]


def _offset_valid_slots(emitted: list[int], offset: int) -> list[int]:
    """Guarded offset (the contract): only non-sentinel picks receive the offset."""
    return [(-1 if e < 0 else e + offset) for e in emitted]


def test_offset_preserves_sentinel_negative_control() -> None:
    block_size = QLI_V3_CAND_BLOCK_SIZE  # 8
    width, topk, offset = 16, 10, 5
    # Pool = block 0 only -> 8 in-pool visible slots; topk=10 cannot be filled, so
    # two -1 sentinels are emitted (visible ∩ pool = 8 < topk).
    pool = frozenset({0})
    score_row = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] + [100.0 - i for i in range(8)]
    score = torch.tensor(score_row, dtype=torch.float32)
    # offset=0 yields the raw picks + -1 sentinels (index emission before the offset site)
    pre = sorted(_semantic_layer_emission(score, width, pool, block_size, topk, 0))
    valid = frozenset(i for i in pre if i >= 0)
    assert valid == frozenset(range(8)), f"pre-offset picks={sorted(valid)}"
    assert pre.count(-1) == topk - len(valid) == 2, f"sentinels={pre.count(-1)}"

    golden = _golden_emission(valid, topk, offset)
    assert golden == sorted([-1, -1, 5, 6, 7, 8, 9, 10, 11, 12]), f"golden={golden}"

    # Old blind rule: no sentinel survives and a fake index rowOff-1 appears.
    blind = sorted(_offset_all_slots(pre, offset))
    assert blind != golden, "the blind offset was not caught"
    assert -1 not in blind, f"the blind rule emitted sentinels: {blind}"
    assert (offset - 1) in blind, "the blind rule did not fabricate rowOff-1"

    # Fixed guarded rule: real picks shifted, sentinels preserved.
    guarded = sorted(_offset_valid_slots(pre, offset))
    assert guarded == golden, f"guarded={guarded} != golden={golden}"
    assert guarded.count(-1) == topk - len(valid), "the guard did not preserve every sentinel"
    assert {g - offset for g in guarded if g != -1} == set(valid)

    # No sentinel -> the guard must be a no-op (offsets are not suppressed).
    full = sorted(valid)
    assert _offset_valid_slots(full, offset) == _offset_all_slots(full, offset)

    print("[ok] offset preserves -1 sentinel: blind Adds FAILS; guarded rule matches golden")


def test_sortall_capacity_negative_control() -> None:
    """The arch22 mode=2 candidate sort must pad to a ``SortAll``-lossless length.

    ``SortAll`` 4-way merges its ``logits_num / 32`` ``Sort32`` groups with
    ``repeatTimes = groups // 4``; a group count that is not a multiple of four at
    any non-terminal step leaves the trailing ``groups % 4`` groups unread, so they
    are silently dropped. Sorting a candidate row at a raw 64-multiple block count
    (192/320/576/640/704/896/1152/1600/1984) therefore loses real candidates, while
    padding to the smallest lossless length >= ``candBlocks``
    (64/128/256/384/512/1024/1536/2048) keeps every candidate in the first
    ``candBlocks`` slots. Pure CPU: no NPU, no compiled operator, no UB-copy model.
    """
    # Capacity fingerprint: exactly the 32*m*4**k counts in (0, 2048].
    assert sortall_lossless_lengths() == (32, 64, 96, 128, 256, 384, 512, 1024, 1536, 2048), sortall_lossless_lengths()

    # Every legal candidate count (64*k, k=1..32) maps to a lossless pad length that
    # (a) is itself lossless, (b) covers the row, and (c) is already minimal when
    # the row is lossless.
    lossless_rows = []
    for k in range(1, 33):
        blocks = 64 * k
        pad = sortall_align_length(blocks)
        assert sortall_is_lossless(pad), (blocks, pad)
        assert blocks <= pad <= QLI_V3_CAND_TOPK_BLOCKS_MAX, (blocks, pad)
        if sortall_is_lossless(blocks):
            assert pad == blocks, (blocks, pad)
            lossless_rows.append(blocks)
    assert lossless_rows == [64, 128, 256, 384, 512, 1024, 1536, 2048], lossless_rows

    # A raw (unpadded) sort of a non-lossless multiple drops whole groups, so fewer
    # than ``candBlocks`` candidates survive -> CountGE's [0, candBlocks) window can no
    # longer be all-real. The padded sort keeps every group.
    for blocks in (192, 320, 576, 640, 704, 896, 1152, 1600, 1984):
        surviving = sortall_surviving_groups(blocks)
        assert len(surviving) < blocks // QLI_V3_SORT_GROUP, (blocks, surviving)
        pad = sortall_align_length(blocks)
        assert len(sortall_surviving_groups(pad)) == pad // QLI_V3_SORT_GROUP, (blocks, pad)

    # Concrete loss: a raw 192-block sort keeps only groups 0..3 (elements 0..127),
    # so the globally-highest candidate cannot be selected when it sits in the
    # dropped group 5 (elements 160..191). Simulate the reduction over (score, index)
    # pairs -- only surviving groups' elements reach the merge.
    blocks = 192
    group = QLI_V3_SORT_GROUP
    dropped = sorted(set(range(blocks // group)) - set(sortall_surviving_groups(blocks)))
    assert dropped == [4, 5], dropped

    scores = [float(blocks - e) for e in range(blocks)]  # distinct real candidates
    scores[160] = float(blocks + 1000)  # the globally-best candidate is in dropped group 5
    all_pairs = sorted(((scores[e], e) for e in range(blocks)), reverse=True)
    surviving = set(sortall_surviving_groups(blocks))
    surviving_pairs = sorted(((scores[e], e) for e in range(blocks) if e // group in surviving), reverse=True)
    assert all_pairs[0][1] in range(160, 192), all_pairs[0]
    assert surviving_pairs[0][1] not in range(160, 192), surviving_pairs[0]
    assert surviving_pairs[0][0] < all_pairs[0][0], (surviving_pairs[0], all_pairs[0])
    print("[ok] SortAll capacity: raw sorts drop whole groups; padding to a lossless length keeps every candidate")


def test_sortall_edge_cases() -> None:
    """``sortall_align_length`` no-candidate error and the ``limit=0`` boundary."""
    assert sortall_lossless_lengths(0) == (), sortall_lossless_lengths(0)

    # 2049 exceeds every lossless candidate length within the default limit.
    try:
        sortall_align_length(QLI_V3_CAND_TOPK_BLOCKS_MAX + 1)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("sortall_align_length returned a pad beyond the limit")

    # An explicit empty limit admits no candidate either.
    try:
        sortall_align_length(1, limit=0)
    except QliA3ReferenceConfigError:
        pass
    else:
        raise AssertionError("sortall_align_length accepted an empty candidate set")
    print("[ok] SortAll edge cases: no-candidate error + empty limit")


def test_config_rejects() -> None:
    bad = [
        {"compress_ratio": 4, "topk": 4},
        {"compress_ratio": 2, "topk": 4, "quant_mode": 0},
        {"compress_ratio": 2, "topk": 4, "mask_mode": 0},
        {"compress_ratio": 2, "topk": 4, "candidate_mode": 1, "candidate_block_size": 4},
        {"compress_ratio": 2, "topk": 4, "candidate_mode": 2, "candidate_topk_blocks": 63},
        {"compress_ratio": 2, "topk": 0},
        {"compress_ratio": 2, "topk": 4, "head_dim": 64},
        {"compress_ratio": 2, "topk": 4, "candidate_mode": 9},
        {"compress_ratio": 2, "topk": 4, "key_num_heads": 2},
        {"compress_ratio": 2, "topk": 2049},
    ]
    for kwargs in bad:
        try:
            QliA3Config(**kwargs)
        except QliA3ReferenceConfigError:
            pass
        else:
            raise AssertionError(f"QliA3Config accepted an out-of-contract config: {kwargs}")
    assert QliA3Config(compress_ratio=2, topk=2048, candidate_mode=1, candidate_topk_blocks=2048).topk == 2048
    # Pinned current behaviour: with candidate_mode=OFF the candidate_topk_blocks
    # field is not consulted, so an out-of-contract block count is accepted there.
    off_cfg = QliA3Config(compress_ratio=2, topk=4, candidate_mode=3, candidate_topk_blocks=63)
    assert off_cfg.candidate_topk_blocks == 63, "OFF mode must skip the candidate_topk_blocks check"
    print("[ok] config gate rejects every out-of-contract field")


def test_empty_bf16_contract() -> None:
    assert_a3_values_output_empty(torch.empty(0, dtype=torch.bfloat16))
    for bad in (torch.zeros(1, dtype=torch.bfloat16), torch.empty(0, dtype=torch.float32)):
        try:
            assert_a3_values_output_empty(bad)
        except QliA3ReferenceConfigError:
            pass
        else:
            raise AssertionError(f"empty-BF16 contract accepted {bad.dtype} numel={bad.numel()}")
    print("[ok] A3 empty-BF16 values contract")


def test_end_to_end_pipeline() -> None:
    """Run one small full chain: int8 q/k -> dot -> rectified -> w*q_scale -> scores
    -> causal mask -> source block pool -> consumer top-k, checking the pool holds.

    Every earlier check isolates one stage with a hand-built FP32 score matrix; this
    is the only place the stages are threaded together, so a rounding-induced
    boundary drift between them would surface here.
    """
    torch.manual_seed(4321)
    rows, heads = 2, 2
    block_size = QLI_V3_CAND_BLOCK_SIZE
    num_blocks = 65
    width = num_blocks * block_size  # 520 compressed rows
    head_dim = QLI_V3_INDEX_HEAD_DIM

    query = torch.randint(-127, 128, (rows, heads, head_dim), dtype=torch.int8)
    key = torch.randint(-127, 128, (rows, width, head_dim), dtype=torch.int8)
    # Non-negative weights keep the FP32 scores from collapsing onto the ReLU zero
    # plateau, where a top-k boundary would legitimately be a tie; the contract
    # itself does not restrict the sign of ``weights``.
    weights = torch.rand(rows, heads, dtype=torch.float32).to(torch.float16)
    query_scale = torch.rand(rows, heads, dtype=torch.float32).to(torch.float16)
    key_scale = torch.rand(rows, width, dtype=torch.float32).to(torch.float16)
    validate_a3_dtypes(query, key, weights, query_scale, key_scale)

    dot = int8_dot_int32(query, key)
    rectified = rectified_scaled_qk(dot)
    weight_scale = head_weight_scale_product(weights, query_scale)
    scores = indexer_scores(rectified, weight_scale, key_scale)
    assert scores.dtype == torch.float32 and tuple(scores.shape) == (rows, width)

    visible = causal_visible_counts(1, committed_rows=width, residual=0, query_length=rows)
    masked = apply_causal_mask(scores, visible)
    assert bool(torch.isfinite(masked[0, 0])), "row 0 must have at least one visible column"

    source_cfg = QliA3Config(compress_ratio=1, topk=4, candidate_mode=1, candidate_topk_blocks=64)
    consumer_cfg = QliA3Config(compress_ratio=1, topk=4, candidate_mode=2, candidate_topk_blocks=64)
    pools = source_candidate_blocks_per_row(masked, visible, source_cfg)
    consumed = consumer_token_topk_per_row(masked, visible, pools, consumer_cfg)

    for row in range(rows):
        # 65 valid blocks, a 64-slot pool: exactly one block is excluded, so the
        # consumer restriction below is non-vacuous.
        assert len(pools[row]) == 64, (row, len(pools[row]))
        excluded = set(range(num_blocks)) - pools[row]
        assert len(excluded) == 1, (row, sorted(excluded))
        for j in consumed[row]:
            assert j < int(visible[row].item()), ("consumer escaped the causal boundary", row, j)
            assert (j // block_size) in pools[row], ("consumer escaped the source pool", row, j, sorted(excluded))
    print("[ok] end-to-end: int8 q/k -> scores -> causal mask -> source pool -> consumer top-k")


