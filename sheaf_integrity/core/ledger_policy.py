"""
Ledger write-back policy engine.

Decision: NOT an auto-repair path.
  - Relative CRC flags self_consistent_wrong_vs_ledger
  - Write-back is a *policy action* with an explicit key
  - Default = refuse

This module is the product boundary so agents cannot "helpfully" overwrite SoT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

POLICY_ALLOW = "FOUNDER_POLICY_ALLOW_WRITEBACK"


@dataclass
class WriteBackRequest:
    entity_ids: list[str]
    reason: str
    requested_by: str = "agent"


@dataclass
class WriteBackResult:
    allowed: bool
    applied: bool
    detail: str
    entities_touched: list[str]


def evaluate_writeback(
    relative_status: str | None,
    request: WriteBackRequest,
    policy_key: str | None,
) -> WriteBackResult:
    if relative_status != "self_consistent_wrong_vs_ledger":
        return WriteBackResult(
            allowed=False,
            applied=False,
            detail=f"write-back only valid for self_consistent_wrong_vs_ledger (got {relative_status})",
            entities_touched=[],
        )
    if policy_key != POLICY_ALLOW:
        return WriteBackResult(
            allowed=False,
            applied=False,
            detail="refused: missing FOUNDER_POLICY_ALLOW_WRITEBACK — human/policy gate required",
            entities_touched=[],
        )
    return WriteBackResult(
        allowed=True,
        applied=False,  # caller must still execute; this only authorizes
        detail=f"authorized for {len(request.entity_ids)} entities: {request.reason}",
        entities_touched=list(request.entity_ids),
    )


def apply_authorized_writeback(
    feats: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    entity_node_ids: list[str],
    authorization: WriteBackResult,
) -> dict[str, np.ndarray]:
    if not authorization.allowed:
        raise PermissionError(authorization.detail)
    out = {k: v.copy() for k, v in feats.items()}
    for nid in entity_node_ids:
        if nid in reference and nid in out:
            out[nid] = reference[nid].copy()
    return out
