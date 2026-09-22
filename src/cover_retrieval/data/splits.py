"""Deterministic, WID-disjoint split assignment.

Protocol (see notes/decisions.md D-001, D-008):

1. Sort WIDs lexicographically.
2. Permute them with ``numpy.random.default_rng(seed)``.
3. Walk the permutation; skip WIDs whose recordings fail validation; fill roles in
   the fixed order calibration -> query -> distractor -> validation -> train.

Filtering *while* walking a seeded permutation of the full sorted list gives the
same result as filtering first and then taking a seeded permutation-ordered sample
of the valid pool, and it lets the data be fetched in priority order on slow links.
The assignment depends only on the seed, the sorted IDs and file validity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

ROLE_ORDER: tuple[str, ...] = ("calibration", "query", "distractor", "validation", "train")


def seeded_wid_order(wids: Sequence[str], seed: int) -> list[str]:
    """Lexicographically sort ``wids`` then apply a seeded permutation."""
    ordered = sorted(wids)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[i] for i in permutation]


def role_sizes(split_config: Mapping[str, object]) -> dict[str, int | None]:
    """Role -> number of WIDs. ``None`` for train means "all remaining valid WIDs"."""
    sizes: dict[str, int | None] = {}
    for role in ROLE_ORDER:
        value = split_config.get(f"n_{role}")
        sizes[role] = None if value in (None, "all") else int(value)  # type: ignore[arg-type]
    return sizes


@dataclass
class SplitAssignment:
    roles: dict[str, list[str]]  # role -> WIDs in permutation order
    excluded: dict[str, str] = field(default_factory=dict)  # WID -> reason
    unavailable: list[str] = field(default_factory=list)  # WIDs not examined/fetched

    def role_of(self) -> dict[str, str]:
        return {wid: role for role, wids in self.roles.items() for wid in wids}


def assign_roles(
    ordered_wids: Sequence[str],
    sizes: Mapping[str, int | None],
    check_wid: Callable[[str], str | None],
    *,
    stop_when_full: bool = True,
) -> SplitAssignment:
    """Fill roles along ``ordered_wids``.

    ``check_wid(wid)`` returns ``None`` if the WID is usable, otherwise a reason
    string. A reason starting with ``"unavailable"`` means the files are simply not
    present locally (e.g. partial download); such WIDs are recorded separately.
    """
    roles: dict[str, list[str]] = {role: [] for role in ROLE_ORDER}
    assignment = SplitAssignment(roles=roles)
    role_iter = iter(ROLE_ORDER)
    current = next(role_iter)

    def full(role: str) -> bool:
        size = sizes[role]
        return size is not None and len(roles[role]) >= size

    for wid in ordered_wids:
        while current is not None and full(current):
            current = next(role_iter, None)
        if current is None:
            if stop_when_full:
                break
            assignment.unavailable.append(wid)
            continue
        reason = check_wid(wid)
        if reason is None:
            roles[current].append(wid)
        elif reason.startswith("unavailable"):
            assignment.unavailable.append(wid)
        else:
            assignment.excluded[wid] = reason

    for role in ROLE_ORDER:
        size = sizes[role]
        if size is not None and len(roles[role]) < size:
            raise RuntimeError(
                f"not enough usable WIDs for role {role!r}: {len(roles[role])}/{size}. "
                "Fetch more data (scripts/download_datacos.py) or lower the split sizes."
            )
    return assignment


def check_disjoint(assignment: SplitAssignment) -> None:
    """Raise if any WID appears in more than one role."""
    seen: dict[str, str] = {}
    for role, wids in assignment.roles.items():
        for wid in wids:
            if wid in seen:
                raise AssertionError(f"{wid} is in both {seen[wid]} and {role}")
            seen[wid] = role
