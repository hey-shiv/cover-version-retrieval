"""Deterministic, WID-disjoint split assignment and manifest construction."""

import filecmp
import shutil

import pytest
from scripts_helpers import run_script

from cover_retrieval.data.manifests import load_protocol, load_split_tracks
from cover_retrieval.data.splits import (
    ROLE_ORDER,
    assign_roles,
    check_disjoint,
    role_sizes,
    seeded_wid_order,
)
from cover_retrieval.data.synthetic import make_synthetic_datacos

WIDS = [f"W_{i}" for i in range(200)]
SIZES = {"calibration": 3, "query": 4, "distractor": 5, "validation": 6, "train": 10}


def test_seeded_order_is_deterministic_and_input_order_free():
    a = seeded_wid_order(WIDS, 20260817)
    b = seeded_wid_order(list(reversed(WIDS)), 20260817)
    assert a == b
    assert sorted(a) == sorted(WIDS)
    assert seeded_wid_order(WIDS, 1) != a


def test_assign_roles_fills_in_order_and_is_disjoint():
    order = seeded_wid_order(WIDS, 0)
    assignment = assign_roles(order, SIZES, lambda w: None)
    check_disjoint(assignment)
    assert {r: len(w) for r, w in assignment.roles.items()} == SIZES
    flat = [w for role in ROLE_ORDER for w in assignment.roles[role]]
    assert flat == order[: len(flat)]


def test_invalid_and_unavailable_wids_are_skipped():
    order = seeded_wid_order(WIDS, 0)
    bad = set(order[:5:2])
    missing = {order[1]}

    def check(wid):
        if wid in bad:
            return "corrupt"
        if wid in missing:
            return "unavailable: not downloaded"
        return None

    assignment = assign_roles(order, SIZES, check)
    used = set(assignment.role_of())
    assert not used & (bad | missing)
    assert set(assignment.excluded) == bad
    assert assignment.unavailable == [order[1]]


def test_not_enough_wids_raises():
    with pytest.raises(RuntimeError, match="not enough"):
        assign_roles(WIDS[:10], SIZES, lambda w: None)


def test_train_all_takes_remaining():
    sizes = {**SIZES, "train": None}
    assignment = assign_roles(seeded_wid_order(WIDS, 0), sizes, lambda w: None)
    assert len(assignment.roles["train"]) == len(WIDS) - sum(v for k, v in SIZES.items() if k != "train")
    assert (
        role_sizes(
            {"n_calibration": 1, "n_query": 1, "n_distractor": 1, "n_validation": 1, "n_train": "all"}
        )["train"]
        is None
    )


def test_manifests_are_deterministic_and_protocols_valid(smoke_config, tmp_path):
    make_synthetic_datacos(smoke_config["paths"]["datacos_root"], n_coveranalysis_works=100, seed=3)
    run_script("build_manifests", smoke_config)
    first = tmp_path / "first"
    shutil.copytree(smoke_config["paths"]["manifests_dir"], first)
    run_script("build_manifests", smoke_config)
    for name in (
        "coveranalysis_splits.csv",
        "dev_queries.csv",
        "dev_candidates.csv",
        "calibration_candidates.csv",
    ):
        assert filecmp.cmp(first / name, f"{smoke_config['paths']['manifests_dir']}/{name}", shallow=False)

    manifests = smoke_config["paths"]["manifests_dir"]
    dev = load_protocol(manifests, "dev")
    n_query, n_distractor = smoke_config["splits"]["n_query"], smoke_config["splits"]["n_distractor"]
    assert len(dev.queries) == n_query
    assert len(dev.candidates) == 2 * (n_query + n_distractor)
    relevance = dev.relevance_matrix()
    assert (relevance.sum(axis=1) == 1).all()  # exactly one partner per query
    assert not (relevance & dev.self_mask()).any()  # never the query itself
    # query = lexicographically lowest PID in its WID
    for q in dev.queries:
        partners = [c.pid for c in dev.candidates if c.wid == q.wid]
        assert q.pid == min(partners)

    roles = {role: {t.wid for t in load_split_tracks(manifests, role)} for role in ROLE_ORDER}
    for i, a in enumerate(ROLE_ORDER):
        for b in ROLE_ORDER[i + 1 :]:
            assert not roles[a] & roles[b], f"{a}/{b} share WIDs"
    calibration = load_protocol(manifests, "calibration")
    assert {t.wid for t in calibration.queries} == roles["calibration"]
    assert not {t.wid for t in calibration.candidates} & (
        roles["query"] | roles["distractor"] | roles["train"]
    )
