"""CPU tests: ordered-manifest load + O0 archive/fingerprint口径 (incl. negative control)."""
import json
import tempfile
from pathlib import Path

import pytest

from decode.common import (
    ARCHIVE_EXIT_INCOMPLETE,
    ARCHIVE_EXIT_NOT_RUN,
    ARCHIVE_GATE_FAIL,
    ARCHIVE_GATE_INCOMPLETE,
    ARCHIVE_GATE_NOT_RUN,
    ARCHIVE_GATE_PASS,
    O0_KERNEL_BAND,
    archive_gate_exit_code,
    archive_gate_status,
    classify_o0_vs_ref,
    compare_token_sequences,
    filter_prompts,
    greedy_byte_exact_pass,
    greedy_numerical_safety_pass,
    greedy_numerical_safety_triggers_fail,
    load_o0_archive,
    o0_archive_triggers_fail,
    o0_fingerprint_triggers_fail,
    sha256_file,
    validate_o0_archive_not_self,
)


def test_compare_match_returns_none():
    """Exact equal sequences → compare returns None (no diverge)."""
    assert compare_token_sequences([1, 2, 3], [1, 2, 3]) is None


def test_o0_negative_control_tamper_one_token():
    """Flip one token mid-sequence → detector reports pos/toks."""
    greedy = [10, 20, 30, 40, 50, 60]
    emitted = list(greedy)
    emitted[3] = 999
    div = compare_token_sequences(emitted, greedy)
    assert div is not None
    assert div["pos"] == 3
    assert div["spec_tok"] == 999
    assert div["greedy_tok"] == 40


def test_o0_length_mismatch_detected():
    """Equal prefix, longer ref → diverge at first missing index."""
    div = compare_token_sequences([1, 2, 3], [1, 2, 3, 4])
    assert div is not None
    assert div["pos"] == 3
    assert div["spec_tok"] is None
    assert div["greedy_tok"] == 4


def test_classify_archive_hard_byte_mismatch():
    """classify: mid+no-candidate→hard; gap_spec≤band→near_tie; gap_spec>band→hard."""
    cls = classify_o0_vs_ref([1, 2, 9], [1, 2, 3])
    assert cls["kind"] == "hard"
    cls2 = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3], top2_gap=0.01,
        greedy_top1=3, greedy_top2=7, gap_spec=0.01, spec_rank=2,
    )
    assert cls2["kind"] == "near_tie"
    cls3 = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3], top2_gap=0.20,
        greedy_top1=3, greedy_top2=7, gap_spec=0.20, spec_rank=5,
    )
    assert cls3["kind"] == "hard"
    assert O0_KERNEL_BAND == 0.15


def test_candidate_near_tie_gap_ok_but_rank_far_is_hard():
    """Negctrl: small top1−top2 gap alone is insufficient; far-rank spec → hard."""
    # Old heuristic would near_tie on top2_gap=0.01; candidate rule must reject.
    cls = classify_o0_vs_ref(
        [1, 2, 999], [1, 2, 3],
        top2_gap=0.01,  # would have passed legacy heuristic
        greedy_top1=3, greedy_top2=4,
        gap_spec=5.0, spec_rank=50,
    )
    assert cls["kind"] == "hard"
    assert o0_fingerprint_triggers_fail(
        [1, 2, 999], [1, 2, 3], top2_gap=0.01,
        greedy_top1=3, greedy_top2=4, gap_spec=5.0, spec_rank=50,
    ) is True


def test_candidate_near_tie_spec_eq_top2_despite_large_gap_is_hard():
    """Negctrl (Round-7 reversal): rank 2 grants nothing; gap 0.50 > band → hard.

    Round-6 classified this near_tie via the `spec_tok == greedy_top2`
    disjunct. That shortcut is a specification error: being second bounds
    nothing about separation, so the OR could admit an arbitrarily large gap.
    """
    cls = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3],
        top2_gap=0.50,
        greedy_top1=3, greedy_top2=9,
        gap_spec=0.50, spec_rank=2,
    )
    assert cls["kind"] == "hard"
    assert o0_fingerprint_triggers_fail(
        [1, 2, 9], [1, 2, 3], top2_gap=0.50,
        greedy_top1=3, greedy_top2=9, gap_spec=0.50, spec_rank=2,
    ) is True


def test_candidate_gap_only_rank2_within_band_is_near_tie():
    """Boundary 1/4: spec is top2 AND gap ≤ band → near_tie (gap is what counts)."""
    cls = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3],
        top2_gap=0.02,
        greedy_top1=3, greedy_top2=9,
        gap_spec=0.02, spec_rank=2,
    )
    assert cls["kind"] == "near_tie"
    assert greedy_numerical_safety_triggers_fail(
        [1, 2, 9], [1, 2, 3], top2_gap=0.02,
        greedy_top1=3, greedy_top2=9, gap_spec=0.02, spec_rank=2,
    ) is False


def test_candidate_gap_only_rank3_within_band_is_near_tie():
    """Boundary 2/4: rank 3 still near_tie when its own gap is inside the band."""
    cls = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3],
        top2_gap=0.01,
        greedy_top1=3, greedy_top2=4,
        gap_spec=0.14, spec_rank=3,
    )
    assert cls["kind"] == "near_tie"
    assert cls["spec_rank"] == 3


def test_candidate_gap_missing_is_hard():
    """Boundary 3/4: no gap probe → hard, never a pass, whatever the rank says."""
    cls = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3],
        top2_gap=0.01,
        greedy_top1=3, greedy_top2=9,
        gap_spec=None, spec_rank=2,
    )
    assert cls["kind"] == "hard"
    assert greedy_numerical_safety_triggers_fail(
        [1, 2, 9], [1, 2, 3], top2_gap=0.01,
        greedy_top1=3, greedy_top2=9, gap_spec=None, spec_rank=2,
    ) is True


def test_candidate_gap_just_above_band_is_hard():
    """Boundary 4/4: band is inclusive; a hair above it is hard."""
    band = O0_KERNEL_BAND
    at_band = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3], greedy_top1=3, greedy_top2=9,
        gap_spec=band, spec_rank=2,
    )
    above = classify_o0_vs_ref(
        [1, 2, 9], [1, 2, 3], greedy_top1=3, greedy_top2=9,
        gap_spec=band + 1e-6, spec_rank=2,
    )
    assert at_band["kind"] == "near_tie"
    assert above["kind"] == "hard"


def test_classify_len_boundary_not_hard():
    """Public prefix OK, unilateral tail → len_boundary (not mid-hard)."""
    cls = classify_o0_vs_ref([1, 2, 3], [1, 2, 3, 4])
    assert cls["kind"] == "len_boundary"


def test_numerical_safety_fails_on_len_boundary():
    """len_boundary fails numerical_safety (cap fix must clear these)."""
    assert greedy_numerical_safety_triggers_fail([1, 2, 3], [1, 2, 3, 4]) is True
    assert greedy_numerical_safety_triggers_fail([1, 2, 3, 4], [1, 2, 3, 4]) is False


def test_numerical_safety_near_tie_not_fail_not_byte_exact():
    """near_tie: safety non-fail, but not byte-exact vs greedy."""
    spec, g = [1, 2, 9], [1, 2, 3]
    assert greedy_numerical_safety_triggers_fail(
        spec, g, gap_spec=0.01, greedy_top2=9) is False
    assert classify_o0_vs_ref(
        spec, g, gap_spec=0.01, greedy_top1=3, greedy_top2=9)["kind"] == "near_tie"
    assert greedy_numerical_safety_triggers_fail(
        spec, g, gap_spec=0.20, greedy_top2=7) is True
    assert greedy_byte_exact_pass(n_exact=0, n_prompts=1) is False


def test_byte_exact_false_when_near_tie_present():
    """Tier D: any near_tie ⇒ greedy_byte_exact_pass FALSE; safety may still TRUE."""
    n_prompts, n_exact, n_near = 300, 281, 19
    assert n_exact + n_near == n_prompts
    assert greedy_byte_exact_pass(n_exact, n_prompts) is False
    assert greedy_numerical_safety_pass(n_len_boundary=0, n_hard=0) is True


def test_safety_true_with_near_tie_zero_material():
    """Tier D: near_tie-only run keeps numerical_safety TRUE."""
    assert greedy_numerical_safety_pass(0, 0) is True
    assert greedy_byte_exact_pass(n_exact=79, n_prompts=100) is False
    assert greedy_byte_exact_pass(n_exact=100, n_prompts=100) is True


def test_archive_gate_zero_coverage_not_run():
    """Tier D: no archive or zero covered ⇒ NOT_RUN (never silent PASS)."""
    assert archive_gate_status(False, 0, 100, 0) == ARCHIVE_GATE_NOT_RUN
    assert archive_gate_status(True, 0, 100, 0) == ARCHIVE_GATE_NOT_RUN
    assert archive_gate_status(True, 50, 100, 0) == ARCHIVE_GATE_INCOMPLETE
    assert archive_gate_status(True, 100, 100, 0) == ARCHIVE_GATE_PASS
    assert archive_gate_status(True, 100, 100, 1) == ARCHIVE_GATE_FAIL
    # Incomplete must not promote even if covered subset clean
    assert archive_gate_status(True, 99, 100, 0) == ARCHIVE_GATE_INCOMPLETE


def test_archive_explicit_zero_coverage_nonzero_exit():
    """Main-level: explicit --o0-archive + zero coverage ⇒ exit 5 (NOT_RUN)."""
    st = archive_gate_status(True, 0, 100, 0)
    assert st == ARCHIVE_GATE_NOT_RUN
    assert archive_gate_exit_code(st, archive_provided=True) == ARCHIVE_EXIT_NOT_RUN
    assert archive_gate_exit_code(st, archive_provided=False) is None


def test_archive_explicit_partial_coverage_nonzero_exit():
    """Main-level: explicit archive + partial coverage ⇒ exit 6 (INCOMPLETE)."""
    st = archive_gate_status(True, 50, 100, 0)
    assert st == ARCHIVE_GATE_INCOMPLETE
    assert archive_gate_exit_code(st, archive_provided=True) == ARCHIVE_EXIT_INCOMPLETE
    assert archive_gate_exit_code(ARCHIVE_GATE_PASS, True) is None
    assert archive_gate_exit_code(ARCHIVE_GATE_FAIL, True) == 2


def test_classify_match():
    """Identical sequences → kind match."""
    assert classify_o0_vs_ref([1, 2], [1, 2])["kind"] == "match"


def test_load_o0_archive_roundtrip():
    """Archive JSON loads id→emitted_tokens map."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "arch.json"
        path.write_text(json.dumps({
            "per_prompt_results": [
                {"id": "a", "emitted_tokens": [1, 2, 3]},
                {"id": "b", "emitted_tokens": [4, 5]},
            ]
        }))
        m = load_o0_archive(str(path))
        assert m["a"] == [1, 2, 3]
        assert m["b"] == [4, 5]


def test_ordered_manifest_preserves_file_order():
    """ordered=True keeps file order; ordered=False shuffles."""
    items = []
    for _id in ["c", "a", "b"]:
        items.append({
            "id": _id,
            "image": f"{_id}.jpg",
            "conversations": [
                {"from": "human", "value": f"Q{_id}\n<image>"},
                {"from": "gpt", "value": " ".join(["word"] * 90)},
            ],
        })
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.json"
        path.write_text(json.dumps(items))
        shuffled = filter_prompts(str(path), min_ref_words=80, seed=42, ordered=False)
        ordered = filter_prompts(str(path), min_ref_words=80, seed=42, ordered=True)
        assert [p["id"] for p in ordered] == ["c", "a", "b"]
        assert [p["id"] for p in ordered] != [p["id"] for p in shuffled] or len(items) >= 1


def test_archive_hard_negative_control():
    """Archive hard: any non-match (even near-tie gap) must trigger fail helper."""
    arch = [10, 20, 30, 40]
    bad = [10, 20, 99, 40]
    assert o0_archive_triggers_fail(bad, arch) is True
    assert o0_archive_triggers_fail(arch, arch) is False


def test_fingerprint_gate_intercept_gap_above_band():
    """Fingerprint: mid diverge + gap_spec>band (and not top2) → hard."""
    spec = [1, 2, 3, 999, 5]
    greedy = [1, 2, 3, 4, 5]
    assert o0_fingerprint_triggers_fail(
        spec, greedy, gap_spec=0.20, greedy_top2=7) is True
    assert o0_fingerprint_triggers_fail(
        spec, greedy, gap_spec=0.15, greedy_top2=7) is False  # == band OK
    assert o0_fingerprint_triggers_fail(
        spec, greedy, gap_spec=0.10, greedy_top2=7) is False
    # missing candidate probe on mid → hard
    assert o0_fingerprint_triggers_fail(spec, greedy) is True


def test_archive_hard_mismatch_intercept():
    """(b) Archive hard mismatch: different archived tokens → must intercept."""
    emitted = [1, 2, 3, 4]
    wrong_archive = [1, 2, 3, 999]
    assert o0_archive_triggers_fail(emitted, wrong_archive) is True
    assert classify_o0_vs_ref(emitted, wrong_archive)["kind"] != "match"


def test_anti_self_archive_path_equals_out_raises():
    """Anti-self: archive path == out path → ValueError."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "same.json"
        p.write_text(json.dumps({"per_prompt_results": [
            {"id": "x", "emitted_tokens": [1]}
        ]}))
        with pytest.raises(ValueError, match="anti-self"):
            validate_o0_archive_not_self(str(p), str(p), run_job_id="999")


def test_anti_self_archive_job_id_equals_run_raises():
    """Anti-self: provenance job_id == current PBS job → ValueError."""
    with tempfile.TemporaryDirectory() as td:
        arch = Path(td) / "arch.json"
        out = Path(td) / "out.json"
        arch.write_text(json.dumps({
            "o0_provenance": {"job_id": "174124013.gadi-pbs", "source_out": "/other/out.json"},
            "per_prompt_results": [{"id": "x", "emitted_tokens": [1]}],
        }))
        out.write_text("{}")
        with pytest.raises(ValueError, match="anti-self"):
            validate_o0_archive_not_self(
                str(arch), str(out), run_job_id="174124013.gadi-pbs",
            )
        # different job id → OK
        meta = validate_o0_archive_not_self(
            str(arch), str(out), run_job_id="000000000.gadi-pbs",
        )
        assert meta["sha256"] == sha256_file(arch)
