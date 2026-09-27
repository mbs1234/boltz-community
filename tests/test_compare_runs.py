"""Tests for boltz-compare: RMSD measures, replicate grouping, deltas, and the CLI."""

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import pytest
from click.testing import CliRunner

from boltz.scripts.compare_runs import COLUMNS, Thresholds, compare_dirs, main

if TYPE_CHECKING:
    from pathlib import Path

# A 40-residue helix-like backbone with a six-ring ligand (plus O and N) sitting
# in a pocket beside residues ~15-25. The ring lies in the xy plane.
N_RES = 40
RING = [
    (1.4 * math.cos(k * math.pi / 3), 1.4 * math.sin(k * math.pi / 3), 0.0)
    for k in range(6)
]
LIGAND_XYZ = np.array([*RING, (2.8, 0.0, 0.0), (-2.8, 0.0, 0.0)]) + np.array(
    [7.0, 0.0, 30.0]
)
LIGAND_ELEMENTS = ["C"] * 6 + ["O", "N"]


def _backbone() -> np.ndarray:
    i = np.arange(N_RES)
    return np.stack([2.3 * np.cos(i * 1.75), 2.3 * np.sin(i * 1.75), 1.5 * i], axis=1)


def _structure(ca: np.ndarray, ligand: np.ndarray | None) -> gemmi.Structure:
    st = gemmi.Structure()
    st.add_model(gemmi.Model(1))
    model = st[0]
    chain = gemmi.Chain("A")
    for i, xyz in enumerate(ca):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(i + 1, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        residue.het_flag = "A"
        for name, offset in (("CA", 0.0), ("CB", 1.5)):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(xyz[0] + offset, xyz[1], xyz[2])
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    if ligand is not None:
        lig_chain = gemmi.Chain("B")
        residue = gemmi.Residue()
        residue.name = "LIG1"
        residue.seqid = gemmi.SeqId(1, " ")
        residue.entity_type = gemmi.EntityType.NonPolymer
        residue.het_flag = "H"
        for k, (element, xyz) in enumerate(zip(LIGAND_ELEMENTS, ligand)):
            atom = gemmi.Atom()
            atom.name = f"{element}{k + 1}"
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(*xyz)
            residue.add_atom(atom)
        lig_chain.add_residue(residue)
        model.add_chain(lig_chain)
    st.setup_entities()
    return st


def _write_prediction(
    root: Path,
    name: str,
    st: gemmi.Structure,
    confidence: dict | None = None,
    affinity: dict | None = None,
    fmt: str = "cif",
) -> None:
    folder = root / f"boltz_results_{name}" / "predictions" / name
    folder.mkdir(parents=True)
    path = folder / f"{name}_model_0.{fmt}"
    if fmt == "cif":
        st.make_mmcif_document().write_file(str(path))
    else:
        st.write_pdb(str(path))
    (folder / f"confidence_{name}_model_0.json").write_text(
        json.dumps(confidence or {})
    )
    if affinity is not None:
        (folder / f"affinity_{name}.json").write_text(json.dumps(affinity))


def _rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    )


def _only_row(reports):
    assert len(reports) == 1
    return reports[0]


class TestRmsd:
    """RMSD measures remove rigid motion and measure real differences."""

    def test_identical_structures(self, tmp_path):
        """Identical predictions give zero protein and ligand RMSD."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(tmp_path / "ref", "sys", st)
        _write_prediction(tmp_path / "test", "sys", st)
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.ligand_chain == "B"
        assert report.values["ca_rmsd_cross_mean"] == pytest.approx(0.0, abs=1e-3)
        assert report.values["ligand_rmsd_cross_mean"] == pytest.approx(0.0, abs=1e-3)
        assert report.flags == []

    def test_rigid_motion_is_removed(self, tmp_path):
        """Rotating and translating the whole complex changes nothing."""
        rotation, shift = _rotation(0.7), np.array([12.0, -4.0, 3.0])
        ref = _structure(_backbone(), LIGAND_XYZ)
        moved = _structure(
            _backbone() @ rotation.T + shift, LIGAND_XYZ @ rotation.T + shift
        )
        _write_prediction(tmp_path / "ref", "sys", ref)
        _write_prediction(tmp_path / "test", "sys", moved)
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["ca_rmsd_cross_mean"] == pytest.approx(0.0, abs=2e-3)
        assert report.values["ligand_rmsd_cross_mean"] == pytest.approx(0.0, abs=2e-3)
        assert report.values["ligand_rmsd_named_cross_mean"] == pytest.approx(
            0.0, abs=2e-3
        )

    def test_ligand_shift_is_measured(self, tmp_path):
        """Moving the ligand 1.5 Å out of the ring plane gives 1.5 Å ligand RMSD."""
        _write_prediction(tmp_path / "ref", "sys", _structure(_backbone(), LIGAND_XYZ))
        shifted = LIGAND_XYZ + np.array([0.0, 0.0, 1.5])
        _write_prediction(tmp_path / "test", "sys", _structure(_backbone(), shifted))
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["ca_rmsd_cross_mean"] == pytest.approx(0.0, abs=1e-3)
        assert report.values["ligand_rmsd_cross_mean"] == pytest.approx(1.5, abs=1e-3)
        assert report.values["ligand_rmsd_named_cross_mean"] == pytest.approx(
            1.5, abs=1e-3
        )

    def test_symmetric_swap(self, tmp_path):
        """Swapping two equivalent ring carbons only moves the name-matched RMSD."""
        _write_prediction(tmp_path / "ref", "sys", _structure(_backbone(), LIGAND_XYZ))
        swapped = LIGAND_XYZ.copy()
        swapped[[1, 5]] = swapped[[5, 1]]
        _write_prediction(tmp_path / "test", "sys", _structure(_backbone(), swapped))
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["ligand_rmsd_cross_mean"] == pytest.approx(0.0, abs=1e-3)
        assert report.values["ligand_rmsd_named_cross_mean"] > 0.5

    def test_pdb_files(self, tmp_path):
        """Predictions written as PDB are read too."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(tmp_path / "ref", "sys", st, fmt="pdb")
        _write_prediction(tmp_path / "test", "sys", st, fmt="pdb")
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["ligand_rmsd_cross_mean"] == pytest.approx(0.0, abs=1e-3)

    def test_protein_only(self, tmp_path):
        """A system without a ligand gets one row with no ligand values."""
        st = _structure(_backbone(), None)
        _write_prediction(tmp_path / "ref", "sys", st)
        _write_prediction(tmp_path / "test", "sys", st)
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.ligand_chain == ""
        assert "ligand_rmsd_cross_mean" not in report.values


class TestGrouping:
    """Replicates are grouped per system, including seed-suffixed names."""

    def test_seed_suffix_and_within_platform_spread(self, tmp_path):
        """Seeds group into one system, and within-platform spread is reported."""
        base = _structure(_backbone(), LIGAND_XYZ)
        nudged = _structure(_backbone(), LIGAND_XYZ + np.array([0.0, 0.0, 0.4]))
        _write_prediction(tmp_path / "ref" / "run1", "sys", base)
        _write_prediction(tmp_path / "ref" / "run2", "sys", nudged)
        _write_prediction(tmp_path / "test", "sys_seed1", base)
        _write_prediction(tmp_path / "test", "sys_seed2", nudged)
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert (report.system, report.n_ref, report.n_test) == ("sys", 2, 2)
        assert report.values["ligand_rmsd_within_ref"] == pytest.approx(0.4, abs=1e-3)
        assert report.values["ligand_rmsd_within_test"] == pytest.approx(0.4, abs=1e-3)

    def test_missing_systems_are_reported(self, tmp_path):
        """Systems on only one side are listed, not compared."""
        st = _structure(_backbone(), LIGAND_XYZ)
        for side, names in (("ref", ("a", "b")), ("test", ("a", "c"))):
            for name in names:
                _write_prediction(tmp_path / side, name, st)
        reports, only_ref, only_test = compare_dirs(tmp_path / "ref", tmp_path / "test")
        assert [r.system for r in reports] == ["a"]
        assert (only_ref, only_test) == (["b"], ["c"])


class TestDeltasAndFlags:
    """Confidence and affinity differences are reported and flagged."""

    def test_affinity_difference_is_flagged(self, tmp_path):
        """A 1.0 log-unit affinity difference exceeds the default 0.5 tolerance."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(
            tmp_path / "ref",
            "sys",
            st,
            {"confidence_score": 0.90, "iptm": 0.80},
            {
                "affinity_pred_value": -1.0,
                "affinity_probability_binary": 0.70,
                "binder_chain": "B",
            },
        )
        _write_prediction(
            tmp_path / "test",
            "sys",
            st,
            {"confidence_score": 0.88, "iptm": 0.81},
            {
                "affinity_pred_value": -2.0,
                "affinity_probability_binary": 0.75,
                "binder_chain": "B",
            },
        )
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["affinity_pred_value_delta"] == pytest.approx(-1.0)
        assert report.values["affinity_probability_binary_delta"] == pytest.approx(0.05)
        assert report.values["confidence_score_delta"] == pytest.approx(-0.02)
        assert len(report.flags) == 1
        assert "affinity" in report.flags[0]

    def test_affinity_without_binder_chain(self, tmp_path):
        """Affinity files from older Boltz versions (no binder_chain) still match."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(
            tmp_path / "ref", "sys", st, affinity={"affinity_pred_value": 0.5}
        )
        _write_prediction(
            tmp_path / "test", "sys", st, affinity={"affinity_pred_value": 0.6}
        )
        report = _only_row(compare_dirs(tmp_path / "ref", tmp_path / "test")[0])
        assert report.values["affinity_pred_value_delta"] == pytest.approx(0.1)

    def test_custom_thresholds(self, tmp_path):
        """A ligand shift beyond a tighter ligand tolerance is flagged."""
        _write_prediction(tmp_path / "ref", "sys", _structure(_backbone(), LIGAND_XYZ))
        shifted = LIGAND_XYZ + np.array([0.0, 0.0, 1.5])
        _write_prediction(tmp_path / "test", "sys", _structure(_backbone(), shifted))
        reports, _, _ = compare_dirs(
            tmp_path / "ref", tmp_path / "test", thresholds=Thresholds(ligand_rmsd=1.0)
        )
        assert any("ligand RMSD" in flag for flag in _only_row(reports).flags)


class TestCli:
    """The boltz-compare command writes a CSV and prints a summary."""

    def test_writes_report(self, tmp_path):
        """The CSV has the documented columns and one row per system."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(
            tmp_path / "ref", "sys", st, affinity={"affinity_pred_value": -1.0}
        )
        _write_prediction(
            tmp_path / "test", "sys", st, affinity={"affinity_pred_value": -1.1}
        )
        out = tmp_path / "report.csv"
        result = CliRunner().invoke(
            main,
            [
                "--ref",
                str(tmp_path / "ref"),
                "--test",
                str(tmp_path / "test"),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Compared 1 systems" in result.output
        assert "Ligand RMSD" in result.output
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert tuple(rows[0].keys()) == COLUMNS
        assert rows[0]["system"] == "sys"
        assert float(rows[0]["affinity_pred_value_delta"]) == pytest.approx(-0.1)

    def test_no_overlap_is_an_error(self, tmp_path):
        """Pointing at folders with no systems in common fails clearly."""
        st = _structure(_backbone(), LIGAND_XYZ)
        _write_prediction(tmp_path / "ref", "a", st)
        _write_prediction(tmp_path / "test", "b", st)
        result = CliRunner().invoke(
            main, ["--ref", str(tmp_path / "ref"), "--test", str(tmp_path / "test")]
        )
        assert result.exit_code != 0
        assert "No systems appear in both" in result.output
