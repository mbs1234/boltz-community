"""Find and read Boltz prediction outputs.

Shared by the ``boltz-compare`` and ``boltz-queue`` tools. For each input ``<name>``,
``boltz predict`` writes::

    <out_dir>/boltz_results_<input stem>/predictions/<name>/
        <name>_model_<k>.cif (or .pdb)     ranked structures; model_0 is the best
        confidence_<name>_model_<k>.json   confidence scores for each model
        affinity_<id>.json                 affinity prediction, when requested

Discovery and the JSON files need only the standard library. Reading structures
needs ``gemmi`` and ``numpy`` (both Boltz dependencies), imported when used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

STRUCTURE_SUFFIXES = (".cif", ".pdb")

CONFIDENCE_KEYS = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

AFFINITY_KEYS = (
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_probability_binary1",
    "affinity_pred_value2",
    "affinity_probability_binary2",
)


@dataclass
class Prediction:
    """The outputs Boltz wrote for one input in one run."""

    name: str
    folder: Path
    models: list[Path]
    confidence: dict[str, Any] = field(default_factory=dict)
    affinity: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def top_model(self) -> Path:
        """Return the best-ranked structure (model_0)."""
        return self.models[0]

    def affinity_for(self, chain: str | None) -> dict[str, Any] | None:
        """Return the affinity result for a binder chain.

        Older Boltz versions don't record the binder chain in the affinity file.
        When there's exactly one affinity result, it's returned for any chain.
        """
        if chain is not None and chain in self.affinity:
            return self.affinity[chain]
        if len(self.affinity) == 1:
            return next(iter(self.affinity.values()))
        return None


def _model_rank(path: Path, name: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(name)}_model_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_prediction(folder: Path) -> Prediction | None:
    """Read one ``predictions/<name>/`` folder, or return None if it has no models."""
    name = folder.name
    ranked = []
    for path in folder.iterdir():
        if path.suffix in STRUCTURE_SUFFIXES:
            rank = _model_rank(path, name)
            if rank is not None:
                ranked.append((rank, path))
    if not ranked:
        return None
    models = [path for _, path in sorted(ranked)]

    confidence: dict[str, Any] = {}
    confidence_path = folder / f"confidence_{name}_model_0.json"
    if confidence_path.is_file():
        confidence = json.loads(confidence_path.read_text())

    affinity: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("affinity_*.json")):
        data = json.loads(path.read_text())
        affinity[str(data.get("binder_chain", ""))] = data

    return Prediction(name, folder, models, confidence, affinity)


def find_predictions(root: Path) -> list[Prediction]:
    """Return every prediction under ``root``, searched recursively."""
    predictions = []
    for folder in sorted(root.glob("**/predictions/*")):
        if folder.is_dir():
            prediction = read_prediction(folder)
            if prediction is not None:
                predictions.append(prediction)
    return predictions


@dataclass
class Ligand:
    """Heavy atoms of one ligand chain."""

    names: list[str]
    elements: list[str]
    coords: np.ndarray


@dataclass
class Structure:
    """The parts of a model that comparisons need.

    ``rep_atoms`` maps each polymer residue to one representative atom (CA for
    amino acids, C1' for nucleotides), keyed by (chain, number, insertion code,
    residue name). ``polymer_atoms`` holds every polymer heavy atom, with the
    residue key of each in ``polymer_atom_residue``.
    """

    rep_atoms: dict[tuple[str, int, str, str], np.ndarray]
    polymer_atoms: np.ndarray
    polymer_atom_residue: list[tuple[str, int, str, str]]
    ligands: dict[str, Ligand]


def _is_polymer_residue(residue_name: str) -> bool:
    import gemmi

    info = gemmi.find_tabulated_residue(residue_name)
    return info is not None and (info.is_amino_acid() or info.is_nucleic_acid())


def _is_water(residue_name: str) -> bool:
    import gemmi

    info = gemmi.find_tabulated_residue(residue_name)
    return info is not None and info.is_water()


def read_structure(path: Path) -> Structure:
    """Read the first model of a .cif or .pdb file written by Boltz."""
    import gemmi
    import numpy as np

    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    model = st[0]

    rep_atoms = {}
    polymer_atoms = []
    polymer_atom_residue = []
    ligands = {}
    for chain in model:
        residues = [r for r in chain if not _is_water(r.name)]
        if any(_is_polymer_residue(r.name) for r in residues):
            for residue in residues:
                key = (
                    chain.name,
                    residue.seqid.num,
                    residue.seqid.icode.strip(),
                    residue.name,
                )
                for atom in residue:
                    xyz = [atom.pos.x, atom.pos.y, atom.pos.z]
                    polymer_atoms.append(xyz)
                    polymer_atom_residue.append(key)
                    if atom.name in ("CA", "C1'") and key not in rep_atoms:
                        rep_atoms[key] = np.array(xyz)
        elif residues:
            names, elements, coords = [], [], []
            for residue in residues:
                for atom in residue:
                    names.append(f"{residue.name}:{residue.seqid.num}:{atom.name}")
                    elements.append(atom.element.name)
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            ligands[chain.name] = Ligand(names, elements, np.array(coords))

    return Structure(
        rep_atoms,
        np.array(polymer_atoms).reshape(-1, 3),
        polymer_atom_residue,
        ligands,
    )
