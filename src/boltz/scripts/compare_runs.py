"""Compare Boltz predictions made on two platforms, for example A100 vs Mac Studio.

Exposed as the ``boltz-compare`` CLI entry point::

    boltz-compare --ref a100_results/ --test mac_results/ --out report.csv

Every ``predictions/<name>/`` folder under ``--ref`` or ``--test`` counts as one
replicate of system ``<name>``. A seed suffix such as ``_seed3`` is stripped
before grouping, so seeds of one system are compared together. For each system
the tool reports how far the two platforms' top-ranked models are apart (protein
CA RMSD and ligand RMSD after superposing on the binding pocket), and the
difference in confidence and affinity outputs.

Diffusion sampling is random, so predictions differ even on one platform. When
a side has at least two replicates, the same measures are reported *within*
that side. The question to answer is whether cross-platform differences stay
within that seed-to-seed variation. The flag thresholds are starting points;
what counts as acceptable is a scientific judgment for the user.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING

import click

from boltz.scripts.prediction_outputs import (
    Prediction,
    Structure,
    find_predictions,
    read_structure,
)

if TYPE_CHECKING:
    import numpy as np

REPORTED_CONFIDENCE = (
    "confidence_score",
    "iptm",
    "ligand_iptm",
    "complex_plddt",
    "complex_iplddt",
)
REPORTED_AFFINITY = ("affinity_pred_value", "affinity_probability_binary")
MIN_ALIGN_ATOMS = 3
MIN_CORRELATION_POINTS = 3
NAN = float("nan")


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R`` and ``t`` such that ``mobile @ R.T + t`` best fits ``target``."""
    import numpy as np

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = target_center - mobile_center @ rotation.T
    return rotation, translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square deviation between two matched coordinate arrays."""
    import numpy as np

    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _matched_rep_coords(
    test: Structure, ref: Structure, keys: list | None = None
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    keys = [
        k for k in (keys if keys is not None else ref.rep_atoms) if k in test.rep_atoms
    ]
    if not keys:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.array([test.rep_atoms[k] for k in keys]),
        np.array([ref.rep_atoms[k] for k in keys]),
    )


def protein_rmsd(test: Structure, ref: Structure) -> float:
    """CA (or C1') RMSD over matched residues after optimal superposition."""
    mobile, target = _matched_rep_coords(test, ref)
    if len(mobile) < MIN_ALIGN_ATOMS:
        return NAN
    rotation, translation = kabsch(mobile, target)
    return rmsd(mobile @ rotation.T + translation, target)


def pocket_residues(ref: Structure, chain: str, cutoff: float) -> list:
    """Residues with any heavy atom within ``cutoff`` Å of the ligand in ``ref``."""
    import numpy as np

    ligand = ref.ligands[chain]
    if len(ref.polymer_atoms) == 0:
        return []
    diff = ref.polymer_atoms[:, None, :] - ligand.coords[None, :, :]
    close = ((diff**2).sum(axis=-1) <= cutoff**2).any(axis=1)
    near = {ref.polymer_atom_residue[i] for i in np.flatnonzero(close)}
    return [key for key in ref.rep_atoms if key in near]


def _named_rmsd(
    names: list[str], moved: np.ndarray, ref_names: list[str], ref_coords: np.ndarray
) -> float:
    index = {name: i for i, name in enumerate(ref_names)}
    if len(index) != len(ref_names) or sorted(names) != sorted(ref_names):
        return NAN
    return rmsd(moved, ref_coords[[index[name] for name in names]])


def _symmetry_tolerant_rmsd(
    elements: list[str],
    moved: np.ndarray,
    ref_elements: list[str],
    ref_coords: np.ndarray,
) -> float:
    """RMSD letting atoms of the same element match in any order (Hungarian).

    This handles symmetric groups (a flipped phenyl ring, equivalent carboxylate
    oxygens) without needing bonds. It can understate the difference for poses
    that are genuinely different but similar in shape, so compare it with the
    name-matched RMSD, which never understates.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if sorted(elements) != sorted(ref_elements):
        return NAN
    elements_arr = np.array(elements)
    ref_elements_arr = np.array(ref_elements)
    total = 0.0
    for element in set(ref_elements):
        mine = moved[elements_arr == element]
        theirs = ref_coords[ref_elements_arr == element]
        cost = ((mine[:, None, :] - theirs[None, :, :]) ** 2).sum(axis=-1)
        rows, cols = linear_sum_assignment(cost)
        total += float(cost[rows, cols].sum())
    return math.sqrt(total / len(ref_elements))


def ligand_rmsd(
    test: Structure, ref: Structure, chain: str, cutoff: float
) -> tuple[float, float]:
    """Return (name-matched, symmetry-tolerant) ligand RMSD after a pocket fit."""
    if chain not in test.ligands or chain not in ref.ligands:
        return NAN, NAN
    mobile, target = _matched_rep_coords(test, ref, pocket_residues(ref, chain, cutoff))
    if len(mobile) < MIN_ALIGN_ATOMS:
        mobile, target = _matched_rep_coords(test, ref)
    if len(mobile) < MIN_ALIGN_ATOMS:
        return NAN, NAN
    rotation, translation = kabsch(mobile, target)
    lig, ref_lig = test.ligands[chain], ref.ligands[chain]
    moved = lig.coords @ rotation.T + translation
    return (
        _named_rmsd(lig.names, moved, ref_lig.names, ref_lig.coords),
        _symmetry_tolerant_rmsd(lig.elements, moved, ref_lig.elements, ref_lig.coords),
    )


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _mean(values: list[float]) -> float:
    values = _finite(values)
    return statistics.fmean(values) if values else NAN


def _max(values: list[float]) -> float:
    values = _finite(values)
    return max(values) if values else NAN


def _sd(values: list[float]) -> float:
    values = _finite(values)
    return statistics.stdev(values) if len(values) > 1 else NAN


@dataclass
class SystemReport:
    """Comparison results for one system and one ligand chain (or none)."""

    system: str
    ligand_chain: str
    n_ref: int
    n_test: int
    values: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class Thresholds:
    """Flag a system when a difference exceeds these starting-point tolerances."""

    ligand_rmsd: float = 2.0
    ca_rmsd: float = 2.0
    affinity: float = 0.5
    probability: float = 0.2


def group_by_system(
    predictions: list[Prediction], seed_suffix: str
) -> dict[str, list[Prediction]]:
    """Group predictions by name, ignoring a seed suffix."""
    pattern = re.compile(seed_suffix) if seed_suffix else None
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        key = pattern.sub("", prediction.name) if pattern else prediction.name
        groups[key].append(prediction)
    return dict(groups)


class _StructureCache:
    def __init__(self) -> None:
        self._cache: dict[Path, Structure] = {}

    def get(self, prediction: Prediction) -> Structure:
        path = prediction.top_model
        if path not in self._cache:
            self._cache[path] = read_structure(path)
        return self._cache[path]


def _scalar(source: dict | None, key: str) -> float:
    value = (source or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else NAN


def compare_system(
    system: str,
    refs: list[Prediction],
    tests: list[Prediction],
    cache: _StructureCache,
    pocket_cutoff: float,
    thresholds: Thresholds,
) -> list[SystemReport]:
    """Compare one system's reference and test replicates."""
    ref_structs = [cache.get(p) for p in refs]
    test_structs = [cache.get(p) for p in tests]
    cross_pairs = list(product(test_structs, ref_structs))
    within = {
        "ref": list(combinations(ref_structs, 2)),
        "test": list(combinations(test_structs, 2)),
    }

    cross_ca = [protein_rmsd(t, r) for t, r in cross_pairs]
    protein = {
        "ca_rmsd_cross_mean": _mean(cross_ca),
        "ca_rmsd_cross_max": _max(cross_ca),
        "ca_rmsd_within_ref": _mean([protein_rmsd(b, a) for a, b in within["ref"]]),
        "ca_rmsd_within_test": _mean([protein_rmsd(b, a) for a, b in within["test"]]),
    }
    confidence = {}
    for key in REPORTED_CONFIDENCE:
        ref_values = [_scalar(p.confidence, key) for p in refs]
        test_values = [_scalar(p.confidence, key) for p in tests]
        confidence[f"{key}_ref"] = _mean(ref_values)
        confidence[f"{key}_test"] = _mean(test_values)
        confidence[f"{key}_delta"] = (
            confidence[f"{key}_test"] - confidence[f"{key}_ref"]
        )
        confidence[f"{key}_ref_sd"] = _sd(ref_values)

    chains = sorted(set().union(*(s.ligands for s in ref_structs)))
    reports = []
    for chain in chains or [""]:
        report = SystemReport(
            system, chain, len(refs), len(tests), {**protein, **confidence}
        )
        if chain:
            cross = [ligand_rmsd(t, r, chain, pocket_cutoff) for t, r in cross_pairs]
            within_ref = [
                ligand_rmsd(b, a, chain, pocket_cutoff)[1] for a, b in within["ref"]
            ]
            within_test = [
                ligand_rmsd(b, a, chain, pocket_cutoff)[1] for a, b in within["test"]
            ]
            report.values.update(
                {
                    "ligand_rmsd_cross_mean": _mean([c[1] for c in cross]),
                    "ligand_rmsd_cross_max": _max([c[1] for c in cross]),
                    "ligand_rmsd_named_cross_mean": _mean([c[0] for c in cross]),
                    "ligand_rmsd_within_ref": _mean(within_ref),
                    "ligand_rmsd_within_test": _mean(within_test),
                }
            )
            for key in REPORTED_AFFINITY:
                ref_values = [_scalar(p.affinity_for(chain), key) for p in refs]
                test_values = [_scalar(p.affinity_for(chain), key) for p in tests]
                report.values[f"{key}_ref"] = _mean(ref_values)
                report.values[f"{key}_test"] = _mean(test_values)
                report.values[f"{key}_delta"] = (
                    report.values[f"{key}_test"] - report.values[f"{key}_ref"]
                )
                report.values[f"{key}_ref_sd"] = _sd(ref_values)
        report.flags = _flags(report.values, thresholds)
        reports.append(report)
    return reports


def _flags(values: dict[str, float], thresholds: Thresholds) -> list[str]:
    checks = (
        ("ca_rmsd_cross_mean", thresholds.ca_rmsd, "protein CA RMSD {v:.2f} Å > {t}"),
        (
            "ligand_rmsd_cross_mean",
            thresholds.ligand_rmsd,
            "ligand RMSD {v:.2f} Å > {t}",
        ),
        (
            "affinity_pred_value_delta",
            thresholds.affinity,
            "affinity Δ {v:+.2f} (log10 µM) beyond ±{t}",
        ),
        (
            "affinity_probability_binary_delta",
            thresholds.probability,
            "binder probability Δ {v:+.2f} beyond ±{t}",
        ),
    )
    flags = []
    for key, limit, text in checks:
        value = values.get(key, NAN)
        if math.isfinite(value) and abs(value) > limit:
            flags.append(text.format(v=value, t=limit))
    return flags


COLUMNS = (
    "system",
    "ligand_chain",
    "n_ref",
    "n_test",
    "ca_rmsd_cross_mean",
    "ca_rmsd_cross_max",
    "ca_rmsd_within_ref",
    "ca_rmsd_within_test",
    "ligand_rmsd_cross_mean",
    "ligand_rmsd_cross_max",
    "ligand_rmsd_named_cross_mean",
    "ligand_rmsd_within_ref",
    "ligand_rmsd_within_test",
    *(
        f"{k}_{s}"
        for k in REPORTED_AFFINITY
        for s in ("ref", "test", "delta", "ref_sd")
    ),
    *(
        f"{k}_{s}"
        for k in REPORTED_CONFIDENCE
        for s in ("ref", "test", "delta", "ref_sd")
    ),
    "flags",
)


def write_csv(reports: list[SystemReport], path: Path) -> None:
    """Write one row per system and ligand chain."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for report in reports:
            row = {
                "system": report.system,
                "ligand_chain": report.ligand_chain,
                "n_ref": report.n_ref,
                "n_test": report.n_test,
                "flags": "; ".join(report.flags),
            }
            for key in COLUMNS:
                if key in report.values:
                    value = report.values[key]
                    row[key] = "" if not math.isfinite(value) else round(value, 4)
            writer.writerow(row)


def _median(values: list[float]) -> float:
    values = _finite(values)
    return statistics.median(values) if values else NAN


def _fmt(value: float, unit: str = "", digits: int = 2) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}{unit}"


def summarize(reports: list[SystemReport], ligand_tol: float) -> list[str]:
    """Return human-readable summary lines."""
    lines = []

    def get(key: str) -> list[float]:
        return [r.values.get(key, NAN) for r in reports]

    def median_of(key: str) -> str:
        return _fmt(_median(get(key)), " Å")

    lines.append(
        f"Protein CA RMSD, test vs ref: median {median_of('ca_rmsd_cross_mean')} "
        f"(seed-to-seed within ref {median_of('ca_rmsd_within_ref')}, "
        f"within test {median_of('ca_rmsd_within_test')})"
    )
    ligand = _finite(get("ligand_rmsd_cross_mean"))
    if ligand:
        under = sum(v <= ligand_tol for v in ligand)
        lines.append(
            f"Ligand RMSD, test vs ref: median {median_of('ligand_rmsd_cross_mean')}; "
            f"{under}/{len(ligand)} under {ligand_tol} Å "
            f"(seed-to-seed within ref {median_of('ligand_rmsd_within_ref')}, "
            f"within test {median_of('ligand_rmsd_within_test')})"
        )
    pairs = [
        (r.values["affinity_pred_value_ref"], r.values["affinity_pred_value_test"])
        for r in reports
        if math.isfinite(r.values.get("affinity_pred_value_ref", NAN))
        and math.isfinite(r.values.get("affinity_pred_value_test", NAN))
    ]
    if pairs:
        mean_abs = statistics.fmean(abs(t - r) for r, t in pairs)
        line = (
            f"Affinity (log10 IC50, µM): mean |Δ| {mean_abs:.3f} "
            f"over {len(pairs)} systems"
        )
        refs, tests = zip(*pairs)
        if (
            len(pairs) >= MIN_CORRELATION_POINTS
            and statistics.pstdev(refs) > 0
            and statistics.pstdev(tests) > 0
        ):
            line += (
                f"; Pearson r across systems {statistics.correlation(refs, tests):.3f}"
            )
        lines.append(line)
        prob = _finite([abs(v) for v in get("affinity_probability_binary_delta")])
        if prob:
            lines.append(f"Binder probability: mean |Δ| {statistics.fmean(prob):.3f}")
    for key in REPORTED_CONFIDENCE:
        deltas = _finite(get(f"{key}_delta"))
        if deltas:
            lines.append(f"{key}: mean Δ {statistics.fmean(deltas):+.3f}")
    flagged = [r for r in reports if r.flags]
    lines.append(f"Flagged: {len(flagged)} of {len(reports)} rows")
    for r in flagged:
        where = f" (ligand chain {r.ligand_chain})" if r.ligand_chain else ""
        lines.append(f"  {r.system}{where}: {'; '.join(r.flags)}")
    return lines


def compare_dirs(
    ref_dir: Path,
    test_dir: Path,
    pocket_cutoff: float = 10.0,
    seed_suffix: str = r"_seed\d+$",
    thresholds: Thresholds | None = None,
) -> tuple[list[SystemReport], list[str], list[str]]:
    """Compare two result folders.

    Returns the reports plus the system names found only in ref or only in test.
    """
    thresholds = thresholds or Thresholds()
    refs = group_by_system(find_predictions(ref_dir), seed_suffix)
    tests = group_by_system(find_predictions(test_dir), seed_suffix)
    cache = _StructureCache()
    reports = []
    for system in sorted(set(refs) & set(tests)):
        reports.extend(
            compare_system(
                system, refs[system], tests[system], cache, pocket_cutoff, thresholds
            )
        )
    return reports, sorted(set(refs) - set(tests)), sorted(set(tests) - set(refs))


@click.command()
@click.option(
    "--ref",
    "ref_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Reference predictions, for example from the A100 servers.",
)
@click.option(
    "--test",
    "test_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Predictions to check, for example from a Mac Studio.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("compare_report.csv"),
    show_default=True,
    help="CSV report, one row per system and ligand chain.",
)
@click.option(
    "--pocket-cutoff",
    type=float,
    default=10.0,
    show_default=True,
    help=(
        "Residues within this many Å of the ligand define the pocket that is "
        "superposed before measuring ligand RMSD."
    ),
)
@click.option(
    "--seed-suffix",
    default=r"_seed\d+$",
    show_default=True,
    help=(
        "Regular expression removed from prediction names before grouping "
        "replicates. Use '' to disable."
    ),
)
@click.option(
    "--ligand-rmsd-tol",
    type=float,
    default=2.0,
    show_default=True,
    help="Flag ligand RMSD above this (Å).",
)
@click.option(
    "--ca-rmsd-tol",
    type=float,
    default=2.0,
    show_default=True,
    help="Flag protein CA RMSD above this (Å).",
)
@click.option(
    "--affinity-tol",
    type=float,
    default=0.5,
    show_default=True,
    help="Flag affinity differences beyond this (log10 units; 0.5 is ~3-fold IC50).",
)
@click.option(
    "--probability-tol",
    type=float,
    default=0.2,
    show_default=True,
    help="Flag binder-probability differences beyond this.",
)
def main(
    ref_dir: Path,
    test_dir: Path,
    out: Path,
    pocket_cutoff: float,
    seed_suffix: str,
    ligand_rmsd_tol: float,
    ca_rmsd_tol: float,
    affinity_tol: float,
    probability_tol: float,
) -> None:
    """Compare Boltz predictions from two platforms (for example A100 vs Mac)."""
    thresholds = Thresholds(ligand_rmsd_tol, ca_rmsd_tol, affinity_tol, probability_tol)
    reports, only_ref, only_test = compare_dirs(
        ref_dir, test_dir, pocket_cutoff, seed_suffix, thresholds
    )
    if not reports:
        msg = f"No systems appear in both {ref_dir} and {test_dir}."
        raise click.ClickException(msg)
    systems = {r.system for r in reports}
    click.echo(f"Compared {len(systems)} systems ({len(reports)} rows).")
    if only_ref:
        click.echo(f"Only in ref (not compared): {', '.join(only_ref)}")
    if only_test:
        click.echo(f"Only in test (not compared): {', '.join(only_test)}")
    for line in summarize(reports, ligand_rmsd_tol):
        click.echo(line)
    write_csv(reports, out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
