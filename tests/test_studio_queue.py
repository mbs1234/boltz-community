"""Tests for boltz-queue: staging inputs, the worker, and the housekeeping commands.

A fake `boltz` script stands in for Boltz. It writes Boltz-shaped outputs, and an
optional `fake:` block in the job YAML makes it fail, sleep, or skip affinity.
"""

from __future__ import annotations

import csv
import json
import os
import plistlib
import shlex
import sys
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from boltz.scripts.studio_queue import (
    Queue,
    claim_job,
    cli,
    host_name,
    read_sidecar,
    ready_jobs,
)

FAKE_BOLTZ = r"""
import json, sys, time
from pathlib import Path
import yaml

args = sys.argv[1:]
assert args[0] == "predict", args
source = Path(args[1])
out = Path(args[args.index("--out_dir") + 1])
data = yaml.safe_load(source.read_text())
fake = data.get("fake") or {}
started = time.time()
time.sleep(float(fake.get("sleep", 0)))
if fake.get("fail"):
    print("fake failure", file=sys.stderr)
    sys.exit(3)
name = source.stem
pred = out / f"boltz_results_{name}" / "predictions" / name
pred.mkdir(parents=True, exist_ok=True)
(pred / f"{name}_model_0.cif").write_text("data_fake\n")
(pred / f"confidence_{name}_model_0.json").write_text(
    json.dumps(
        {"confidence_score": 0.9, "iptm": 0.8, "ligand_iptm": 0.7, "complex_plddt": 0.8}
    )
)
for prop in data.get("properties") or []:
    if "affinity" in prop and not fake.get("skip_affinity"):
        binder = prop["affinity"]["binder"]
        (pred / f"affinity_{name}.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": -1.5,
                    "affinity_probability_binary": 0.6,
                    "binder_chain": binder,
                }
            )
        )
(out / "fake_args.json").write_text(
    json.dumps({"argv": args, "input": data, "started": started, "ended": time.time()})
)
"""

PROTEIN = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEA"


@pytest.fixture
def fake_boltz(tmp_path: Path) -> str:
    """Command line for the fake boltz script."""
    script = tmp_path / "fake_boltz.py"
    script.write_text(FAKE_BOLTZ)
    return shlex.join([sys.executable, str(script)])


@pytest.fixture
def queue(tmp_path: Path) -> Queue:
    """Return an initialized, empty queue."""
    root = tmp_path / "queue"
    result = CliRunner().invoke(cli, ["init", str(root)])
    assert result.exit_code == 0, result.output
    return Queue(root)


def _job_yaml(
    folder: Path,
    name: str,
    msa: str | None = "empty",
    affinity: bool = False,
    fake: dict | None = None,
) -> Path:
    data: dict = {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": PROTEIN, "msa": msa}}],
    }
    if affinity:
        data["sequences"].append(
            {"ligand": {"id": "B", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"}}
        )
        data["properties"] = [{"affinity": {"binder": "B"}}]
    if fake:
        data["fake"] = fake
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _add(queue: Queue, *args: str):
    return CliRunner().invoke(cli, ["add", str(queue.root), *args])


def _run_worker(queue: Queue, fake_boltz: str, *args: str, once: bool = True):
    command = [
        "worker",
        str(queue.root),
        "--no-caffeinate",
        "--poll",
        "0.05",
        "--settle",
        "0",
        "--boltz-cmd",
        fake_boltz,
    ]
    if once:
        command.append("--once")
    return CliRunner().invoke(cli, [*command, *args])


def _fake_args(queue: Queue, name: str) -> dict:
    return json.loads((queue.results / name / "fake_args.json").read_text())


class TestInitAndAdd:
    """Creating a queue and adding YAML jobs."""

    def test_init_creates_layout(self, queue):
        """Init makes every folder plus a README."""
        for folder in (
            queue.inbox,
            queue.running,
            queue.done,
            queue.failed,
            queue.results,
            queue.shared_files,
        ):
            assert folder.is_dir()
        assert (queue.root / "README.txt").is_file()

    def test_add_copies_referenced_msa(self, queue, tmp_path):
        """A referenced MSA is copied into shared/files and the path rewritten."""
        (tmp_path / "inputs" / "msa").mkdir(parents=True)
        (tmp_path / "inputs" / "msa" / "target.a3m").write_text(
            ">query\n" + PROTEIN + "\n"
        )
        job = _job_yaml(tmp_path / "inputs", "job", msa="./msa/target.a3m")
        result = _add(queue, str(job))
        assert result.exit_code == 0, result.output
        staged = yaml.safe_load((queue.inbox / "job.yaml").read_text())
        msa = staged["sequences"][0]["protein"]["msa"]
        assert msa.startswith("../shared/files/")
        assert msa.endswith("_target.a3m")
        assert (queue.inbox / msa).resolve().read_text().startswith(">query")
        assert read_sidecar(queue.inbox / "job.yaml")["attempt"] == 0

    def test_add_with_seeds(self, queue, tmp_path):
        """--seeds makes one job per seed with a --seed argument."""
        result = _add(queue, str(_job_yaml(tmp_path, "job")), "--seeds", "1,2")
        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in queue.jobs_in(queue.inbox)) == [
            "job_seed1.yaml",
            "job_seed2.yaml",
        ]
        assert read_sidecar(queue.inbox / "job_seed2.yaml")["boltz_args"] == [
            "--seed",
            "2",
        ]

    def test_missing_reference_is_an_error(self, queue, tmp_path):
        """A job whose MSA file doesn't exist is refused."""
        result = _add(queue, str(_job_yaml(tmp_path, "job", msa="missing.a3m")))
        assert result.exit_code != 0
        assert "referenced file not found" in result.output
        assert queue.jobs_in(queue.inbox) == []

    def test_duplicate_names_are_refused(self, queue, tmp_path):
        """The same job name can't be queued twice."""
        job = _job_yaml(tmp_path, "job")
        assert _add(queue, str(job)).exit_code == 0
        result = _add(queue, str(job))
        assert result.exit_code != 0
        assert "already in the queue" in result.output

    def test_non_boltz_yaml_is_refused(self, queue, tmp_path):
        """A YAML without a sequences list isn't a Boltz input."""
        path = tmp_path / "notes.yaml"
        path.write_text("title: not a job\n")
        result = _add(queue, str(path))
        assert result.exit_code != 0
        assert "isn't a Boltz input" in result.output


class TestMakeInputs:
    """Turning a target template and a ligand CSV into jobs."""

    @pytest.fixture
    def template(self, tmp_path) -> Path:
        """Return a target template with an MSA next to it."""
        (tmp_path / "target.a3m").write_text(">query\n" + PROTEIN + "\n")
        return _job_yaml(tmp_path, "target", msa="target.a3m")

    @staticmethod
    def _csv(
        path: Path,
        rows: list[tuple[str, str]],
        header: tuple[str, str] = ("name", "smiles"),
    ) -> Path:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_into_queue(self, queue, template, tmp_path):
        """One job per ligand, each with the ligand and an affinity request."""
        ligands = self._csv(
            tmp_path / "ligands.csv",
            [
                ("tyr", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
                ("phe", "N[C@@H](Cc1ccccc1)C(=O)O"),
            ],
        )
        result = CliRunner().invoke(
            cli,
            ["make-inputs", str(template), str(ligands), "--queue", str(queue.root)],
        )
        assert result.exit_code == 0, result.output
        assert sorted(p.stem for p in queue.jobs_in(queue.inbox)) == [
            "target_phe",
            "target_tyr",
        ]
        data = yaml.safe_load((queue.inbox / "target_tyr.yaml").read_text())
        assert data["sequences"][-1] == {
            "ligand": {"id": "L", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"}
        }
        assert data["properties"] == [{"affinity": {"binder": "L"}}]
        assert len(list(queue.shared_files.iterdir())) == 1  # the MSA is stored once

    def test_to_folder_uses_absolute_paths(self, template, tmp_path):
        """With --out, MSA paths are made absolute so the files work from anywhere."""
        ligands = self._csv(
            tmp_path / "ligands.csv", [("tyr", "N[C@@H](Cc1ccc(O)cc1)C(=O)O")]
        )
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli,
            [
                "make-inputs",
                str(template),
                str(ligands),
                "--out",
                str(out),
                "--no-affinity",
            ],
        )
        assert result.exit_code == 0, result.output
        data = yaml.safe_load((out / "target_tyr.yaml").read_text())
        assert Path(data["sequences"][0]["protein"]["msa"]).is_absolute()
        assert "properties" not in data

    def test_ligand_id_clash(self, queue, tmp_path):
        """The ligand's chain ID must not already be used by the template."""
        template = _job_yaml(tmp_path, "target")
        ligands = self._csv(tmp_path / "ligands.csv", [("tyr", "CCO")])
        result = CliRunner().invoke(
            cli,
            [
                "make-inputs",
                str(template),
                str(ligands),
                "--queue",
                str(queue.root),
                "--ligand-id",
                "A",
            ],
        )
        assert result.exit_code != 0
        assert "already uses chain ID" in result.output

    def test_missing_column(self, queue, template, tmp_path):
        """A CSV without the SMILES column is refused with the columns it has."""
        ligands = self._csv(
            tmp_path / "ligands.csv", [("tyr", "CCO")], header=("name", "structure")
        )
        result = CliRunner().invoke(
            cli,
            ["make-inputs", str(template), str(ligands), "--queue", str(queue.root)],
        )
        assert result.exit_code != 0
        assert "no column 'smiles'" in result.output

    def test_names_that_collide_after_cleaning(self, queue, template, tmp_path):
        """Names that become identical once cleaned are refused."""
        ligands = self._csv(
            tmp_path / "ligands.csv", [("cmpd 1", "CCO"), ("cmpd_1", "CCN")]
        )
        result = CliRunner().invoke(
            cli,
            ["make-inputs", str(template), str(ligands), "--queue", str(queue.root)],
        )
        assert result.exit_code != 0
        assert "repeat" in result.output

    def test_invalid_smiles(self, queue, template, tmp_path):
        """With RDKit installed, SMILES it can't parse are refused."""
        pytest.importorskip("rdkit")
        ligands = self._csv(
            tmp_path / "ligands.csv", [("good", "CCO"), ("bad", "C1CC(")]
        )
        result = CliRunner().invoke(
            cli,
            ["make-inputs", str(template), str(ligands), "--queue", str(queue.root)],
        )
        assert result.exit_code != 0
        assert "bad" in result.output


class TestWorker:
    """Running jobs with the fake boltz."""

    def test_runs_job_to_done(self, queue, tmp_path, fake_boltz):
        """A job runs, lands in done/, and gets a summary with its results."""
        (tmp_path / "target.a3m").write_text(">query\n" + PROTEIN + "\n")
        assert (
            _add(
                queue, str(_job_yaml(tmp_path, "job", msa="target.a3m", affinity=True))
            ).exit_code
            == 0
        )
        result = _run_worker(queue, fake_boltz)
        assert result.exit_code == 0, result.output
        assert (queue.done / "job.yaml").is_file()
        assert (queue.results / "job" / "DONE").is_file()
        summary = json.loads((queue.results / "job" / "summary.json").read_text())
        assert summary["confidence"]["confidence_score"] == 0.9
        assert summary["affinity"]["B"]["affinity_pred_value"] == -1.5
        assert summary["provenance"]["python"]
        args = _fake_args(queue, "job")
        assert args["argv"][args["argv"].index("--accelerator") + 1] == "mps"
        msa = Path(args["input"]["sequences"][0]["protein"]["msa"])
        assert msa.is_absolute()
        assert msa.is_file()

    def test_job_and_worker_arguments_reach_boltz(self, queue, tmp_path, fake_boltz):
        """Worker options after -- and per-job seeds are passed on, job options last."""
        assert (
            _add(queue, str(_job_yaml(tmp_path, "job")), "--seeds", "7").exit_code == 0
        )
        result = _run_worker(queue, fake_boltz, "--", "--diffusion_samples", "3")
        assert result.exit_code == 0, result.output
        argv = _fake_args(queue, "job_seed7")["argv"]
        assert argv[-4:] == ["--diffusion_samples", "3", "--seed", "7"]

    def test_failure_is_retried_then_failed(self, queue, tmp_path, fake_boltz):
        """A failing job is retried, then moved to failed/ with the log tail."""
        assert (
            _add(queue, str(_job_yaml(tmp_path, "job", fake={"fail": True}))).exit_code
            == 0
        )
        result = _run_worker(queue, fake_boltz, "--retries", "1")
        assert result.exit_code == 0, result.output
        assert (queue.failed / "job.yaml").is_file()
        error = (queue.failed / "job.error.txt").read_text()
        assert "exited with code 3" in error
        assert "fake failure" in error
        assert read_sidecar(queue.failed / "job.yaml")["attempt"] == 2

    def test_missing_affinity_is_a_failure(self, queue, tmp_path, fake_boltz):
        """A job that asked for affinity but got none doesn't count as done."""
        job = _job_yaml(tmp_path, "job", affinity=True, fake={"skip_affinity": True})
        assert _add(queue, str(job)).exit_code == 0
        assert _run_worker(queue, fake_boltz, "--retries", "0").exit_code == 0
        assert "affinity was requested" in (queue.failed / "job.error.txt").read_text()

    def test_leftover_jobs_are_requeued_on_start(self, queue, tmp_path, fake_boltz):
        """Jobs left in this Mac's running/ folder by a crash are run again."""
        assert _add(queue, str(_job_yaml(tmp_path, "job"))).exit_code == 0
        claim_job(queue, queue.inbox / "job.yaml", host_name())
        result = _run_worker(queue, fake_boltz)
        assert result.exit_code == 0, result.output
        assert "left over from an earlier run" in result.output
        assert (queue.done / "job.yaml").is_file()

    def test_finished_result_is_not_overwritten(self, queue, fake_boltz):
        """A job whose name already has a finished result is refused."""
        _job_yaml(queue.inbox, "job")
        (queue.results / "job").mkdir()
        (queue.results / "job" / "DONE").write_text("earlier\n")
        assert _run_worker(queue, fake_boltz).exit_code == 0
        assert (
            "already holds a finished result"
            in (queue.failed / "job.error.txt").read_text()
        )

    def test_drain_flag_stops_the_worker(self, queue, tmp_path, fake_boltz):
        """With control/drain present, the worker exits without taking jobs."""
        assert _add(queue, str(_job_yaml(tmp_path, "job"))).exit_code == 0
        (queue.control / "drain").touch()
        result = _run_worker(queue, fake_boltz, once=False)
        assert result.exit_code == 0, result.output
        assert "drain flag" in result.output
        assert (queue.inbox / "job.yaml").is_file()

    def test_timeout(self, queue, tmp_path, fake_boltz):
        """A job that runs past --timeout-hours is stopped and failed."""
        assert (
            _add(queue, str(_job_yaml(tmp_path, "job", fake={"sleep": 30}))).exit_code
            == 0
        )
        started = time.monotonic()
        result = _run_worker(
            queue, fake_boltz, "--timeout-hours", "0.0003", "--retries", "0"
        )
        assert result.exit_code == 0, result.output
        assert time.monotonic() - started < 20
        assert "time limit" in (queue.failed / "job.error.txt").read_text()

    def test_slots_run_jobs_at_the_same_time(self, queue, tmp_path, fake_boltz):
        """With --slots 2, two jobs run concurrently."""
        for name in ("one", "two"):
            assert (
                _add(
                    queue, str(_job_yaml(tmp_path, name, fake={"sleep": 1.5}))
                ).exit_code
                == 0
            )
        assert _run_worker(queue, fake_boltz, "--slots", "2").exit_code == 0
        one, two = _fake_args(queue, "one"), _fake_args(queue, "two")
        assert one["started"] < two["ended"]
        assert two["started"] < one["ended"]


class TestClaiming:
    """Low-level claiming rules that keep Macs from running the same job."""

    def test_only_one_claim_wins(self, queue):
        """Claiming the same file twice succeeds once."""
        _job_yaml(queue.inbox, "job")
        first = claim_job(queue, queue.inbox / "job.yaml", "mac1")
        second = claim_job(queue, queue.inbox / "job.yaml", "mac2")
        assert first == queue.running / "mac1" / "job.yaml"
        assert second is None

    def test_fresh_files_wait_to_settle(self, queue):
        """A file still being copied (recently modified) isn't picked up yet."""
        job = _job_yaml(queue.inbox, "job")
        assert ready_jobs(queue, settle_s=60) == []
        os.utime(job, (time.time() - 120, time.time() - 120))
        assert ready_jobs(queue, settle_s=60) == [job]


class TestHousekeeping:
    """status, summarize, requeue and launchd-plist."""

    def test_status_summarize_and_requeue(self, queue, tmp_path, fake_boltz):
        """After one success and one failure, the reports and requeue behave."""
        assert (
            _add(queue, str(_job_yaml(tmp_path, "good", affinity=True))).exit_code == 0
        )
        assert (
            _add(queue, str(_job_yaml(tmp_path, "bad", fake={"fail": True}))).exit_code
            == 0
        )
        assert _run_worker(queue, fake_boltz, "--retries", "0").exit_code == 0

        status = CliRunner().invoke(cli, ["status", str(queue.root)])
        assert status.exit_code == 0, status.output
        assert "done: 1" in status.output
        assert "failed: 1" in status.output
        assert "exited with code 3" in status.output

        summary = CliRunner().invoke(cli, ["summarize", str(queue.root)])
        assert summary.exit_code == 0, summary.output
        with (queue.root / "summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert [
            (r["name"], r["binder_chain"], r["affinity_pred_value"]) for r in rows
        ] == [("good", "B", "-1.5")]

        requeue = CliRunner().invoke(cli, ["requeue", str(queue.root), "--all-failed"])
        assert requeue.exit_code == 0, requeue.output
        assert (queue.inbox / "bad.yaml").is_file()
        assert read_sidecar(queue.inbox / "bad.yaml")["attempt"] == 0
        assert not (queue.failed / "bad.error.txt").exists()

    def test_requeue_from_a_stopped_mac(self, queue):
        """Jobs stuck on a Mac that's gone can be returned to the inbox."""
        _job_yaml(queue.inbox, "job")
        claim_job(queue, queue.inbox / "job.yaml", "old-mac")
        result = CliRunner().invoke(
            cli, ["requeue", str(queue.root), "--from-host", "old-mac"]
        )
        assert result.exit_code == 0, result.output
        assert (queue.inbox / "job.yaml").is_file()

    def test_launchd_plist(self, queue, tmp_path):
        """The LaunchAgent runs this queue's worker and restarts it after a crash."""
        out = tmp_path / "agent.plist"
        result = CliRunner().invoke(
            cli,
            [
                "launchd-plist",
                str(queue.root),
                "--worker-args",
                "--slots 2",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        plist = plistlib.loads(out.read_bytes())
        assert plist["ProgramArguments"][-4:] == [
            "worker",
            str(queue.root.absolute()),
            "--slots",
            "2",
        ]
        assert plist["KeepAlive"] == {"SuccessfulExit": False}
