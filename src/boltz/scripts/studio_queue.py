"""A shared-folder job queue for running Boltz on several Macs.

Exposed as the ``boltz-queue`` CLI entry point. A queue is a folder, local or on a
network share that every Mac mounts::

    <queue>/
      inbox/            new jobs: Boltz YAML files (use `boltz-queue add`)
      running/<host>/   jobs a Mac has claimed
      done/  failed/    finished inputs; failures also get <name>.error.txt
      results/<name>/   Boltz output for the job, plus summary.json and a DONE marker
      shared/files/     MSA and template files that jobs reference
      logs/<host>.log   one log per worker
      control/          create `drain` (all Macs) or `drain-<host>` to stop new work

Each Mac runs ``boltz-queue worker <queue>``. A worker claims a job by renaming
it from inbox/ into its own running/ folder. Renaming is atomic, so two Macs
never run the same job.

Paths: Boltz resolves relative ``msa`` and template paths against the YAML's
folder, so in an inbox YAML they are relative to inbox/. ``add`` and
``make-inputs`` copy referenced files into shared/files/ and write paths like
``../shared/files/<hash>_<name>``. Before running a job, the worker makes those
paths absolute against its own mount of the queue, so mount points may differ
between Macs.

Per-job Boltz options (for example ``--seed``) live in a sidecar file
``<name>.job.json`` next to the YAML.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable

import click
import yaml

from boltz.scripts.prediction_outputs import (
    AFFINITY_KEYS,
    CONFIDENCE_KEYS,
    find_predictions,
)

JOB_SUFFIXES = (".yaml", ".yml")
SIDECAR_SUFFIX = ".job.json"
DONE_MARKER = "DONE"
LOG_TAIL_LINES = 60
TERMINATE_GRACE_S = 30
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

QUEUE_README = """\
This folder is a boltz-queue job queue.

  inbox/            new jobs (Boltz YAML files); add them with `boltz-queue add`
  running/<host>/   jobs a Mac is working on
  done/  failed/    finished inputs; failures have a <name>.error.txt
  results/<name>/   Boltz output, summary.json and the Boltz log for each job
  shared/files/     MSA and template files the jobs use
  logs/             one log per Mac
  control/          create a file named `drain` (or `drain-<host>`) to stop new work

Commands: boltz-queue status | summarize | requeue | add | make-inputs | worker
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_name(name: str) -> str:
    """Turn arbitrary text into a safe job name (letters, digits, ., _, -)."""
    cleaned = _UNSAFE_NAME_CHARS.sub("_", name.strip()).strip("._")
    if not cleaned:
        msg = f"Can't make a job name from {name!r}."
        raise ValueError(msg)
    return cleaned


def host_name() -> str:
    """Return this Mac's short host name, safe for use as a folder name."""
    return sanitize_name(socket.gethostname().split(".")[0] or "host")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def sidecar_for(job_yaml: Path) -> Path:
    """Return the sidecar path that belongs to a job YAML."""
    return job_yaml.with_name(job_yaml.stem + SIDECAR_SUFFIX)


def read_sidecar(job_yaml: Path) -> dict[str, Any]:
    """Read a job's sidecar, or return an empty one."""
    path = sidecar_for(job_yaml)
    return json.loads(path.read_text()) if path.is_file() else {}


def _move_job(job_yaml: Path, dest_dir: Path) -> Path:
    """Move a job and its sidecar into ``dest_dir`` (sidecar first)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_for(job_yaml)
    if sidecar.exists():
        sidecar.replace(dest_dir / sidecar.name)
    dest = dest_dir / job_yaml.name
    job_yaml.replace(dest)
    return dest


@dataclass(frozen=True)
class Queue:
    """Paths inside a queue folder."""

    root: Path

    @property
    def inbox(self) -> Path:  # noqa: D102
        return self.root / "inbox"

    @property
    def running(self) -> Path:  # noqa: D102
        return self.root / "running"

    @property
    def done(self) -> Path:  # noqa: D102
        return self.root / "done"

    @property
    def failed(self) -> Path:  # noqa: D102
        return self.root / "failed"

    @property
    def results(self) -> Path:  # noqa: D102
        return self.root / "results"

    @property
    def shared_files(self) -> Path:  # noqa: D102
        return self.root / "shared" / "files"

    @property
    def logs(self) -> Path:  # noqa: D102
        return self.root / "logs"

    @property
    def control(self) -> Path:  # noqa: D102
        return self.root / "control"

    def running_for(self, host: str) -> Path:
        """Return the running/ folder of one Mac."""
        return self.running / host

    def create(self) -> None:
        """Create the queue folders (safe to repeat)."""
        for folder in (
            self.inbox,
            self.running,
            self.done,
            self.failed,
            self.results,
            self.shared_files,
            self.logs,
            self.control,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.txt"
        if not readme.exists():
            readme.write_text(QUEUE_README)

    def is_initialized(self) -> bool:
        """Return True if the queue folders exist."""
        return self.inbox.is_dir() and self.results.is_dir() and self.running.is_dir()

    def jobs_in(self, folder: Path) -> list[Path]:
        """Return the job YAMLs in a folder (ignoring hidden temp files)."""
        if not folder.is_dir():
            return []
        return sorted(
            p
            for p in folder.iterdir()
            if p.suffix in JOB_SUFFIXES and not p.name.startswith(".") and p.is_file()
        )

    def names_in_use(self) -> set[str]:
        """Job names that are waiting, running, done, or have a finished result."""
        names = {p.stem for p in self.jobs_in(self.inbox)} | {
            p.stem for p in self.jobs_in(self.done)
        }
        if self.running.is_dir():
            for host_dir in self.running.iterdir():
                names |= {p.stem for p in self.jobs_in(host_dir)}
        if self.results.is_dir():
            names |= {p.parent.name for p in self.results.glob(f"*/{DONE_MARKER}")}
        return names


def open_queue(queue_dir: Path) -> Queue:
    """Return an initialized queue, or raise a ClickException."""
    queue = Queue(queue_dir.expanduser())
    if not queue.is_initialized():
        msg = (
            f"{queue.root} isn't a boltz-queue folder. "
            f"Create it with: boltz-queue init {queue.root}"
        )
        raise click.ClickException(msg)
    return queue


# ---------------------------------------------------------------------------
# Boltz input files
# ---------------------------------------------------------------------------


def load_job_yaml(path: Path) -> dict[str, Any]:
    """Load a Boltz YAML input and check it has a sequences list."""
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        msg = f"{path.name} isn't valid YAML: {e}"
        raise ValueError(msg) from e
    if not isinstance(data, dict) or not isinstance(data.get("sequences"), list):
        msg = f"{path.name} isn't a Boltz input (it has no 'sequences' list)."
        raise ValueError(msg)  # noqa: TRY004 - bad file content, not a caller error
    return data


def path_references(data: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Return (mapping, key) for every file path in a Boltz input.

    These are ``msa`` entries of sequences (except ``empty``) and the ``cif`` /
    ``pdb`` entries of templates, the fields Boltz resolves against the YAML's
    folder.
    """
    refs = []
    for item in data.get("sequences") or []:
        if isinstance(item, dict):
            for entity in item.values():
                msa = entity.get("msa") if isinstance(entity, dict) else None
                if isinstance(msa, str) and msa.strip().lower() != "empty":
                    refs.append((entity, "msa"))
    for template in data.get("templates") or []:
        if isinstance(template, dict):
            refs.extend(
                (template, key)
                for key in ("cif", "pdb")
                if isinstance(template.get(key), str)
            )
    return refs


def wants_affinity(data: dict[str, Any]) -> bool:
    """Return True if a Boltz input asks for an affinity prediction."""
    return any(
        isinstance(p, dict) and "affinity" in p for p in data.get("properties") or []
    )


def absolutize_references(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Return a copy of ``data`` with relative paths resolved against ``base_dir``."""
    data = copy.deepcopy(data)
    for mapping, key in path_references(data):
        path = Path(mapping[key]).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        mapping[key] = str(path)
    return data


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def stage_references(
    data: dict[str, Any], source_dir: Path, queue: Queue
) -> dict[str, Any]:
    """Copy files a job references into the queue; rewrite paths relative to inbox/.

    Files already inside the queue are referenced where they are. Other files
    are copied to shared/files/<hash>_<name>, so every Mac can read them and the
    same file is stored once.
    """
    data = copy.deepcopy(data)
    root = queue.root.resolve()
    inbox = queue.inbox.resolve()
    for mapping, key in path_references(data):
        source = Path(mapping[key]).expanduser()
        if not source.is_absolute():
            source = source_dir / source
        source = source.resolve()
        if not source.is_file():
            msg = f"referenced file not found: {source}"
            raise ValueError(msg)
        if source.is_relative_to(root):
            dest = source
        else:
            queue.shared_files.mkdir(parents=True, exist_ok=True)
            dest = queue.shared_files.resolve() / f"{_file_hash(source)}_{source.name}"
            if not dest.exists():
                tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
                shutil.copyfile(source, tmp)
                tmp.replace(dest)
        mapping[key] = os.path.relpath(dest, inbox)
    return data


def write_job(
    queue: Queue, name: str, data: dict[str, Any], sidecar: dict[str, Any]
) -> Path:
    """Put a job in the inbox: sidecar first, then the YAML, each atomically."""
    queue.inbox.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        queue.inbox / f"{name}{SIDECAR_SUFFIX}", json.dumps(sidecar, indent=2)
    )
    path = queue.inbox / f"{name}.yaml"
    _atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    return path


def parse_seeds(seeds: str | None) -> list[int]:
    """Parse a comma-separated list of seeds."""
    if not seeds:
        return []
    try:
        return [int(s) for s in seeds.split(",") if s.strip()]
    except ValueError as e:
        msg = f"--seeds must be comma-separated integers, got {seeds!r}"
        raise click.BadParameter(msg) from e


def _plan_jobs(
    base_name: str,
    data: dict[str, Any],
    seeds: list[int],
    boltz_args: list[str],
    source: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    jobs = []
    for seed in seeds or [None]:
        name = base_name if seed is None else f"{base_name}_seed{seed}"
        args = [*boltz_args, *([] if seed is None else ["--seed", str(seed)])]
        sidecar = {"boltz_args": args, "attempt": 0, "added": _now(), "source": source}
        jobs.append((name, data, sidecar))
    return jobs


def _write_planned(
    queue: Queue, planned: list[tuple[str, dict[str, Any], dict[str, Any]]]
) -> None:
    in_use = queue.names_in_use()
    names = [name for name, _, _ in planned]
    clashes = sorted({n for n in names if n in in_use or names.count(n) > 1})
    if clashes:
        msg = (
            f"Job names already in the queue (or repeated): {', '.join(clashes)}. "
            "Rename the inputs, or use `boltz-queue requeue` for failed jobs."
        )
        raise click.ClickException(msg)
    for name, data, sidecar in planned:
        write_job(queue, name, data, sidecar)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def claim_job(queue: Queue, job_yaml: Path, host: str) -> Path | None:
    """Move a job into running/<host>/; return None if another Mac claimed it first."""
    dest_dir = queue.running_for(host)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job_yaml.name
    try:
        job_yaml.rename(dest)
    except FileNotFoundError:
        return None
    sidecar = sidecar_for(job_yaml)
    with contextlib.suppress(FileNotFoundError):
        sidecar.rename(dest_dir / sidecar.name)
    return dest


def ready_jobs(queue: Queue, settle_s: float) -> list[Path]:
    """Inbox jobs not modified for ``settle_s`` seconds, oldest first."""
    now = time.time()
    ready = []
    try:
        candidates = queue.jobs_in(queue.inbox)
    except OSError:
        return []
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if now - mtime >= settle_s:
            ready.append((mtime, path.name, path))
    return [path for _, _, path in sorted(ready)]


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def collect_provenance() -> dict[str, Any]:
    """Versions and hardware to record with every result."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if platform.mac_ver()[0]:
        info["macos"] = platform.mac_ver()[0]
    for dist in ("boltz-community", "torch"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            info[dist] = importlib.metadata.version(dist)
    with contextlib.suppress(
        importlib.metadata.PackageNotFoundError, json.JSONDecodeError, TypeError
    ):
        direct = json.loads(
            importlib.metadata.distribution("boltz-community").read_text(
                "direct_url.json"
            )
            or "{}"
        )
        commit = direct.get("vcs_info", {}).get("commit_id")
        if commit:
            info["boltz_commit"] = commit
            info["boltz_source"] = direct.get("url")
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            info["chip"] = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
    return info


def default_boltz_cmd() -> list[str]:
    """Return the boltz command next to this Python, or the one on PATH."""
    sibling = Path(sys.executable).parent / "boltz"
    if sibling.exists():
        return [str(sibling)]
    found = shutil.which("boltz")
    if found:
        return [found]
    msg = (
        "Can't find the boltz command. "
        "Install Boltz in this environment or pass --boltz-cmd."
    )
    raise click.ClickException(msg)


@dataclass
class RunningJob:
    """A job this worker has started."""

    name: str
    yaml_path: Path
    result_dir: Path
    command: list[str]
    proc: subprocess.Popen
    log_file: IO[str]
    started: float
    started_iso: str
    wants_affinity: bool
    attempt: int


@dataclass
class Worker:
    """Runs jobs from a queue on this Mac."""

    queue: Queue
    boltz_cmd: list[str]
    boltz_args: list[str] = field(default_factory=list)
    host: str = field(default_factory=host_name)
    slots: int = 1
    retries: int = 1
    timeout_s: float = 0.0
    poll_s: float = 10.0
    settle_s: float = 15.0
    once: bool = False
    caffeinate: bool = True
    echo: Callable[[str], None] = click.echo
    active: dict[str, RunningJob] = field(default_factory=dict)
    stop_requested: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def log(self, message: str) -> None:
        """Print a line and append it to this Mac's log in the queue."""
        line = f"{_now()} [{self.host}] {message}"
        self.echo(line)
        with contextlib.suppress(OSError):
            self.queue.logs.mkdir(parents=True, exist_ok=True)
            with (self.queue.logs / f"{self.host}.log").open("a") as f:
                f.write(line + "\n")

    # -- main loop --------------------------------------------------------

    def run(self) -> int:
        """Process jobs until stopped (or until the inbox is empty with --once)."""
        self._wait_for_queue()
        self.queue.running_for(self.host).mkdir(parents=True, exist_ok=True)
        self.provenance = collect_provenance()
        self._requeue_own_running_jobs()
        keep_awake = self._start_caffeinate()
        previous = self._install_signal_handlers()
        self.log(
            f"worker started: slots={self.slots} retries={self.retries} "
            f"command={shlex.join(self.boltz_cmd)} args={shlex.join(self.boltz_args)}"
        )
        try:
            self._loop()
        finally:
            self._restore_signal_handlers(previous)
            if keep_awake is not None:
                keep_awake.terminate()
        self.log("worker stopped")
        return 0

    def _loop(self) -> None:
        next_scan = 0.0
        while True:
            if self._reap():
                next_scan = 0.0  # a slot just freed up: look for the next job now
            if self.stop_requested:
                self._stop_all()
                return
            draining = self._drain_requested()
            if (
                not draining
                and len(self.active) < self.slots
                and time.monotonic() >= next_scan
            ):
                next_scan = time.monotonic() + self.poll_s
                self._fill_slots()
            if not self.active:
                if draining:
                    self.log("drain flag found in control/; stopping")
                    return
                if self.once and not ready_jobs(self.queue, self.settle_s):
                    return
            self._sleep(0.5 if self.active else min(self.poll_s, 5.0))

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.stop_requested and time.monotonic() < end:
            time.sleep(min(0.2, max(0.0, end - time.monotonic())))

    def _wait_for_queue(self) -> None:
        waited = 0.0
        while not self.queue.is_initialized():
            if self.once:
                msg = (
                    f"{self.queue.root} isn't a boltz-queue folder "
                    "(run `boltz-queue init` first)."
                )
                raise click.ClickException(msg)
            if waited % 300 == 0:
                self.echo(
                    f"Waiting for the queue folder {self.queue.root} "
                    "(not mounted or not initialized yet)"
                )
            time.sleep(10)
            waited += 10

    def _drain_requested(self) -> bool:
        return (self.queue.control / "drain").exists() or (
            self.queue.control / f"drain-{self.host}"
        ).exists()

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous = {}

        def handler(signum: int, _frame: object) -> None:
            self.stop_requested = True
            self.echo(
                f"Received signal {signum}; "
                "stopping jobs and returning them to the inbox"
            )

        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError):  # not in the main thread
                previous[signum] = signal.signal(signum, handler)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(signum, handler)

    def _start_caffeinate(self) -> subprocess.Popen | None:
        """Keep the Mac awake while this worker runs, without changing settings."""
        if (
            not self.caffeinate
            or sys.platform != "darwin"
            or not shutil.which("caffeinate")
        ):
            return None
        command = ["caffeinate", "-is", "-w", str(os.getpid())]
        return subprocess.Popen(command)  # noqa: S603

    # -- claiming and starting jobs -------------------------------------------

    def _requeue_own_running_jobs(self) -> None:
        """Requeue jobs left in running/<this host>/ by a crash or restart."""
        for job in self.queue.jobs_in(self.queue.running_for(self.host)):
            _move_job(job, self.queue.inbox)
            self.log(
                f"returned {job.stem} to the inbox "
                "(left over from an earlier run on this Mac)"
            )

    def _fill_slots(self) -> None:
        for job in ready_jobs(self.queue, self.settle_s):
            if len(self.active) >= self.slots or self.stop_requested:
                return
            claimed = claim_job(self.queue, job, self.host)
            if claimed is None:
                continue
            try:
                self._start(claimed)
            except Exception as e:  # noqa: BLE001 - keep the worker running
                self._fail(claimed, str(e), log_path=None, retry=False)

    def _start(self, job_yaml: Path) -> None:
        name = job_yaml.stem
        sidecar = read_sidecar(job_yaml)
        result_dir = self.queue.results / name
        if (result_dir / DONE_MARKER).exists():
            msg = (
                f"results/{name} already holds a finished result; "
                "rename the job to run it again"
            )
            raise ValueError(msg)
        data = load_job_yaml(job_yaml)
        resolved = absolutize_references(data, self.queue.inbox.resolve())
        missing = [
            m[k] for m, k in path_references(resolved) if not Path(m[k]).is_file()
        ]
        if missing:
            msg = f"referenced file(s) not found: {', '.join(missing)}"
            raise ValueError(msg)

        input_dir = result_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_yaml = input_dir / f"{name}.yaml"
        input_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False))

        command = [
            *self.boltz_cmd,
            "predict",
            str(input_yaml),
            "--out_dir",
            str(result_dir),
        ]
        if (result_dir / f"boltz_results_{name}").exists():
            command.append("--override")  # an earlier attempt left partial output
        command += [*self.boltz_args, *sidecar.get("boltz_args", [])]

        attempt = int(sidecar.get("attempt", 0))
        started_iso = _now()
        (result_dir / "job.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "host": self.host,
                    "started": started_iso,
                    "attempt": attempt,
                    "command": command,
                },
                indent=2,
            )
        )
        log_file = (result_dir / "boltz.log").open("a")
        log_file.write(
            f"\n===== {started_iso} on {self.host}, attempt {attempt + 1}: "
            f"{shlex.join(command)}\n"
        )
        log_file.flush()
        proc = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        self.active[name] = RunningJob(
            name,
            job_yaml,
            result_dir,
            command,
            proc,
            log_file,
            time.monotonic(),
            started_iso,
            wants_affinity(data),
            attempt,
        )
        self.log(
            f"started {name} (attempt {attempt + 1}, "
            f"{len(self.active)}/{self.slots} slots busy)"
        )

    # -- finishing jobs ---------------------------------------------------------

    def _reap(self) -> int:
        """Finish jobs whose process ended (or ran out of time); return how many."""
        finished = 0
        for job in list(self.active.values()):
            code = job.proc.poll()
            if code is None:
                if self.timeout_s and time.monotonic() - job.started > self.timeout_s:
                    self._terminate(job)
                    self._finish(
                        job,
                        None,
                        f"stopped after the {self.timeout_s / 3600:.2f} h time limit",
                    )
                    finished += 1
                continue
            self._finish(job, code)
            finished += 1
        return finished

    def _finish(
        self, job: RunningJob, code: int | None, reason: str | None = None
    ) -> None:
        del self.active[job.name]
        job.log_file.close()
        runtime = time.monotonic() - job.started
        if reason is None and code != 0:
            reason = f"boltz exited with code {code}"
        if reason is None:
            try:
                reason = self._check_outputs(job)
                if reason is None:
                    self._succeed(job, runtime)
                    return
            except Exception as e:  # noqa: BLE001 - keep the worker running
                reason = f"couldn't read Boltz's output: {e}"
        self._fail(
            job.yaml_path, reason, log_path=job.result_dir / "boltz.log", retry=True
        )

    def _check_outputs(self, job: RunningJob) -> str | None:
        predictions = [
            p for p in find_predictions(job.result_dir) if p.name == job.name
        ]
        if not predictions:
            return "boltz finished but wrote no structure"
        if job.wants_affinity and not predictions[0].affinity:
            return "affinity was requested but boltz wrote no affinity result"
        return None

    def _succeed(self, job: RunningJob, runtime: float) -> None:
        prediction = next(
            p for p in find_predictions(job.result_dir) if p.name == job.name
        )
        summary = {
            "name": job.name,
            "status": "done",
            "host": self.host,
            "started": job.started_iso,
            "finished": _now(),
            "runtime_s": round(runtime, 1),
            "attempt": job.attempt + 1,
            "command": job.command,
            "provenance": self.provenance,
            "top_model": str(prediction.top_model.relative_to(job.result_dir)),
            "confidence": {
                k: prediction.confidence[k]
                for k in CONFIDENCE_KEYS
                if k in prediction.confidence
            },
            "affinity": prediction.affinity,
        }
        (job.result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (job.result_dir / DONE_MARKER).write_text(summary["finished"] + "\n")
        _move_job(job.yaml_path, self.queue.done)
        self.log(f"done {job.name} in {runtime / 60:.1f} min")

    def _fail(
        self, job_yaml: Path, reason: str, log_path: Path | None, retry: bool
    ) -> None:
        name = job_yaml.stem
        sidecar = read_sidecar(job_yaml)
        attempt = int(sidecar.get("attempt", 0)) + 1
        sidecar.update(
            {"attempt": attempt, "last_error": reason, "last_host": self.host}
        )
        _atomic_write_text(sidecar_for(job_yaml), json.dumps(sidecar, indent=2))
        if retry and attempt <= self.retries:
            _move_job(job_yaml, self.queue.inbox)
            self.log(
                f"failed {name} ({reason}); returned to the inbox "
                f"for retry {attempt} of {self.retries}"
            )
            return
        _move_job(job_yaml, self.queue.failed)
        details = [
            f"{name} failed on {self.host} at {_now()} after {attempt} attempt(s)",
            f"reason: {reason}",
        ]
        if log_path is not None:
            details += ["", f"last lines of {log_path}:", _tail(log_path)]
        (self.queue.failed / f"{name}.error.txt").write_text("\n".join(details) + "\n")
        self.log(f"failed {name}: {reason} (moved to failed/)")

    @staticmethod
    def _terminate(job: RunningJob) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(job.proc.pid, signal.SIGTERM)
        try:
            job.proc.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(job.proc.pid, signal.SIGKILL)
            job.proc.wait()

    def _stop_all(self) -> None:
        for job in list(self.active.values()):
            self._terminate(job)
            job.log_file.close()
            del self.active[job.name]
            _move_job(job.yaml_path, self.queue.inbox)
            self.log(f"stopped {job.name} and returned it to the inbox")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Run Boltz jobs from a shared folder on one or more Macs."""


queue_argument = click.argument(
    "queue_dir", type=click.Path(file_okay=False, path_type=Path)
)


@cli.command()
@queue_argument
def init(queue_dir: Path) -> None:
    """Create a queue folder (safe to repeat)."""
    queue = Queue(queue_dir.expanduser())
    queue.create()
    click.echo(
        f"Queue ready at {queue.root}. "
        f"Add jobs with `boltz-queue add {queue.root} job.yaml`."
    )


@cli.command()
@queue_argument
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--seeds", help="Comma-separated seeds: one job per seed, named <name>_seed<k>."
)
@click.option(
    "--boltz-args",
    default="",
    help='Extra boltz predict options for these jobs, e.g. "--diffusion_samples 5".',
)
def add(
    queue_dir: Path, files: tuple[Path, ...], seeds: str | None, boltz_args: str
) -> None:
    """Add Boltz YAML inputs to the queue.

    Files the inputs reference (MSAs, templates) are copied into the queue so
    every Mac can read them.
    """
    queue = open_queue(queue_dir)
    seed_list = parse_seeds(seeds)
    planned = []
    for path in files:
        try:
            data = load_job_yaml(path)
            staged = stage_references(data, path.parent.resolve(), queue)
            base = sanitize_name(path.stem)
        except ValueError as e:
            msg = f"{path}: {e}"
            raise click.ClickException(msg) from e
        planned += _plan_jobs(
            base, staged, seed_list, shlex.split(boltz_args), str(path)
        )
    _write_planned(queue, planned)
    click.echo(f"Added {len(planned)} job(s) to {queue.inbox}")


def _read_ligand_rows(
    csv_path: Path, name_col: str, smiles_col: str
) -> list[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for col in (name_col, smiles_col):
            if col not in columns:
                msg = (
                    f"{csv_path.name} has no column {col!r} "
                    f"(columns: {', '.join(columns)})"
                )
                raise click.ClickException(msg)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            name, smiles = (
                (row.get(name_col) or "").strip(),
                (row.get(smiles_col) or "").strip(),
            )
            if not name and not smiles:
                continue
            if not name or not smiles:
                msg = f"{csv_path.name} line {line_number}: needs a name and a SMILES"
                raise click.ClickException(msg)
            rows.append((name, smiles))
    if not rows:
        msg = f"{csv_path.name} has no ligands"
        raise click.ClickException(msg)
    return rows


def _invalid_smiles(rows: list[tuple[str, str]]) -> list[str] | None:
    """Names whose SMILES RDKit can't parse, or None if RDKit isn't installed."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    return [name for name, smiles in rows if Chem.MolFromSmiles(smiles) is None]


def _entity_ids(data: dict[str, Any]) -> set[str]:
    ids = set()
    for item in data.get("sequences") or []:
        for entity in item.values() if isinstance(item, dict) else []:
            value = entity.get("id") if isinstance(entity, dict) else None
            ids |= set(value) if isinstance(value, list) else {value}
    return {str(i) for i in ids if i is not None}


def _load_template(template: Path, ligand_id: str, affinity: bool) -> dict[str, Any]:
    try:
        base = load_job_yaml(template)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if ligand_id in _entity_ids(base):
        msg = (
            f"The template already uses chain ID {ligand_id!r}; "
            "pick another with --ligand-id."
        )
        raise click.ClickException(msg)
    if affinity and wants_affinity(base):
        msg = (
            "The template already has an affinity property; "
            "remove it or use --no-affinity."
        )
        raise click.ClickException(msg)
    return base


def _check_smiles(rows: list[tuple[str, str]]) -> None:
    bad = _invalid_smiles(rows)
    if bad is None:
        click.echo("Note: RDKit isn't installed here, so the SMILES weren't checked.")
    elif bad:
        msg = f"RDKit can't parse the SMILES for: {', '.join(bad)}"
        raise click.ClickException(msg)


def _ligand_job_names(rows: list[tuple[str, str]], prefix: str) -> list[str]:
    try:
        names = [f"{prefix}{sanitize_name(name)}" for name, _ in rows]
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        msg = f"Ligand names repeat after cleaning: {', '.join(repeated)}"
        raise click.ClickException(msg)
    return names


def _with_ligand(
    base: dict[str, Any], ligand_id: str, smiles: str, affinity: bool
) -> dict[str, Any]:
    data = copy.deepcopy(base)
    data["sequences"].append({"ligand": {"id": ligand_id, "smiles": smiles}})
    if affinity:
        properties = data.get("properties") or []
        data["properties"] = [*properties, {"affinity": {"binder": ligand_id}}]
    return data


def _write_to_folder(jobs: list[tuple[str, dict[str, Any]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [f"{n}.yaml" for n, _ in jobs if (out_dir / f"{n}.yaml").exists()]
    if existing:
        msg = f"Files already exist in {out_dir}: {', '.join(existing)}"
        raise click.ClickException(msg)
    for name, data in jobs:
        (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    click.echo(f"Wrote {len(jobs)} input file(s) to {out_dir}")


@cli.command("make-inputs")
@click.argument(
    "template", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "ligands_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--queue",
    "queue_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Add the jobs to this queue.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Write YAML files here instead.",
)
@click.option(
    "--ligand-id", default="L", show_default=True, help="Chain ID to give the ligand."
)
@click.option(
    "--affinity/--no-affinity",
    default=True,
    show_default=True,
    help="Predict affinity for the ligand.",
)
@click.option(
    "--name-col",
    default="name",
    show_default=True,
    help="CSV column with ligand names.",
)
@click.option(
    "--smiles-col", default="smiles", show_default=True, help="CSV column with SMILES."
)
@click.option(
    "--prefix",
    default=None,
    help="Job-name prefix (default: the template's file name and '_').",
)
@click.option(
    "--seeds",
    help="Comma-separated seeds (needs --queue): one job per ligand and seed.",
)
@click.option(
    "--boltz-args",
    default="",
    help="Extra boltz predict options for these jobs (needs --queue).",
)
def make_inputs(
    template: Path,
    ligands_csv: Path,
    queue_dir: Path | None,
    out_dir: Path | None,
    ligand_id: str,
    affinity: bool,
    name_col: str,
    smiles_col: str,
    prefix: str | None,
    seeds: str | None,
    boltz_args: str,
) -> None:
    """Make one Boltz input per ligand from a target TEMPLATE and a LIGANDS_CSV.

    TEMPLATE is a normal Boltz YAML with the target (protein, its MSA, any
    cofactors or constraints) but without the ligand being screened.
    LIGANDS_CSV has a name and a SMILES column.
    """
    if (queue_dir is None) == (out_dir is None):
        msg = "Give exactly one of --queue or --out."
        raise click.UsageError(msg)
    if out_dir is not None and (seeds or boltz_args):
        msg = "--seeds and --boltz-args need --queue."
        raise click.UsageError(msg)
    base = _load_template(template, ligand_id, affinity)
    rows = _read_ligand_rows(ligands_csv, name_col, smiles_col)
    _check_smiles(rows)
    prefix = f"{sanitize_name(template.stem)}_" if prefix is None else prefix
    names = _ligand_job_names(rows, prefix)

    queue = open_queue(queue_dir) if queue_dir is not None else None
    try:
        if queue is not None:
            base = stage_references(base, template.parent.resolve(), queue)
        else:
            base = absolutize_references(base, template.parent.resolve())
    except ValueError as e:
        msg = f"{template}: {e}"
        raise click.ClickException(msg) from e
    jobs = [
        (name, _with_ligand(base, ligand_id, smiles, affinity))
        for name, (_, smiles) in zip(names, rows)
    ]
    if queue is None:
        _write_to_folder(jobs, out_dir)
        return
    planned = []
    for name, data in jobs:
        planned += _plan_jobs(
            name, data, parse_seeds(seeds), shlex.split(boltz_args), str(ligands_csv)
        )
    _write_planned(queue, planned)
    click.echo(f"Added {len(planned)} job(s) to {queue.inbox}")


@cli.command()
@queue_argument
@click.option(
    "--slots",
    default=1,
    show_default=True,
    type=click.IntRange(1, 8),
    help="Jobs to run at once on this Mac.",
)
@click.option(
    "--retries",
    default=1,
    show_default=True,
    type=click.IntRange(0, 10),
    help="Retries before a job goes to failed/.",
)
@click.option(
    "--timeout-hours",
    default=0.0,
    show_default=True,
    type=float,
    help="Stop a job after this long (0 = no limit).",
)
@click.option(
    "--poll",
    default=10.0,
    show_default=True,
    type=float,
    help="Seconds between inbox checks.",
)
@click.option(
    "--settle",
    default=15.0,
    show_default=True,
    type=float,
    help="Skip inbox files changed within this many seconds.",
)
@click.option(
    "--use-msa-server",
    is_flag=True,
    help=(
        "Let Boltz fetch missing MSAs from the public ColabFold server "
        "(this sends protein sequences to it)."
    ),
)
@click.option(
    "--boltz-cmd",
    default=None,
    help="Command that runs Boltz (default: boltz next to this Python).",
)
@click.option(
    "--caffeinate/--no-caffeinate",
    default=True,
    show_default=True,
    help="Keep the Mac awake while working.",
)
@click.option(
    "--once",
    is_flag=True,
    help="Exit when the inbox is empty instead of waiting for more jobs.",
)
@click.argument("boltz_args", nargs=-1, type=click.UNPROCESSED)
def worker(
    queue_dir: Path,
    slots: int,
    retries: int,
    timeout_hours: float,
    poll: float,
    settle: float,
    use_msa_server: bool,
    boltz_cmd: str | None,
    caffeinate: bool,
    once: bool,
    boltz_args: tuple[str, ...],
) -> None:
    """Run jobs from the queue on this Mac.

    Options after `--` go to every `boltz predict` call, for example:
    boltz-queue worker /Volumes/boltz-queue -- --diffusion_samples 5
    """
    args = list(boltz_args)
    if "--accelerator" not in args:
        args = ["--accelerator", "mps", *args]
    if use_msa_server and "--use_msa_server" not in args:
        args.append("--use_msa_server")
    runner = Worker(
        queue=Queue(queue_dir.expanduser()),
        boltz_cmd=shlex.split(boltz_cmd) if boltz_cmd else default_boltz_cmd(),
        boltz_args=args,
        slots=slots,
        retries=retries,
        timeout_s=timeout_hours * 3600,
        poll_s=poll,
        settle_s=settle,
        once=once,
        caffeinate=caffeinate,
    )
    sys.exit(runner.run())


def _elapsed(started_iso: str | None) -> str:
    if not started_iso:
        return ""
    with contextlib.suppress(ValueError):
        minutes = (
            datetime.now(timezone.utc) - datetime.fromisoformat(started_iso)
        ).total_seconds() / 60
        return f" for {minutes:.0f} min"
    return ""


@cli.command()
@queue_argument
def status(queue_dir: Path) -> None:
    """Show waiting, running, done and failed jobs."""
    queue = open_queue(queue_dir)
    click.echo(f"Queue: {queue.root}")
    click.echo(f"  waiting: {len(queue.jobs_in(queue.inbox))}")
    for host_dir in sorted(p for p in queue.running.iterdir() if p.is_dir()):
        jobs = queue.jobs_in(host_dir)
        click.echo(f"  running on {host_dir.name}: {len(jobs)}")
        for job in jobs:
            info_path = queue.results / job.stem / "job.json"
            info = json.loads(info_path.read_text()) if info_path.is_file() else {}
            click.echo(f"    {job.stem}{_elapsed(info.get('started'))}")
    click.echo(f"  done: {len(queue.jobs_in(queue.done))}")
    failed = queue.jobs_in(queue.failed)
    click.echo(f"  failed: {len(failed)}")
    for job in failed[-10:]:
        reason = read_sidecar(job).get("last_error", "")
        click.echo(f"    {job.stem}: {reason}")
    flags = (
        sorted(p.name for p in queue.control.glob("drain*"))
        if queue.control.is_dir()
        else []
    )
    if flags:
        click.echo(f"  drain flags set: {', '.join(flags)} (delete them to resume)")


SUMMARY_COLUMNS = (
    "name",
    "host",
    "runtime_min",
    "attempts",
    "started",
    "finished",
    *CONFIDENCE_KEYS,
    "binder_chain",
    *AFFINITY_KEYS,
    "top_model",
)


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "name": summary.get("name"),
        "host": summary.get("host"),
        "runtime_min": round(summary.get("runtime_s", 0) / 60, 2),
        "attempts": summary.get("attempt"),
        "started": summary.get("started"),
        "finished": summary.get("finished"),
        "top_model": summary.get("top_model"),
        **{k: summary.get("confidence", {}).get(k) for k in CONFIDENCE_KEYS},
    }
    affinity = summary.get("affinity") or {}
    if not affinity:
        return [base]
    return [
        {
            **base,
            "binder_chain": chain or values.get("binder_chain", ""),
            **{k: values.get(k) for k in AFFINITY_KEYS},
        }
        for chain, values in sorted(affinity.items())
    ]


@cli.command()
@queue_argument
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="CSV path (default: <queue>/summary.csv).",
)
def summarize(queue_dir: Path, out: Path | None) -> None:
    """Collect finished jobs' confidence and affinity results into one CSV."""
    queue = open_queue(queue_dir)
    summaries = [
        json.loads(p.read_text()) for p in sorted(queue.results.glob("*/summary.json"))
    ]
    out = out or queue.root / "summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerows(_summary_rows(summary))
    click.echo(f"Wrote {len(summaries)} finished job(s) to {out}")
    if not summaries:
        return
    by_host: dict[str, list[float]] = {}
    for summary in summaries:
        by_host.setdefault(summary.get("host", "?"), []).append(
            summary.get("runtime_s", 0) / 60
        )
    for host, minutes in sorted(by_host.items()):
        average = sum(minutes) / len(minutes)
        click.echo(f"  {host}: {len(minutes)} job(s), average {average:.1f} min")
    with contextlib.suppress(ValueError, KeyError, TypeError):
        first = min(datetime.fromisoformat(s["started"]) for s in summaries)
        last = max(datetime.fromisoformat(s["finished"]) for s in summaries)
        hours = (last - first).total_seconds() / 3600
        if hours > 0:
            rate = len(summaries) / hours
            click.echo(f"  throughput: {rate:.1f} jobs/hour over {hours:.1f} h")


@cli.command()
@queue_argument
@click.argument("names", nargs=-1)
@click.option("--all-failed", is_flag=True, help="Requeue every failed job.")
@click.option(
    "--from-host",
    default=None,
    help="Return jobs stuck in running/<HOST>/ (only if that Mac's worker is stopped).",
)
def requeue(
    queue_dir: Path, names: tuple[str, ...], all_failed: bool, from_host: str | None
) -> None:
    """Return failed jobs (or a stopped Mac's jobs) to the inbox."""
    queue = open_queue(queue_dir)
    moved = 0
    if from_host is not None:
        for job in queue.jobs_in(queue.running_for(sanitize_name(from_host))):
            _move_job(job, queue.inbox)
            moved += 1
    failed = {job.stem: job for job in queue.jobs_in(queue.failed)}
    selected = list(failed) if all_failed else list(names)
    unknown = [n for n in selected if n not in failed]
    if unknown:
        msg = f"Not in failed/: {', '.join(unknown)}"
        raise click.ClickException(msg)
    for name in selected:
        job = failed[name]
        sidecar = read_sidecar(job)
        sidecar.update({"attempt": 0})
        sidecar.pop("last_error", None)
        _atomic_write_text(sidecar_for(job), json.dumps(sidecar, indent=2))
        _move_job(job, queue.inbox)
        (queue.failed / f"{name}.error.txt").unlink(missing_ok=True)
        moved += 1
    click.echo(f"Returned {moved} job(s) to the inbox")


@cli.command("launchd-plist")
@queue_argument
@click.option(
    "--label",
    default="org.boltz.queue-worker",
    show_default=True,
    help="LaunchAgent label.",
)
@click.option(
    "--worker-args",
    default="",
    help='Options for the worker, e.g. "--slots 2 --use-msa-server".',
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the plist here.",
)
def launchd_plist(
    queue_dir: Path, label: str, worker_args: str, output: Path | None
) -> None:
    """Print a LaunchAgent that runs the worker at login and restarts it on a crash.

    Nothing is installed; the printed instructions say how to install it.
    """
    log_path = Path.home() / "Library" / "Logs" / f"{label}.log"
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin"}
    if os.environ.get("BOLTZ_CACHE"):
        env["BOLTZ_CACHE"] = os.environ["BOLTZ_CACHE"]
    plist = {
        "Label": label,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "boltz.scripts.studio_queue",
            "worker",
            str(queue_dir.expanduser().absolute()),
            *shlex.split(worker_args),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "EnvironmentVariables": env,
    }
    text = plistlib.dumps(plist).decode()
    target = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if output is not None:
        output.write_text(text)
    else:
        click.echo(text)
    click.echo(
        f"\nTo install (you run these):\n"
        f"  save the plist as {target}\n"
        f"  launchctl bootstrap gui/$(id -u) {target}\n"
        f"To stop and remove:\n"
        f"  launchctl bootout gui/$(id -u)/{label}\n"
        f"Worker output goes to {log_path}.\n"
        "The agent starts when you log in, so the Mac must log in after a reboot,\n"
        "and a network queue folder must mount at login.",
        err=True,
    )


if __name__ == "__main__":
    cli()
