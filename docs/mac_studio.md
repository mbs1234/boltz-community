# Running Boltz on the lab's Mac Studios

This fork adds three things for running Boltz-2 co-folding and affinity jobs on
Apple Silicon Macs:

- a **setup script** that installs Boltz on a Mac Studio and checks the GPU;
- **`boltz-queue`**, a job queue that lets several Macs share one list of jobs,
  including a helper that turns a target plus a ligand spreadsheet into jobs;
- **`boltz-compare`**, which checks Mac results against reference results, such
  as the same jobs run on the A100 servers.

> **Status (2026-09-27):** these tools were written and tested on an Intel Mac,
> using a stand-in for Boltz and synthetic structures. They have not yet been run
> on a Mac Studio. [`HANDOFF.md`](../HANDOFF.md) lists what still needs checking.

## 1. Set up each Mac Studio (once)

1. Install Apple's Command Line Tools if the Mac doesn't have them. A window pops up
   to confirm:

   ```bash
   xcode-select --install
   ```

2. Download this repository and run the setup script:

   ```bash
   git clone https://github.com/mbs1234/boltz-community.git
   bash boltz-community/scripts/mac/setup_mac_studio.sh
   ```

The script installs everything into `~/boltz/venv`. It then runs a small
protein + ligand + affinity test on the GPU. The first run downloads several GB of
model weights to `~/.boltz`. When it finishes, it records the versions it used in
`~/boltz/setup-info.txt`. It never changes system settings.

**Use identical code on every Studio.** Install the same tag or commit on each one,
for example `BOLTZ_REF=<tag or commit> bash boltz-community/scripts/mac/setup_mac_studio.sh`.
Run the script with `--help` for all options, or with `--dry-run` to see what it
would do without doing it.

To use Boltz in a new Terminal window afterwards:

```bash
source ~/boltz/venv/bin/activate
```

## 2. Run a single prediction

```bash
boltz predict my_complex.yaml --accelerator mps
```

Inputs are ordinary Boltz YAML files, the same ones you use on the A100 servers.

**About MSAs.** Boltz needs an MSA for each protein.
- `--use_msa_server` fetches MSAs from the public ColabFold server. That sends your
  protein sequences to an outside service, and the server limits how much each
  user can request.
- For repeated work on the same target, reuse a saved MSA instead. Boltz keeps the
  MSAs it fetched in `boltz_results_<name>/msa/`. Point `msa:` at that file.

## 3. The job queue

A queue is a folder. Any Mac that can see the folder can run a worker that takes
jobs from it. With one Mac, the folder can be on that Mac's own disk. With several,
use a folder they all mount: a lab file server, or File Sharing turned on for one of
the Studios.

### Create the queue and start workers

```bash
boltz-queue init /Volumes/boltz-queue        # once
boltz-queue worker /Volumes/boltz-queue      # on each Studio, in its own Terminal window
```

The worker takes the oldest waiting job and runs `boltz predict` on the Mac GPU.
It then takes the next job, and so on. Some details:

- **Staying awake:** the worker keeps the Mac awake while it runs, so no energy
  settings need changing.
- **Several jobs at once:** `--slots 2` runs two jobs at a time. Whether that's
  faster depends on the job size, so measure it.
- **Boltz options:** options after `--` go to every job, for example
  `boltz-queue worker /Volumes/boltz-queue -- --diffusion_samples 5`.
- **MSA server:** `--use-msa-server` lets Boltz fetch missing MSAs. See the privacy
  note above.
- **Stopping one worker:** press Ctrl-C. The jobs it was running go back to the
  queue.
- **Stopping all workers politely:** create a file named `drain` in the queue's
  `control/` folder. Workers take no new jobs and exit once their current jobs
  finish. A file named `drain-<mac name>` stops just that Mac. Delete the file to
  resume.

### Add jobs

From Boltz YAML files:

```bash
boltz-queue add /Volumes/boltz-queue complex1.yaml complex2.yaml
boltz-queue add /Volumes/boltz-queue complex1.yaml --seeds 1,2,3   # one job per seed
```

`add` copies any MSA or template files the YAML refers to into the queue, so every
Mac can read them.

For a **ligand series** against one target:
1. Write a target YAML with the protein, its MSA and any cofactors, but without the
   ligand. [`examples/mac_studio/target.yaml`](../examples/mac_studio/target.yaml)
   shows one.
2. Make a spreadsheet saved as CSV with `name` and `smiles` columns, like
   [`examples/mac_studio/ligands.csv`](../examples/mac_studio/ligands.csv).
3. Create the jobs:

   ```bash
   boltz-queue make-inputs target.yaml ligands.csv --queue /Volumes/boltz-queue
   ```

That makes one job per ligand, named like `target_<ligand name>`, and asks for an
affinity prediction for each.
- `--no-affinity` skips the affinity prediction.
- `--seeds` makes several runs per ligand.
- `--out folder/` writes the YAML files to a folder instead of queueing them.

Every SMILES is checked with RDKit before anything is added.

### Follow progress and collect results

```bash
boltz-queue status /Volumes/boltz-queue       # waiting, running (per Mac), done, failed
boltz-queue summarize /Volumes/boltz-queue    # writes summary.csv and prints jobs/hour
```

- **`summary.csv`** has one row per finished job (per binder). It includes the
  confidence scores (`confidence_score`, `iptm`, `ligand_iptm`, `complex_plddt`, …),
  the affinity outputs (`affinity_pred_value`, `affinity_probability_binary`), the
  run time, and which Mac ran the job.
- **Each job's full Boltz output** is in `results/<job>/`. That folder also holds
  `summary.json`, which records the exact Boltz command and software versions, and
  the Boltz log.
- **Failed jobs** are retried once, then moved to `failed/` with a
  `<job>.error.txt` explaining why.
  - After fixing the cause, run `boltz-queue requeue /Volumes/boltz-queue <job>`,
    or `--all-failed` to requeue every failed job.
  - If a Mac died with jobs still claimed, return them with
    `boltz-queue requeue /Volumes/boltz-queue --from-host <mac name>`, but only
    once that Mac's worker is stopped.

### Start workers automatically (optional)

`boltz-queue launchd-plist /Volumes/boltz-queue` prints a macOS LaunchAgent, plus
the two commands to install it. You run those commands yourself. The agent starts
the worker when you log in, and restarts it if it crashes. For this to survive a
reboot, the Mac must log in on its own and the queue folder must mount at login.

## 4. Compare Mac results with A100 results

1. Copy the A100 `boltz_results_*` folders onto a Studio. Keep them outside this
   repository.
2. Run the same inputs on the Mac, with the same MSA files, flags and seeds. The
   queue's `--seeds` option is convenient here.
3. Compare the two sets:

   ```bash
   boltz-compare --ref a100_results/ --test mac_results/ --out report.csv
   ```

The tool pairs up predictions by name. A seed suffix such as `_seed2` is ignored
when grouping. For each system it reports:

- **Protein CA RMSD:** how far the Mac's top model is from the A100's.
- **Ligand RMSD:** measured after superposing the binding pocket. It's given two
  ways:
  - atom-by-atom by name;
  - symmetry-tolerant, so a flipped ring doesn't count as an error.
- **Output differences:** in the confidence scores and the affinity outputs.
- **The same measures within each side,** when a side has several seeds.
  Predictions differ from seed to seed even on one machine. The real question is
  whether Mac-vs-A100 differences are larger than that seed-to-seed spread.

Rows that exceed the starting-point thresholds are flagged:
- ligand RMSD 2 Å
- protein RMSD 2 Å
- affinity difference 0.5 log units
- binder-probability difference 0.2

Change the thresholds with `--ligand-rmsd-tol`, `--ca-rmsd-tol`, `--affinity-tol`
and `--probability-tol`. What counts as acceptable is a scientific call.

## 5. Good practice

- **Keep research data out of this repository.** It's public. Store inputs, MSAs,
  ligand lists and results elsewhere, or in a `local/` folder inside it, which git
  ignores.
- **Hold macOS updates on the Studios** until the setup script's smoke test passes
  on the new version. One M5 Max failed on large inputs until a macOS update fixed
  it.
- **Keep all three Studios on the same version of this repository.** Each job's
  `summary.json` records the versions actually used.
