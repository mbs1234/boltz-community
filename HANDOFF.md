# Handoff: Boltz-2 co-folding on the lab's Mac Studios

_Written 2026-09-27 at the end of the first Claude Code session, which did evaluation and planning from the Intel
iMac. Nothing was run on a Studio yet. Keep §9 (status log) current._

## 0. Starting the next session

1. Merge this PR, then clone the repo on the Studio chosen as the development machine:
   `gh repo clone mbs1234/boltz-community`
2. Open that folder in Claude Code. `CLAUDE.md` loads automatically.
3. A good first prompt: _"Read HANDOFF.md and start Phase 0 on this Studio."_

## 1. Goal and context

- The user runs Boltz-2 co-folding (protein + ligand) and affinity prediction for research, currently on external
  A100 servers. That time is paid for and capped per week.
- The lab has three M1 Ultra Mac Studios (128 GB each) that cost nothing extra to run and can be dedicated to
  co-folding.
- **Goal:** move as much routine work as possible to the Studios while keeping results trustworthy. Keep the A100
  quota for the largest complexes, urgent jobs, and periodic spot checks.
- **Metric:** validated jobs per week across the three Studios. Speed improvements matter because each one converts
  directly into weekly capacity: 1.5× faster means 50% more jobs per week at no extra cost.
- The user has results from systems they routinely run on the A100s. Those serve as the reference for validation.

## 2. Decisions so far

| Topic | Decision |
|---|---|
| Base repo | Fork Novel-Therapeutics/boltz-community, not fnachon/boltz (see §3). Created as public `mbs1234/boltz-community` from `401f181` (v2.10.12) on 2026-09-27 |
| Delivery | Feature branch plus a PR inside this fork. The user reviews before merging to `main` |
| Job input | Both: a YAML inbox folder, plus a helper that turns a target YAML and a ligand CSV into job YAMLs |
| Sharing between Studios | **Not decided.** Design around a shared folder (lab NAS, or macOS File Sharing on one Studio). The same design must also work on one Studio's local disk |
| Hard tier (custom Metal GPU code / MLX) | Not started. Depends on profiling results from Phase 1 |

## 3. Findings from the evaluation

### Repo lineage
- **fnachon/boltz** is boltz-community v2.10.10 plus one commit, `75b25f7`. That commit works around MPS checkpoint
  loading on PyTorch builds that lack `torch.mps.current_device`. It also maps `--accelerator gpu` to `mps` on Macs.
  - It is now 9 commits behind the parent.
  - The parent's `9cf2135` (#16) always stages checkpoints on CPU, which supersedes most of it.
  - The only idea worth taking is auto-selecting MPS on a Mac (Phase 4).
  - Don't port the commit as-is. Its `current_device` shim is installed early in `predict()`, which turns its own
    CPU-fallback branches into dead code.
- **Novel-Therapeutics/boltz-community** is actively maintained, with about 160 commits of fixes beyond upstream Boltz.
  It has a large CPU test suite, is published on PyPI as `boltz-community`, and accepts outside PRs.

### State of Mac (MPS) support
- **Enabled, not optimized.** `--accelerator mps` forces fp32 (no bf16-mixed), uses a single device, and disables
  pinned memory. Autocast goes through `autocast_device_type()`, and `torch.mps.empty_cache()` is guarded. There is no
  Metal-specific code.
  - The cuEquivariance triangle kernels are CUDA-only, so MPS runs the plain PyTorch path.
- **`--flash_attn` (SDPA)** is off by default and not gated by device. It is untested on MPS.
- **Rigid alignment:** `weighted_rigid_align` calls `torch.linalg.svd` and `torch.det` on MPS tensors at every
  diffusion step. Depending on the torch version these run natively or fall back to the CPU. Profiling will show which.
- **Test coverage is thin.**
  - `tests/test_mps.py` covers only a 10-residue peptide, and that peptide plus a tyrosine ligand with affinity.
  - Those tests are excluded from CI and run by hand.
  - The regression tests (golden outputs) run only on CUDA or CPU.
  - No Mac-vs-NVIDIA accuracy comparison has been published.
- **Only benchmark on record:** a single data point, removed from the README in `550c5b8`. An M3 MacBook Air (fp32)
  vs an NVIDIA T4 (bf16):

  | Case | M3 Air | T4 |
  |---|---|---|
  | 10-residue peptide + ligand, with affinity | 114 s | 107 s |
  | 163-residue protein + ligand, with affinity | 251 s | 144 s |
  | 260-residue protein + ligand, with affinity | 383 s | 193 s |
  | 163-residue protein, structure only | 74 s | 57 s |

- **Version sensitivity (parent issue #9):** an M5 Max crashed with `Failed to allocate private MTLBuffer for size 0`
  on a roughly 440-residue Fab. A macOS update from 26.3.1 to 26.4.1 fixed it. Pin macOS and PyTorch versions, and
  re-run the smoke test after any update.
- **`OMP: Error #15` on macOS:** run `boltz-fix-macos-libomp` after installing or upgrading torch or scikit-learn.

### Upstream `.gitignore` bug (fixed in this fork's first PR)
- The parent's `63ae28c` (2026-03-04) merged `boltz_results_*/` and `.idea/` into a single line,
  `boltz_results_*/.idea/`. As a result, Boltz output folders were not ignored.
- This is a candidate for an upstream PR, but ask the user first.

### Anthropic's Boltz-2 optimization kit
- Location: <https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/boltz2>. Apache-2.0,
  unmaintained, and not accepting PRs.
- **Can't run on Macs.**
  - It is NVIDIA-only: driver 580 or newer, compute capability 8.0 or higher, with configs for A100, H100 and H200.
  - It is pinned to stock boltz 2.2.1, not this fork.
  - It relies on Triton, CUDA graphs, NVRTC and cuEquivariance.
  - It leaves the affinity pass unchanged.
- **Device-independent ideas from its `CHANGES.md` worth trying here (Phase 4):**
  - Pairformer eval-mode shortcuts: skip multiplying by the all-ones dropout mask and adding all-zero mask terms
    (`resid`, `mask2`).
  - Skip the template module when every template is a dummy (`templ_skip`).
  - Hoist step-invariant work out of the diffusion loop, computing it once per sample (`dit_hoist`, `atom_glue_hoist`).
  - In the MSA module, cast once, reuse mask counts and divide in place. Skip parameter initialization that the
    checkpoint load overwrites (`waste_*`).
  - Write outputs on a background process, and prefetch the next input's featurization (`writer_overlap`, `prefetch`).
  - Cut peak memory: row-chunk the pair transition and diffusion conditioning, free dead conditioning outputs early,
    and recompute relative-position encodings lazily (`xl_trans`, `xl_cond`, `xl_free`, `relpos_lazy`).
  - Compute the 3×3 SVD and determinant for rigid alignment on the GPU with a small closed-form or Jacobi routine
    (`align_jacobi64`). This only matters if MPS falls back to CPU for SVD.
- **Its validation standard is a good model:** `exact` means byte-identical to stock; `fast` means within stock's
  seed-to-seed variation.

## 4. Hardware notes (M1 Ultra Mac Studio)

- **Specs:** 20-core CPU, 48- or 64-core GPU (check which), 128 GB unified memory, 800 GB/s memory bandwidth.
  - By default the GPU can use roughly three-quarters of RAM. Check with `torch.mps.recommended_max_memory()` if the
    installed PyTorch has it.
- **No native bf16 on M1-generation GPUs.** Reduced precision here means fp16, whose narrower range can overflow in
  attention logits and pair activations.
  - NVIDIA runs Boltz-2 in bf16-mixed, while the Mac currently runs fp32.
- **No matrix-multiply hardware.** An A100's tensor cores give it roughly an order of magnitude more peak throughput
  for the matrix math that dominates Boltz, plus 2–2.5× the memory bandwidth.
  - Don't expect A100 speed per job. Make each Studio as fast as it can be, and run all three in parallel.
- **The Ultra is two M1 Max dies joined together.** GPU work often doesn't scale fully across both halves.
  - Measure on the Ultra itself.
  - Test whether 1, 2 or 3 concurrent jobs per Studio give the best throughput.
- **Unattended runs:**
  - Prevent idle sleep while jobs run with `caffeinate -is -w <pid>`, which the worker should launch. This needs no
    change to system settings.
  - Ask the user to defer macOS updates until a smoke test passes on the new version.

## 5. Plan

Each phase ends with a go/no-go check with the user.

### Phase 0: Set up the development Studio
- [ ] Write `scripts/mac/setup_mac_studio.sh` (spec §6.1) and run it.
- [ ] Record the environment in `$BOLTZ_HOME/setup-info.txt`: macOS version, chip, GPU cores, torch version, boltz
      commit.
- [ ] Run `pytest tests/test_mps.py -m mps -v` and the CPU suite.
- [ ] Keep one Studio as the development machine, so benchmarks aren't disturbed by production runs.

### Phase 1: Baseline and trust
- [ ] Pick a reference set with the user: 5–10 representative systems (typical protein sizes and ligand types, with
      and without affinity, plus any constraints or templates they use), and one or two large systems.
- [ ] Collect A100 references for each system:
  - the exact inputs, including MSA files (Boltz saves them in `boltz_results_<name>/msa/`)
  - the Boltz version, flags and seeds
  - the full outputs
- [ ] Get matched A100 runs where possible. The cleanest comparison is a few A100 re-runs using the same
      boltz-community commit as the Mac, with several seeds each. This costs only a small amount of quota.
- [ ] Add ground truth where available (crystal poses, assay data). Agreement with the A100 shows consistency, not
      correctness.
- [ ] Build `boltz-compare` (spec §6.3) and run it against the reference set.
- [ ] Measure wall time per job and its breakdown: model load, featurization, trunk, diffusion, confidence, affinity.
      Profile with `torch.profiler` and Instruments (Metal System Trace) to find the top operations.
- [ ] Measure throughput with 1, 2 and 3 concurrent jobs per Studio.
- [ ] Report to the user: Mac-vs-A100 agreement, an estimate of jobs per week, and where the time goes. Commit a
      report only if it contains no research data. The user sets the acceptance tolerances.
- **Gate:** can Mac results replace A100 runs for routine work, and at what tolerance?

### Phase 2: Run the three Studios as one small cluster
- [ ] Build `boltz-queue` (spec §6.2), including the YAML inbox and the CSV helper.
- [ ] Settle the sharing method with the user: NAS, File Sharing on one Studio, or an SSH hub.
- [ ] Provide a LaunchAgent so workers restart after a reboot. The user installs it.
- **Gate:** does it run unattended for a week?

### Phase 3: Routing
- Send routine jobs and ligand series to the Studios.
- Send the largest complexes, urgent jobs and periodic spot checks to the A100.

### Phase 4: Moderate-tier speed work, driven by profiling
Each change must show a measured gain on real jobs and pass validation within tolerance. Otherwise revert it.
- [ ] Auto-select MPS for the default `--accelerator gpu` on Apple Silicon. Implement it cleanly with tests; see §3
      for why not to port `75b25f7` directly.
- [ ] Fix hotspots found in Phase 1: CPU fallbacks, `.item()` and other host syncs, memory peaks.
- [ ] Try `--flash_attn` (SDPA) on MPS and check correctness and speed.
- [ ] Try fp16 on M1 layer by layer, keeping sensitive operations in fp32. Validate.
- [ ] Port the device-independent ideas from Anthropic's kit (§3).

### Phase 5: Hard tier, only if Phase 1 profiling justifies it
- **Amdahl check first.** For example, if triangle attention is 40% of runtime, a kernel that makes it 3× faster gives
  about 1.4× overall.
- **Route A (try first on M1): custom Metal kernels inside the existing PyTorch code.**
  - Forward pass only, plugged in where `use_kernels` switches to cuEquivariance on CUDA.
  - Mechanism: `torch.mps.compile_shader` for inline Metal (check the API on the installed PyTorch), or a C++ /
    Objective-C++ extension.
  - Order of work:
    1. Triangle attention: flash-attention-style online softmax with pair bias, never materializing the full
       N×N×N logits.
    2. Triangle multiplication: fuse the LayerNorm, gating and projections around the contraction.
    3. A fused LayerNorm + transition.
  - Tune on the M1 Ultra itself.
- **Route B: port the inference forward pass to MLX** (Apple's framework).
  - Covers the trunk, diffusion, confidence and affinity modules: several thousand lines.
  - A hybrid is also possible, with MLX for only the trunk and diffusion.
  - Choose this only if profiling shows that general PyTorch-on-MPS overhead dominates.
  - Costs: a second implementation to maintain, every upstream fix mirrored by hand, and every feature re-validated.

## 6. Tool specs

### 6.1 Setup script: `scripts/mac/setup_mac_studio.sh`
- **Preflight:** check for macOS on arm64; warn below macOS 14.
- **Python environment:**
  - Install `uv` if it's missing: via Homebrew if present, otherwise the official installer. Print what it does.
  - Create a Python 3.12 venv at `$BOLTZ_HOME/venv` (default `~/boltz`).
- **Install this fork at a pinned ref:**
  `boltz-community @ git+https://github.com/mbs1234/boltz-community.git@$BOLTZ_REF`.
  - The default ref is `main`. Use a tag or commit so all three Studios run identical code.
  - For development, install editable from a checkout instead (`BOLTZ_SRC=/path`).
- **Post-install checks:** run `boltz-fix-macos-libomp`, then verify `torch.backends.mps.is_available()`.
- **Smoke test:** a small protein + ligand + affinity job with `msa: empty`, `--accelerator mps` and reduced steps.
  - The first run downloads several GB of weights to `$BOLTZ_CACHE` (default `~/.boltz`).
- **Records:** write `$BOLTZ_HOME/setup-info.txt` and print next steps.
- **Options:** `--dry-run` prints commands without running them.
- **Constraint:** never change system settings.

### 6.2 Job queue: `boltz-queue` (entry point → `src/boltz/scripts/studio_queue.py`)
Folder layout under a queue root, which can be a shared folder or a local one:
```
queue/
  inbox/            drop .yaml/.yml jobs here (use `boltz-queue add`, or write via temp name + rename)
  running/<host>/   jobs claimed by that Studio
  done/  failed/    finished inputs; failures also get <name>.error.txt
  results/<name>/   boltz --out_dir for the job, plus summary.json and a DONE marker
  shared/files/     MSA and template files that jobs reference (copied in by add / make-inputs)
  logs/<host>.log
  control/          drain flags: `drain` (all workers) or `drain-<host>`
```

**Claiming jobs**
- A claim is an atomic `os.rename(inbox/x.yaml → running/<host>/x.yaml)`. The worker that loses a race gets
  `FileNotFoundError` and skips the job.
- Only claim files older than `--settle` seconds (default 15), to skip half-copied files.

**File paths**
- Relative `msa` and template `cif`/`pdb` paths in an inbox YAML mean "relative to `inbox/`", matching Boltz's own
  rule of resolving relative to the YAML's folder.
- At claim time the worker writes `results/<name>/input/<name>.yaml` with those paths made absolute against its own
  mount of the queue, so it works even when mount points differ between Studios. `msa: empty` is left alone.
- `add` and `make-inputs` resolve references relative to the source file. Files outside the queue are copied to
  `shared/files/<sha12>_<basename>` (deduplicated by content hash), and the references are rewritten to
  `../shared/files/...`.

**Per-job options**
- Sidecar file `<name>.job.json`, containing `{"boltz_args": [...], "attempt": n}`. It is written before the YAML so a
  claim never sees the YAML without its sidecar.
- `add --seeds 1,2,3` writes `<stem>_seed<k>.yaml` plus a sidecar with `--seed k` for each seed.

**Worker:** `boltz-queue worker QUEUE [--slots N] [--retries 1] [--timeout-hours H] [--use-msa-server]
[--boltz-cmd boltz] -- [extra boltz args]`
- Defaults to `--accelerator mps`.
- **On start:** requeues anything left in `running/<its own host>/` (crash recovery).
- **On SIGINT/SIGTERM:** stops child processes, requeues their jobs and exits.
- **Drain flags** stop new claims, and the worker exits when its current jobs finish.
- If the queue folder isn't mounted yet (for example right after a reboot), it waits and retries.
- Runs `caffeinate -is -w <pid>` on macOS unless `--no-caffeinate` is given.
- **On success:**
  - Parse the top model and affinity outputs into `results/<name>/summary.json`: name, host, start and end times,
    runtime, attempt, arguments, provenance (boltz and torch versions, macOS version, chip), confidence metrics and
    affinity values.
  - Write the `DONE` marker and move the input to `done/`.
- **On failure:** retry up to N times, then move the input to `failed/` with the last log lines in `<name>.error.txt`.
- **Duplicates:** a job whose `results/<name>/DONE` already exists is refused.
- **`--use-msa-server` is opt-in.** It sends sequences to the public ColabFold server, which raises privacy concerns
  and has usage limits. Prefer precomputed or reused MSAs.

**Other subcommands**
- `init`
- `add FILES [--seeds]`
- `make-inputs TEMPLATE.yaml LIGANDS.csv [--queue Q | --out DIR] [--affinity] [--ligand-id L] [--name-col name]
  [--smiles-col smiles]`
- `status`
- `summarize [--out summary.csv]`
- `requeue [--failed] [NAMES]`
- `launchd-plist`: prints a LaunchAgent for the user to install.

**Tests:** use a fake `--boltz-cmd` script that writes Boltz-shaped outputs. Cover claims, races, retries, requeueing,
path rewriting and duplicate names. No torch needed.

### 6.3 Comparison tool: `boltz-compare` (entry point → `src/boltz/scripts/compare_runs.py`)
**Usage:** `boltz-compare --ref A100_DIR --test MAC_DIR [--out report.csv] [--pocket-cutoff 10]
[--seed-suffix '_seed\d+$'] [thresholds]`

**Discovery**
- Every `predictions/<name>/` folder containing `<name>_model_0.cif` or `.pdb` counts as one replicate.
- Replicates are grouped by name after stripping the seed suffix.
- Names missing from either side are reported.

**Metrics per system**
- **Protein:** Cα RMSD after Kabsch superposition. Residues are matched by chain, residue number and residue name.
- **Each ligand chain:** superpose on the pocket Cα atoms, meaning residues within the cutoff of the ligand in the
  reference structure. If fewer than 3 pocket residues are found, use all Cα atoms. Then report ligand heavy-atom RMSD
  two ways:
  - name-matched
  - symmetry-tolerant, using a per-element Hungarian assignment via scipy
- **Confidence:** differences in `confidence_score`, `iptm`, `ligand_iptm`, `complex_plddt` and `complex_iplddt`.
- **Affinity:** differences in `affinity_pred_value` and `affinity_probability_binary`.
- For every metric, report the cross-platform mean and the spread within each platform (which needs at least 2 seeds
  per side). The key question: do cross-platform differences stay within seed-to-seed variation?

**Output**
- A tidy CSV with one row per system × ligand chain.
- A printed summary: medians, the fraction of systems with ligand RMSD under 2 Å, the correlation of affinity values
  across systems, and a list of flagged systems.
- The built-in thresholds are only starting points; the user sets the real tolerances.

**Dependencies and tests:** uses gemmi, numpy and scipy, which are already Boltz dependencies. Test with synthetic
structures: a rotated copy must give RMSD 0, and a known ligand shift must give the expected RMSD.

## 7. Validation protocol

- **Kernel level:** compare each new kernel against the PyTorch operation it replaces, using real activations dumped
  from runs.
  - Cover many sizes, including sizes that aren't multiples of the tile size, and masked inputs.
  - Tolerances: about 1e-5 relative in fp32. Looser for fp16, with explicit bounds.
- **Model level:** same inputs, MSAs and flags, with several seeds per system on both platforms.
  - Ligand pose RMSD, Cα RMSD, confidence differences and affinity differences must stay within seed-to-seed
    variation. This is the same bar as Anthropic's `fast` mode.
- **Science level:** a benchmark set with ground truth, such as the user's systems with crystal structures or assay
  data, supplemented by public sets such as PoseBusters where helpful.
- **Ownership and records:** tolerances are the user's scientific call. Record the versions and settings of every
  validation run.

## 8. Open questions for the user

- How will the Studios share files: a NAS, macOS File Sharing, or SSH?
- Which systems make up the reference set?
- Which Boltz version and flags do the A100 runs use? Did their MSAs come from the public server or from local
  databases?
- What acceptance tolerances apply?
- Do the Studios have 48 or 64 GPU cores?
- Should generally useful fixes, such as the `.gitignore` fix, be sent upstream to Novel-Therapeutics/boltz-community?

## 9. Status log

- **2026-09-27:**
  - Evaluated fnachon/boltz, Novel-Therapeutics/boltz-community and Anthropic's Boltz-2 kit by reading their code,
    tests, history and issues. Nothing was run.
  - Forked the parent as `mbs1234/boltz-community` at `401f181` (v2.10.12).
  - Added `CLAUDE.md` and `HANDOFF.md`, and fixed `.gitignore`.
  - **Next:** Phase 0 on the development Studio.
