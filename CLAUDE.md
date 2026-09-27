# CLAUDE.md

This is **mbs1234/boltz-community**, a public fork of
[Novel-Therapeutics/boltz-community](https://github.com/Novel-Therapeutics/boltz-community), which is itself a
maintained community fork of [jwohlwend/boltz](https://github.com/jwohlwend/boltz) (Boltz-1 / Boltz-2).

**Project goal:** run Boltz-2 protein–ligand co-folding and affinity prediction on the lab's three Apple Silicon
Mac Studios (M1 Ultra, 128 GB each), moving routine work off paid, quota-limited external A100 servers. Success is
measured in trustworthy results per week across the three Studios, not in the speed of a single job.

**Read [HANDOFF.md](HANDOFF.md) before starting any work.** It holds the plan, the findings behind it, the tool specs,
the validation protocol, open questions, and a status log. When you finish a piece of work, update its checkboxes and
add a line to the status log.

## Rules

- **This repo is public. Never commit research data.** That covers real input YAMLs, MSAs, ligand lists and SMILES,
  predictions, and reference results from the A100 servers. Keep data outside the repo (for example `~/boltz-data/`),
  or in `local/`, which is git-ignored. Only synthetic or public test data belongs in `tests/`.
- Work on a feature branch and open pull requests **inside this fork**:
  `gh pr create --repo mbs1234/boltz-community --base main`. Without `--repo`, `gh` can target the parent project.
  Never open PRs or issues on Novel-Therapeutics/boltz-community or jwohlwend/boltz unless the user asks.
- Keep edits to files inherited from upstream small, and put new code in new files where possible, so syncing with
  the parent stays easy: `git fetch upstream && git merge upstream/main`. Add the remote first if it's missing:
  `git remote add upstream https://github.com/Novel-Therapeutics/boltz-community.git`.
- Don't change macOS system, energy, or update settings on the Studios yourself. Recommend them to the user.
- Any speed change needs the validation described in HANDOFF.md §7 before it merges. Report measured numbers, not
  expectations.

## Environment

- Target hardware: M1 Ultra Mac Studios, using the Apple GPU through PyTorch's MPS backend. To check which machine
  you're on, run `sysctl -n machdep.cpu.brand_string` and `system_profiler SPDisplaysDataType | grep Cores`.
- The user's desk machine is an Intel iMac. It can't run Boltz on the GPU, because PyTorch stopped shipping Intel-Mac
  builds after 2.2. Don't use it for benchmarks or MPS tests.
- To set up a Studio, run `scripts/mac/setup_mac_studio.sh`. For development, use
  `BOLTZ_SRC=$PWD bash scripts/mac/setup_mac_studio.sh` to get an editable install.
  Otherwise, use a Python 3.12 venv and `pip install -e ".[test]"`.

## Tools added by this fork (see `docs/mac_studio.md`)

- `boltz-queue` (`src/boltz/scripts/studio_queue.py`): a shared-folder job queue for several Macs, with
  `make-inputs` to turn a target and a ligand CSV into jobs.
- `boltz-compare` (`src/boltz/scripts/compare_runs.py`): compares Mac predictions against A100 references.
- `src/boltz/scripts/prediction_outputs.py`: finds and reads Boltz output folders; both tools use it.
- These were written on the Intel iMac and have **not yet run on a Studio or on real Boltz output**. The checklist in
  HANDOFF.md §6.4 comes first.

## Tests

- CPU suite, the same as upstream CI: `pytest -m "not slow and not regression" -v --tb=short`
- This fork's tools (fast, no torch or GPU needed): `pytest tests/test_compare_runs.py tests/test_studio_queue.py -q`
- Mac GPU tests, run manually on a Studio: `pytest tests/test_mps.py -m mps -v`
- The markers `slow`, `regression`, and `mps` are defined in `pyproject.toml`. New tools should get CPU-runnable
  tests that don't need torch or a GPU (for example, a fake `boltz` command).

## Key code locations

- `src/boltz/main.py` → `predict()`: accelerator handling. MPS forces fp32 and a single device. This is also where
  checkpoint loading and the `pin_memory` choice live.
- `src/boltz/model/layers/triangular_attention/` and `src/boltz/model/layers/triangular_mult.py`: the triangle
  operations, whose cost grows with the cube of complex size. On CUDA they use cuEquivariance kernels
  (`use_kernels`); everywhere else they run as plain PyTorch.
- `src/boltz/model/layers/attentionv2.py` and `pairformer.py`: the optional `--flash_attn` (PyTorch SDPA) path.
- `src/boltz/model/loss/diffusionv2.py` → `weighted_rigid_align`: an SVD and determinant on every diffusion step.
- `src/boltz/model/modules/utils.py` → `autocast_device_type`: the MPS autocast helper.
- `src/boltz/data/parse/schema.py` → `parse_boltz_schema`: resolves relative `msa` and template `cif`/`pdb` paths
  against the input YAML's folder.
- `src/boltz/data/write/writer.py`: output file names (`<name>_model_<k>.cif`, `confidence_<name>_model_<k>.json`,
  `affinity_<id>.json`, …).
- `src/boltz/scripts/`: command-line tools shipped as entry points (see `[project.scripts]` in `pyproject.toml`).
