#!/usr/bin/env bash
# Set up Boltz (this fork) for GPU (MPS) predictions on an Apple Silicon Mac.
#
# Usage:
#   bash scripts/mac/setup_mac_studio.sh [--dry-run] [--skip-smoke-test]
#
# Settings (environment variables, all optional):
#   BOLTZ_HOME      folder for the Python environment and setup notes   (default: ~/boltz)
#   BOLTZ_REF       branch, tag or commit of this fork to install        (default: main)
#   BOLTZ_REPO      git URL to install from  (default: https://github.com/mbs1234/boltz-community.git)
#   BOLTZ_SRC       path to a local checkout to install in editable mode instead (for development)
#   BOLTZ_CACHE     where Boltz keeps model weights                      (default: ~/.boltz)
#   PYTHON_VERSION  Python version for the environment                   (default: 3.12)
#
# What it does: installs uv (a Python package manager) if it's missing, creates a
# Python environment, installs Boltz, fixes duplicate OpenMP libraries, checks the
# GPU, runs a small protein + ligand + affinity test on it, and writes the versions
# it used to $BOLTZ_HOME/setup-info.txt. Safe to re-run. It never changes system
# settings.
#
# Use the same BOLTZ_REF (ideally a tag or commit) on every Mac so they all run
# identical code.

set -euo pipefail

DRY_RUN=0
SMOKE_TEST=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-smoke-test) SMOKE_TEST=0 ;;
    -h | --help)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

BOLTZ_HOME="${BOLTZ_HOME:-$HOME/boltz}"
BOLTZ_REF="${BOLTZ_REF:-main}"
BOLTZ_REPO="${BOLTZ_REPO:-https://github.com/mbs1234/boltz-community.git}"
BOLTZ_SRC="${BOLTZ_SRC:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="$BOLTZ_HOME/venv"
PY="$VENV/bin/python"

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# --- This Mac ---------------------------------------------------------------
say "Checking this Mac"
[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS."
if [ "$(uname -m)" != "arm64" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    warn "This Mac isn't Apple Silicon ($(uname -m)); continuing only because of --dry-run."
  else
    die "This Mac isn't Apple Silicon ($(uname -m)). Boltz's Mac GPU support needs an M-series chip."
  fi
fi
macos_version="$(sw_vers -productVersion)"
if [ "${macos_version%%.*}" -lt 14 ]; then
  warn "macOS $macos_version: macOS 14 or newer is recommended for PyTorch's Mac GPU backend."
fi
# git (to install from GitHub) and otool/install_name_tool/codesign (for the
# OpenMP fix) come with Apple's Command Line Tools.
if ! xcode-select -p >/dev/null 2>&1; then
  msg="Apple's Command Line Tools are needed. Install them with: xcode-select --install (then re-run this script)."
  if [ "$DRY_RUN" = 1 ]; then warn "$msg"; else die "$msg"; fi
fi
chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
memory_gb=$(($(sysctl -n hw.memsize) / 1073741824))
gpu_cores="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ {print $2; exit}')"
echo "  $chip, ${gpu_cores:-?}-core GPU, ${memory_gb} GB memory, macOS $macos_version"

# --- uv ---------------------------------------------------------------------
say "Checking for uv (Python package manager)"
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
elif command -v brew >/dev/null 2>&1; then
  echo "  Installing uv with Homebrew"
  run brew install uv
  UV="$(command -v uv || echo uv)"
else
  echo "  Installing uv with its official installer (https://docs.astral.sh/uv/)"
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$HOME/.local/bin/uv"
fi
echo "  using $UV"

# --- Python environment and Boltz --------------------------------------------
say "Python $PYTHON_VERSION environment at $VENV"
run mkdir -p "$BOLTZ_HOME"
if [ -x "$PY" ]; then
  echo "  reusing the existing environment"
else
  run "$UV" venv --python "$PYTHON_VERSION" "$VENV"
fi

if [ -n "$BOLTZ_SRC" ]; then
  say "Installing Boltz from the local checkout $BOLTZ_SRC (editable)"
  run "$UV" pip install --python "$PY" -e "${BOLTZ_SRC}[test]"
else
  say "Installing Boltz from $BOLTZ_REPO at $BOLTZ_REF"
  run "$UV" pip install --python "$PY" --reinstall-package boltz-community \
    "boltz-community[test] @ git+${BOLTZ_REPO}@${BOLTZ_REF}"
fi

say "Fixing duplicate OpenMP libraries (prevents 'OMP: Error #15'; safe to repeat)"
run "$VENV/bin/boltz-fix-macos-libomp" ||
  warn "The OpenMP fix didn't finish. If Boltz later fails with 'OMP: Error #15', run $VENV/bin/boltz-fix-macos-libomp again."

say "Checking the GPU (PyTorch MPS backend)"
run "$PY" -c "import torch; assert torch.backends.mps.is_available(), 'MPS is not available'; print('  PyTorch', torch.__version__, '- Mac GPU available')"

# --- Smoke test ---------------------------------------------------------------
if [ "$SMOKE_TEST" = 1 ]; then
  say "Small protein + ligand + affinity test on the GPU (the first run downloads several GB of model weights)"
  smoke_dir="$BOLTZ_HOME/smoke-test"
  run mkdir -p "$smoke_dir"
  if [ "$DRY_RUN" = 0 ]; then
    # Protein G B1 domain (56 residues) with tyrosine; no MSA, so no network use.
    cat >"$smoke_dir/smoke.yaml" <<'YAML'
version: 1
sequences:
  - protein:
      id: A
      sequence: MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
      msa: empty
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
YAML
  fi
  run "$VENV/bin/boltz" predict "$smoke_dir/smoke.yaml" --out_dir "$smoke_dir" \
    --accelerator mps --override --recycling_steps 1 --sampling_steps 20 \
    --diffusion_samples 1 --sampling_steps_affinity 20 --diffusion_samples_affinity 1
  if [ "$DRY_RUN" = 0 ]; then
    result="$smoke_dir/boltz_results_smoke/predictions/smoke"
    [ -f "$result/smoke_model_0.cif" ] || die "The smoke test wrote no structure; see the output above."
    [ -f "$result/affinity_smoke.json" ] || die "The smoke test wrote no affinity result; see the output above."
    echo "  structure and affinity written to $result"
  fi
fi

# --- Record what was installed -------------------------------------------------
say "Recording setup details in $BOLTZ_HOME/setup-info.txt"
if [ "$DRY_RUN" = 0 ]; then
  {
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "computer: $(scutil --get ComputerName 2>/dev/null || hostname)"
    echo "chip: $chip"
    echo "gpu_cores: ${gpu_cores:-unknown}"
    echo "memory_gb: $memory_gb"
    echo "macos: $macos_version"
    echo "boltz_ref: ${BOLTZ_SRC:-$BOLTZ_REF}"
    "$PY" - <<'PYINFO'
import importlib.metadata as md
import json
import platform

import torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
dist = md.distribution("boltz-community")
print("boltz_community:", dist.version)
direct = json.loads(dist.read_text("direct_url.json") or "{}")
commit = direct.get("vcs_info", {}).get("commit_id")
if commit:
    print("boltz_commit:", commit)
try:
    print("mps_recommended_max_memory_gb:", round(torch.mps.recommended_max_memory() / 2**30, 1))
except Exception:  # not available in every PyTorch version
    pass
PYINFO
  } >"$BOLTZ_HOME/setup-info.txt"
  sed 's/^/  /' "$BOLTZ_HOME/setup-info.txt"
fi

cat <<EOF

Done. To use Boltz in a new Terminal window:
  source "$VENV/bin/activate"
  boltz predict input.yaml --accelerator mps

The job queue, the ligand-CSV helper and the A100 comparison tool are described in
docs/mac_studio.md in the repository.

Recommended (these are yours to set; this script doesn't change them):
  - Hold macOS updates on the Studios until you've re-run this script's smoke test
    on the new version; updates have changed Mac GPU behaviour before.
  - boltz-queue keeps the Mac awake while it works; no energy settings are needed.
EOF
