#!/usr/bin/env bash
set -euo pipefail

# Repository dir
REPO_DIR="${1:-$(pwd)}"

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
PIP_BIN="$REPO_DIR/.venv/bin/pip"
ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
  else
    echo "Error: Python/pip not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Run sync env from $REPO_DIR
echo "cd $REPO_DIR && uv sync"
cd "$REPO_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync

# Determine Python major.minor version from the env
echo ""$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Determine the torch version/CUDA build actually resolved by uv, so the
# conda env mirrors it instead of installing a different, stale pin.
TORCH_VERSION_FULL=$("$PY_BIN" -c 'import torch; print(torch.__version__)')
TORCH_VERSION="${TORCH_VERSION_FULL%%+*}"
TORCH_CUDA_TAG="${TORCH_VERSION_FULL#*+}"
if [ "$TORCH_CUDA_TAG" = "$TORCH_VERSION_FULL" ]; then
  TORCH_CUDA_TAG="cpu"
fi
echo "Detected torch=$TORCH_VERSION (index tag: $TORCH_CUDA_TAG)"

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --format freeze"
PIP_FREEZE=$(uv pip list --format freeze)

# Remove problematic packages — torch will pull in the correct
# nvidia-*/cuda versions itself, pinning them causes solver conflicts
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch==' | grep -vE '^(nvidia-|cuda-toolkit==|cuda-bindings==|cuda-pathfinder==|triton==)')

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

# Write conda YAML
{
  echo "name: leworldmodel"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - pip:"
  # ---- PyTorch (matches the build resolved by uv) ----
  echo "    - \"--index-url=https://download.pytorch.org/whl/$TORCH_CUDA_TAG\""
  echo '    - "--extra-index-url=https://pypi.org/simple"'
  echo "    - \"torch==$TORCH_VERSION\""

  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no additional pip packages detected"
  else
    while IFS= read -r pkg; do
      [ -z "$pkg" ] && continue
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

# Deactivate current uv env
echo "deactivate"
deactivate || true