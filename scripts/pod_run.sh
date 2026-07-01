#!/usr/bin/env bash
# RunPod one-shot setup and verification for IndicF5 streaming (branch: pod).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/IndicF5-streaming}"
BRANCH="${BRANCH:-pod}"
PORT="${PORT:-8000}"
NFE_STEP="${NFE_STEP:-16}"

cd "$REPO_DIR"

echo "=== [1/6] git fetch + checkout $BRANCH ==="
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

echo "=== [2/6] venv ==="
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -U pip

echo "=== [3/6] pin transformers + install deps ==="
pip install 'transformers>=4.40,<4.50'
pip install -e .
pip install fastapi "uvicorn[standard]"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA available — ensuring torch cu124 wheels"
  pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi

echo "=== [4/6] environment checks ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo "=== [5/6] pod_verify (load + synthesize) ==="
python scripts/pod_verify.py --device cuda --nfe-step "$NFE_STEP" --output /tmp/pod_verify.wav

echo "=== [6/6] start server on 0.0.0.0:$PORT ==="
echo "Open RunPod HTTP proxy for port $PORT"
exec python tts_server.py --host 0.0.0.0 --port "$PORT" --device cuda --nfe-step "$NFE_STEP"
