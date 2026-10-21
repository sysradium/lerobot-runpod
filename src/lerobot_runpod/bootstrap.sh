# Runs inside the pod (as root), shipped base64-encoded in $BOOT. Parameters
# arrive as env vars: HF_TOKEN DATASET POLICY_ARGS REPO STEPS BATCH SAVE_FREQ LOG_FREQ
# MAX_HOURS DEADLINE FAIL_GRACE_S LEROBOT_REF EXTRAS.
# Markers (@@...@@) drive the local log follower.

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# RunPod injects a pod-scoped RUNPOD_API_KEY; it may terminate its own pod via
# GraphQL (the REST endpoint answers 403 for it).
self_destruct() {
  echo "@@SELFDESTRUCT $1@@"
  curl -s -X POST -H "Content-Type: application/json" \
    "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" >/dev/null
}

# Never let the script exit on its own: RunPod restarts an exited container,
# which would rerun the whole bootstrap and bill for it again.
fail() {
  echo "@@FAILED $1@@"
  echo "pod stays up ${FAIL_GRACE_S}s for debugging, then deletes itself"
  sleep "$FAIL_GRACE_S"
  self_destruct failed
  sleep infinity
  exit 1
}

# DEADLINE is absolute (set when the pod was created), so a container restart
# can't extend it.
[ -n "$RUNPOD_API_KEY" ] || echo "@@WARN no RUNPOD_API_KEY: this pod cannot delete itself@@"
(
  left=$((DEADLINE - $(date +%s)))
  [ "$left" -gt 0 ] && sleep "$left"
  echo "@@FAILED timeout: hit the ${MAX_HOURS}h limit@@"
  self_destruct timeout
) &

echo "@@STAGE install@@"
# torchcodec must match the image's torch 2.8
{ python -m pip install -q uv &&
  uv pip install --system --break-system-packages -q \
    "lerobot[$EXTRAS] @ git+https://github.com/huggingface/lerobot@$LEROBOT_REF" "torchcodec==0.7.*"; } ||
  fail "install"
# torchcodec needs system ffmpeg libs; without them lerobot falls back to slower pyav
if ! ldconfig -p | grep -q libavutil; then
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null; } ||
    echo "@@WARN ffmpeg install failed, video decoding falls back to pyav@@"
fi

echo "@@STAGE env@@"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# lerobot silently falls back to CPU, which would burn GPU-hours for nothing
python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda); torch.zeros(1).cuda()" ||
  fail "cuda unavailable on this host"

echo "@@STAGE download@@"
hf download --repo-type dataset "$DATASET" --local-dir "$HF_LEROBOT_HOME/$DATASET" >/dev/null ||
  fail "download"

echo "@@STAGE train@@"
# video decoding becomes the bottleneck on fast GPUs with the default 4 workers
workers=$(($(nproc) - 1))
[ "$workers" -gt 12 ] && workers=12
echo "dataloader workers: $workers (of $(nproc) cpus)"
# bf16 autocast (fp32 master weights, so the saved model is unchanged). Newer lerobot
# takes --accelerator.mixed_precision; 0.6.0 hands Accelerate mixed_precision=None for
# policies without a dtype field (e.g. ACT), and Accelerate then reads this env var.
precision=()
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)"; then
  echo "precision: fp32 (GPU has no bf16)"
elif lerobot-train --help 2>/dev/null | grep -q -- "--accelerator.mixed_precision"; then
  precision=(--accelerator.mixed_precision=bf16)
  echo "precision: bf16 (--accelerator.mixed_precision)"
else
  export ACCELERATE_MIXED_PRECISION=bf16
  echo "precision: bf16 (ACCELERATE_MIXED_PRECISION)"
fi
lerobot-train \
  "${precision[@]}" \
  --num_workers="$workers" \
  --dataset.repo_id="$DATASET" \
  $POLICY_ARGS \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id="$REPO" \
  --policy.private=true \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="${LOG_FREQ:-100}" \
  --output_dir="$HOME/out/run" ||
  fail "train rc=$?"
echo "@@DONE@@"
# the model is on the Hub now; give the local follower time to see @@DONE@@
sleep 120
self_destruct done
sleep infinity
