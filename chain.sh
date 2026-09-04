#!/usr/bin/env bash
cd /nfs/triton.dgpsrv/local/2024/karthikm/expressive_motion_gen/motion-diffusion-model
while kill -0 3363992 2>/dev/null; do sleep 20; done
echo "=== main run finished, starting E2-shape ===" 
uv run python run_e2_shape.py
echo "=== starting guidance sweep ==="
uv run python run_guidance.py
echo "=== ALL EXPERIMENTS COMPLETE ==="
