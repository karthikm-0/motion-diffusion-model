#!/usr/bin/env bash
cd /nfs/triton.dgpsrv/local/2024/karthikm/expressive_motion_gen/motion-diffusion-model
while pgrep -f "python3 run_guidance.py|python3 run_e4_seeds.py" >/dev/null; do sleep 20; done
echo "=== starting E2-shape (rerun after misplaced file) ==="
uv run python run_e2_shape.py
echo "=== ALL DONE ==="
