# Plan F Phase 2 Main Training Launch (2026-05-06)

Pre-flight passed. Main training launch status:

- V7: launched on GPU2 via tmux `planf_v7v8_0506`
- V8: launched on GPU3 via tmux `planf_v7v8_0506`
- V6_NOISE: queued via tmux `planf_v6noise_wait_0506`, auto-starts on GPU1 when GPU1 is free
- Steps: 160000 per arm
- Python: `/home/qujiaxiang/.conda/envs/rae/bin/python`

Logs:

- `review/0505/local/runs/V7/train.log`
- `review/0505/local/runs/V8/train.log`
- `review/0505/local/runs/V6_NOISE/train.log`
- `review/0506/operator/logs/v6noise_wait_gpu1_20260506.log`

## Direct GPU1 Launch Update

Per operator request, `V6_NOISE` was started directly on GPU1 via tmux `planf_v6noise_0506`, without waiting for the existing GPU1 process to finish.
