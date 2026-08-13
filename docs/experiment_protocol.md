# Experiment protocol

1. Phase 0: unit/integration tests + smoke-test.
2. Phase 1: `overfit-small` on fixed crops.
3. Phase 2A: architecture screen (attn / f / z) on HQ-only.
4. Phase 3: single-seed S1 train (`s1_hq_f8z4.yaml` with real manifest).
5. Phase 4: three seeds.
6. Only then registration audit + optional LR probe (S2).

Never open test until freeze-candidate credentials exist.
