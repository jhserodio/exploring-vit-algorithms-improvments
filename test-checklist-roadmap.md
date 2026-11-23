# Execution checklist — ViT research engine

**Canonical scientific roadmap:** [roadmap.md](roadmap.md). This file is a concise validation tracker, not a duplicate copy of the full 10-phase roadmap.

Last audited: **2026-11-23**. Target branch: `feat/testing-background`.

## Verified or implemented

- [x] Four CPU unit tests plus Rust build, `cargo fmt`, `cargo clippy` and CLI help passed on [2026-10-08 GitHub Actions](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/37717374470).
- [x] DINOv2 Base model-specific forward and checkpoint loader code is present (not a proven real-checkpoint result).
- [x] Preprocessing fixtures, tensor shape/CHW/dtype and pixel normalization checks are implemented.
- [x] CUDA 12.8.1 / SM 120 **configured** in Docker; actual compatibility not verified.
- [x] `vit-gpu` executes a distinct CUDA inference command; `vit-cpu` retains CPU-only tests.
- [x] Default container avoids `SYS_ADMIN`; shared SELinux relabeling and read-only input mounts documented.
- [x] Synthetic fixtures, model weights and datasets are handled separately; sensitive weights/data are gitignored.

## Open acceptance gates

- [ ] Updated commit passes CPU GitHub Actions (including any new tests).
- [ ] Run actual DINOv2 checkpoint on CPU: output finite with shape `[1,768]`.
- [ ] Run real checkpoint on the RTX 5070 Ti and save environment via `scripts/collect_env.sh`.
- [ ] Independent, identical-normalized-input PyTorch versus Candle comparison passes declared tolerance.
- [ ] Introduce versioned experiment configuration and patient-/slide-safe dataset manifests (Phase 1).
- [ ] Implement model-independent adapter contract, registry and second backbone (Phase 2).
- [ ] Add reference mathematical operation tests and CPU/GPU numerical parity (Phase 3).
- [ ] Evaluate OSCC only on appropriately labeled data and leak-free patient splits (Phase 4).
- [ ] Collect controlled CUDA benchmarks before proposing algorithmic speedups (Phases 5–7).

## Where to look

- [PHASE0.md](PHASE0.md): commands, SELinux notes, model weights, CPU and GPU evidence.
- [roadmap.md](roadmap.md): all scientific goals, phases 0–9, and sprint S01–S10.
- [GitHub Actions](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions): CI history.

**Rule:** `[x]` indicates that the *specific stated subtask* has been implemented or has explicit verification evidence. It does not mean the parent phase has reached its exit gate.
