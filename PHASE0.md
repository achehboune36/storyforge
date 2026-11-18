# Phase 0 — Benchmark results

Run 2026-09-18/19 on RTX 3060 12 GB, WSL2 (24 GB RAM, 16 GB swap).
Raw data in `bench/results/`. Decisions live in `config.json`.

## Decisions

| Question | Answer | From |
|---|---|---|
| Image model | **Klein 4B Q8_0** + Qwen3-4B | E1/E2 |
| Resolution | 768×768 (24 s/image) | E4 |
| Steps | 20 — tunable to 12 (~15 s) pending visual check | E3 |
| LLM | `llama3.2:3b`, `num_ctx` **4096** | E6 |
| GPU strategy | **Asymmetric** — ComfyUI resident, LLM cycles | E5–E9 |

## Headline numbers

```
Klein 9B Q6  1024²  100.8s  4.94 s/it  11412 MiB
Klein 4B Q6  1024²   43.0s  2.08 s/it  10459 MiB
Klein 4B Q8  1024²   35.1s  1.76 s/it  10415 MiB
Klein 4B Q8  768²    21.9s  1.03 s/it   9157 MiB   <- chosen
```

Steps scale linearly (~1 s fixed + 1.7 s/step). Resolution cost tracks megapixels.
Previews fired on **24/24** runs.

```
Ollama evict     0.30 s     ComfyUI /free      1.01 s
Ollama load      ~4 s       ComfyUI reload    ~25 s
llama3.2:3b   2546 MiB @4k ctx / 3079 MiB @8k ctx
              124 chunk/s solo, 44 co-resident
```

## Three findings that changed the plan

**1. Q8_0 beats Q6_K on every axis** — 18% faster, better quality, lower peak VRAM,
despite being a 0.9 GB larger file. The workload is compute-bound and Q8_0's flat
8-bit layout dequantises far more cheaply than Q6_K's multi-scale k-quant blocks.
The plan had recommended Q6 on file-size reasoning. That reasoning was wrong.

**2. ComfyUI never releases VRAM on idle** — it holds 9519 MiB between generations
against an 11007 MiB working peak. The hoped-for "idles at ~6 GB" was false, which
is why `num_ctx` had to drop to 4096 to buy headroom.

**3. The binary COEXIST/SWAP framing was wrong.** The two handoffs differ by ~80×
in cost, so the answer is neither: keep the expensive side (ComfyUI) resident and
cycle the cheap side (Ollama). ~4.3 s per turn instead of ~30 s.

## Harness bugs found (and fixed)

- **Cache hits.** ComfyUI serves an identical workflow instantly from cache, which
  looks exactly like a 0 s generation. Bit E5 and then E7. Seeds must vary.
- **Double-counted steps.** Both `progress` and `progress_state` fire per step, so
  counting frames halved `s/it`. Now derived from sampler step *values*.
- **CSV column shift** from appending new-format rows under an old header.
  Display-only; per-run sampler files were always correct.

## Not yet measured

- **E8** (mid-stream kickoff vs serial) — moot under the asymmetric strategy, since
  the LLM is evicted before images start. Revisit only if we move to co-residency.
- Whether 12 steps holds quality. Images at 8/12/16/20 with seed 777 are in
  `ComfyUI/output/` awaiting a human eye.
