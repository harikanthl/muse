# Prior art: muse_bf16_proj (skinny BF16 GEMM)

Research agent report, collected 2026-10-31 (web sources, verified unless flagged inside). Part of the Open Kernel Challenge notes; see ../moe-field-guide.md for the synthesis.

## Prior-art findings: skinny BF16 GEMM on H200 (verified unless flagged)

**Q1. Why cuBLAS loses at small M / what beats it**
- ONNX Runtime PR #31478 (H200, SM90): for M≤8, N≤1024 cuBLAS "picks a 1x1 CTA grid" and spends ~5 µs on a few hundred KiB of weights. Their split-K GEMV (grid = N/32 × splitK, FP32 partials to workspace, `__threadfence` + atomic counter, **last CTA reduces in fixed order inside the same kernel**) hit 2.1 µs standalone but 10.6 µs inside the real model — author blames cold weights/L2. Directly relevant: your L2 is cold. https://github.com/microsoft/onnxruntime/pull/31478
- Stream-K paper (A100): for small, bandwidth-bound shapes its large blocking "does not compete well against cuBLAS". https://arxiv.org/pdf/2301.03598
- TRT-LLM `tinygemm2` (source read: `cpp/tensorrt_llm/kernels/tinygemm2/`): 384 threads (8 TMA-DMA warps + 4 MMA warps), `__launch_bounds__(384,1)`, TILE 16 weight-rows × 8 batch × 64 K, **16 stages**, `cp.async.bulk.tensor.2d` with `SWIZZLE_128B`, `mma.sync.m16n8k16` + ldmatrix, grid (N/16, ceil(M/8)), PDL attribute set, no split-K, plain row-major W (no pre-layout). FlashInfer ships it as `tinygemm_bf16` (M ideally 1–8). https://docs.flashinfer.ai/generated/flashinfer.gemm.tinygemm_bf16.html
- TRT-LLM `cudaCoreGemm` (`kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu`): pure FMA, templated TILE_M = M for **M ≤ 16**, TILE_N=2, 128 threads, 128-bit loads, each thread strides K, warp-shuffle + smem reduction, grid = N/2 (no split-K). FP8 and 16F inputs confirmed; bf16 branch not checked.
- Achieved-bandwidth datapoints: Hazy megakernel 78% of H100 HBM at batch 1 vs ≤50% for vLLM/SGLang (https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); Cohere megakernel 62% SoL (https://cohere.com/blog/megakernels); STREAM copy on H100 SXM 3067/3352 GB/s = 91% (https://docs.nvidia.com/gh200-superchip-benchmark-guide.pdf); H800 plain loads 91% of peak, TMA ≈ same bandwidth as ld.global with +170 cycles latency (https://arxiv.org/html/2501.12084v2). No published per-M % for M=1/16/32 BF16 GEMM on H100 — **unverified**.
- Load-depth finding (H100 SXM5): plain-load bandwidth peaks at ~2 outstanding 16B loads/thread and falls ~35% at 8; cp.async reaches 2.1–2.9× plain-load throughput at high depth. https://arxiv.org/abs/2608.15764

**Q2. Weight pre-layout**
- Marlin: offline repack into 16×64 tiles, `64207531` interleave, so one 16B load per thread feeds `mma.sync` with no index math (A10 kernel speedup ~3.87× at M≤32 → 1.5× at M=128). https://arxiv.org/html/2408.11743v1
- Machete: pre-shuffle via CuTe because ldmatrix has no 4-bit form; sub-32 batch left as "future work". https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
- FlashInfer `interleave_for_tma_wgmma_up`: reorders global rows so one 128×128 SWIZZLE_128B TMA box = one wgmma tile; TMA does the XOR swizzle itself. For 16-bit W, TMA already yields the wgmma layout from row-major — **no source quantifies a gain from pre-shuffling bf16 weights**.

**Q3. M=1..8: CUDA cores vs tensor cores**
- Industry does both: TRT-LLM uses FMA (`cudaCoreGemm`) and tensor cores (`tinygemm2`) at tiny M. "Can Tensor Cores Benefit Memory-Bound Kernels? (No!)" bounds the gain at ≤1.33× — FP64 only. https://arxiv.org/abs/2502.16851
- A10G hand GEMV: 84.7% DRAM with 128-bit loads, cuBLAS ~95% at big N, 1.36× win at small N. https://kyrieblunders.bearblog.dev/the-kernels-that-were-there/
- Split-K vs single pass at M=1 on H100: no verified measurement; ORT's result cuts both ways.

**Q4. FP32 router**
- Henry/Tang/Heinecke: 2-term split = 16 mantissa bits, 3 products; measured accuracy order bf16x2(3 products) < SGEMM < bf16x3(6 products); dropped-term average ≈4.4e-9·|a0b0| but worst-case needs 7 products. https://ar5iv.labs.arxiv.org/html/1904.06376 → 2-term/3-product gives ~1.5e-5 per-element relative error, comfortably under 1e-4 rel-L2 (as the baseline already proves).
- cuBLAS BF16x9: compute capability **10.0/10.3 only**, needs Blackwell scaling hardware; not Hopper. https://docs.nvidia.com/cuda/cublas/index.html ; https://arxiv.org/html/2605.16617v1 (BF16x6 moves the same bytes as x9).
- Bytes: hi+lo = 4 B/elem = FP32; floor stays 1.8 µs. Compression: DFloat11 exponent entropy ≈2.6/8 bits (≈17% of FP32), decode kernels reach 76.5% of peak BW (L40S) and ~2× slower at batch 1 — net loss on 8 MB. https://arxiv.org/pdf/2504.11651 ; https://arxiv.org/pdf/2603.17435

**Q5. Cold L2 / ordering**
- H100 L2 = 2 partitions; far hit ≈1.6–2× near latency (H800: 258 vs 414 cycles); 2 GPCs per L2 sub-NUMA node; advice "pin data local to SMs" but **no bandwidth measurement** for streaming. https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth ; https://www.alphaxiv.org/abs/2607.19922
- VRAM channel hash is non-XOR, not derivable analytically (SGDRC, https://arxiv.org/pdf/2407.13996). Partition camping on H200 HBM3e: **unverified, no source**.

**Q6. Overheads**
- H100 dummy launch: 2.1 µs stream, 1.3 µs CUDA graph (Hazy). PDL hides launch steps but `cudaGridDependencySynchronize` is coarse. NVIDIA (Greg): graphs remove steps 1–2, graphs+PDL "completely eliminate dependent launch latency" — no numbers. https://forums.developer.nvidia.com/t/kernel-switch-latency-successive-kernels-switch-latency/309504

## Top 5 actionable ideas
1. One kernel, one wave: tinygemm2-style 1 CTA/SM, 384 threads, 16-stage TMA pipeline (~64 KB+ in flight/SM); use in-kernel last-CTA split-K reduction (ORT pattern) only where N/16 < ~264 CTAs (o-proj N=1024, router N=256, unpooler M≤8).
2. M≤16: FMA path with 128-bit loads but ≤2–4 outstanding loads/thread or cp.async/TMA (per 2608.15764); tensor cores buy nothing.
3. Router: pre-split W into interleaved hi/lo in prepare(); FP32 FMA for M≤8, 3-product bf16 tensor-core for M>8. Don't chase <4 B/elem.
4. prepare(): tile W so each CTA's K-strip is one contiguous, 128B-swizzle-ready region → 1D `cp.async.bulk`, sequential DRAM, zero shuffles (gain plausible, unquantified).
5. PDL + graph: set `programmaticStreamSerialization`, issue weight TMA before `griddepcontrol.wait` (CUTLASS ex. 63); never add a second reduce kernel (≈1.3 µs ≈ 10–35% of a 3.5–14 µs call).

## Dead ends
cuBLAS BF16x9 (Blackwell only); Stream-K; lossless compression of router; bf16 weight pre-shuffle for wgmma layout (TMA already does it); L2-reuse rasterization (weights read once); expecting >~91% of peak (realistic floor ≈4.3–4.4 TB/s on H200 — H200 figure itself unverified).
