# Concept list: what to learn, in order

Written 2026-10-31 during the Open Kernel Challenge kernel track. Leaderboard figures are from that evening. See ../moe-field-guide.md for the synthesis.

The four research agents are running. Here's the concept list to work through now, in the order you'll need them. Each line says what to learn and the fastest practical way to get it.

## Tier 1: the mental model (do today, ~2 hours)

1. **Roofline and arithmetic intensity.** FLOPs per byte vs. the machine's ridge point (H200: 989 TFLOP/s ÷ 4.8 TB/s ≈ 206). Every problem here sits left of the ridge, so time = bytes ÷ bandwidth. Practice: compute the floor for each shape in `TASK.md` by hand and check against the table.
2. **Little's law for memory.** Bandwidth × latency = bytes that must be in flight (~3.8 MB chip-wide, ~29 KB per SM). This is why pipelining and occupancy matter more than clever math.
3. **The GPU execution hierarchy.** Thread → warp (32) → CTA/block → SM (132 of them) → grid. Registers → shared memory (228 KB/SM) → L2 (50 MB) → HBM. Know the latency of each level roughly (registers ~1 cycle, smem ~30, L2 ~250, HBM ~600–800).
4. **Coalescing and sectors.** A warp's loads are served in 32-byte sectors; 128-byte lines. Scattered 64-byte reads waste half a line. PMPP ch. 6.
5. **What a kernel launch costs.** ~2–4 µs each, even in a CUDA graph. Count the kernels in SGLang's profile for bs=1 (16 kernels, 80 µs, 11 µs floor). Fusion is the first optimization, not the last.

## Tier 2: the number formats (1 hour)

6. **BF16, FP32, FP8 E4M3, E2M1 bit layouts.** Draw them. Know that BF16 = FP32 with the low 16 mantissa bits cut off, so FP32 = hi(BF16) + lo(BF16) nearly exactly.
7. **NVFP4 as a three-level scale.** value = e2m1 × fp8_scale(per 16) × fp32_scale(per expert). 0.5625 bytes per weight. Why dequant to BF16 is exact (2-bit × 4-bit significand fits in 8).
8. **Relative L2 error.** How the judge measures correctness, and why 1.5% is generous for the MoE but 0.01% is tight for the router.

## Tier 3: the kernel-writing toolkit (2 days, hands on)

9. **Triton basics.** `tl.program_id`, `tl.load/store` with masks, `tl.dot`, `tl.arange`, block pointers, `num_stages`/`num_warps`. Write one GEMV and one tiled GEMM. Triton is what you'll ship first.
10. **CUDA graphs and capturability.** No `.item()`, no `.cpu()`, no data-dependent shapes or allocations. Learn to pre-allocate buffers in `prepare()` and index into them.
11. **Split-K and Stream-K.** How to make one 28 MB matrix occupy 132 SMs when M=1: slice K, accumulate partials in FP32, reduce. Stream-K balances the slices.
12. **Tensor core shape rules.** Hopper `wgmma`: M is fixed at 64, N from 8 to 256, K=16 per instruction (BF16). This is why "put tokens on N, weights on M" (MonoMoE's trick) avoids padding at small batch.
13. **Software pipelining.** Multi-stage buffers: load tile k+2 while computing tile k. `cp.async` and TMA (`cp.async.bulk`). Triton's `num_stages` does this for you.
14. **Bit tricks for dequant.** `lop3` (any 3-input boolean), `prmt` (byte permute), shifting a nibble into a BF16's exponent field. Read Marlin's `dequant.h`.
15. **Online softmax and split-KV.** The flash-decoding pattern: each CTA handles a slice of keys, keeps (max, sum, acc), then merge with log-sum-exp. Needed for the attention problem and the decode layer.

## Tier 4: the measurement discipline (ongoing)

16. **Reading the judge's profile.** `profiles/*.txt` names each kernel and its share. Kill the small ones first.
17. **Paired-ratio timing and confidence intervals.** Gains inside ±1.5% are noise. Don't chase them.
18. **Using `prepare()` as a lab.** 300 untimed seconds on the real H200, with `print()` returned in the log. Benchmark many configs in one check run.

## Where to read each

| Concepts | Source |
|---|---|
| 1–5 | PMPP 4th ed. ch. 4–6 (`~/Documents/Books/pmpp.pdf`); the roofline section of the briefing page |
| 6–8 | `muse_moe_w4a16/nvfp4.py` and `checks.py` (read the code, it's short) |
| 9, 13 | Triton tutorials 01–03 (vector add, fused softmax, matmul), ~2 hours total |
| 10 | `baseline.py`: see how it pre-allocates `cache1/cache2/cache3` |
| 11–12 | Colfax Hopper GEMM paper (link on the briefing page) |
| 14 | Marlin `dequant.h` and the bit diagram on the briefing page |
| 15 | Flash-decoding blog post (PyTorch/Tri Dao, 2023), then FA3 paper section on FP8 |

Skip for now: CUTLASS templates, CuTe layouts, PTX inline assembly, warp specialization. Those are week-2 tools, and Triton gets you to ~2.3× without them.
