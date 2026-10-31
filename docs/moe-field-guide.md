# The Muse kernel field guide

A study guide and playbook for the Open Kernel Challenge (Meta's Muse Spark Open on NVIDIA H200), written 2026-10-30 from five research sweeps. It is organised so you can read the first three sections in an evening and then use the rest as a reference while writing kernels.

Raw sources, with URLs, live in `research/`. Everything below that is a number was either measured by the judge, read from a primary source, or is marked *derived*.

---

## 0. Where things stand

| Item | Status |
|---|---|
| Kernel track | Open. Closes **Nov 4, 11:59 pm PT**. No prize money. Top 50 on the overall kernel board get first place in the serving-run queue. |
| Serving track | **Nov 5 noon PT to Nov 19.** This is where the $100K is: $50K + a signed H200 for 1st, $30K 2nd, $10K 3rd, smaller prizes to 10th. 8×H200 node, 90-minute runs, 1 scored + 3 check runs per day. |
| Compute | None needed locally. Every run executes on the organisers' H200s. Per problem per day: 10 scored runs, 20 check runs, reset 00:00 UTC. |
| Our runs so far | 1 check run on `muse_moe_w4a16` (the baseline, to get a profile). 19 checks and 10 scored left today on it; full allowance on the other four. |
| Repo | `muse_moe_w4a16/` is the `okc init` output plus the baseline profile. `submission.py` is still the plain PyTorch version. |

### The five problems

| Problem | What it is | Floor is set by |
|---|---|---|
| `muse_bf16_proj` | Dense projections: qkv_gate 2304×8192, o 8192×1024, pooler 4096×8192, unpooler 8192×4096, router 256×8192 FP32. M from 1 to 256. Baseline cuBLAS. | Weight bytes ÷ 4.8 TB/s. Pooler 67 MB → 14 µs; o-proj 16.8 MB → 3.5 µs; router 8.4 MB → 1.8 µs. |
| `muse_attention_decode` | GQA decode, 16 q heads / 2 KV heads per GPU, head_dim 64, FP8 E4M3 paged KV with page_size 1, ~80K context, q_len 1 or 8. Sliding-window (2048, RoPE) and full-attention layers. Fused QK-norm, cache write, output norm × sigmoid(gate). Baseline FA3 + Triton norms. | Full attention at batch 32 reads ~655 MB → ~126 µs. Sliding window reads 2048 tokens → ~3 µs, so it is launch-bound. |
| `muse_moe_w4a16` | Top-8 of 256 experts, 32 local, width 4096, SwiGLU, NVFP4 weights, RMSNorm on each expert output. Baseline SGLang Marlin. | Touched experts × 28.3 MB ÷ 4.8 TB/s ≈ 5.9 µs per expert. |
| `muse_decode_layer` | Everything above plus every norm and residual, one layer, TP8/EP8. | Sum of the parts, minus whatever fusion removes. |
| `muse_prefill_layer` | Same layer at prefill shapes: 753 new tokens on 80K context ×4, and 16K-token chunks. Dominated by one big grouped NVFP4 GEMM. | Compute-bound for the MoE GEMM. Different game. |

### Leaderboard snapshot (evening of Oct 30)

| Problem | Entries | #1 | #50 | #200 | Median |
|---|---|---|---|---|---|
| muse_moe_w4a16 | 1212 | 2.76× | 2.43× | 2.34× | 1.81× |
| muse_attention_decode | 1158 | 3.22× | 2.69× | 2.59× | 1.79× |
| muse_decode_layer | 1077 | 2.42× | 2.24× | 2.15× | 1.65× |
| muse_bf16_proj | 1507 | 1.65× | 1.57× | 1.54× | 1.24× |
| muse_prefill_layer | 721 | 2.03× | 1.82× | 1.73× | 1.49× |

**Confirmed from the rules page.** Only four problems count on the overall board: bf16_proj, moe_w4a16, attention_decode and decode_layer. Prefill has its own board and does not count. The board is computed server-side; the public API shows only rank and problems solved. On Oct 30, 45 of the top 50 had all four solved, and one team at #14 had only one.

**Strategy that follows from the numbers.** The overall board ranks a combined decode-step speedup across problems, and the top entries have submissions on four problems. One great MoE kernel does not get into the top 50; coverage does. Realistic 5-day target: ~2.3–2.4× on MoE, ~1.55× on bf16_proj, plus entries on attention and decode_layer. The MoE kernel is also most of `muse_decode_layer`, and it is exactly what you would patch into SGLang for the serving track.

**Integrity rules worth repeating.** The organisers will reread and rerun the top entries on held-out data. Disqualifying: tampering with timing, work outside the timed call, caching outputs, lower precision than allowed, tuning to synthetic data. Allowed: lossless repacking in `prepare()`, dequantizing inside kernels, fusing anything, tuning configs.

---

## 1. The physics that decides everything

Every decode-shape problem here sits far to the left of the roofline ridge. Time equals bytes divided by bandwidth, and the only questions are how close to peak bandwidth you stream and how much fixed overhead you pay.

### 1.1 Roofline

- H200: 4.8 TB/s HBM3e, 989 TFLOP/s dense BF16, 132 SMs, 50 MB L2, 228 KB shared memory per SM.
- Ridge point: 989 ÷ 4.8 ≈ **206 FLOP per byte**.
- One MoE expert: 3 × 4096 × 4096 weights × 0.5625 B ≈ 28.3 MB. With m tokens, arithmetic intensity ≈ 3.6·m FLOP/B. You would need **m ≈ 58 tokens per expert** before compute mattered. Scored shapes give 1 to ~30.
- Marlin, the baseline, reaches ~1.5 TB/s here. That is the entire 3.5–7× gap between SGLang and the floor.

### 1.2 What "peak" really means

- STREAM-style reads reach 91–94% of nominal on H100 SXM5 and H800, and NVIDIA's own GH200 guide shows 91–93%. Treat **~92%, about 4.4 TB/s on H200**, as the practical ceiling. The floors in `TASK.md` are therefore ~10% below what anyone can reach.
- H200 memory: 6 HBM3e stacks × 16 channels × 2 pseudo-channels = **192 pseudo-channels** on a 6144-bit bus. Each pseudo-channel is 32-bit, burst length 8, so the atom is **32 bytes**, the same as an L2 sector. *Derived:* ≈25 GB/s per pseudo-channel; a kernel has to keep all 192 busy. The address-to-channel hash is undocumented and non-XOR, so "partition camping" tuning has no public basis on Hopper. The only safe guidance is contiguous streams, all SMs active.
- Memory clock is static (3201 MHz) at any power cap. The SM clock holds 1980 MHz for memory-bound work unless capped. H200 needs ~550 W average to sustain peak HBM traffic; memory alone draws ~220 W. If a profile ever shows ~130 W, the kernel is not saturating HBM.

### 1.3 Little's law: bytes in flight

Bandwidth × latency = bytes that must be outstanding at all times.

- H100 unloaded HBM latency ≈ 353 ns; loaded, 600–800 ns (H800 measured ~656 cycles; near/far L2 miss 555/744 cycles).
- *Derived:* 4.8 TB/s × 350–700 ns = **1.7–3.4 MB device-wide, 13–26 KB per SM**, every moment.
- **The load-depth trap (H100 SXM5, arXiv 2608.15764):** plain `ld.global` bandwidth peaks at ~2 outstanding 16-byte loads per thread (3.1 TB/s) and **falls ~35% at 8 per thread** (2.0 TB/s) with the same DRAM bytes. `cp.async` sustains 2.1–2.9× the matched plain-load throughput at depth. TMA reaches ~88% on H800 with 1–4 CTAs/SM but adds ~170 cycles of latency.
- Lesson: deep pipelines use **asynchronous copies into shared memory**, not more unrolled register loads. Aim for 2–3 outstanding 8–16 KB bulk copies per SM, and at most 2–4 plain loads per thread elsewhere. TRT-LLM's `tinygemm2` (16 TMA stages, ~64 KB in flight per SM) is this rule made concrete.

### 1.4 The ALU budget

At full bandwidth each SM consumes 18.4 B per clock (4.8 TB/s ÷ 132 ÷ 1.98 GHz). A Hopper SM has 128 FP32 lanes per clock.

- *Derived:* **~7 ALU ops per streamed byte, ~3.5 per NVFP4 value.**
- Dequantization fits if it is ~1–2 ops per value (section 2.3). Entropy decoding at several integer ops per byte is marginal; Doubleword's fused tANS work reaches the same conclusion, and ZipServ's fixed-length coding lost to cuBLAS on A100/H800 where bandwidth is abundant.
- Extra ALU work is free in power terms until the chip nears 700 W.

### 1.5 Fixed costs

- Kernel launch on H100: ~2.1 µs on a stream, **~1.3 µs inside a CUDA graph**. At bs=1 the MoE floor is 11.4 µs and SGLang runs 16 kernels; launches alone are roughly 2× the floor. On `bf16_proj`, a second reduce kernel costs 10–35% of a 3.5–14 µs call.
- Programmatic dependent launch (PDL, sm_90+): moves the next kernel's start from ~1 µs after the previous one ends to ~0.25 µs before. Triton supports it: `launch_pdl=True` on the host, `tl.extra.cuda.gdc_wait()` / `gdc_launch_dependents()` in the kernel. CUTLASS uses `griddepcontrol`. Issue weight loads before the wait.
- Graph replay fixed cost (~4 µs) is subtracted from both sides by the judge.
- L2 is **cold** before every replay. H200's 50 MB L2 could otherwise hold an entire expert; it never will.
- Megakernel results for calibration: Hazy's Llama-1B kernel reached 78% of H100 bandwidth at batch 1 vs ≤50% for vLLM/SGLang; Cohere's MoE megakernel 62% of speed-of-light at batch 1 vs 39% for vLLM. Both rank wave quantization, false dependencies and weight prefetch above raw launch overhead.

### 1.6 Coalescing and sectors

- L2 line 128 B, sector 32 B. A warp's loads are served per sector.
- Random 64-byte reads (the attention problem's page_size=1 KV rows) waste at least half a line's worth of fetch granularity in the worst case. **Nobody has published random-64B gather bandwidth for H100/H200.** It has to be measured; one check run with a microbenchmark inside `prepare()` does it.

---

## 2. Number formats

### 2.1 Bit layouts

| Format | Sign | Exponent | Mantissa | Note |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | |
| BF16 | 1 | 8 | 7 | FP32 with the low 16 mantissa bits cut off |
| FP8 E4M3 | 1 | 4 | 3 | NVFP4 block scales, FP8 KV cache |
| E2M1 (FP4) | 1 | 2 | 1 | Magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} only |

### 2.2 NVFP4 is a three-level scale

value = e2m1 × fp8_scale (one per 16 elements) × fp32_scale (one per expert, or per tensor).

- 0.5 B per element + 1/16 B of scale = **0.5625 B per element**. Scales are 12.5% of the bytes.
- Hopper has no FP4 tensor cores (Blackwell added `cvt.e2m1x2` and FP4 MMA). On Hopper you dequantize to BF16 in registers and run BF16 MMA. vLLM serves NVFP4 on Hopper this way through Marlin.
- The dequant is **exact**: E2M1 has 2 significant bits, E4M3 has 4, the product has at most 6, and BF16 holds 8. So "dequantize inside the kernel" is lossless and rule-safe.

### 2.3 The dequant bit trick

E2M1 is a bit-prefix of BF16. Shift the 3 magnitude bits into BF16 bits 8:6 and OR the sign into bit 15. The result is exactly e2m1 × 2⁻¹²⁶, including 0 and 0.5 (subnormal handling comes out right). Fold the 2¹²⁶ into the FP8 block scale in `prepare()`, and whatever does not fit there into the per-expert FP32 scale applied once at the end.

- Cost: one `lop3` (any 3-input boolean) plus one `hfma2` per **pair** of values, operating on 2 BF16 lanes per 32-bit register. That is well inside the 3.5-ops-per-value budget.
- `prmt` (byte permute) with its 3-bit selectors is also an exact 8-entry lookup table for the magnitude bits, 4 lookups per instruction.
- Repack the nibbles in `prepare()` so the low and high nibbles of each byte belong to K positions that line up with the MMA fragment, and no interleave shuffle is needed in the hot loop. Marlin's 16×64 tile repack with the `64207531` interleave is the precedent; Microsoft's [e0,e2,e4,e6,e1,e3,e5,e7] ordering saves logic ops the same way.
- Reference points: Marlin's `dequant.h` (LOP3 with `(0xf0&0xcc)|0xaa` truth tables, PRMT with magic constants, one `__hsub2`/`__hmul2` per pair). LiquidGEMM reaches 2 instructions per 4 elements for int4→int8 on Hopper by overlapping dequant with TMA and MMA. No published work searches the E2M1→BF16 sequence automatically; a CPU exhaustive search over 256 byte values is feasible and would be new.

### 2.4 FP32 on BF16 tensor cores (the router)

- FP32 = b0 + 2⁻⁸·b1 + 2⁻¹⁶·b2 exactly, with three BF16 slices.
- The router must be within 0.01% relative L2. With activations already exact in BF16, only the weight needs splitting. *Derived:* 2 slices give ~2⁻¹⁷ ≈ 8e-6 relative error (measured ~1.5e-5 per element in the hi+lo literature); 3 slices ≈ FP32; 1 slice (2⁻⁹ ≈ 2e-3) fails.
- The baseline already uses hi+lo for M>8 and plain FP32 for M≤8. That proves 2 slices pass.
- hi+lo is 4 B per element, the same as FP32, so the floor stays 1.8 µs. Reading only 2 of 3 stored slices would save 33% of router bytes but is a grey area under "no precision reduction"; storage stays lossless either way. Compressing the router below 4 B/element is a dead end: decoders run slower than the 1.8 µs they would save.
- cuBLAS BF16x9 emulation is compute capability 10.0/10.3 only. Not Hopper.

### 2.5 How the judge measures error

Relative L2, |out − ref| / |ref|, against an FP32 reference. MoE: 1.5% overall, any row within 5× its bound, near-zero rows measured against a tenth of the average row. SGLang's MoE kernel sits at 0.55%, so there is margin, but it cannot be spent on precision. bf16_proj: 1% generally, 0.01% for the router, and some rows of x are a single power-of-two so the output must equal a weight column within 0.1%. Attention: a 3–15% band, which sounds generous until FP8 P·V accumulation over 80K tokens is involved (section 3.3).

---

## 3. Problem playbooks

### 3.1 `muse_moe_w4a16`: routed experts

**What the baseline does** (from `profiles/baseline.*.txt`): two Marlin launches at 56–96% of the time, surrounded by 5–14 small PyTorch kernels (casts, index gathers, reductions, `moe_align_block_size`, `act_and_mul`, post-expert norm). At bs=1: 80 µs total, 45 µs Marlin, 11.4 µs floor. At bs=256: 619 µs, 95% Marlin, 109 µs floor. Marlin streams at ~1.5 TB/s.

**The design the research points to.**

1. **Weight-major, tokens on N.** Hopper `wgmma` fixes M at 64 and allows N from 8 to 256. Put the expert weights on M and the tokens on N, so a 1–30-token batch wastes no padding. This is MonoMoE's central trick (FlashInfer `csrc/fused_moe/monomoe/`, evaluated on H200, 1.17–1.54× over vLLM's Triton grouped GEMM at batch 1–8). Its code is hard-wired to E=256, N=512, K=2048, M≤8, block-FP8, so borrow the design, not the code.
2. **One persistent launch, one CTA per SM.** Routing (histogram + prefix scan), GEMM1 with SwiGLU in the epilogue, GEMM2, and the RMSNorm + weighted sum, with global-memory counters or flags between stages. GEMM1's output is 8 KB per token in BF16 and can stay in shared memory or L2. This removes ~12 launches, which at bs=1 is more than the whole budget.
3. **Stream-K / split-K for bs=1.** One or two experts carry all the work and must cover 132 SMs. Split K for GEMM2 (accumulate FP32, in-kernel last-CTA reduce in fixed order). GEMM1 is awkward because SwiGLU needs the full K sum first; split along N instead, or split K and apply SwiGLU after the reduce.
4. **Deep async pipeline.** TMA or `cp.async.bulk` of contiguous 8–16 KB weight tiles, 3–5 stages, laid out in `prepare()` so each CTA's strip is one contiguous, 128B-swizzle-ready region. Zero in-kernel shuffles. Sequential DRAM runs keep row buffers open.
5. **Dequant as in 2.3.** Budget ~1 op per value.
6. **Skewed routing.** Routings make this GPU the busiest, so some experts get many tokens. Schedule the heaviest experts first (longest-processing-time-first) or use an atomic work queue like Cohere's megakernel. CUTLASS's grouped scheduler, which hands equal tile counts to each CTA, is the wrong model here.

**Experiments, in order of value per check run.**

| ID | Experiment | Why first |
|---|---|---|
| E0 | Print from `prepare()` which FlashInfer / Marlin / CuTe DSL kernels the judge image has. 1 check run. | Could save days if a usable NVFP4 path exists. |
| E5 | Use `prepare()` as a lab: 300 untimed seconds on the real H200, `print()` comes back in the log. Benchmark dozens of tile / split / stage configs under CUDA-graph replay in one run. | One check run replaces ~50. Keep it to check runs; hard-code the winners in the scored file. |
| E4 | Superoptimize the E2M1→BF16 sequence on CPU, verified exhaustively over 256 byte values. | Costs no runs. Possibly new work. |
| E3/E6 | One repack in `prepare()`: nibble order for the cheapest decode, and contiguous per-CTA tiles for bulk copies. | Lossless permutations, explicitly allowed. |
| E2 | The fused persistent kernel. | The big win at small batch. |
| E1 | Lossless scale compression. The judge's generator yields E2M1 codes at exactly 4.0 bits (incompressible) and FP8 scales with 29 distinct values, 3.71 bits of entropy. Ceiling: 6% fewer bytes; a practical 5-bit codebook gives ~4%. | Last. Build the codebook from the scales that arrive, never from the generator, and fall back above 32 distinct values. |

**Dead ends:** Huffman/rANS in-kernel (≤0.6 TB/s decode on H100 vs 4.8 needed; Cloudflare's Unweight added 41% latency at batch 1). LUT-based matmul (FLUTE, LUT-GEMM, T-MAC: Ampere, CPU or 3-bit targets). Lattice vector quantization (changes the format). Partition-camping tuning (no Hopper evidence). nvCOMP hardware decompression (Blackwell). Hilbert tile ordering for L2 reuse (every weight byte is read once).

### 3.2 `muse_bf16_proj`: skinny GEMM

The field is tightly packed (1.65× top, 1.54× at #200), so this is about being within a few percent of the bandwidth floor at every M.

**Why cuBLAS loses.** For M≤8 and small N it picks a tiny grid (ONNX Runtime saw a 1×1 CTA grid on H200) and leaves the chip idle for ~5 µs on a few hundred KB.

**The template: TRT-LLM `tinygemm2`** (`cpp/tensorrt_llm/kernels/tinygemm2/`, shipped in FlashInfer as `tinygemm_bf16`). One CTA per SM, 384 threads as 8 TMA-DMA warps feeding 4 MMA warps, tile 16 weight rows × 8 batch × 64 K, **16 pipeline stages**, `cp.async.bulk.tensor.2d` with `SWIZZLE_128B`, `mma.sync.m16n8k16` + `ldmatrix`, PDL attribute set, no split-K, plain row-major weights. TRT-LLM's `cudaCoreGemm` is the pure-FMA alternative for M≤16: 128 threads, 128-bit loads, warp-shuffle reduction, grid = N/2.

**The plan.**

1. One kernel, one wave, tinygemm2-style, for the big weights (qkv_gate, pooler, unpooler at M>8).
2. Where N/16 gives fewer than ~264 CTAs (o-proj N=1024, router N=256, unpooler rows 0–1023 at M≤8), split K across all SMs and reduce **inside the same kernel**: `__threadfence`, atomic counter, last CTA sums partials in fixed order. Never add a second kernel.
3. M≤16: FMA path with 128-bit loads, ≤2–4 outstanding per thread, or async copies. Tensor cores buy nothing in a memory-bound GEMV; the "Can Tensor Cores Benefit Memory-Bound Kernels? (No!)" result bounds the gain at ≤1.33× even for FP64.
4. Router: pre-split W into interleaved hi/lo BF16 planes in `prepare()`; FP32 FMA for M≤8, 3-product BF16 MMA with FP32 accumulation above.
5. `prepare()`: tile W so each CTA's K-strip is contiguous and swizzle-ready, then stream with 1D bulk copies. Plausible gain, unquantified. Note that for 16-bit weights TMA already produces the wgmma layout from row-major, so pre-shuffling for layout alone has no documented gain; the win is contiguity.
6. PDL on every call.

**Dead ends:** BF16x9 (Blackwell), Stream-K (its own paper says it does not compete with cuBLAS at small bandwidth-bound shapes), router compression, L2-reuse rasterization, expecting more than ~91% of peak.

### 3.3 `muse_attention_decode`

**Why the baseline is beatable.** FA3's `num_splits_heuristic` returns 1 split when `total_mblocks ≥ 0.8 × num_SMs` or when `num_n_blocks ≤ 4`, and otherwise picks the smallest split that reaches 85% of the best wave efficiency. A 2026 BSC preprint (arXiv 2604.00028) found this leaves **>90% of SMs idle** in low-head-count decode, and patching it gave 21–24%. With 2 KV heads per GPU, this problem is exactly that regime.

**The design.**

1. **Lean Attention** (MLSys '25) is the best fit: a Stream-K-style fixed grid (2 CTAs per SM), contiguous key ranges per CTA ("LeanTiles", 256 tokens optimal for head_dim 64), and an in-kernel host-CTA reduction through flags. No separate combine kernel. 1.73× over flash-decoding on average. FlashInfer's plan step (longest-first greedy assignment, persistent fixed grid, deterministic merge, "write-through" for unsplit requests) is the other reference.
2. **GQA packing.** 8 q heads × 8 verify tokens = **64 rows**, exactly one `wgmma` M=64 tile per KV head. K and V get read once per KV head instead of 8 times. For q_len 1 pad 8 rows to 64 or use the CUDA-core path with the KV tile in shared memory shared by all 8 heads, as FlashInfer's decode kernel does (`bdy` = GQA group size, 16-byte vectors per thread, min chunk 256 tokens, `exp2` with log2e folded into the scale).
3. **page_size=1 gather.** TMA is fixed-stride on Hopper; `tile::gather4` is Blackwell-only. Use `cp.async` of 64-byte rows with the page indices prefetched into shared memory, deep but with the load-depth rule from 1.3. FlashInfer reports the decode gap between page_size 1 and dense within 1%. Measure random-64B bandwidth first (one check run in `prepare()`).
4. **Numerics.** Fold log2e into the softmax scale and use `exp2`. Keep FP32 accumulators and **two-level FP32 accumulation for P·V over 80K tokens**: vLLM's FP8-KV testing saw needle-in-a-haystack accuracy collapse from 91% to 13% before that fix. FA3 rounds Q to FP8 with block quantization and Hadamard incoherent processing; here, keep Q in FP8 only if the band allows. FP8 wgmma needs k-major V, which means an in-kernel LDSM/STSM transpose.
5. **Tiny shapes.** Sliding-window layers have a ~3 µs floor and pay ~1.3 µs per launch. Fuse QK-norm, RoPE, cache store and the output norm × gate into the attention kernel's prologue and epilogue, or at least chain them with PDL. Use EVICT_FIRST hints on the KV streams (FlashMLA does; it reaches ~90% of H800 bandwidth with split-KV + combine overlapped by PDL and 64×64 TMA sub-tiles).

**Dead ends:** TMA gather (Blackwell), XQA as a page_size-1 reference (minimum 16), FA3's default split heuristic, FlashDecoding++'s unified max (prior-dependent), looking for published random-64B bandwidth (none exists).

### 3.4 `muse_decode_layer`

SGLang's Muse layer is **already fused at the obvious seams**: QK-norm + RoPE + KV store, FusedRMSNormGate, the boundary norm + residual kernels, a post-expert finalize, and the router GEMV + top-k on a side stream joined before the experts. Chasing more small fusions is low-value. The remaining gains:

1. The MoE block as one persistent kernel (3.1), Cohere-style: router GEMM → top-k → work queue → weight-major NVFP4 expert tiles → atomic reduce, with global integer barriers. Cohere found dependency-affinity placement 1–2% *slower* than round-robin, so don't over-engineer placement.
2. PDL on every remaining boundary, everything in one graph.
3. Fold the token gather/permute into the expert GEMM's loads (MPK measured the separate gather at up to 11% of SGLang's MoE time at batch 1).
4. Prefetch the next op's weights (pooler, unpooler) into L2 or shared memory during spin-waits.
5. Fuse pooler GEMM with the pre-expert norm, and the unpooler into the finalize.

Open templates: Cohere's megakernel (single CUDA file, SM90a, BF16 MoE at batch 1–8), Hazy's Megakernels repo (interpreter + per-SM instruction lists, dense only), MPK (compiler-generated, fixed worker SMs with FIFO queues).

### 3.5 `muse_prefill_layer`

Different regime: the MoE GEMM is large and compute matters. Hopper has **no ready NVFP4 tensor-core path**:

- CUTLASS examples 55/69 (mixed-input grouped GEMM) upcast int4 in registers, but the scale group must be K or a multiple of tile-K. NVFP4's group of 16 does not fit the collective's scale path; the scales have to be applied inside a custom E2M1→BF16 converter.
- Machete is int4/int8 only. FlashInfer's `cake_w4a16` is exactly BF16×NVFP4 but sm_100a/103a only. DeepGEMM's FP4 paths are SM100. Cohere's W4A8 changes activation dtype, likely rule-incompatible.
- Marlin NVFP4 (the baseline) uses `mma.sync`; Machete's authors measured Marlin losing ~37% of Hopper peak at large M.

Plan, if time allows after the kernel track: CUTLASS ex. 69 with a custom converter and group-16 scales applied in the converter; a persistent tile scheduler with per-expert tile counts and hot experts first (DeepGEMM's masked grouped GEMM is the CUDA-graph-safe shape); SwiGLU in the epilogue. This is week-2 work.

---

## 4. Cross-domain tricks

Asked for: quantum physics and computing, physics and electronics generally, and competitive programming. Here is what transferred and what did not.

### 4.1 Quantum computing

Quantum hardware is out: the judge runs offline in a fixed image, the bottleneck is moving classical bytes, and loading classical data into a quantum state is the known weak point (Tang's dequantization results show why). What transfers is the **GPU simulation toolkit quantum physicists built**, which is memory-bound work with the same shape as ours.

| Quantum-side idea | Kernel equivalent | Status |
|---|---|---|
| Gate fusion in state-vector simulators (cuQuantum: 5-qubit fusion, 2.35–3.0 TB/s on H100) | Fuse dequant, GEMM, SwiGLU, norm into one pass over memory | Core of E2. Only cost is register pressure. |
| Tensor-network slicing (cuTensorNet) | Split-K: sum of independent slices, tiny reduce for GEMV | Core of the bs=1 plan. |
| Contraction-path hyper-optimizer | Sample many configs, keep the cheapest: the `prepare()` tuner | E5. |
| Qubit-index permutation (cuStateVec bit-swap) | Lossless relabeling of weight layouts in `prepare()` | E3. |
| Circuit synthesis (minimize gate count) | Superoptimize the dequant instruction sequence | E4. |
| Entropy (Shannon / von Neumann) | Lossless weight compression | Ceiling measured at 6%. E1, last. |
| Quantum-inspired sampling, low-rank approximation | Approximate matmul | Rejected: lower precision. |

### 4.2 Physics and electronics

- **Lattice QCD is the best analogy.** QUDA's Dirac operator is bandwidth-bound at arithmetic intensity ~0.7 and its 15-year-old trick, "reconstruct-12/8", stores part of each SU(3) matrix and rebuilds the rest from unitarity. Principle: FLOPs are cheap, bytes are expensive. Limit: it needs an algebraic constraint, and random E2M1 codes have none. Only the scales are compressible.
- **DRAM device physics.** Long contiguous runs per CTA keep row buffers open; 32-byte atoms across 192 pseudo-channels. Translates to E6: contiguous per-CTA tiles, every SM streaming.
- **Temporal blocking (stencils)** needs reuse across time steps. Weights are touched once. No transfer.
- **Mandelbrot/Julia sets**: compute-heavy, no memory traffic. No mechanism carries over.
- **Space-filling curves (Hilbert, Morton)** are real GPU GEMM tooling for L2 reuse, up to 15% on AMD in HipKittens. Small here since weights are read once; worth one variant in the tuner for spreading concurrent CTAs across channels. Real payoff is in attention, where 8 q heads share a KV head.
- **Power/clock diagnosis.** nvidia-smi averages over idle gaps; use nsys GPU metrics at 100 kHz to see whether the kernel actually saturates HBM.

### 4.3 Competitive programming

The CP literature on constant factors is a parallel to kernel work. The one resource whose topics are exactly ours is Sergey Slotin's *Algorithms for Modern Hardware* (en.algorithmica.org): read the memory and SIMD chapters and mentally swap "AVX2 lane" for "CUDA thread".

| CP trick | Kernel equivalent | Where |
|---|---|---|
| Fast I/O: read the whole buffer, parse in registers | TMA bulk copies of whole tiles, dequant in registers | MoE, bf16_proj |
| Precompute everything query-independent | Everything into `prepare()`: layouts, scale tables, expert maps | All |
| Bitset / SWAR: 64 things per instruction | 8 nibbles per register, `lop3` + `prmt` | MoE dequant |
| Branchless select | Mask-and-blend instead of `if` inside a warp | Routing, causal masks |
| Four Russians: table over bit-groups | `prmt` as a 16-entry byte table, 4 lookups per instruction | Dequant |
| Mo's algorithm with Hilbert order | Group the 8 q heads sharing a KV head into one CTA | Attention |
| Sqrt decomposition / offline sorting | `moe_align_block_size`: sort tokens by expert | MoE routing |
| Lazy propagation | Defer the RMSNorm until the last split-K partial lands | MoE, decode_layer |
| Meet in the middle | Split-K: two half-problems, cheap merge | bs=1 shapes |

Does not transfer: asymptotic cleverness (the algorithm here is settled, only the constant is in play), compiler-flag and strict-aliasing hacks, `#pragma GCC target`.

The meta-lesson is the same in both worlds: measure, find the one hot loop, pack data so one instruction does many things, order work for locality, move everything possible offline. Our hot loop is "stream weights and dequant"; `prepare()` is the offline pass.

---

## 5. Study plan

### Tier 1: the mental model (one evening)

1. **Roofline and arithmetic intensity.** Compute the floor for each shape in `TASK.md` by hand and check against the table.
2. **Little's law.** Bandwidth × latency = bytes in flight. Why pipelining and async copies beat clever math.
3. **The execution hierarchy.** Thread → warp (32) → CTA → SM (132) → grid. Registers → shared memory (228 KB) → L2 (50 MB) → HBM. Rough latencies: registers ~1 cycle, smem ~30, L2 ~250, HBM ~600–800.
4. **Coalescing and sectors.** 32-byte sectors, 128-byte lines.
5. **What a launch costs.** Count the kernels in the bs=1 baseline profile. Fusion is the first optimization.

### Tier 2: number formats (one hour)

6. Draw BF16, FP32, E4M3, E2M1. See that BF16 is FP32 truncated, so FP32 = hi + lo.
7. NVFP4 as a three-level scale; why dequant to BF16 is exact.
8. Relative L2 error; why 1.5% is generous and 0.01% is tight.

### Tier 3: the toolkit (two days, hands on)

9. **Triton basics.** `tl.program_id`, masked `tl.load/store`, `tl.dot`, `num_stages`/`num_warps`. Write a GEMV and a tiled GEMM. Triton ships first.
10. **CUDA graphs.** No `.item()`, `.cpu()`, data-dependent shapes or allocations. Pre-allocate in `prepare()`.
11. **Split-K and Stream-K.** Make one 28 MB matrix occupy 132 SMs at M=1.
12. **`wgmma` shape rules.** M=64 fixed, N 8–256, K=16 per BF16 instruction. Hence tokens on N.
13. **Software pipelining.** Load tile k+2 while computing k. `cp.async`, TMA. Triton's `num_stages`.
14. **Bit tricks.** `lop3`, `prmt`, nibble into the BF16 exponent field. Marlin's `dequant.h`.
15. **Online softmax and split-KV.** Each CTA keeps (max, sum, acc); merge with log-sum-exp.
16. **PDL.** `launch_pdl=True`, `gdc_wait()`, `gdc_launch_dependents()`.

### Tier 4: measurement discipline (ongoing)

17. Read `profiles/*.txt` after every passing run. Kill the small kernels first.
18. Gains inside ±1.5% are noise. Don't chase them.
19. `prepare()` is the lab: 300 untimed seconds, `print()` returned in the log tail. Many configs per check run.
20. Eager timing misleads at these sizes. Always time CUDA-graph replays.

### Where to read each

| Concepts | Source |
|---|---|
| 1–5 | PMPP 4th ed. ch. 4–6 (`~/Documents/Books/pmpp.pdf`); ch. 9–11 for histogram, reduction, scan |
| 6–8 | `muse_moe_w4a16/nvfp4.py`, `checks.py`, `reference.py` |
| 9, 13 | Triton tutorials 01–03, and tutorial 11 for PDL |
| 10 | `baseline.py` (how it pre-allocates buffers) |
| 11–12 | Colfax Hopper GEMM paper; MonoMoE paper for the tokens-on-N argument |
| 14 | Marlin `dequant.h`; section 2.3 above |
| 15 | PyTorch flash-decoding blog; FA3 paper, FP8 section; Lean Attention paper |
| 16 | Triton docs, `triton.language.extra.cuda` |

Skip for the kernel track: CUTLASS templates, CuTe layouts, hand-written PTX, warp specialization. Triton gets to ~2.3× on MoE. Those are week-2 tools, and the prefill problem is where they become necessary.

---

## 6. Order of work

1. **E0**: one check run that prints what the judge image offers.
2. Baseline Triton MoE kernel: fused routing, bit-trick dequant, GEMM1 + SwiGLU, GEMM2 + norm. Validate on CPU in a Linux container before spending runs.
3. **E4** on CPU in parallel; **E5** tuner in 1–2 check runs.
4. **E3/E6** repack, then **E2** fusion, weight-major with tokens on N, Stream-K at bs=1.
5. `bf16_proj`: tinygemm2-style kernel plus in-kernel split-K reduce for the small-N weights. Quick win for board coverage.
6. `attention_decode`: Lean-Attention-style persistent kernel with GQA packing. Microbenchmark random-64B gather first.
7. `decode_layer`: drop the MoE kernel in, PDL everywhere.
8. **E1** and prefill only if time remains.

---

## 7. Sources

Primary sources with URLs are in the `research/` files:

- `research/briefing-moe-w4a16.md`: the roofline, ALU budget and reading list for the MoE problem.
- `research/attention-decode.md`: FA3 heuristics, FlashInfer, FlashMLA, Lean Attention, XQA, FP8 numerics.
- `research/skinny-bf16-gemm.md`: ONNX Runtime split-K GEMV, tinygemm2, cudaCoreGemm, Marlin and Machete layouts, FP32 splitting, load-depth study.
- `research/layer-fusion-and-prefill.md`: Hazy, Cohere and MPK megakernels, MonoMoE, Triton PDL, CUTLASS mixed-input, skewed routing.
- `research/physics-ee-crossover.md`: HBM3e organisation, power and clocks, Little's law numbers, entropy coding ceilings, BF16 splitting, LUT dequant.
- `research/quantum-crossover.md` and `research/prior-work-and-physics-analogies.md`: the quantum, fractal and lattice-QCD sweep and the prior work for each experiment.
- `research/competitive-programming-tricks.md`: Algorithmica, bitsets, Hilbert-ordered Mo, Four Russians.
- `research/concept-list.md`: the original tiered concept list.

Things that were looked for and **not found** anywhere, so they must be measured: random 64-byte gather bandwidth on Hopper; per-M achieved bandwidth for skinny BF16 GEMM on H100; HBM3e partition-camping behaviour; longest-processing-time or Stream-K scheduling across MoE experts.
