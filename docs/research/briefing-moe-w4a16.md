# Briefing: the physics of muse_moe_w4a16

Written 2026-10-31 during the Open Kernel Challenge kernel track. Leaderboard figures are from that evening. See ../moe-field-guide.md for the synthesis.

I've paused the kernel work to write this. I couldn't search inside the PMPP PDF (no PDF text tool on this Mac), so the chapter pointers below come from the 4th edition's standard contents.

---

# Briefing: the physics of `muse_moe_w4a16` and how to get on the board

## 1. The problem in one equation

There are 256 experts, and each token is routed to 8 of them. Your GPU holds experts 0–31, which is 1/8 of the model (8-way expert parallelism). For each token routed to one of your local experts:

```
h     = x · W13ᵀ            [4096] → [8192]   (gate | up)
inter = silu(gate) ⊙ up     [4096]            (SwiGLU)
y     = inter · W2ᵀ         [4096] → [4096]
out  += g · RMSNorm(y ⊙ (1+w))
```

The weights are **NVFP4**:
- **Element:** E2M1, which is 1 sign bit, 2 exponent bits and 1 mantissa bit. The only magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
- **Block scale:** every 16 elements share an FP8 E4M3 scale.
- **Global scale:** each expert has one FP32 scale.
- **Cost:** 0.5 bytes per weight plus 1/16 byte for scales, so 0.5625 B per element.

**One expert is 3 × 4096 × 4096 × 0.5625 B ≈ 28.3 MB.**

## 2. Roofline: why this is a pure memory-bandwidth problem

On the busiest GPU, a batch of `bs` tokens produces about `bs` local routes, spread over 32 experts. That's only **1 to ~30 tokens per expert**.

- **Arithmetic intensity:** each expert does 2·m·50.3M FLOPs over 28.3 MB, so AI ≈ 3.6·m FLOP/byte.
- **H200 ridge point:** 989 TFLOP/s ÷ 4.8 TB/s ≈ 206 FLOP/byte.
- **Consequence:** you'd need m ≈ 58 tokens per expert before compute mattered. Every scored shape sits far left of the ridge.

So the score is decided by one thing: **how close you can stream the touched experts' weights to 4.8 TB/s.** The time floor is (touched experts × 28.3 MB) ÷ 4.8 TB/s, or about 5.9 µs per expert.

Marlin reaches about 1.5 TB/s here. That explains SGLang's 3.5–7× gap to speed of light.

## 3. The three constraints that actually bind

**(a) Little's law: bytes in flight.**
- The H200 needs about 4.8 TB/s × ~0.8 µs latency ≈ 3.8 MB in flight across the chip.
- That's ~29 KB outstanding per SM, all the time.
- In practice this means deep multi-stage pipelining (cp.async or TMA loads into shared memory, 3–5 stages) and enough CTAs or warps to cover the latency.
- PMPP ch. 4 (latency hiding, occupancy) and ch. 6 (coalescing, thread coarsening) cover the fundamentals. Section 6.4's checklist is the useful part.

**(b) The ALU budget for dequantization. This is the subtle part, and probably where the top entries win.**
- At full bandwidth each SM eats 18.4 B per clock (4.8 TB/s ÷ 132 SMs ÷ 1.98 GHz). That's ~33 FP4 elements per clock per SM.
- A Hopper SM has ~128 FP32 lanes and 64 INT32 lanes per clock.
- So you get **roughly 4 ALU operations per weight element** to turn a nibble into a scaled BF16 value. A naive dequant (unpack, table lookup, convert, multiply) blows that budget, and the kernel turns ALU-bound.
- Hopper has no FP4 hardware. That arrived with Blackwell's `cvt.e2m1x2`. So you need bit tricks:
  - Put the nibble's 3 magnitude bits straight into a BF16's exponent and mantissa fields (`mag<<6 | sign<<15`). The result is exactly the E2M1 value × 2⁻¹²⁶.
  - Fold the 2¹²⁶ factor, minus a few powers of two, into the scale. Fold the rest into the per-expert global scale, applied once at the end.
  - Use `prmt` (byte permute) and `lop3` (any 3-input logic function in one instruction) to handle 2 elements per 32-bit operation.
  - Repack the nibbles in `prepare()` so the low and high nibble of a byte belong to K-halves that line up. Then no interleave shuffle is needed.
- This exact-arithmetic argument should satisfy the "no lower precision" rule. E2M1 has 2 significant bits and E4M3 has 4. Their product has ≤ 6 significant bits, which fits in BF16's 8, so the dequant is exact.
- NVIDIA's own Marlin NVFP4 kernel works this way. Its docs mention global scales pre-adjusted for "skip-flop E2M1 conversion" ([TensorRT-Edge-LLM Marlin MoE](https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api/kernels/moeMarlin.html)).

**(c) Parallelism at tiny batch, and launch overhead.**
- At bs=1, only 1–2 local experts have any work. That one 28 MB expert has to be spread across all 132 SMs, using fine N-tiles and/or split-K.
- Split-K is easy for GEMM2 (accumulate in FP32). It's awkward for GEMM1, because SwiGLU is nonlinear and needs the full K sum first.
- The bs=1 floor is 11 µs. SGLang's profile shows ~35 µs of tiny PyTorch kernels (dtype casts, index gathers, reductions) at 1–6 µs each.
- **Fusion is free speed at small batch.** The obvious fusions:
  - Routing and token compaction fused into GEMM1. This is a histogram plus prefix scan, PMPP ch. 9 and 11.
  - SwiGLU in GEMM1's epilogue.
  - The RMSNorm and weighted sum after GEMM2. This is a reduction, PMPP ch. 10, and it's harder because the norm needs the full 4096-wide row.
- Hopper's programmatic dependent launch lets the next kernel's prologue overlap the previous kernel's tail.

**Judge details that close loopholes:**
- L2 is cold on every replay. H200's 50 MB L2 could otherwise hold an entire expert.
- Inputs are fresh on every call.
- Timing uses CUDA graph replay, so no host syncs and no shapes that depend on data values.
- Error must stay within 1.5% relative L2. SGLang's kernel sits at 0.55%, so there's margin, but you can't spend it on lower precision.

## 4. The competitive landscape (live data)

| problem | entries | #1 | #50 | #200 | median |
|---|---|---|---|---|---|
| muse_moe_w4a16 | 1212 | 2.76× | 2.43× | 2.34× | 1.81× |
| muse_attention_decode | 1158 | 3.22× | 2.69× | 2.59× | 1.79× |
| muse_decode_layer | 1077 | 2.42× | 2.24× | 2.15× | 1.65× |
| muse_bf16_proj | 1507 | 1.65× | 1.57× | 1.54× | 1.24× |
| muse_prefill_layer | 721 | 2.03× | 1.82× | 1.73× | 1.49× |

**What matters strategically:**
1. **Only serving-track places pay.** The $100K prize pool is paid on the serving board (Nov 5–19). The kernel boards pay nothing.
2. **What the kernel boards get you is serving priority.** The top 50 on the overall kernel board when it closes get first place in the queue for 8×H200 serving runs.
3. **The overall board ranks your combined decode-step speedup across problems.** The top 8 almost all have entries on 4 problems. One great MoE kernel won't get you into the top 50; coverage will.
4. **The MoE kernel is reusable.** It's most of `muse_decode_layer`, and it's exactly the code you'd patch into SGLang for the serving track.
5. **Realistic 5-day target:** about 2.3–2.4× on MoE, ~1.55× on bf16_proj (dense GEMMs, cuBLAS-adjacent and easy to approach), and entries on attention and decode_layer. That puts a top-50 overall finish in reach.

## 5. Reading list, in priority order

1. **PMPP (4th ed., in `~/Documents/Books/pmpp.pdf`):**
   - ch. 4: compute architecture, warps, occupancy
   - ch. 5: memory hierarchy and tiling
   - ch. 6: coalescing, latency hiding, thread coarsening. The most important chapter for this problem.
   - ch. 9–11: histogram, reduction, scan. These are the routing and norm pieces.
2. **Hopper GEMM mechanics** (TMA, WGMMA, warp specialization, persistent kernels):
   - [Colfax: Hopper GEMM with CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/) and [their paper](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf)
   - [PyTorch: CUTLASS Ping-Pong kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
   - [Persistent GEMM in CuTe DSL](https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/). CuTe DSL is available in the judge's image.
3. **FP4 on Hopper:**
   - [TensorRT-LLM PR #14775](https://github.com/NVIDIA/TensorRT-LLM/pull/14775), the W4A16 NVFP4 fallback for Hopper via a Triton dequant
   - The [NVFP4 dequant](https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api/kernels/nvfp4_dequant.html) docs
4. **vLLM's [fused_moe](https://docs.vllm.ai/api/vllm/model_executor/layers/fused_moe/fused_moe.html):**
   - The canonical Triton MoE kernel: sorted token IDs, per-expert M-tiles padded to BLOCK_M, configs tuned by M and E
   - It also has a small-M path that skips the alignment step
5. **Your local copies:** `muse_moe_w4a16/baseline.py` (what you're racing) and `reference.py` (what's checked).

## 6. Agent37 sandboxes and GPU credits

You don't need them for this:
- `okc check` and `okc submit` already run on the organizers' H200s for free.
- The Auto-research-agent README says the same thing about OKC: no GPU needed, evaluation runs remotely.
- It doesn't say whether Agent37 sandboxes have GPUs at all.
- Claiming them means opening a public PR with your email in it. That's your call, and I won't do it on your behalf.

The README also mentions a third-party H200 queue (app.getsupers.com) with 5 free runs. I haven't verified it, and it's small next to the 20 free check runs a day you already have.

---

**Next step:** I'll continue the Triton kernel: fused routing, the bit-trick NVFP4 dequant, GEMM1 with SwiGLU, then GEMM2 with the norm. I'll check the logic on CPU in a Linux container (Docker is installed) before spending check runs. We've used 1 of today's 20.
