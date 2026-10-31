# Prior art: muse_attention_decode

Research agent report, collected 2026-10-31 (web sources, verified unless flagged inside). Part of the Open Kernel Challenge notes; see ../moe-field-guide.md for the synthesis.

## Prior-art findings: muse_attention_decode (H200, GQA 8:1, hd64, FP8 KV, page_size 1)

### Q1. Split-KV / flash-decoding
- Flash-Decoding (FA2.2+) = split keys into chunks, each writes O + LSE, cheap reduce; "up to 8x" at long context. https://pytorch.org/blog/flash-decoding/
- FA3 `hopper/heuristics.h` `num_splits_heuristic`: returns 1 split if `total_mblocks >= 0.8*num_SMs` (except KV head > 50 MB L2 and non-causal), returns 1 if `num_n_blocks <= 4`, else picks the smallest split count reaching 85% of best wave efficiency `n_waves/ceil(n_waves)`. Combine kernel (`flash_fwd_combine_kernel.h`) stages all LSEs in smem, uses natural `expf/logf` (not exp2), 4-stage cp.async pipeline over partials, -inf guards. https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h, https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_combine_kernel.h
- A 2026 BSC preprint found FA3's heuristic leaves >90% of SMs idle in low-head-count decode; patching it gave 21–24% kernel gain. https://arxiv.org/abs/2604.00028
- FlashInfer: CPU plan step sizes chunks as `L_kv = Σ⌈l_q/T_q⌉·l_kv / #CTA`, longest-first greedy onto CTAs with cost α·l_q+β·l_kv; persistent fixed-grid kernels (CUDA-graph safe); deterministic merge without atomics; "write-through" skips the reduce for unsplit requests. Decode CUDA-core kernel: 16-byte vectors/thread (FP8 hd64 → vec_size 16, bdx 4... computed as `max(16/sizeof, HD/32)`), `bdy` = GQA group so K/V tile in smem is shared by all 8 q-heads, `bdz` splits within block, `ptx_exp2` with log2e folded into scale, min chunk 256 tokens. https://arxiv.org/html/2501.01005v2, https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/include/flashinfer/attention/decode.cuh
- FlashMLA (Hopper): ~3 TB/s on H800 (≈90% of 3.35 TB/s) with `splitkv_mla` + `combine` overlapped via PDL, tile scheduler, 64×64 TMA sub-tiles so GEMM starts per sub-copy, EVICT_FIRST hints. https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
- Lean Attention (MLSys'25): stream-K style, fixed grid (2 CTAs/SM), contiguous "LeanTile" ranges (256 tokens optimal for hd64), in-kernel host-CTA reduce via flags, 1.73x avg over FlashDecoding. https://arxiv.org/html/2405.10480
- Published FP8-decode bandwidth numbers: none found. vLLM reports FP8 KV halves ITL slope (54% of BF16) on H100 but no GB/s. https://vllm.ai/blog/fp8-kvcache

### Q2. GQA packing into M
- FlashInfer "head-group fusion": q heads fused with query-length dim; tensor-core (prefill template) GQA decode 2–3x over vLLM; CUDA-core version reaches only "40%+" bandwidth. https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- XQA: `HEAD_GRP_SIZE` default 8, `CACHE_ELEM_ENUM=2` (E4M3) default, `SPEC_DEC` with `M_TILESIZE 32`, "SWAP AB" for fixed small q_len, `ENABLE_PDL=2` aggressive PDL in `mha_sm90.cu` (TMA+QGMMA). Exact M mapping not in README (unverified). https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/kernels/xqa/defines.h
- NVIDIA: runtime steps when G×(1+D) crosses multiples of 128 (tile M). Your 8×8=64 rows fits one wgmma M=64 tile exactly. https://developer.nvidia.com/blog?p=122024

### Q3. page_size=1
- Hopper TMA is fixed-stride only; `tile::gather4` requires SM100+ (CCCL/Mojo docs; PTX text not read directly). https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_prefetch_tensor.html
- FlashInfer: decode gap sparse(page 1) vs dense "within 1%"; prefill ~10% (cp.async smaller tiles). Page indices prefetched to smem. https://arxiv.org/html/2501.01005v2
- XQA `TOKENS_PER_PAGE` allowed 16/32/64/128 — no page_size 1.
- Sector facts: 128 B line, 32 B sectors; V100 user test showed single-sector miss fetches 64 B from HBM2. H100/HBM3 random-64B gather bandwidth: NOT FOUND — must microbenchmark. H100 LDG study: bandwidth peaks at ~2 outstanding loads/thread, falls 35% at 8; cp.async 2.1–2.9x plain loads at high depth. https://forums.developer.nvidia.com/t/behavior-of-l1-l2-caches/255293, https://hgpu.org/?p=31133

### Q4. FP8 numerics
- FA3 FP8: block quantization + Hadamard incoherent processing (RMSE 2.4e-2 → 9.1e-3), FP8 wgmma needs k-major V → in-kernel LDSM/STSM transpose, FP32 accumulators. https://arxiv.org/html/2407.08608v2
- Long-context FP8 P·V accuracy collapse (91%→13% NIAH) fixed by two-level FP32 accumulation (flash-attention#104). Relevant at 80K. https://vllm.ai/blog/fp8-kvcache
- FlashDecoding++ unified-max (ϕ fixed, recompute fallback) is mathematically exact but relies on priors. https://arxiv.org/html/2311.01282v4

### Q5. Tiny shapes
- H100 launch ≈2.1 µs (1.3 µs with graphs); megakernel 78% BW; PDL deemed "too coarse". https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Cohere decode megakernel: ATTN_DECODE/ATTN_COMBINE + work-stealing ATTN_DRAIN, 62% SoL at BS1. https://cohere.com/blog/megakernels
- PDL needs sm_90; used in FlashMLA and XQA.

### Q6. Papers
Lean Attention, FlashInfer, POD-Attention (prefill+decode co-scheduling, less relevant) are rule-compatible (exact, read all keys). FlashDecoding++ borderline.

### Top 5 actionable ideas
1. Single persistent kernel over all 132 SMs with tensor-core M=64 tile (8 heads × 8 q, or 8 heads padded) per KV head, Lean-Attention-style contiguous KV ranges + in-kernel flag reduce (kills combine launch and wave quantization).
2. Deep cp.async pipeline of 64-byte rows with page indices prefetched to smem (FlashInfer pattern); tune outstanding depth per the H100 LDG finding; microbenchmark random-64B gather first.
3. Fold log2e into scale, exp2, FP32 two-level accumulation for 80K P·V (avoid FA3's FP8 accuracy bug).
4. Fuse QK-norm/RoPE/cache-store/out-norm into prologue/epilogue; use PDL or a single launch for sliding-window shapes (~3 µs floor vs ~2 µs per launch).
5. EVICT_FIRST on KV streams, fixed-grid CUDA-graph-friendly design; keep Q as FP8 only if accuracy band allows (FA3 quantizes Q).

### Dead ends
- TMA gather on Hopper (Blackwell-only).
- XQA as reference for page_size 1 (unsupported).
- FA3 default split heuristic (under-splits at low head count).
- FlashDecoding++ unified max (prior-dependent, no clear win).
- Searching for published H100 random-64B bandwidth — none exists; measure it.
