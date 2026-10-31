# Prior work per experiment, plus fractals, lattices and lattice QCD

Written 2026-10-31 during the Open Kernel Challenge kernel track. Leaderboard figures are from that evening. See ../moe-field-guide.md for the synthesis.

I found prior work for almost every experiment. The most important find is a paper from August that builds nearly the kernel I was planning.

## Prior work, experiment by experiment

**E2 (fused persistent kernel): done already, on this GPU.**
- [**MonoMoE**](https://arxiv.org/abs/2609.04244) (Gong et al., Aug 2026) is evaluated **on H200 only**.
- Its three costs of standard MoE decode match my analysis:
  - padding and preprocessing when few tokens reach each expert
  - short-lived grids that underuse bandwidth
  - quantization, activation and reduction run as separate stages
- **Design: weight-major.** The decode tokens sit on the tensor core's N dimension. CTAs are split over expert-weight tiles, and one persistent launch does routing through reduction.
- **Results:** up to 1.54× over vLLM's Triton grouped GEMM. It's FP8, not NVFP4, but the code is open source in FlashInfer (`csrc/fused_moe/monomoe`).
- **Takeaway:** put tokens on N, not M. Hopper's `wgmma` instruction fixes M at 64 but allows N as small as 8, so a 1–30-token batch wastes no padding. That replaces the M-tile padding I had planned.
- **Megakernel precedent:** Hazy Research's Llama-1B megakernel hit ~78% of H100 bandwidth at batch 1, against roughly half that for standard stacks ([Cohere's summary](https://cohere.com/blog/megakernels)). [MPK](https://arxiv.org/html/2512.22219v1) automates the same idea and reports up to 1.7× lower latency.

**E4 (superoptimized dequant): the gap looks real.**
- [**LiquidGEMM**](https://arxiv.org/pdf/2509.01229) reaches **2 instructions per 4 elements** for W4A8 on Hopper. It overlaps dequant with TMA loads and tensor-core math in a pipeline.
- That beats my ≤2-ops-per-element target by 4×. It's int4→int8, not E2M1→BF16, but it shows how far bit-level design can go.
- **Closest search work:** [STOKE](https://arxiv.org/pdf/1211.0557) does stochastic superoptimization, but on x86 CPU code. Cambridge's [SIP](https://www.cl.cam.ac.uk/research/srg/netos/euromlsys2024/slides/S1_3__24.pdf) runs simulated annealing over SASS schedules inside Triton.
- **Gap:** neither one searches the bit-twiddling itself. I found no published automated search for an E2M1→BF16 decode sequence.

**E1 (lossless compression): prior work favors the cheap option.**
- [**ZipServ**](https://arxiv.org/abs/2603.17435) found that **fixed-length codes beat entropy-optimal codes on GPUs**, because decoding stays constant-time and parallel.
- That backs a 5-bit scale codebook over Huffman coding. [DFloat11](https://huggingface.co/papers/2504.11651) and Cloudflare's Unweight show the approach works losslessly on BF16.
- [**FLUTE**](https://arxiv.org/pdf/2407.10960) is a guide for decoding through a shared-memory lookup table. It duplicates the table to avoid bank conflicts and gets 2–4× at batch < 32.
- FLUTE also uses **Stream-K** partitioning, a balanced form of split-K. That's the answer to the bs=1 problem where one expert has to cover all 132 SMs.

**A kernel to check:** [flashrt/grouped-moe-gemv](https://huggingface.co/kernels/flashrt/grouped-moe-gemv) does grouped NVFP4 W4A16 matrix-vector products for MoE. It's listed for Blackwell only and publishes no benchmarks, so it's a reference at most.

## Fractals and lattices

**Mandelbrot and Julia sets: no.** They're iterated complex maps. That's compute-heavy work with almost no memory traffic, the opposite of this problem. I found no mechanism that carries over, and I'd rather say so than invent one.

**Fractal space-filling curves (Hilbert, Morton): a real technique, small payoff here.**
- GPU GEMMs order their tiles along these curves for L2 reuse. [One write-up](https://indii.org/blog/gpu-matrix-multiply/) beats cuBLAS at one size with Hilbert ordering. [HipKittens](https://arxiv.org/pdf/2511.08083) reports up to 15% from cache-aware tile order on AMD.
- **Why it's small here:** each weight byte is read exactly once, so L2 has almost nothing to reuse. The only angle left is spreading concurrent CTAs across HBM channels, a cheap variant to add to the E5 tuner.

**Lattice QCD: the best analogy in physics.**
- QUDA's Dirac operator is bandwidth-bound at an arithmetic intensity of ~0.7, worse than ours.
- QUDA's main trick is **"reconstruct-12/8"**: store only part of each SU(3) matrix and rebuild the rest in registers from unitarity ([arXiv:1011.0024](https://arxiv.org/pdf/1011.0024), [1309.2256](https://arxiv.org/pdf/1309.2256)). Physicists have relied on this for 15 years.
- **Mapping:** this is the E1 idea in physics form. It's the general principle that FLOPs are cheap and bytes are expensive.
- **Limit:** their reconstruction relies on a symmetry. Our E2M1 codes are uniformly random, so they have none to exploit, and only the scales can be compressed.

**Lattice vector quantization (E8 lattices, QuIP#): rejected.** It changes the weight format, and the rules require NVFP4 to stay NVFP4.

## Revised plan
1. **New E0 (1 check run):** print from inside `prepare()` whether the judge image's FlashInfer includes MonoMoE, and which NVFP4 kernels it has. Building on it could save days.
2. **Kernel design changes:**
   - weight-major layout, tokens on N
   - Stream-K for bs=1
   - LiquidGEMM-style overlap of loads, dequant and math
3. E4 and E5 as before. Add a Hilbert/channel-spread CTA order as one variant in the tuner.
