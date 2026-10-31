# Cross-domain: HBM3e physics, power, Little's law, entropy coding, numerics, LUT dequant

Research agent report, collected 2026-10-31 (web sources, verified unless flagged inside). Part of the Open Kernel Challenge notes; see ../moe-field-guide.md for the synthesis.

**Cross-domain research: weight-streaming kernels on H200** (verified unless marked *unverified/derived*)

**1. HBM3e device physics**
- H200 = 6 HBM3e stacks × 16 ch × 2 pseudo-channels = 192 PCs on a 6144-bit bus; H100 SXM used 5 stacks/5120-bit (https://www.hwcooling.net/en/nvidias-new-fastest-ai-gpu-h200-with-141gb-of-hbm3e-memory/). HBM3 PCs are 32-bit, BL8 → 32-byte atom, matching the L2 sector (https://www.synopsys.com/designware-ip/technical-bulletin/hbm3-ip-dwtb.html). *Derived*: 4.8 TB/s ÷ 192 ≈ 25 GB/s per PC; any kernel must keep all 192 PCs busy. Page size per PC ≈1 KB (gem5 config only, *unverified vs JESD238*); tFAW/tRRD/turnaround numbers for HBM3e not publicly documented.
- Achievable: STREAM-style reads reach 91–94% of 3.35 TB/s on H100 SXM5 (arXiv 2608.15764), 91% on H800 (https://arxiv.org/html/2501.12084v1), 91–93% in NVIDIA's own GH200 STREAM guide (https://docs.nvidia.com/gh200-superchip-benchmark-guide.pdf). Treat ~92% (≈4.4 TB/s on H200) as the practical ceiling.
- Partition camping on Hopper: no public measurement; the Hopper dissection papers explicitly could not resolve the address→L2-partition mapping. *Unverified* — only guidance is "contiguous, all-SMs-active".

**2. Power/clocks**
- Ujeniya et al. (https://arxiv.org/pdf/2604.11391): memory clock is strictly static (3201 MHz on H200) at every power cap; SM clock stays at 1980 MHz boost for memory-bound STriad unless capped. But H200 needs ≥~550 W average to sustain peak STriad bandwidth and degrades at ≤300 W caps; H200 memory power alone saturates ≈220 W (H100 ≈125 W). **Your 127 W reading is inconsistent with sustained peak HBM3e traffic** — likely nvidia-smi averaging over idle gaps between short calls (nvidia-smi reports a requested state; use nsys GPU-metrics at 100 kHz: https://forums.developer.nvidia.com/t/sm-frequency-reported-in-nsight-compute/264271). Chips and Cheese saw H100 PCIe dip to 1395 MHz under bandwidth tests at 350 W (https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth).
- Extra ALU is free until total ≈700 W; *derived budget*: 132 SMs × 128 FP32 lanes × 1.98 GHz ≈ 33 T lane-ops/s ÷ 4.8 TB/s ≈ **7 ALU ops per streamed byte** (≈3.5 per NVFP4 value). Dequant fits; entropy decode (several INT ops/byte) is marginal — Doubleword says the same (https://resources.doubleword.ai/resources/pushing-memory-bound-kernels-beyond-the-speed-of-light-with-lossless-decompression).

**3. Little's law / in-flight bytes**
- H100 unloaded HBM latency ≈353 ns, L2 ≈145 ns (https://arxiv.org/pdf/2608.15764); H800 ≈656 cycles, near/far L2 miss 555/744 cycles (2501.12084). *Derived*: 4.8 TB/s × 350–700 ns = 1.7–3.4 MB device-wide ≈ **13–26 KB in flight per SM**.
- Key finding (2608.15764): plain LDG bandwidth peaks at ~2 outstanding loads/thread (3111 GB/s) and **falls ~35% at 8/thread** (1990 GB/s) with DRAM bytes unchanged; cp.async sustains 2.1–2.9× the matched LDG bandwidth at 1–8 warps/SM. TMA reached >1800 GB/s (≈88%) on H800 with 1–4 CTAs/SM but adds ≈170 cycles latency (2501.12084). Lesson: don't over-unroll per-thread loads; use 2–3 outstanding 8–16 KB bulk copies per SM.

**4. Entropy coding**
- Ceilings: DietGPU rANS decode ≈592 GB/s on H100 (https://arxiv.org/pdf/2606.18900); DFloat11 reaches 76.5% and DietGPU 43.7% of L40S bandwidth, decode alone 1.56–3.44× the GEMM time (ZipServ, https://arxiv.org/html/2603.17435v1). Cloudflare Unweight on H100: 22% smaller, **+41% latency at batch 1** (https://blog.cloudflare.com/unweight-tensor-compression). "Approaching Shannon Bound" fuses rANS into GEMM tiles on H200 but is "decoder-bound at small batch" (https://arxiv.org/html/2606.15789).
- Fixed-length wins: ZipServ's 3-bit exponent codeword + three 64-bit bitmaps per 8×8 tile, decode via popc/shuffle, no LUT, ≈11.3 bits/elem (1.4×), 1.31–2.21× over cuBLAS on 4090/L40S but **does not beat cuBLAS on A100/H800** (abundant bandwidth). Doubleword's fused tANS on FP8 exponents hit 993 GB/s logical on a 4090 (1.09× amplification, max 1.15× for FP8).
- All of this needs real weight statistics (BF16 exponents ≈2.6–3 bits entropy). i.i.d. random nibbles/FP8 bytes are incompressible; NVFP4 scales are only 12.5% of bytes.

**5. Numerical analysis**
- Exact split FP32 = b0 + 2⁻⁸b1 + 2⁻¹⁶b2 (3 BF16, lossless); BF16x9 keeps all 9 products, BF16x6 drops 3 and is safe only if exponents lie in ≈[−110,111] (https://arxiv.org/pdf/2605.16617; Henry et al. https://www.arxiv.org/pdf/1904.06376). cuBLAS BF16x9 is **CC 10.0/10.3 only — not Hopper** (https://docs.nvidia.com/cuda/cublas/index.html).
- *Derived*: with activations already exact BF16, w·x needs only weight slices: 2 slices → rel. error ≈2⁻¹⁷ ≈ 8e-6, 3 slices ≈ FP32. Both clear 1e-4 rel-L2 with FP32 accumulation; 1 slice (2⁻⁹ ≈ 2e-3) fails.

**6. Physics analogies**
- QUDA gauge reconstruct (18→12/8 reals, https://arxiv.org/pdf/1011.0024): works because SU(3) unitarity is an algebraic constraint — weights have none → no transfer.
- State-vector gate fusion (cuQuantum: 5-qubit fusion on H100, 2.35–3.0 TB/s, https://arxiv.org/pdf/2308.01999): transfers as "one pass over memory for several ops" = fuse dequant+GEMV+router+norm; the only cost is register pressure.
- cuTensorNet slicing = split-K: sum of independent slices, increases FLOPs (https://docs.nvidia.com/cuda/cuquantum/22.03.0/_sources/cutensornet/overview.rst.txt). Transfers as split-K across CTAs to guarantee ≥132×N contiguous streams; reduction cost is tiny for GEMV.
- Temporal blocking (AN5D, https://arxiv.org/abs/2001.01473v2): needs reuse across time steps; weights are touched once → no transfer.

**7. Electronics/LUT angle**
- Marlin's LOP3 (`(0xf0&0xcc)|0xaa`) + PRMT-with-magic-constant + one `__hsub2`/`__hmul2` per 2 values is the standard 4-bit→BF16 path (https://huggingface.co/RedHatAI/quantization/blob/refs%2Fpr%2F3/gptq_marlin/dequant.h). *Derived*: E2M1 is a bit-prefix of BF16 — shift the 3 magnitude bits to bits 8:6, OR sign to bit 15, fold 2¹²⁶ into the FP8 block scale → exact including 0/0.5; one LOP3 + one HFMA2 per pair. PRMT's 3-bit byte selector also implements an exact 8-entry LUT for the 3 magnitude bits (PTX ISA, section text not fetched — *semantics from memory*).
- vLLM serves NVFP4 on Hopper via Marlin W4A16 (dequant→BF16 mma) since Hopper lacks FP4 tensor cores (https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/discussions/12). FLUTE is Ampere-tailored; LUT-GEMM targets BCQ 3-bit (A100 FFN 0.225–0.269 ms); LUT Tensor Core is simulated hardware (https://arxiv.org/pdf/2408.06003). None demonstrates bandwidth-bound NVFP4+16-elem FP8 scales on H100.

**Top 7 (plausibility × payoff)**
1. TMA/cp.async.bulk producer with 2–3 outstanding 8–16 KB copies per SM, ≤2 plain loads/thread elsewhere (avoids the 35% LDG post-peak drop). Rule-safe.
2. Fuse every per-call memory pass (dequant, GEMV, router, scale) into one kernel; PDL to overlap prologue with prior tail (~2–3 µs/launch on H100, https://forums.developer.nvidia.com/t/too-much-time-for-kernel-launch-latency/233645/3). Rule-safe.
3. Verify power/clock with nsys 100 kHz metrics; if H200 truly averages 127 W you are not at peak bandwidth — fix concurrency first. Diagnostic, rule-safe.
4. FP32 router stored as 3 exact BF16 planes (repack); compute with BF16 tensor-core MMAs, 3 slices (≈FP32) or 2 slices (8e-6) — the 2-slice read saves 33% of router bytes but is a gray area under "no precision reduction"; storage stays lossless.
5. Split-K/slicing so every call issues ≥1 CTA per SM with contiguous 32 B-aligned streams covering all 192 PCs; needed most for 10 MB calls (~2 µs of streaming vs µs-scale launch).
6. E2M1→BF16 via bit-prefix shift + scale-folded 2¹²⁶ (≈1 op per value, ~15% of the 7-ops/byte budget). Rule-safe.
7. ZipServ-style fixed-length bitmap exponent coding for BF16 weights (1.4×, branch-free) — only if test tensors have real exponent statistics; measure entropy first. Lossless, rule-safe; ALU-marginal on Hopper (ZipServ lost to cuBLAS on H800).

**Rejected**
- Huffman/rANS in-kernel: ≤0.6 TB/s decode on H100 vs 4.8 TB/s needed; +41% batch-1 latency (Cloudflare).
- Compressing NVFP4 nibbles, FP8 KV, or FP8 scales: random data incompressible; even real FP8 ≤1.15×; scales are 12.5% of bytes.
- cuBLAS BF16x9/Ozaki emulation: Blackwell-only for FP32; Ozaki FP64 irrelevant.
- Gauge reconstruction / temporal blocking: no algebraic constraint, no temporal reuse in weights.
- LUT-based matmul (FLUTE/LUT-GEMM/T-MAC): Ampere/CPU/3-bit targets; Hopper dequant+MMA is already bandwidth-bound.
- Partition-camping tuning: no Hopper evidence; mapping undocumented.
- nvCOMP hardware decompression engine: Blackwell-only.
