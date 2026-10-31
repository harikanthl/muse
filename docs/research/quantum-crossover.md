# Quantum physics and quantum computing: what transfers to GPU kernels

Written 2026-10-31 during the Open Kernel Challenge kernel track. Leaderboard figures are from that evening. See ../moe-field-guide.md for the synthesis.

I've finished the research. There's a real bridge between quantum physics and GPU silicon, but it runs through the mathematics quantum computing uses, not through quantum hardware.

# Quantum physics → GPU silicon: what transfers, and what to test

## The short answer about quantum computers

**Don't use them for this, even free ones:**
- **Nothing to run there.** The judge runs your file offline inside a fixed Docker image with no network, so a QPU can't be in the loop.
- **Wrong kind of problem.** The bottleneck is moving 28 MB of classical weights through HBM. Loading classical data into a quantum state is the known weak point of quantum algorithms. That's exactly why Ewin Tang could "dequantize" the quantum recommendation-system speedup into a classical algorithm with only polynomial slowdown ([Chia, Gilyén, Li, Lin, Tang, Wang, J. ACM 2022](https://ar5iv.labs.arxiv.org/html/1910.06151)).
- **Tiny budgets.** IBM's free plan gives minutes of QPU time per month. As far as I know, Google's hardware is limited to approved research partners, though its Cirq simulator is free. Neither changes the math above.

**What does transfer:** the tools quantum physicists built to simulate quantum systems on GPUs. State-vector simulation and tensor-network contraction are memory-bound workloads with the same structure as ours, and their tricks map directly onto this problem.

## The bridges

| Quantum-side idea | GPU-side equivalent | Experiment |
|---|---|---|
| **Entropy** (Shannon / von Neumann) | Lossless weight compression. [DFloat11](https://huggingface.co/papers/2504.11651) Huffman-codes BF16 exponents, cutting ~30% with bit-exact output. | **E1** |
| **Gate fusion** in state-vector sims: fewer passes over the state vector means less bandwidth ([cuQuantum paper](https://arxiv.org/pdf/2308.01999), [CUDA-Q docs](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html)) | Fuse routing, GEMM1, SwiGLU, GEMM2 and the norm into one persistent kernel | **E2** |
| **Qubit-index permutation** (cuStateVec's bit-swap and permutation APIs) | Lossless relabeling of weight layouts: permute K the same way in W and x, so the dot product doesn't change | **E3** |
| **Circuit synthesis** (minimizing gate count) | Superoptimize the FP4→BF16 decode as a sequence of `lop3` / `prmt` instructions | **E4** |
| **Contraction-path hyper-optimization and slicing** ([cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/22.07.1/cutensornet/overview.html): sample many paths, keep the cheapest; slicing creates independent work) | Tile shape and split-K search. Slicing is split-K. | **E5** |
| **Device physics** (DRAM row buffers, bank parallelism) | Long contiguous runs per CTA keep DRAM row buffers open; spread work across banks | **E6** |
| **Quantum-inspired sampling** (Tang) | Approximate low-rank matmul | **Rejected:** the rules forbid lower precision |

## The experiments, ranked by expected value per check run

**E5: run a remote lab inside `prepare()`. Highest leverage.**
- `prepare()` gets 300 untimed seconds on the judge's real H200, and `print()` output comes back in the log (last 16,000 characters).
- One check run can therefore benchmark dozens of tile, split-K and pipeline-stage configs under CUDA-graph replay. That's like cuTensorNet's hyper-optimizer, which samples many contraction paths and keeps the cheapest.
- **Integrity line:** this is config tuning, not precomputing outputs. To stay clearly inside the rules, I'd use it only in `okc check` runs, then hard-code the winning configs in the scored submission.

**E4: superoptimize the dequantization.**
- Earlier I estimated a budget of about 4 ALU ops per weight. The goal is ≤2.
- The search runs entirely on CPU, and it can be verified exhaustively: each byte has only 256 possible values, so every candidate sequence gets tested against all of them.
- Precedents exist: Marlin builds its int4 decode from `lop3` with compile-time truth tables ([dequant.h](https://huggingface.co/RedHatAI/quantization/blob/refs%2Fpr%2F3/gptq_marlin/dequant.h)). Microsoft reorders int4 elements as [e0,e2,e4,e6,e1,e3,e5,e7] to save logic ops ([paper](https://arxiv.org/pdf/2211.10017)). I found no published automated search, so this could be new work.

**E3 + E6: one repack in `prepare()` that does two jobs.**
- **Order the nibbles** so the decode from E4 is cheapest.
- **Lay out memory** so each CTA's weight tile is a single contiguous chunk of tens of KB, fetched with TMA bulk copies. That maximizes DRAM row-buffer hits.
- Both are lossless permutations, which the rules allow.

**E2: one fused persistent kernel.**
- GEMM1's output is tiny, 8 KB per token in BF16, so it can stay in shared memory or L2 instead of going back to HBM.
- Putting the whole chain in one kernel removes about 12 kernel launches. At bs 1, launches eat roughly 3× the entire 11.4 µs budget.
- **Scheduling:** run the most heavily loaded experts first (longest-processing-time-first) to shrink the tail on skewed routings.

**E1: lossless scale compression. Measured: small ceiling, real risk.**
- I computed it from the judge's own weight generator.
- **E2M1 codes:** drawn uniformly at random, so exactly 4.0 bits each. Incompressible.
- **FP8 block scales:** only 29 distinct values and **3.71 bits of entropy** out of 8.
- **Ceiling:** at most **6.0% fewer bytes**. A practical 5-bit codebook saves about 4%, and only if decoding through a shared-memory lookup is close to free.
- **Integrity line:** build the codebook from whatever scales arrive, never from the generator's known distribution, and fall back when a layer has more than 32 distinct values. A scheme tuned to synthetic data is the kind of thing the final human review removes.
- Verdict: try it last, as a refinement.

## Order of work

1. Build the baseline Triton kernel.
2. Run **E4** on CPU in parallel; it costs no check runs.
3. Run the **E5** in-`prepare()` tuner: 1–2 check runs replace roughly 50 single-config runs.
4. **E3/E6** repack.
5. **E2** fusion.
6. **E1** only if there's time left.

Other sources: [Ewin Tang's line of work](https://alphaxiv.org/abs/1910.06151), [DRAM organization and row-buffer behavior](https://par.nsf.gov/servlets/purl/10347486), [GPU memory microbenchmarking](https://arxiv.org/pdf/1509.02308), [Marlin MoE (TensorRT-Edge-LLM)](https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api/kernels/moeMarlin.html).
