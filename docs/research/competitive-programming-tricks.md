# Competitive-programming tricks that transfer to kernels

Written 2026-10-31 during the Open Kernel Challenge kernel track. Leaderboard figures are from that evening. See ../moe-field-guide.md for the synthesis.

Here's the competitive-programming sweep. The CP world has a parallel literature on constant-factor speedups, and several ideas map onto this problem.

## What transfers

**1. The CP bible for this: Algorithmica's *Algorithms for Modern Hardware*** ([en.algorithmica.org](https://en.algorithmica.org/))
- Written by Sergey Slotin, a Codeforces grandmaster. It's the one CP resource whose topics are exactly ours: caching, SIMD, pipelining, memory bandwidth.
- Its case studies include a SIMD-wide segment tree 10× faster than Fenwick, and a blocked, vectorized matmul tutorial. His [Codeforces blog](https://mirror.codeforces.com/blog/sslotin?locale=ru) announces the chapters.
- Read the HPC chapters on memory (bandwidth vs latency, prefetching, cache lines) and SIMD. Swap "AVX2 lane" for "CUDA thread" mentally and most of it carries.

**2. The bitset mindset: do 64 things per instruction** ([Chilli's blog](https://codeforces.com/blog/entry/77480))
- CP programmers pack bits into 64-bit words to get ~64× over scalar loops. On a GPU the same instinct is "pack 8 FP4 nibbles into a 32-bit register and operate on all of them with one `lop3`/`prmt`." That's exactly the dequant trick we need.
- Branchless counting (`cnt += !(a&b)` instead of `if`) is the CP version of avoiding warp divergence.

**3. Hilbert-curve ordering of queries: a real CP trick that's also a GPU trick** ([gepardo's blog](https://codeforces.com/blog/entry/61203))
- Mo's algorithm sorts offline queries along a Hilbert curve so consecutive queries touch nearby memory. Reported 1.0–1.4× normally, up to 7.7× when queries are sparse.
- GPU GEMMs rasterize tiles along the same curves for L2 reuse. For us the payoff is small (every weight byte is read once), but the principle "order your work to keep what's hot in cache" applies to the attention problem, where 8 query heads share one KV head.

**4. Four Russians: trade a lookup table for arithmetic** ([Albrecht, Bard, Hart](https://arxiv.org/abs/0811.1714), [CUDA version](https://vestniken.bmstu.ru/eng/catalog/mathsim/hidden/16.html))
- Precompute a table over small bit-groups, then process k bits per lookup instead of 1. This is the ancestor of FLUTE and T-MAC, which do LUT-based 4-bit matmul.
- The M4RI paper's conclusion is telling: the actual bottlenecks are memory reads and writes and data locality. Same as here.
- On GPU, the lookup is `prmt` (a 16-entry byte table in two registers, 4 lookups per instruction) or shared memory. Our dequant could be a 16-entry table keyed by the nibble.

**5. Sqrt decomposition and offline batching**
- The CP habit of sorting work so that related items are adjacent is `moe_align_block_size` in disguise: sort tokens by expert so each expert's rows are contiguous.
- "Offline" thinking also applies to `prepare()`: do every reordering, padding and precomputation there, where it's untimed.

## Specific CP tricks worth stealing

| CP trick | Kernel equivalent | Where it applies |
|---|---|---|
| Fast I/O (read the whole buffer, parse in registers) | TMA bulk copies of whole tiles, parse (dequant) in registers | MoE, bf16_proj |
| Precompute everything that doesn't depend on the query | Precompute in `prepare()`: layouts, scale tables, expert maps | All problems |
| Bitset / SWAR | Pack 8 nibbles per register; `lop3`+`prmt` | MoE dequant |
| Branchless select | Mask-and-blend instead of `if` inside warps | Routing, causal masks |
| Reverse iteration / loop order for cache | Loop over K in the order the weights were laid out | All GEMMs |
| Hilbert/Mo ordering | Group 8 q-heads sharing a KV head into one CTA | Attention |
| Lazy propagation (defer work until needed) | Defer the RMSNorm until the last split-K partial lands | MoE, decode_layer |
| Meet in the middle | Split-K: two half-problems, cheap merge | bs=1 shapes |

## What doesn't transfer

- **Asymptotic cleverness (segment trees, FFT, suffix automata).** Our problem is fixed-size dense linear algebra; the algorithm is settled. Only the constant factor is in play.
- **Strict-aliasing and compiler-flag hacks** ([blazingio post](https://codeforces.com/topic/131287/diff/en4/en5)): CPU-specific, and Triton/CUDA give direct control anyway.
- **Pragmas like `#pragma GCC target("avx2")`**: no equivalent; you pick instructions explicitly.

## The useful meta-lesson from CP

Top CP speedups came from a few habits, not dozens of tricks:
1. Measure, find the one hot loop, make it tight.
2. Pack data so one instruction does many things.
3. Order work for cache locality.
4. Move everything you can offline.

That's also the kernel plan. Our hot loop is "stream weights and dequant," and `prepare()` is our offline pass.

Sources also consulted: [gepardo's Hilbert Mo thread](https://codeforces.com/topic/61548/en4), [M4RI (ACM TOMS Algorithm 898)](https://core.ac.uk/works/393403), [Four Russians for RNA folding](https://dcmpx.remotevs.com/org/arxiv/PL/pdf/1307.7820).
