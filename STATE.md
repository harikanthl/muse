# Where we are (saved 2026-10-30, ~8 pm PT)

## Status
- Research phase done. Synthesis: `docs/moe-field-guide.md`. Sources: `docs/research/`.
- No kernel written yet. `muse_moe_w4a16/submission.py` is still the plain PyTorch starter.
- Runs used: 1 check run on `muse_moe_w4a16` (the baseline, for its profile). Everything else untouched.
- okc CLI: `~/.okc/bin/okc` (not on PATH). Logged in; rules accepted by the user.

## Deadlines
- Kernel track closes **Nov 4, 11:59 pm PT** (08:00 UTC Nov 5). ~125 h left at save time.
- Serving track: Nov 5 noon PT to Nov 19. Prizes are paid only there.

## Rules facts confirmed from the site (openkernelchallenge.dev/llms.txt and /rules)
- Kernel boards carry no prizes. Top 50 on the overall kernel board at close get serving-queue priority. Top contestants may be offered a Meta interview.
- Overall board counts 4 problems: bf16_proj, moe_w4a16, attention_decode, decode_layer. **Prefill does not count.**
- Overall board is server-computed (combined decode-step speedup); public API shows only rank and problems solved. On Oct 30: 45 of the top 50 had all 4 solved; one team at #14 had only 1.
- Limits: 10 scored + 20 check runs per problem per day, reset 00:00 UTC. Identical code returns the earlier result. 15-minute cap per run.
- Router exception: FP32 router weights may run as BF16 hi+lo with FP32 accumulation at any batch size.
- Attention queries may be rounded to FP8 E4M3, nothing coarser.
- Final review reruns top entries on scored and nearby shapes; much slower nearby shapes get a closer look.
- The Auto Research app (if running) shares the same per-problem allowance.

## Next steps, in order
1. E0: one check run on muse_moe_w4a16 that prints from `prepare()` which kernels the judge image has (FlashInfer, MonoMoE, Marlin, CuTe DSL, TileLang).
2. Triton MoE kernel: fused routing, bit-trick NVFP4 dequant, GEMM1+SwiGLU, GEMM2+norm. Weight-major, tokens on N, split for bs=1.
3. Quick entries on bf16_proj, attention_decode, decode_layer for overall-board coverage.
4. Then iterate on MoE (the `prepare()` tuner, repack, fusion), reuse in decode_layer.
