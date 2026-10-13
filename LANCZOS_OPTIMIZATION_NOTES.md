# CUDA Lanczos Optimization Notes

Date: 2026-05-26

This is the merged working plan for optimizing the CUDA Lanczos path. It
combines the useful parts of the Claude and Codex reviews, with priorities
updated after both reviews agreed that the native Blackwell build issue is the
first blocker.

No profiling conclusions should be treated as final until the CUDA artifacts
are rebuilt for the actual GPU architecture and a fresh baseline is captured.

## Code Surface

| File | Role |
|---|---|
| `common/lanczos/lanczos.c` | Top-level Lanczos iteration |
| `common/lanczos/lanczos_matmul.c` | `mul_sym_NxN_NxB`, `mul_MxN_NxB` |
| `common/lanczos/gpu/lanczos_matmul_gpu.c` | GPU matrix setup and SpMV orchestration |
| `common/lanczos/gpu/lanczos_vv.c` | Host wrappers for vector-vector kernels |
| `common/lanczos/gpu/lanczos_kernel.cu` | `mask`, `xor`, `inner_prod`, `outer_prod` kernels |
| `cub/spmv_engine.cu` | CSR XOR SpMV kernel and autotune |
| `cub/Makefile` | Builds `spmv_engine.so` and `sort_engine.so` |

## Critical Finding: Native sm_120 Build Is Broken

The working GPU is an RTX 5070, compute capability 12.0. The checked CUDA
artifacts were not native Blackwell:

```bash
cuobjdump --list-elf cub/spmv_engine.so
# ELF file 1: spmv_engine.1.sm_90.cubin
# ELF file 2: spmv_engine.2.sm_90.cubin

cuobjdump --list-elf lanczos_kernel.fatbin
# ELF file 1: lanczos_kernel.1.sm_90.cubin
```

The top-level Makefile can ask nvcc for `sm_120`:

```make
lanczos_kernel.fatbin:
	$(NVCC) -arch sm_$(SM) -fatbin -DVBITS=$(VBITS) -o $@ $<
```

But the CUB submake is called as:

```make
cd cub && make WIN=$(WIN) WIN64=$(WIN64) VBITS=$(VBITS) sm=$(SM)0 && cd ..
```

For `CUDA=120`, that passes `sm=1200` into `cub/Makefile`.

`cub/Makefile` currently uses substring matches such as:

```make
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_20,code=\"sm_20,compute_20\"
endif
```

`SM_ARCH=1200` therefore matches the old `200` case and tries to emit
`compute_20/sm_20`, which modern CUDA has removed. If stale artifacts remain,
Lanczos can silently keep using an older `spmv_engine.so`.

This must be fixed before interpreting any performance data.

### Build Fix

Required:

- Add explicit handling for `SM_ARCH=1200` or `SM_ARCH=120`, producing
  `compute_120,sm_120`.
- Replace all substring architecture matching in `cub/Makefile` with exact
  matching. Future architectures must not accidentally match older entries.
- Make the top-level `cub/built` target sensitive to at least:
  - `cub/Makefile`
  - `cub/spmv_engine.cu`
  - `cub/spmv_engine.h`
  - `cub/sort_engine.cu`
  - `cub/sort_engine.h`
  - `VBITS`
  - selected `SM`
- Consider passing `sm=$(SM)` rather than `sm=$(SM)0`, or rename the CUB-side
  variable to make the expected format explicit.

Verification:

```bash
make clean
make all CUDA=120 VBITS=64
cuobjdump --list-elf cub/spmv_engine.so
cuobjdump --list-ptx cub/spmv_engine.so
cuobjdump --list-elf lanczos_kernel.fatbin
cuobjdump --dump-resource-usage cub/spmv_engine.so
cuobjdump --dump-resource-usage lanczos_kernel.fatbin
```

Expected result: native `sm_120` cubins for both `cub/spmv_engine.so` and
`lanczos_kernel.fatbin`.

## Baseline Instrumentation Before Kernel Work

Add lightweight timing or NVTX ranges before tuning. Without per-direction
labels, normal SpMV and transpose SpMV are difficult to separate in nsys.

Useful labels:

- `mul_core`
- `mul_trans_core`
- each `spmv_engine_run`
- dense-row handling in `mul_packed_gpu`
- dense-row handling in `mul_packed_trans_gpu`
- `vv_mul_BxN_NxB`
- `vv_mul_NxB_BxB_acc`
- `lanczos_kernel_outer_prod`
- `lanczos_kernel_inner_prod`

Baseline capture:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --stats=true \
  ./msieve -nc2 "select_density=100" -g 0
```

For kernel details:

```bash
ncu --set full --kernel-name regex:csr_spmv_xor_warpmerge_kernel ./msieve ...
ncu --set full --kernel-name regex:lanczos_kernel_outer_prod ./msieve ...
ncu --set full --kernel-name regex:lanczos_kernel_inner_prod ./msieve ...
```

Track:

- Normal SpMV time vs transpose SpMV time.
- DRAM throughput.
- L2 hit rate.
- Atomic throughput and atomic stalls.
- Achieved occupancy and register/shared-memory limits.
- Stall reasons.
- Launch count per iteration.

## Per-Iteration Shape

At default `VBITS=64`, each Lanczos iteration roughly does:

| Call | Count | Notes |
|---|---:|---|
| `mul_sym_NxN_NxB` | 1 | One multiply by `A`, then one multiply by `A^T` |
| `vv_mul_BxN_NxB` | 2 common, more for checks | Outer products producing `VBITS x VBITS` data |
| `vv_mul_NxB_BxB_acc` | 3-4 common | Full-vector updates into `vnext` and `x` |
| `vv_mask`, `vv_xor`, `vv_copy`, clears | several | Streaming vector kernels |

SpMV is expected to dominate, but confirm this after native `sm_120` artifacts
are built.

## Priority Order

### 1. Fix Native sm_120 Build

This is prerequisite work, not an optimization experiment.

If the code is running `sm_90` cubins or PTX-JITed code on `sm_120` hardware,
then register counts, occupancy, scheduling, cache behavior, and any ncu
conclusions are not a clean baseline.

### 2. Add Timing Labels / NVTX

This is also prerequisite work.

The key split is normal SpMV vs transpose SpMV. The two directions may have
different row-length distributions and different bottlenecks. The optimization
choice depends on knowing which direction costs more.

### 3. Baseline With nsys / ncu

Use a repeatable matrix, ideally a checkpoint restart or a fixed `-nc2`
workload. Record:

- GPU model and compute capability.
- CUDA toolkit version.
- `CUDA` and `VBITS` build settings.
- `block_nnz`.
- matrix dimensions, nonzeros, dense rows.
- number of normal and transpose matrix blocks.
- total iteration time and kernel breakdown.

### 4. Sweep `block_nnz`

`block_nnz` is already exposed through the `-nc2` argument string:

```text
block_nnz=N
```

Suggested initial sweep:

```text
block_nnz=256000000
block_nnz=512000000
block_nnz=1000000000
block_nnz=1750000000
```

Why this matters:

- Smaller blocks reduce the active slice of `x[col]` for normal SpMV.
- That can improve L2 locality for random gathers.
- But smaller blocks also increase launch count, rowptr traffic, and repeated
  accumulation into the output vector.

Rough L2-fit target for RTX 5070:

- L2 size is about 48 MB.
- At `VBITS=64`, each `x` entry is 8 bytes.
- About 6 million `x` entries fit in L2.
- With a typical NFS matrix average around 200 nonzeros per column, this
  corresponds to about 1.2 billion nonzeros per block.

The current default, `block_nnz=1750000000`, sits roughly 50% above that
back-of-envelope L2-fit ceiling. The locality hypothesis therefore predicts
that the `256M` to `1B` end of the sweep should improve SpMV time. If it does
not, then launch overhead, rowptr traffic, atomics, or the actual row/column
distribution is dominating the simple L2 model.

Measure total iteration time, not only L2 hit rate. The best point is a balance
between locality and launch/accumulation overhead.

### 5. Lane-0 `csr_upper_bound` + Warp Broadcast

Current `csr_spmv_xor_warpmerge_kernel` computes:

```cuda
int row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
uint32_t row_end = rowptr[row + 1];
```

All lanes in a warp compute the same starting row. That duplicates the binary
search and rowptr loads 32 times.

Low-risk experiment:

```cuda
int row;
uint32_t row_end;
if (lane == 0) {
    row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
    row_end = rowptr[row + 1];
}
row = __shfl_sync(0xffffffffu, row, 0);
row_end = __shfl_sync(0xffffffffu, row_end, 0);
```

Do the same kind of lane-0/broadcast treatment for the row-advance path if it
stays warp-uniform.

Expected impact is small, but the patch is localized and correctness risk is
low.

### 6. SPMV Tile / Direction Tuning

Current `spmv_engine.cu` has one global autotune decision:

```c++
struct SpmvEngine { int threads_per_block; int warp_items; bool tuned; };
```

`spmv_autotune()` runs once and reuses the result for all normal and transpose
blocks.

Experiment:

- Add explicit override:

```text
spmv_warp_items=256
spmv_warp_items=512
spmv_warp_items=1024
spmv_warp_items=2048
```

- Then allow a direction split:

```text
spmv_warp_items=normal,transpose
spmv_warp_items=512,1024
```

If this matters, replace global one-shot tuning with per-direction or per-block
tuning. Per-direction tuning is likely the useful first split. Per-block tuning
only becomes interesting if `block_nnz` is reduced enough to create many more
matrix blocks.

## Conditional Next Steps

These depend on what the native baseline shows.

### If Atomics Are a Bottleneck: Reduce SpMV Atomics

Current SpMV uses `atomicXor` for every segment result. This is simple and
correct because:

- rows can straddle multiple warp tiles inside one kernel, and
- multiple matrix blocks accumulate into the same output vector.

Possible directions:

- Hybrid owned-row path:
  - For rows fully owned by a single warp tile, do non-atomic `y[row] ^= sum`.
  - For boundary or long rows, keep `atomicXor`.
- Row-per-warp or row-per-CTA kernel:
  - One warp/CTA owns a row and stores without atomics.
  - Use fallback for very long rows.

Important correction:

Do not assume the pre-SpMV `cuMemsetD8` can simply be removed. `mul_packed_gpu`
and `mul_packed_trans_gpu` clear the output once, then multiple matrix-block
SPMV launches XOR-accumulate into the same output vector. A plain store from
one block would overwrite contributions from earlier blocks.

Removing or reducing the memset requires a different accumulation scheme, not
just replacing atomics with stores.

One viable redesign:

- The first matrix-block SpMV initializes the output vector and uses plain
  stores where it owns a row.
- Later matrix-block SpMVs continue to XOR-accumulate into the initialized
  output vector.
- For rows split across warp tiles or handled by long-row fallback, keep
  atomics or use a second-stage reduction.

The first pass must initialize every output row, including rows with no
nonzeros in the first matrix block. That can mean a row-oriented first-block
kernel that stores zero for empty rows, or a separate compact initialization
for first-block-empty rows. Without that, later `atomicXor` calls would operate
on undefined output values.

For one-block matrices, this can remove the standalone memset and most SpMV
atomics. For two- or three-block matrices, it still removes the standalone
memset and reduces atomic traffic for the first block, but the later blocks
still need safe accumulation.

### If Full-Vector Update Traffic Is Material: Fuse `NxB * BxB` Updates

The Lanczos recurrence performs several full-vector update passes:

```text
vnext ^= v[0] * D
vnext ^= v[1] * E
vnext ^= v[2] * F   // conditional
x     ^= v[0] * X
```

Each pass launches `lanczos_kernel_inner_prod` and scans a large vector.

Experiment:

- Copy all small BxB matrices to one scratch region.
- Launch one fused kernel that updates `vnext` and optionally `x` in one scan.

Potential win:

- Fewer launches.
- Fewer vector reads/writes.

Risk:

- Register pressure. The stage-1 fusion experiments showed that correctness
  can be clean while performance regresses due to lost occupancy. Check
  `cuobjdump --dump-resource-usage` immediately.

### If Outer Product Is Material: Fuse or Rework Outer Products

The common iteration computes:

```text
v[0]^T   * vnext
vnext^T * vnext
```

These are two `vv_mul_BxN_NxB` calls, each scanning `N`.

At `VBITS=64`, the inner `w_x * w_y` loop inside one `outer_prod` call is only
`1 x 1`, so this is not a huge VBITS-loop fusion opportunity. The possible
benefit is saving one whole-vector scan and one launch by computing both outer
products together.

Treat this as a measured experiment, not a guaranteed win.

Also consider a two-stage reduction if `lanczos_kernel_outer_prod` shows global
atomic contention on the tiny `VBITS x VBITS` result:

- kernel 1 writes per-CTA partial results,
- kernel 2 reduces partials.

### If SpMV Is DRAM-Bound After `block_nnz`: Consider L2 Persistence

CUDA access-policy windows can bias L2 retention for the input vector slice.
This is a backstop if `block_nnz` cannot make the active `x` range cache well.

Do this only after measuring native `sm_120` behavior and the `block_nnz`
sweep.

### If Dense Rows Are Surprisingly Large: Consider Stream Overlap

The sparse SPMV loop and dense-row kernels in `mul_packed_gpu` are serialized
on stream 0. At `VBITS=64`, dense rows are likely small, so overlap is probably
not worth engineering first.

Only revisit if nsys shows dense-row kernels are material.

### If Memory Capacity Is the Limiter: Revisit A / A^T Storage

`gpu_matrix_init` builds independent CSR-like structures for normal and
transpose directions. This roughly doubles sparse matrix storage on the GPU.

Sharing storage or using a compact dual-view representation is more about
fitting larger jobs than reducing milliseconds per iteration. It is a larger
engineering project and should not be the first speed optimization.

### If Nothing Else Moves: Matrix Reordering

A matrix permutation or graph-partitioning pass could improve locality for
random gathers into `x`, but this is speculative and invasive:

- must preserve solver correctness,
- needs persisted permutations,
- may require external tooling or substantial preprocessing time.

Only consider after native profiling shows SpMV is still gather/locality bound
and simpler geometry changes fail.

## Things Not Recommended First

- Compressible memory: already tried in this tree and commented as not useful.
- L2 fetch granularity tweak: already tried and commented as not useful for
  `VBITS=256`.
- cuSPARSE: GF(2) XOR is not a supported cuSPARSE semiring.
- CUDA Graphs: potentially useful for launch overhead, but only after the
  kernel sequence is stable.
- Broad fusion before profiling: stage-1 experiments showed fusion can regress
  through register pressure even when it removes a launch.

## Immediate Checklist

1. Patch `cub/Makefile` architecture matching.
2. Patch top-level CUB build dependencies / stale stamp behavior.
3. Clean rebuild with `CUDA=120 VBITS=64`.
4. Verify `sm_120` artifacts with `cuobjdump`.
5. Add NVTX/timing labels around normal SpMV, transpose SpMV, dense rows, and
   vector-vector kernels.
6. Capture nsys baseline on a repeatable `-nc2` workload.
7. Capture ncu on the dominant SPMV kernel.
8. Sweep `block_nnz`.
9. Apply lane-0 `csr_upper_bound` broadcast and remeasure.
10. Choose the next structural kernel experiment from actual bottlenecks.

## Results: 2026-06-09 Baseline + block_nnz Sweep (basement folder)

Setup: RTX 5070 native sm_120 build, CUDA 13.2, VBITS=64, C168_1074_2202
density-100 matrix (9538691 x 9538916, 884.6M sparse nnz + 64 packed dense
rows), 10.5 GB VRAM, benchmark = 5-min `-ncr` checkpoint restarts.

nsys NVTX split (default block_nnz, per iteration):

| Range | ms/iter | share |
|---|---:|---:|
| normal SpMV (A·x) | 84.0 | 70% |
| transpose SpMV (A^T·x) | 28.0 | 23% |
| vector ops + dense rows + memset | ~4 | 3% |

Normal direction is 3x slower than transpose over the same nnz.
Profile: /tmp/lanczos_nsys_baseline.nsys-rep

block_nnz sweep:

| block_nnz | blocks | ms/iter | speedup |
|---|---:|---:|---:|
| 1.75e9 (default) | 1 | 119.7 | 1.00x |
| 1e9 | 1 | 119.9 | 1.00x |
| 512M | 2 | 54.4 | 2.20x |
| **256M** | 4 | **46.6** | **2.57x** |
| 128M | 7 | 60.4 | 1.98x |
| 64M | 14 | 88.4 | 1.35x |

Clear minimum near 256M; L2-locality hypothesis confirmed. Note the sweep
was run at VBITS=64 (8B per x entry); at VBITS=256 the optimum should shift
~4x lower (~64M). Full-solve projection at 256M: ~1h57m vs ~5h default.

Next: finer probe 192M-384M, ncu on normal-direction kernel (atomics vs
DRAM), per-direction warp_items tuning, lane-0 broadcast.

## 2026-06-10: Multi-block Correctness Verification

After a sqrt failure on a solve that had mixed block_nnz experiments and a
likely matrix rebuild, multi-block SpMV was verified two independent ways:

1. `cub/spmv_blocktest.c` — standalone harness (dlopen spmv_engine.so):
   single vs 2/4/7/14 column slices bitwise-match a CPU reference on two
   synthetic shapes including 400K-entry heavy rows.
2. Real-matrix A/B: identical seed checkpoint restarted under default
   (1 block) and block_nnz=256M (4 blocks), both run to the same fixed
   dump boundary (new `dump_interval=N` nfs_args knob in lanczos.c skips
   the timing-based recalibration). The two 381MB checkpoints at
   dim_solved=100033 were byte-identical.

Conclusions:
- Multi-block SpMV and block_nnz=256M are safe for production.
- A `.chk` is only valid for the exact `.mat` it started on. Rebuilding
  the matrix draws fresh random quadratic characters, so "rebuild .mat,
  continue from old .chk" silently produces garbage dependencies that
  fail in sqrt as "algebraic side is not a square" on every dependency.

## 2026-06-10: Cross-VBITS Comparison (same matrix, fresh-start 380s windows)

Matrix: C168 9532121 x 9532347, 884.6M sparse nnz, RTX 5070. Metric is
dims/sec (ms/iter is not comparable across VBITS). msieve-reported host
memory in parens.

| VBITS | block_nnz | blocks | dims/s | ms/iter | mem |
|---:|---|---:|---:|---:|---|
| 64  | default (1 blk) | 1 | 528   | 119.7 | (7.4 GB) |
| 64  | 512M  | 2  | 1162 | 54.4  | |
| 64  | 256M  | 4  | **1350** | 46.6-47.5 | |
| 64  | 128M  | 7  | 1046 | 60.4  | |
| 128 | default | 1 | 743  | 171.3 | (7.5 GB) |
| 128 | 256M  | 4  | 1308 | 97.3  | (7.7 GB) |
| 128 | 128M  | 7  | **1421** | 89.6 | (7.9 GB) |
| 128 | 64M   | 14 | 1020 | 124.8 | (8.4 GB) |
| 256 | default | 1 | 1208 | 211.3 | (8.3 GB) |
| 256 | 128M  | 7  | **1435** | 177.8 | (8.7 GB) |
| 256 | 64M   | 14 | 1162 | 219.6 | (9.1 GB) |
| 256 | 32M   | 28 | 769  | 255.2 | (10.0 GB) |

Findings:

- Tuned optima converge: 1350 / 1421 / 1435 dims/s for VBITS 64/128/256.
  Higher VBITS is worth ~5-6% over VBITS=64 once block_nnz is tuned —
  real but small. The convergence suggests a common memory-system limit.
- Untuned (single-block) the spread is huge: 528 / 743 / 1208. This is
  why higher VBITS is the right advice for users who never set block_nnz:
  wider vectors amortize per-nonzero index traffic.
- Optimal block_nnz by VBITS: 256M / 128M / 128M. A "half-L2 active
  x-window" model predicts 278M / 139M / 70M; the v256 deviation (128M
  measured vs 70M predicted) indicates a per-block overhead floor.
  Candidate dynamic formula for lanczos_matmul_gpu.c:739:
    block_nnz = max(128M, (L2_bytes/2) / sizeof(v_t) * avg_col_weight)
  which reproduces all three measured optima on this card.
- VBITS=256 fits this ~9.5M matrix on 12GB (shared with display), but
  vectors scale 4x vs VBITS=64 — capacity, not speed, is the VBITS=256
  concern for larger matrices on this card.

Next: one ncu pass on the tuned config to identify the converged
bottleneck (gather sectors / atomics / DRAM). If the memory system is
saturated, kernel micro-opts (lane-0 broadcast) won't pay; matrix
reordering would be the remaining lever.

## 2026-06-10: ncu Profile of Tuned Config (VBITS=64, block_nnz=256M)

Captured per-direction via NVTX filters (8 kernels each, --set detailed).

Normal direction (3 full blocks @ ~6.8ms + 1 partial @ ~3ms):
- L2 cache throughput 74% (the saturated unit), L2 hit rate 87.8%
- DRAM only 28% (185 GB/s) — NOT DRAM-bound; block_nnz fix confirmed
- SM 39%, occupancy 89%, L1 hit 8% (random gathers miss L1)
- autotune picked warp_items=512 at this block size (1024 at single-block)

Transpose direction (4 blocks, heterogeneous):
- Block 1 (heavy ideal rows, small x-window): L1 hit 86%, L2 only 15%,
  SM 58% — gathers absorbed by L1 when the window is tiny
- Blocks 2-3: L2 throughput 84-88%, L2 hit 87.5%, DRAM ~32% — same
  L2-bandwidth-bound profile as the normal direction

Conclusion: after block_nnz tuning, both directions sit at the L2
BANDWIDTH ceiling (~75-88% utilization, ~88% hit rate). This explains the
cross-VBITS convergence at ~1400 dims/s. Remaining levers, in order:
1. VBITS=256 in production (+~6%, already measured; full 32B sectors per
   gather and 4x less colidx traffic per dim).
2. Owned-row stores to cut atomicXor L2 round-trips (bounded, ~10-15% of
   L2 ops at most).
3. lane-0 csr_upper_bound broadcast (small).
4. Column-clustering / matrix reordering to raise the 8% L1 hit rate —
   the only large lever left, and the most invasive.
The 2.6x from block_nnz was the structural win; everything left is
constants-level (5-20%).

## Cross-Card Validation Protocol

`vbits_block_sweep.sh` packages the benchmark for other GPUs: copy
msieve.dat.mat + msieve.fb + worktodo.ini from the reference C168 job,
run `./vbits_block_sweep.sh <sm>`, return bench_results.tar.gz.

Predictions to check against the Tesla V100 (sm_70, 6MB L2, 900GB/s HBM2):
1. The block_nnz curve should be much flatter than the 5070's, with the
   optimum at large blocks (256M-1.75B) — the L2-resident window is
   unreachable above the per-block overhead floor.
2. The VBITS=256 vs 64 gap should be much larger than the 5070's +6%
   (full-sector gathers attack the DRAM-bound constraint directly).
3. Total block_nnz gain limited (~1.2-1.5x vs the 5070's 2.6x).
If (1)-(3) hold, the dynamic block_nnz formula's floor behavior is right
for small-L2 cards and per-arch guidance becomes: big-L2 cards tune
block_nnz, small-L2 cards raise VBITS.

## 2026-06-11: Tesla V100 Results (small-L2 validation)

V100-SXM2-32GB (sm_70, 6MB L2, 900GB/s HBM2), CUDA 12.1, 20.2M x 20.2M
matrix, 2.244B sparse nnz, 111 nnz/col. Fresh-start 900s windows.
Reference: user's 51h production solve = VBITS=64 default = 574 ms/iter.

| VBITS | block_nnz | blocks | dims/s | ms/iter | host mem |
|---:|---|---:|---:|---:|---|
| 64  | default(1.75B) | 2 | 109.5 | 577 | 18.7 GB |
| 64  | 512M | 5  | 115.5 | 547 | 19.1 GB |
| 64  | 256M | 9  | 127.5 | 496 | 19.7 GB |
| 64  | 128M | 18 | **141.9** | 445 | 21.2 GB |
| 64  | 64M  | 35 | 136.2 | 464 | 23.9 GB |
| 256 | default(1.75B) | 2 | FAILED internal check at shutdown | | 20.2 GB |
| 256 | 512M | 5  | 256.1 | 997 | 20.5 GB |
| 256 | 256M | 9  | **257.5** | 991 | 21.1 GB |
| 256 | 128M | 18 | 246.3 | 1037 | 22.3 GB |
| 256 | 64M  | 35 | 217.5 | 1173 | 24.7 GB |

Prediction scorecard:
1. "Flat curve, optimum at large blocks" — half right. VBITS=256 is flat
   (512M-128M within 4%), as predicted (its windows can never fit 6MB L2).
   VBITS=64 has a real curve: +30% at 128M (9.2MB window ~ 1.5x L2), so
   partial L2 residency pays even on small-L2 cards.
2. "VBITS=256 gap >> the 5070's +6%" — confirmed dramatically: best-vs-
   best +81% (141.9 -> 257.5 dims/s). Pure transaction efficiency.
3. "Total block_nnz gain 1.2-1.5x" — 1.30x at VBITS=64. Combined
   VBITS=256 + tuned block_nnz vs the production config: 2.35x
   (the 51h solve becomes ~22h).

Dynamic formula validation: block_nnz = max(128M, (L2/2)/sizeof(v_t) *
avg_col_weight) predicts: V100 v64 -> 128M (measured optimum, exact);
V100 v256 -> 128M (within 4.4% of measured best); 5070 v64 -> 278M
(~256M optimum, exact); 5070 v128/v256 -> 128M (measured optima, exact).
Within ~5% of measured best on every card/VBITS tested. VALIDATED on
two architectures at opposite ends of the L2 spectrum.

Unified recommendation: VBITS=256 + dynamic block_nnz wins on both
cards (5070: 1435 dims/s; V100: 257 dims/s) WHERE VRAM PERMITS — this
20.2M matrix needs ~21GB at VBITS=256, so 12GB consumer cards must run
VBITS=64 (+formula) for matrices this size.

ANOMALY to investigate: v256 + default (2 blocks of ~1.75B nnz) failed
the periodic consistency check during graceful shutdown on the V100
("error: corrupt state"). All other v256 runs passed continuous checks
for 8-10 min each. Could be the near-clamp 1.75B block size at VBITS=256
on sm_70, or a shutdown-path quirk. The 5070 ran v256+default cleanly.
Do not ship v256 near the 1.75B clamp until understood; the recommended
formula values (128-256M) are unaffected.

## Tuning Heuristic (validated 2026-06-11, RTX 5070 + Tesla V100; RTX 3060 exception added 2026-09-25)

1. Use the largest VBITS that fits VRAM (256 where possible). Tuned
   speed orders 256 >= 128 > 64 on every card tested; the advantage
   grows as L2 shrinks (+6% on 48MB-L2 5070, +81% on 6MB-L2 V100).
2. Set block_nnz = max(128M, (L2_size/2)/sizeof(v_t) * avg_col_weight).
   Within ~5% of measured optimum on the 5070 and V100 at every VBITS.
   Exception (RTX 3060, 3MB L2, VBITS=256): a single block, i.e.
   block_nnz at or above the matrix's sparse nnz, was ~7% faster than
   the 128M default (single runs, against ~5% run-to-run noise) and uses
   less VRAM. Only measured on a 958M-nnz matrix, where 1024M and 1.75B
   are both one block; large multi-block settings were not tested on the
   3060, and on the V100 at VBITS=256 two 1.75B blocks were slightly
   slower than 128M (243.6 vs 246.3). Does not apply at VBITS=64.
   Known failure (2026-09-27, A100 MPI 1x2, 61M-row piece): the fixed
   128M floor is far too small when the GPU's piece has many rows. See
   "A100 Feedback" at the end; a row-proportional floor was added
   2026-09-28 (k=12, see there).
3. If VRAM is tight: raise block_nnz first (frees per-block rowptr
   replicas and costs little speed at high VBITS); drop VBITS only as
   a last resort (-45% on small-L2 cards).
4. use_managed (matrix > VRAM): max VBITS + default/max block_nnz.
   The bottleneck becomes matrix bytes streamed per dim of progress
   = (4B*nnz + rowptr replicas)/VBITS — big VBITS divides it, big
   blocks minimize replication. (Predicted, not yet benchmarked.)
5. Never rebuild the .mat mid-solve (fresh random quadratic characters
   = different matrix; checkpoints are only valid for the exact .mat
   they started on).

Anomaly follow-up: the v256 + default (1.75B, near-clamp) "corrupt
state" failure on the V100 did NOT reproduce on rerun — clean halt,
243.6 dims/s (1048 ms/iter), completing the v256 table (default costs
~5% vs the 256M optimum; flat curve confirmed). Treated as a transient
one-off; no config restriction, but if "corrupt state" ever appears
again, capture the log and investigate the shutdown-path check.

## Planned: VBITS=512 Test (2026-06-12)

Code supports VBITS up to 512 (VWORDS=8 unrolls present; lanczos.h
whitelist). Predictions to check, written before measurement:

1. Speed: v512 ~= v256 + 0-8%. GPU memory moves in 32B sectors; v_t hits
   exactly one sector at VBITS=256 (gather sector traffic per dim
   plateaus there — v512 issues half the gathers but each is 2 sectors).
   The remaining v512 gain is colidx/rowptr traffic halving per dim.
   If correct, VBITS=256 is the efficiency sweet spot and 512 is only
   worth it where its memory cost is free.
2. Memory: vectors and the dense-row block double vs v256 (~+30% total
   footprint). RTX 5070 + 9.5M C168 matrix: ~13GB needed vs 12GB card —
   expect OOM (which is itself the answer for 12GB cards). V100-32GB +
   20.2M matrix: ~30-31GB at large blocks — borderline; prefer
   NNZ_LIST="default 512000000 256000000 128000000" and watch the new
   per-run vram_*.log sampling (added to vbits_block_sweep.sh).

Commands: 5070: build VBITS=512, run plain skip_matbuild (formula picks
block size). V100: ./vbits_block_sweep.sh 70 512 with the NNZ_LIST above.

## 2026-09-24: Single-copy Matrix, segscan Forward Kernel, 4-bit Vector Tables

Setup: RTX 5070 (12GB, 48MB L2, WSL2 — ~1-4.5GB of VRAM held by Windows),
VBITS=256, C170 job, TD=90 matrix 13170233 x 13170412 (1132M sparse nnz,
958M after GPU dense-row packing) and TD=110 matrix 12307226 x 12307402.
Numbers from `bench_la.sh` (90-120s windows after warmup; ~5% noise).

1. `single_copy=1`: skip the transposed CSR entirely; A^T x scatters
   through A's column-slice blocks (csr_spmv_xor_scatter_kernel: groups
   of VWORDS lanes per nonzero, one 64-bit atomicXor per lane, so each
   warp-wide RED covers whole 32B v_t entries). The block's column range
   is the scatter's output window; atomics are cheap only while it stays
   in L2, so single-copy wants much smaller blocks (48M vs 128-256M).
   At equal block size the scatter beat the stored-transpose gather:
   73ms vs 102ms per iteration (single 48M vs stock 256M).
2. Forward kernel was the bottleneck (170ms/iter at both 4 and 20 blocks):
   small blocks leave ~3-4 nnz per row segment and warpmerge spends a
   32-lane reduction (20 shuffles + 4 atomics) on each. New
   csr_spmv_xor_segscan_kernel: 8 groups of 4 lanes each take a nonzero,
   segmented XOR scan across groups, runs carried between steps, 4 steps
   of loads issued ahead. 170 -> 78.5ms/iter at 38M blocks. It is worse
   than warpmerge at large (DRAM-bound) blocks; see the review follow-up
   below for how the kernel is now chosen.
3. inner_prod (y ^= v * X): 4-bit tables, word-major so a warp's 16
   possible entries per word are bank-disjoint: 128 -> 64 branch-free
   lookups per element. outer_prod (X^T Y): 4-bit pieces with half-warp
   table copies (rotation trick needs one slot per lane): half the shared
   RMWs. Together ~16ms -> ~11ms and ~26 -> 19ms per iteration.
4. Rejected: storing the transpose of the heavy rows (rows < 65536 hold
   ~36% of nnz; transpose would fit uint16 indices). Skipping all heavy-
   row atomics only cut the scatter by 19% (~12ms/iter upper bound)
   before paying for the replacement gather and ~750MB VRAM.

Single-copy block_nnz sweep (segscan forward, TD=90): 24M 1233, 32M 1449,
48M 1484, 64M 1182, 96M 821 dims/s. Default is now (L2/3)/sizeof(v_t) *
avg col weight with a 16M floor = 38M (TD=90) / 46M (TD=110).

| TD | mode | block_nnz | dims/s | full ETA | sparse MB |
|---|---|---|---:|---:|---:|
| 90 | stock | 128M default | OOM | - | 8114 |
| 90 | stock | 256M | 981 (best of 3; 891, 798) | 3h43m | 7713 |
| 90 | stock | 1024M | 833 | 4h23m | 7461 |
| 90 | use_managed=1 | 128M | 334 | 10h56m | host |
| 90 | single_copy (first version) | 48M | 985 | 3h42m | 4660 |
| 90 | single_copy (all changes) | 38M default | 1602 | 2h16m | 4962 |
| 110 | single_copy (all changes) | 46M default | 1635 | 2h05m | 5248 |

VBITS=64 build (both layouts fit at their defaults, so this is the clean
comparison): stock 266M default 825 dims/s (4h26m, 9003MB sparse, 11.8GB
peak) vs single_copy 177M default 1163 dims/s (3h08m, 4602MB, 7.7GB peak).
The v64 single-copy default was not tuned further.

Stock runs peak at ~11.9GB of 12.2GB; their spread (981/891/798 at the
same setting) looks like WDDM paging under VRAM pressure. Untested: stock
+ segscan at small blocks on a card with room for it (needs ~2x rowptr
overhead: every block of each direction carries a full rowptr array).

### Review follow-up (same day)

- Fixed a pre-existing bug: the GPU dense-row part of A^T x XORed dense
  block i into b + VBITS*i instead of b (CPU: lanczos_matmul0.c). Only
  matters with more than one dense block (num_dense_rows > VBITS).
- Kernel choice is now per block (spmv_kernel=auto): segscan below 8
  nonzeros per row, warpmerge above. Measured single-copy, forced
  kernels: 128M blocks (9.7 nnz/row) 558 vs 547 dims/s, 256M (19.4)
  404 vs 402 — a tie there, while at 38M (2.9) segscan is ~2x. Applies
  to two-copy transpose blocks too, so small-block two-copy runs get
  segscan automatically.
- Single-copy block_nnz floor raised to 2*nrows: each block carries a
  full rowptr array (53MB at 13M rows), which on small-L2 cards (V100,
  3060) would otherwise eat most of the saving.
- Tried and reverted: capping the inner/outer product grids at the
  resident block count (to build/fold their shared tables fewer times).
  A/B under identical load: capped 1272/1272, uncapped 1571/1558
  dims/s. The 10000-block grid stays.
- Load note: the 1551 dims/s post-review default (TD=90) was measured
  with ~22 load average from unrelated CPU jobs; 1602 was an idle box.

### RTX 3060 check (2026-09-25, small L2)

RTX 3060 12GB (sm_86, 3MB L2, 28 SMs), native Linux, driver 535,
VBITS=256, same TD=90 matrix, code as of the review follow-up:

| mode | block_nnz | dims/s | full ETA | sparse MB | peak VRAM |
|---|---|---:|---:|---:|---:|
| stock | 1024M | 332 | 11h01m | 7461 | 11.1GB |
| stock | 128M default | 310 | 11h48m | 8114 | 11.8GB |
| single_copy | 26M default (2*nrows floor) | 251 | 14h34m | 5514 | 9.2GB |
| single_copy | 64M | 233 | 15h43m | 4409 | 8.1GB |

With a 3MB L2 the scatter's output window can never be L2-resident.
Single-copy's default was ~19% slower than the two-copy default (251 vs
310) and ~24% slower than the best two-copy setting (332), so there it
is a fit-in-memory option, not a speedup. Smaller blocks still helped
single-copy (26M beat 64M) despite the extra row pointers. Stock at
VBITS=256 was fastest as a single block on this card (table below); the
V100 at VBITS=256 was flat instead. 4-bit inner/outer product kernels:
pre-change commit 106bf9f, stock block_nnz=1024M, 319.2 dims/s vs 331.7
with them (+4%, one run each, within the ~5% noise). On the 5070 there
is no dims/s A/B for them; the profile went from ~16+26 to ~11+19
ms/iter, an estimated 5-7% of an iteration.

Stock block_nnz on small-L2 cards. The matrices differ: the 3060 ran
the TD=90 C170 matrix (13.2M cols, 958M sparse nnz on the GPU, 72.8/col),
so 1024M and 1.75B are both one block there; the V100 ran the 20.2M
matrix (2.24B nnz, 111/col), so 1.75B is two blocks. Block counts in
parentheses; single runs each.

| card | VBITS | 128M window / (L2/2) | 128M | 1024M | 1.75B | best |
|---|---:|---:|---:|---:|---:|---|
| 3060 | 256 | 37x | 310 (8) | 332 (1) | 332 (1) | one block (+7%) |
| 3060 | 64 | 9x | 142.9 | 132.2 (1) | - | 128M (+8%) |
| V100 | 256 | 12x | 246.3 (18) | - | 243.6 (2) | flat (256M 257.5) |
| V100 | 64 | 3x | 141.9 (18) | - | 109.5 (2) | 128M (+30%) |

The window column is relative to half the L2 (the formula's target), so
37x is ~19x the full L2. One large block only won where the 128M floor's
window was that far out of cache (3060 at VBITS=256), and then by ~7%.
Not put in the default formula (one card type, threshold fitted to four
points); see the exception under Tuning Heuristic rule 2 above. 3060
VBITS=64 peak VRAM: 10.7GB at 128M, 9.9GB at 1024M.

## 2026-09-25: Streaming Matrix Blocks From Host Memory

Goal: matrices whose sparse part doesn't fit on the card, without
use_managed's collapse (on WSL managed memory never migrates, and an
LRU cache over a cyclic scan larger than memory misses on every page).
Every iteration uses the whole matrix (single-copy: each block twice),
so overflow blocks are re-copied every iteration; the question is only
whether PCIe keeps up.

Standalone CUDA tests on the RTX 5070 (PCIe 5.0 x16, WSL2, no matrix):

- Pinned host-to-device: 48 GB/s at 16-512 MB chunks; pageable 10.6
  GB/s. Pinned allocation: 8 GB in 8s, 16 GB in 21s, 32 GB in 221s
  (the last with a filtering job holding most of the RAM).
- Copies concurrent with a VRAM-heavy gather kernel: kernel +5.8%,
  copies still 48.5 GB/s.
- Pipeline simulation, 4 GB fake matrix in 16 x 256 MB chunks, 3
  staging buffers, streamed chunks spread evenly. Time per pass is
  about max(compute, streamed bytes / 46 GB/s):

| kernel's matrix rate | 75% streamed | 100% streamed |
|---|---:|---:|
| 197 GB/s | +258% | +325% |
| 61 GB/s (~single-copy at VBITS=256 on the 5070) | +2.0% | +22% |
| 22 GB/s | +0.8% | +1.0% |
| 6 GB/s | ~0% | +1.1% |

Real estimates: single-copy TD=90 reads 4.66 GB twice per 159 ms
iteration (~58 GB/s), two-copy 7.7 GB once per 260 ms (~30 GB/s), the
3060 two-copy ~9 GB/s. VBITS=256 is the best case (matrix bytes per
dimension scale as 1/VBITS). The Lanczos vectors (7 x ncols x 32 B at
VBITS=256) stay on the card and become the VRAM limit.

Implementation (lanczos_matmul_gpu.c): plan_matrix lays out every
block before any is built. Forward blocks come from the column weights
(each spans all rows and takes whole columns up to block_nnz).
Two-copy transpose blocks come from prefix sums of per-row nonzero
counts, running the same grow/shrink size search extract_block_trans
used; that function is gone, and gpu_matrix_init now just collects the
planned row ranges, so two-copy setup no longer rescans every nonzero
per search step (checked block-for-block identical to the old search,
see Review follow-up 2). If the blocks fit in the free memory less
vectors and dense rows (counting 2MB average allocation rounding per
block), they upload as they are built. Otherwise choose_streamed picks
the fewest evenly spaced blocks whose removal makes the resident
blocks plus the staging buffers (up to stream_slots, fewer if needed,
each the largest streamed block, budgeted at 4MB worst-case rounding)
fit, less a margin of 256MB natively or 1.5GB under WDDM. Streamed
blocks are packed straight into pinned memory while building (ordinary
memory, with a warning, if pinning fails). If no layout fits, it warns
and loads everything as before. run_spmv_block makes the default
stream wait on the block's copy event, records a freed event after the
SpMV, then queues copies for the next num_slots-1 schedule entries on a
non-blocking stream, evicting the buffered block used furthest in the
future. The single-copy transpose walks the blocks backwards (reusing
the last streamed blocks; harmless otherwise). With no more streamed
blocks than buffers (only reachable with stream_frac) the blocks are
copied once and stay, reported as 0 MB streamed. Options:
max_gpu_mem=MB (used instead of the free memory, no margin),
stream_frac=F, stream_slots=N.

Validation on a synthetic 1.5M x 1.501M matrix (matgen: rows chosen
with p ~ 1/(r+50), ~60 nonzeros/col, 89M nonzeros), VBITS=256, RTX
5070, filtering job running on the CPU. All runs recovered 12
dependencies and an independent check (every dependency's columns XOR
to zero on every row) passed; SIGTERM checkpoint + -ncr restart with
streaming also passed. Lanczos loop times (5875 iterations, ~19
ms/iter, so per-iteration syncs weigh more than on real matrices):

| config | streamed per pass | slots | loop time |
|---|---:|---:|---:|
| single_copy, 18 blocks of 4M, resident | 0 | - | 113-114s |
| same, stream_frac=0.5 (10 of 18) | 196 MB x2 | 3 | 116s |
| same, max_gpu_mem=640 (8 of 18) | 168 MB x2 | 3 | 114s |
| same, stream_frac=0.75 (14 of 18) | 280 MB x2 | 3 / 6 | 125s / 118s |
| same, stream_frac=1 | 364 MB x2 | 3 / 6 | 144s / 130s |
| two-copy, block_nnz=8M, resident | 0 | - | 87s |
| two-copy, stream_frac=0.5 (9 of 18) | 316 MB | 3 | 89-92s |

### C189 (2026-09-25): a matrix that doesn't fit

C189 matrices, too big for the 12 GB RTX 5070 even single-copy:
TD=90 23.9M cols (2.06B sparse nnz), TD=110 22.3M (2.32B), TD=130
21.2M (2.57B). WSL, 5-min windows (managed 10 min), one run each; the
filtering/matbuild job was using the CPU and RAM for the first runs.

| config | vectors | resident / streamed | dims/s | full ETA |
|---|---:|---:|---:|---:|
| TD=90 VBITS=256 single_copy | 5.1 GB | 14 / 24 of 38 blocks (6.4 GB) | 685 | 9h41m |
| same, stream_slots=6 | 5.1 GB | 11 / 27 (7.2 GB) | 688 | 9h39m |
| TD=90 VBITS=256 two-copy | 5.1 GB | 5 / 24 of 29 (13.3 GB) | 570 | 11h39m |
| TD=90 VBITS=128 single_copy | 2.6 GB | 16 / 7 of 23 (2.9 GB) | 774 | 8h34m |
| TD=110 VBITS=128 single_copy | | (3.7 GB) | 753 | 8h14m |
| TD=130 VBITS=128 single_copy | | (4.2 GB) | 700 | 8h24m |
| TD=90 old binary, two-copy, use_managed=1 | | block_nnz 128M | 197.5 | 33h38m |
| same, block_nnz=1.75B | | | 204.9 | 32h25m |
| TD=90 VBITS=128 single_copy use_managed=1 | | default block_nnz | 184.7 | 35h58m |
| same, block_nnz=1.75B | | | 98.4 | 67h28m |

Streaming is ~3.8x faster than the best use_managed run. Managed
matrix setup took 29 min at 128M blocks (5.6 min at 1.75B); pinning the
streamed blocks took 3-41 s. Streamed bytes per iteration work out to
30-35 GB/s in all three layouts, below the 48 GB/s measured alone (the
matbuild job competing for host memory bandwidth may be part of it; a
2026-10-03 run reached ~50 GB/s with the same margin, see there), so
these runs are copy-bound: speed follows streamed bytes per dimension.
VBITS=128 wins by halving the vectors, which leaves more of the matrix
resident; more staging buffers cost resident blocks. Density barely
mattered (within noise). Single-copy row pointers are ~35% of each
VBITS=256 block (96 MB per block at 23.9M rows), so compressing them
for streamed blocks, larger streamed blocks, or moving rarely used
vectors off the card are the next things to try.

Rechecked after the review fixes (exact single-copy planning, streamed
blocks packed straight into pinned memory), GPU otherwise idle: TD=90
VBITS=128 single_copy 787.3 / 786.9 dims/s with the staging buffers
allocated last and 785.6 / 787.9 with them allocated first (8h26m);
same layout as before (7 of 23 blocks, 2.9 GB streamed). An earlier
718 dims/s run of the same config was noise.

VBITS=64 single_copy (same code, GPU idle): TD=90 fits entirely (8.9 GB
sparse, 180M-nnz blocks) at 572.8 dims/s, 11h35m; TD=110 streams 4 of 11
blocks (3.6 GB) at 413.2 dims/s, 15h00m; TD=130 streams 5 of 11 (5.2 GB)
at 403.7 dims/s, 14h34m. So VBITS=128 with streaming beats even a fully
resident VBITS=64 by ~37%. VBITS=64's default single-copy blocks are
0.9-1 GB, so three staging buffers take ~3 GB and a matrix only ~0.4 GB
over budget (TD=110) streams 3.6 GB; capping block_nnz when streaming
would cut that, but not enough to catch VBITS=128 here.

### Review follow-up 2 (2026-09-28)

Second code review of the streaming code; no protocol bugs, fixes to
the planning around it:
- Layout planned up front for both copies (above). The prefix-sum
  replay of extract_block_trans was checked block-for-block against a
  saved copy of the old search (CPU harness): identical on the
  synthetic matrix (block_nnz 4M-128M, 0 and 240 dense rows) and on the
  2026-09-28 job's TD=90 matrix (18.9M rows) at 128M and 227M.
- Margin under WDDM raised to 1.5GB when streaming (the C189 slowdown
  and driver faults happened with ~250MB free); a matrix that fits
  without the margin still loads whole.
- Fit test uses 2MB average allocation rounding per block instead of
  the 4MB worst case, so it no longer streams matrices the old code
  loaded whole.
- Pinning failure falls back to pageable memory for that block; empty
  blocks no longer ask CUDA for zero bytes; the single_copy note is
  skipped when block_nnz is set by hand.
- End-to-end after the rewrite (2026-09-29): 12 of 13 synthetic suite
  runs recovered 12 dependencies that passed the independent check;
  real TD=90 (2026-09-28 job) two-copy and single-copy streaming runs
  loaded as planned and passed the integrity check. The 13th run
  (stream_frac=0.1) ended "lanczos error: not all columns used".
  Replaying its logged seeds (14f3ca93 9c505a0b, via a scratch-only
  seed override in demo.c) gave the same error at the same dimension
  with no streaming at all, single- and two-copy, so it is that start
  vector on the synthetic matrix: its Lanczos ends ~600 dimensions
  below max_nrows - VBITS, where msieve's end-of-run check is known to
  misfire. Fresh seeds passed that config 5 of 6 times.

Left as documented limitations:
- Staging buffers scale with block size. With big blocks (VBITS=64
  single-copy ~1GB, two-copy under the new row floor: 287M nnz = 1.25GB
  at VBITS=256 on C189) three buffers take 3-4GB and push more of the
  matrix out. Capping block_nnz while streaming would help that but
  adds per-block row-pointer bytes (96MB per block at 24M rows). First
  data on the balance: the 2026-09-29 RTX 5090 section below, where
  1.25x the default block_nnz was best.
- MPI ranks sharing one GPU each plan against the same free memory
  (no regression: the old code ran out of memory there too); give each
  rank a max_gpu_mem share.
- The k=12 row floor changes two-copy VBITS=256 defaults above ~10.7M
  rows without a desktop recheck. On the 12GB 5070 such matrices now
  stream, which confounds a comparison; the 32GB 5090 fits C189 TD=130
  two-copy at VBITS=256 (19.2GB), so bench_la.sh there with
  block_nnz=128000000 vs the default (254M) would check it.

## 2026-09-27: A100 Feedback (Greg) — the 128M Floor Fails on Tall MPI Pieces

Greg ran the new code (b61b5ac) on 2x A100 under MPI with a 1x2 grid.
The first GPU's piece is 61010818 x 30505387, weight 3266389052
(107.08/col). The GPU's sparse average column weight is 76.4, so about
2.33B sparse nnz and an average row weight of ~38. The log line was
"computed block_nnz 128000000 (L2 cache 41943040 bytes, average column
weight 76.4)". VBITS was not stated: the formula gives 100M at VBITS=128
and 50M at 256, so it is at least 128 either way (both hit the floor).
Full-solve ETAs:

| code | mode | block_nnz | kernel | ETA |
|---|---|---|---|---:|
| previous (pre-b61b5ac) | stock | 1.75B (old default) | - | 29h53m |
| new | stock | 128M (new default, floor) | auto | 32h37m |
| new | stock | 1.75B | auto | 28h44m |
| new | stock | 1.75B | warpmerge | same as auto |
| new | stock | not stated (probably 1.75B) | segscan | 29h18m |
| new | single_copy | 122M default (2*nrows floor) | auto | 43h41m |
| new | single_copy | below the floor (L2 fit) | auto | "a little slower still" |
| new | single_copy | 1.75B | auto | 49h47m |

What held up:
- New inner/outer product and vector kernels: -3.9% ETA at matched
  1.75B (29h53m -> 28h44m). This matches the 3060's +4%.
- auto = warpmerge at 1.75B, as expected: blocks there have 29-57 nnz
  per row, far above the segscan threshold of 8. Forced segscan was only
  2% slower on these long rows. No change needed.

What failed: the default block_nnz (+13.5% vs 1.75B) and single_copy
(+52%).

### Diagnosis: per-block cost scales with the piece's rows

Every A*x block spans all nrows: it carries a full rowptr array and XORs
into the whole output vector. Every A^T*x block spans all ncols. So each
block costs about (2*sizeof(v_t) + 4) bytes per output row it touches,
independent of block_nnz. The 128M floor was calibrated on square
matrices with 9.5M-20M rows, and on this piece it gives far fewer
nonzeros per output row per block:

| matrix | rows | nnz per output row per block at 128M |
|---|---:|---:|
| 5070 C168 | 9.54M | 13.4 |
| 5070/3060 C170 | 13.17M | 9.7 |
| V100 | 20.2M | 6.3 |
| Greg A100 (MPI 1x2 piece) | 61.0M | 2.1 (A*x), 4.2 (A^T*x) |

The lowest value we ever tested was ~3 (V100 64M, 5070 C168 v256 32M),
and it lost every time.

Back-of-envelope: 128M gives 19 blocks per direction vs 2 at 1.75B, so
17 extra blocks each way. Assumptions: 80-100% of rows touched per
block, streaming at 1.55-2.0 TB/s HBM, full dimension 2*30.5M = 61M.

| VBITS | iterations | observed extra | est. extra row-side traffic | est. time |
|---:|---:|---:|---:|---:|
| 128 | 479K | 29 ms/iter | 46-56 GB/iter | 23-36 ms |
| 256 | 239K | 58 ms/iter | 86-106 GB/iter | 43-68 ms |

The per-block overhead accounts for the whole loss, so the smaller
window bought ~no L2 locality. That is expected: at 128M the A*x window
is 27MB (v128) or 54MB (v256), and the A^T*x window is twice that
(54/107MB) because rows are half as heavy as columns. That is against a
40MB L2 that is really two 20MB partitions (see the caveat below).

### Why not "just use max block size" on some cards

Our own data argues against dropping the L2 fit:
- 5070 C168 v256: one block was 16% slower than the 128M floor (1208 vs
  1435).
- V100 v256: 256M beat 1.75B by ~6% (257.5 vs 243.6) even though the
  window was 12x L2.
- 5070 C170 v256: 256M beat 1024M (981 vs 833, noisy under WDDM
  pressure).

Only the 3060 v256 (+7%) and this A100 run preferred the largest blocks.
The fix belongs in the floor, not in the L2 targeting.

### Proposed fix (implemented 2026-09-28, k=12)

Make the floor proportional to the piece's rows, in `lanczos_matmul_gpu.c`
where the two-copy path does `computed = MAX(computed, 128000000)`:

    floor = MAX(128M, k * MAX(nrows, ncols) * sizeof(v_t) / 32),  k = 12

In words: a block must gather at least k times as many 32-byte sectors
as it rewrites in its output vector. The sizeof(v_t)/32 factor is there
because a row rewrite costs 2*sizeof(v_t), while a missed gather costs a
full 32B sector at any VBITS. That is why the V100 at v64 still wants
128M. Use p->nrows/p->ncols, i.e. the local MPI piece. Effect of k=12:

| card / matrix | VBITS | current | k=12 floor | new | measured |
|---|---:|---:|---:|---:|---|
| 5070 C168 | 64 | 278M | 29M | 278M | optimum ~256M |
| 5070 C168 | 128 | 139M | 57M | 139M | optimum 128M |
| 5070 C168 | 256 | 128M | 114M | 128M | optimum 128M |
| 5070 C170 TD=90 | 256 | 128M | 158M | 158M | 128M OOM, 256M 981 |
| V100 | 64 | 128M | 61M | 128M | optimum 128M |
| V100 | 256 | 128M | 242M | 242M | optimum 256M (+4.5% vs 128M) |
| 3060 C170 | 64 | 128M | 40M | 128M | 128M, +8% over 1 block |
| 3060 C170 | 256 | 128M | 158M | 158M | 1 block +7% over 128M |
| A100 piece | 128 | 128M | 366M (7 blocks) | 366M | est. ~4% behind 1.75B |
| A100 piece | 256 | 128M | 732M (4 blocks) | 732M | est. ~1.5% behind 1.75B |

Every default that matched a measured optimum is unchanged. The two that
did not (V100 v256, 3060 v256) move in the right direction. On the A100
piece, the estimated loss vs hand-set 1.75B drops from 13.5% to ~1.5-4%.
k=13 would still leave the 5070 C168 v256 alone (124M) and gives the
A100 piece 3 blocks at v256.

Possible refinements:
- Size the two directions separately. A*x windows use the column weight
  and A^T*x windows use the row weight; these differ 2x on a 1x2 MPI
  piece, so one block_nnz cannot suit both.
- Balance block sizes. At 732M the A100 piece splits into
  732/732/732/135M, and the tiny last block still pays full rowptr and
  output costs.

### Single copy: same mechanism, and a usable predictor

Single copy won only where its L2 fit ((L2/3)/sizeof(v_t) * col weight)
cleared its 2*nrows floor:
- 5070: 38M vs 26M floor. Won.
- 3060: 2.4M vs 26M. Lost 19%.
- A100 piece: 33M (v256) vs 122M. Lost 52%.

Proposal: when single_copy=1 is requested and the floor binds, log that
it is expected to be slower than two copies and only saves VRAM. Update
the CLAUDE.md single_copy paragraph with the A100 result.

### Caveat: A100 L2 partitioning (unverified)

The formula takes CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE at face value. On
A100 (and H100), L2 is two partitions, and each caches data for the SMs
attached to it. Data that every SM gathers may therefore be held in both
halves, so the usable capacity would be closer to 20MB. This does not
matter for Greg's run, where the floor dominates, but it would matter if
we ever tune the L2 fit for datacenter cards.

### Next steps

1. Ask Greg for his VBITS, and for ETAs after warmup at block_nnz =
   512M, 768M and 1G (two-copy, auto kernel). This shows whether the
   row floor is enough or the A100 really wants max beyond the per-block
   overhead. It is also our first data on a piece with more than 20M
   rows.
2. Done 2026-09-28: the k=12 row floor (two-copy default is now
   max(L2 fit, 128M, 12*max(nrows,ncols)*sizeof(v_t)/32)). Still to
   re-check on the 5070: C168 defaults should be unchanged, and C170
   v256 moves 128M -> 158M, so check its VRAM and speed.
3. Done 2026-09-28: msieve logs a note when single_copy's L2-sized
   blocks fall below its 2*nrows floor (expect it to be slower than two
   copies, only saving memory), and CLAUDE.md describes both floors.
4. If Greg's sweep does not fit the model: add a startup autotune that
   times a few SpMV iterations at 2-3 block sizes and keeps the fastest.
   - Safe: the host matrix (p->unpacked_cols) lives for the whole
     solve, and block layout does not change results (byte-identical
     checkpoints, 2026-06-10).
   - Cheap: each GPU rebuild is ~minutes against a ~29h solve.
   - Each MPI rank can pick its own size.
   - This is the robust answer for cards we cannot test.

## 2026-09-29: RTX 5090, 41.6M-dim matrix (single copy + streaming)

First real matrix that doesn't fit a 32GB card two-copy: TD=130 of the
job after C189, 41,646,000 x 41,646,178, weight 6.44B (154.5/col). After
reduction: 41,645,760 dims with 4.69B sparse nonzeros at VBITS=256
(41,645,888 and 5.02B at VBITS=128; more dense rows are packed out at
256). Rented RTX 5090 (96MB L2, 31.6GB free) on Vast, native Linux, so
the 256MB streaming margin. bench_5090.sh (single_copy=1 at the default
block_nnz, half and double it; stream_slots=2 if the default streams),
then two more sizes by hand. bench_la.sh, 300s windows, one run each,
GPU otherwise idle; the matrix loaded in ~89s every time. All runs passed
the integrity check.

| VBITS | block_nnz | blocks | streamed | dims/s | full ETA |
|---:|---:|---:|---:|---:|---:|
| 256 | 111M (default) | 40 | 6 blocks, 3.5 GB, 2 slots | 800.3 | 14h27m |
| 256 | same, stream_slots=2 | 40 | same (default already used 2) | 803.2 | 14h24m |
| 256 | **139M (1.25x)** | | 2.8 GB | **820.4** | **14h05m** |
| 256 | 167M (1.5x) | | none | 745.2 | 15h31m |
| 256 | 223M (2x) | | none | 607.0 | 19h03m |
| 256 | 56M (0.5x) | 80 | 26 blocks, 9.7 GB, 2 slots | 522.7 | 22h07m |
| 128 | 236M (default) | | none | 710.7 | 16h16m |
| 128 | 118M (0.5x) | | none | 683.6 | 16h55m |
| 128 | 472M (2x) | | none | 431.3 | 26h49m |

Vectors take 8.9GB at VBITS=256 and 4.4GB at 128, which left 21.4GB and
26.5GB for the sparse blocks. The single_copy 2*nrows floor (83M) didn't
bind at either VBITS (no note).

- VBITS=256 beat 128 by 13-15% even while streaming. That is the
  opposite of the C189/5070 result, where the copies were the limit
  and halving the vectors to keep more of the matrix resident won.
  Here the streamed share is small (2.8-3.5GB of ~20GB). Each streamed
  block is used by both products (snake order reuses the last one or
  two), so roughly 6-7GB moves per ~310-320ms iteration, ~20GB/s. That
  is below the 30-35GB/s the 5070 runs reached, so the copies should be
  mostly hidden (inferred, not measured).
- Block size matters more than streaming: 1.5x fits entirely and still
  loses 9% to 1.25x with streaming; 2x loses 26%. The L2 formula's
  default (L2/3 of columns) is near the optimum. Nudging it up trades
  cache fit for fewer 167MB row-pointer arrays (one per block at 41.6M
  rows) and less streaming, and 1.25x came out 2.5% ahead, within
  noise of the default. No change to the formula from this.
- Small blocks are costly at this height. Halving block_nnz doubles the
  row-pointer arrays (+6.7GB at v256), which nearly tripled the
  streaming and made it the slowest v256 run.
- The planner fell back from 3 to 2 staging buffers at the default,
  since 3 of the 584MB buffers didn't fit, so stream_slots=2 repeated
  that layout. The two runs are within 0.4% of each other, a rough
  measure of run-to-run noise.
- Chosen for the real run: VBITS=256 single_copy=1 block_nnz=139000000.

## 2026-10-03: RTX 5070, 25.2M-dim C190 streaming (and an RTX 3080)

The upload counts and inferred PCIe rates in this section were revised in the
October 4 storage review below. The measured solve times remain valid, but
they do not establish that this run reached the PCIe limit.

Production run on the 12GB RTX 5070 (WSL2), C190, TD=110 from a
multi-density filter: 25,152,405 dims, 2.40B sparse nonzeros (95.5/col)
after packing 128 dense rows. VBITS=128, single_copy=1, default
block_nnz 100.2M (L2 fit; the 2*nrows floor of 50M didn't bind), 24
blocks. 11015 MB free, 2687 MB of vectors, 11565 MB of blocks needed:
12 of 24 blocks (5732 MB) streamed through 3 buffers of 478 MB.

The binary was built 2026-09-25, before Review follow-up 2, so the
WDDM margin was still 256MB (no "leaving N MB free" line; the budget
only adds up with 256MB). Polyselect (msieve -npr, CPU only) ran on 8
cores from ~15 minutes in to the end.

Result: 197,688 iterations, Lanczos loop 11h45m10s (594.5 dims/s,
BLanczosTime 42535; the 0.1% ETA said 12h06m), 32 dependencies,
p89 * p102 on the first dependency, 12h20m total.

Copy rate: 127.2 dims per iteration at 214 ms. With 3 slots the
transpose reuses the last two streamed blocks, so about 22 x 478 MB =
10.5 GiB (11.0 GB) is copied per iteration: ~50 GB/s, the 48 GB/s the
standalone pinned test reached. So this run was copy-bound at the PCIe
limit, and the SpMV work was hidden under the copies.

### What this says about the C189 rates

The C189 runs above implied lower copy rates, computed the same way:

| C189 run (2026-09-25) | copied per iteration | dims/s | implied rate |
|---|---:|---:|---:|
| TD=90 VBITS=256 single_copy, 24 of 38 streamed | ~12.3 GB | 685 | ~33 GB/s |
| TD=90 VBITS=256 two-copy, 24 of 29 | ~13.3 GB | 570 | ~30 GB/s |
| TD=90 VBITS=128 single_copy, 7 of 23 | ~5.0 GB | 774-787 | ~31 GB/s |
| TD=110/130 VBITS=128 single_copy | ~6.5-7.8 GB | 700-753 | ~38-43 GB/s |

- Not the margin: this run had the same 256MB margin and ran at ~50.
- Not CPU load as such: polyselect kept all 8 cores busy during this
  run. Polyselect barely touches memory, though, and the first C189
  runs shared the machine with a matbuild job that does. Host-memory
  traffic slowing the DMA is still a plausible cause.
- The TD=90 VBITS=128 run (the idle recheck at 787) streamed the
  least of any run and may not have been copy-bound; then ~31 GB/s is
  just its copy load, not a ceiling.
- Not settled. To check on the next big job (a C191): bench_la.sh on
  its matrix with the machine idle, then with a matbuild running.

### The 1.5GB WDDM margin: measured ~10% on this matrix

Estimated first (before the test) at ~25%, assuming 3 more streamed
blocks with 3 buffers. Measured the same day with the rebuilt binary
(current code), same matrix, bench_la.sh 300s windows, one run each,
-t 8. ECM stage 2 (about 12 processes, ~1.6GB each) was running on the
CPU for both; ~1.9-2.3GB of VRAM was used by other programs.

| margin | streamed | buffers | copies/iter | ms/iter | dims/s | full ETA | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1536MB (default) | 13 of 24 (6210 MB) | 2 | 25 | 253.5 | 501.9 | 13h55m | 11343 MiB |
| 256MB (max_gpu_mem=10759) | 11 of 24 (5259 MB) | 2 | 21 | 230.7 | 551.7 | 12h40m | 11844 MiB |
| production run above (old binary, ~256MB) | 12 of 24 (5732 MB) | 3 | 22 | 214 | 594.5 | 11h45m | |

- The 1.5GB margin cost 9% in speed (10% in ETA). Less than estimated
  because the planner falls back to 2 buffers in both cases, so only
  two more blocks stream.
- Implied copy rates (2n-1 copies with 2 buffers, 2n-2 with 3): ~49
  GB/s at the 1.5GB margin, ~46 at 256MB (perhaps not fully copy-bound
  there), ~51 in production.
- The production run beat both, with one block more streamed than
  the 256MB run. It had 3 buffers and only polyselect (light on
  memory) on the CPU, against ECM stage 2 (heavy on memory) here. Its
  number is the whole-run average, not a 5-minute window. Not
  separable from these runs, but consistent with host-memory traffic
  slowing the copies.
- Correction: the current code counts the dense rows as 383MB where the
  old binary counted ~639MB, so max_gpu_mem = free - 256 gives a true
  256MB margin but a budget 256MB larger than the old binary's (7689
  vs 7433), not the old budget exactly.
- 5 minutes with ~380MB of VRAM free is not proof that a 12h run is
  safe, but the production run went 11h45m at about the same fill with
  no trouble. The faults on 2026-09-25 came with other programs
  actively using the GPU.
- Options: run with max_gpu_mem = free - 256 by hand, or lower
  STREAM_MARGIN_WDDM (512MB would keep most of the gain). Not changed.

### Revised estimates for 12GB cards (VBITS=128 single copy, ~48 GB/s)

The 2026-10-01 estimates assumed ~33 GB/s and the 1.5GB margin; at
~48 GB/s, still with the 1.5GB margin:
- ~24-25M dims: 8.5-12h (measured, C189 and this run)
- ~30M: ~18 GB copied per iteration, ~1 day (estimate)
- 41.6M (the 5090's TD=130 matrix): 43-54 GB per iteration depending
  on block size, ~3.5-4.5 days (estimate; the 5090 took 14h)
- Hard limits: the vectors (~107 B/col at VBITS=128) and dense rows must
  stay on the card, ~60M dims on 12GB; host RAM is similar at 80GB
  (~4 B per nonzero for the host matrix plus a pinned copy of the
  streamed part, ~50GB for the 41.6M matrix).

### Host memory while building blocks

gpu_matrix_init builds each block on the host before it goes to the
card: an 8-byte-per-nonzero entries array (grown by doubling, so up to
2x the block, and kept at the largest block's size), the radix sort's
two 8-byte arrays (forward blocks only), and the packed block. That is
~28-36 bytes per nonzero of the largest block on top of the host matrix,
for example ~1GB at 28M, ~7-9GB at 250M and ~50-63GB at 1.75B: the cause
of the out-of-memory kills seen when raising block_nnz by hand. It is
gone once the solve starts; after that, larger blocks slightly shrink
the pinned copy (fewer row-pointer arrays). Possible fix, not done:
walking a forward block's columns in order already sorts each row's
column indices, so a count-per-row pass plus a placement pass could
write the CSR block directly (4 bytes per nonzero, no sort). The
transpose blocks could be written directly the same way. Blocks would
be byte-identical.

### RTX 3080 (someone else's run, 2026-09-30)

RTX 3080 10GB (sm_86, 5MB L2), WDDM (1.5GB margin), current code,
VBITS=256 single_copy: 14,150,048 dims, 1.04B sparse nonzeros
(73.4/col). block_nnz was the 2*nrows floor (28.3M; L2 fit far
below, and msieve logged the slower-than-two-copies note). 37 blocks,
15 streamed (2429 MB) through 2 buffers of 162 MB. Lanczos 8h49m,
~446 dims/s, factored (p77 * p96).
- Copies were ~8 GB/s (~4.5 GB per 572 ms iteration), well under PCIe
  4.0, so this run was limited by the SpMV, not the copies.
- The 3060 (3MB L2, nearly the same size matrix: 251 dims/s single
  copy, resident) scaled by memory bandwidth (2.1x) and nonzeros puts
  a resident 3080 at ~490, so streaming cost at most ~10% here.
- Larger single-copy blocks would cut the row pointers (57 MB per
  block, 2.1 GB in all) and so the streaming, but on the 3060 64M
  blocks were 7% slower than the 26M floor, resident. Two copies were
  24-32% faster on the 3060 but would need ~9 GB here and stream ~8 GB
  per iteration. No sweep was run; the default is probably within ~10%
  of the best on this card.

## 2026-10-04: Storage review and next experiments

Implemented actual transfer counters and three opt-in storage changes:
`host_check=1`, `col24=1`, and `row16=1`. All three default off and can be
combined. The matrix file and checkpoint formats are unchanged. Packed blocks
are decoded directly by the CUDA kernels, with per-block/per-group fallbacks
when a smaller representation cannot hold the data. Comparisons use the
retained C190 matrix; results and validation follow below.

### Correction to the inferred PCIe rates above

The current scheduler retains blocks at both turnarounds, including the
transpose-to-forward transition between iterations. Replaying the actual
`sched_distance` and `stream_load` functions with CUDA calls mocked, for 360
block/buffer combinations over five iterations, gives `2*(m-slots)` uploads
per steady-state single-copy iteration when m > slots, not `2*m-(slots-1)`.
Unequal block sizes require counting the actual bytes, not multiplying by the
largest block. Initial loading and two-copy mode have different counts.

For the current-code October 3 measurements, 13 streamed blocks and 2 slots
mean 22 uploads, not 25; 11 blocks and 2 slots mean 18, not 21. The inferred
rates are about 43.5 and 39.1 GB/s, respectively. With the same scheduling
policy, the older production binary's 12 blocks and 3 slots would mean 18
uploads, not 22, or about 42 GB/s. Therefore the earlier claim that C190 had
demonstrably reached the PCIe limit is too strong. Actual HtoD byte counters
and a GPU timeline should distinguish copy traffic from exposed copy stalls.

### Storage opportunities measured on C190

The logged 24 blocks contain 2,402,981,751 sparse entries and have widths
766,448-1,150,002 columns. Their local column indices need only 20 or 21 bits.
Using 24 bits instead of 32 saves 2.238 GiB across the full sparse matrix;
20/21-bit packing would save 3.194 GiB. Decode inside SpMV to retain the space
saving in both resident blocks and staging buffers. Wider blocks need a
32-bit fallback. At the October 3 default sparse budget, a CPU replay of the
planner with 24-bit columns reduces steady-state uploads from 10.27 to 5.98
GiB per iteration. These are modeled byte counts, not measured speedups.

The 24 full row-pointer arrays total 2.249 GiB. A read-only sample of the first
real block reproduced its 100,176,700 nonzeros. Of its 128-row pointer groups,
99.946% fit 16-bit offsets from a shared 32-bit base; the remaining groups
need a 32-bit fallback. With groups of 32, 95.369% fit 8-bit offsets. A first
16/32-bit implementation should save roughly 1.1 GiB across this layout,
subject to measuring all blocks and including metadata and decode costs.

After iteration 3, v0 is only the occasional integrity-check vector. Keeping
it on the host and uploading into the existing, then-idle SpMV scratch vector
reclaims 383.8 MiB at C190/VBITS=128, or about 1.24 GiB at 41.6M/VBITS=256.
Preserve startup behavior, every integrity check, and checkpoint compatibility.

### Future experiment: complete row slabs, one upload for both products

For a vertical partition of A into complete row slabs A_r,

    A^T (A x) = sum_r A_r^T (A_r x).

A slab can stay in its staging buffer while both products run, then be
discarded. This could reduce repeat matrix uploads and shrink the intermediate
A*x vector. Column tiling within each slab is necessary to retain the current
cache locality; it also adds output accumulation traffic and launches, so the
net speedup must be measured. Size slabs by nonzeros to accommodate heavy rows.
The identity does not apply independently to the current column slices:
doing A_i^T(A_i*x_i) for each would omit cross-slice contributions.

Keep this as a separate layout experiment after the smaller storage changes.
Benchmark complete Lanczos iterations, include dense rows, and verify both
matrix products and final dependencies before drawing speed conclusions.

### Validation of the storage implementation

- GPU products matched an independent CPU oracle at VBITS=64, 128, and 256:
  all four combinations of column/row encoding, all three gather modes,
  both directions, and 256/512/1024/2048-item tiles. Cases include empty rows,
  an empty matrix, a 70,000-entry row forcing a wide offset group, and XOR
  accumulation into a nonzero output. Host codec checks cover group/tail
  boundaries, high 32-bit bases, and the largest 24-bit index.
- At VBITS=128, full synthetic solves with both single-copy and two-copy
  streaming recovered dependencies independently verified against the original
  matrix, including the dense and removed rows. A checkpoint written by the
  saved original binary resumed with all three options, then resumed again
  in the original binary and completed with valid dependencies.
- CUDA and CPU builds passed. No multi-GPU/MPI run or full C190 factorization
  has been performed for these changes. Short C190 benchmark runs retain the
  solver's scheduled integrity checks and its final check before checkpointing.

Tests live in `tools/test_spmv_packed.cu` and `tools/test_lanczos_packed.py`.
Integration artifacts from the passing run are in
`bench/packed-validation-17etcn1i/`. Saved executables and build logs are in
`bench/codex-20261004/`; benchmark results append to `bench/results.tsv`.
An initial two-copy synthetic fixture stalled in the existing transpose
planner: `block_nnz=150000` starts its integer growth search at 15 rows,
and `5*(15/4)` is still 15. The fixture now uses 200000; production planning
is unchanged, and the test harness kills its own process if a timeout cannot
stop it gracefully.

### Applying the results to other GPUs

Column savings are one byte per eligible sparse entry, regardless of VBITS;
the host check-vector saving is `number_of_columns * VBITS / 8` bytes per
single-GPU solve. Row savings depend on the number of blocks and the frequency
of wide offset groups. All three options preserve the existing block widths,
so they can reduce storage without enlarging the active vector window.

For the September 29 RTX 5090 layout, 4.69B sparse entries imply about 4.37 GiB
saved by eligible 24-bit columns at VBITS=256, plus about 1.24 GiB from the
check vector. This should be tested at the previously good 111M/139M block
sizes before considering larger blocks merely to make the matrix fit.
These are capacity estimates, not a measured speedup on the 5090. The 5070
timings below do not establish the preferred settings on resident matrices,
older GPUs, or multi-GPU solves; decoding and occasional vector transfers
can cost time even when the storage reduction is useful.

Under a fixed GPU budget the planner spends freed space on additional
resident blocks, so peak VRAM need not fall. Compare the complete matrix/vector
footprint and actual uploaded bytes per iteration as well as `nvidia-smi`.

### C190 measurements, October 4-5

RTX 5070 12GB under WSL, VBITS=128, `-t 8`, default block size 100,176,501,
`single_copy=1 max_gpu_mem=9500` for every run. `bench_la.sh` uses 30 seconds
of warmup and a 180-second measurement window, then stops for an integrity
check and checkpoint. These are single-window measurements and projected
iteration-only full-run times, not completed C190 solves. Startup time and
other applications' VRAM use varied between runs.

| Extra storage options | dims/s | ms/iteration | projected full ETA | streamed blocks/slots | actual uploaded MiB/iteration |
|---|---:|---:|---:|---:|---:|
| Original executable, none | 473.7 | 268.6 | 14h45m | 13/2 | not recorded |
| `host_check=1` | 487.7 | 260.9 | 14h19m | 12/2 | 9557.054 |
| `col24=1` (Oct 4) | 596.1 | 213.5 | 11h43m | 10/2 | 6120.901 |
| `col24=1` (Oct 5 control) | 562.8 | 226.1 | 12h24m | 10/2 | 6120.901 |
| `row16=1` | 484.0 | 262.9 | 14h26m | 12/2 | 8628.064 |
| `col24=1 host_check=1` | 540.2 | 235.6 | 12h56m | 9/2 | 5355.789 |
| `col24=1 row16=1 host_check=1` | 494.6 | 257.3 | 14h07m | 6/2 | 2688.854 |

The original executable predates the counters; its upload estimate is about
10.27 GiB/iteration. Column packing measured 19-26% higher throughput than
the recorded baseline. The 2% row-only difference is too small to establish a speedup from
one run, although the 1114.8 MiB reduction is exact to the logged precision.
The 3% host-check-only difference is also modest; it does demonstrably free
383.8 MiB and remove one streamed block under this budget.
Together the encoded matrix saves 3406.5 MiB; the host check vector adds
383.8 MiB, for about 3.70 GiB less total matrix/vector storage.
The combined format is not the fastest of these runs despite uploading much
less data; capacity savings alone do not establish a speed benefit.

For this 5070/C190 setup, start with `col24=1` at the tested memory budget.
It was the fastest measured setting in both runs. `host_check=1` is available
for another 383.8 MiB of capacity; with column packing it saves 2.61 GiB in
total, but it did not improve the measured speed. Its 4% gap to the same-day
column-only control is smaller than the 5.6% change between the two
column-only runs, so these samples do not establish a reliable slowdown.
The saved and fresh builds have identical GPU kernel instructions.

Keep `row16=1` as an optional capacity tradeoff: all three options saved
3.70 GiB but ran slower than either column-packing configuration. Extra row
decoding is a likely cost to profile before pursuing a faster decoder; lower
upload volume by itself is not evidence of a faster solve. No default storage
settings were changed.

Reproduce a column-only window with the rebuilt `./msieve`:

```bash
./bench_la.sh -a 'single_copy=1 max_gpu_mem=9500 col24=1' -T 180 -W 30 -- -t 8
```

The cap is specific to this test setup. Use a suitable budget and measure on
each other GPU. The counters used for these measurements count explicit matrix uploads,
excluding the first three iteration products; they do not time copies or
measure unified-memory migrations.

An initial host-check-only run passed solver integrity and checkpointing but
its timing result is excluded: editing `bench_la.sh` while Bash was still
executing it broke the wrapper's final parsing. The benchmark script must
remain unchanged during a run.

## 2026-10-05: Claude review follow-up

Claude's review found no packed-block correctness defect in the prior
implementation. It identified an unsafe host-vector abstraction, gaps in the
benchmark/test harness, unnecessary host work, and a plausible cause of the
row16 slowdown. The changes below address those findings in source. Testing
was initially held while the GPU was occupied, then resumed with the user's
authorization. Correctness results and five completed C190 timing windows
are recorded below. The last two windows ran after the user released the
hardware again that evening. All planned local checks are complete. The
existing `./msieve` has not been replaced. A copy is saved as
`bench/codex-review-20261005/msieve-before-review-v128` for comparison.

### Safety and measurement fixes

- `v0` now has a separate check-vector representation. With `host_check=1`
  it owns a plain pinned host buffer; every ordinary GPU vector owns real
  device memory. The two inner-product call sites explicitly upload `v0`
  into the idle SpMV scratch vector, with an operand-alias guard. Checkpoint
  reads/writes use host storage directly, including host-buffer MPI gather
  and scatter helpers. Generic vector operations no longer accept the previous
  device-address-zero representation.
- Device-to-host vector copies write directly to the requested destination,
  eliminating the extra host copy when saving `v0`. Uploading the check buffer
  also avoids copying it into scratch's host mirror. The stored checkpoint
  field order and vector representation on disk are unchanged.
- Transfer accounting moved out of the generic Lanczos loop into the GPU
  forward/transpose pair. It excludes three initial pairs; a fresh solve's
  initialization product is included in that warmup. It excludes the standalone
  forward products used to validate dependencies. One final aggregate reports
  every measured pair in the solve, including short benchmark runs; it is not
  limited to the benchmark's timing window. Resident and managed-memory runs
  produce no explicit-streaming summary.
- `bench_la.sh -b` resolves the checked executable to an absolute path before
  invocation. Each run records that path and SHA-256 in `executable.txt`, and
  appends `binary_path` to its TSV row. A local bare name can no longer select
  an unrelated executable on `PATH`.
- The synthetic integration test creates its own input and polynomial, passes
  `-i` and `-nf`, and runs in its private directory. Restart interoperability
  uses completed periodic checkpoints: old binary finishes, new binary resumes
  an earlier checkpoint and finishes, then old binary resumes the new checkpoint
  and finishes. No progress-polling/SIGTERM race is involved. Each solve's
  dependencies are independently checked against the original synthetic matrix.

The earlier C190 table's copy figures came from the first measured iteration:
those benchmark windows ended before the old logger's next 1024-iteration
report. They are actual upload counts for that sampled pair, not whole-window
averages. The new final aggregate resolves this limitation.

### Host work and row decoding

- Block packing zeros only column padding and raw row counts. The column
  entries are fully overwritten, and compressed row packing writes its own
  tail padding. For C190's roughly 2.4B entries this removes about 9.6 GB of
  redundant zero writes with 32-bit columns, or 7.2 GB with 24-bit columns.
- Two-copy row16 planning computes independent per-column counts with OpenMP,
  using the requested `-t` count, before the serial prefix sum. It retains one
  row-pointer work array; it does not allocate one large array per worker.
  The serial prefix sum and size scan remain. Startup improvement is unmeasured.
- The column width is stored once per block in the plan. Host and test packers
  share `spmv_col24_put` with the decoder's header. Narrow row offsets are packed
  and read through uint32 words, avoiding the uint16 pointer alias in autotuning
  and preserving the little-endian storage layout.
- Warpmerge, segscan and scatter now keep a sequential row cursor with the
  current 128-pointer group's base and payload offset. It reloads those values
  only on a group change. Initial binary search still uses random-access
  decoding. Raw-pointer kernels retain direct reads. This targets repeated
  metadata loads when blocks contain only a few entries per row.

Row-group caching is a performance candidate, not a demonstrated speedup.
Extra live registers and the group-boundary branch could offset its benefit,
and 32-bit word loads may behave differently from 16-bit loads on each GPU.
Compare a normal build against an otherwise identical SpMV build compiled with
`-DSPMV_CACHE_ROW_GROUPS=0`; use separate clean build directories and saved
executables so changing a compiler flag cannot accidentally reuse an old object.
The previous binary is an additional comparison, but changes besides the cache
mean it cannot isolate the cache's contribution.

### Validation plan and scope

1. Compile CPU and CUDA builds in separate directories. Run the fake-executable
   regression in `tools/test_bench_la.py` and host codec checks, including narrow
   tails, wide groups, and sequential/nonmonotone cursor accesses.
2. Run the independent GPU SpMV oracle for VBITS 64, 128 and 256, with caching
   enabled and disabled. Include a sanitizer pass for group/tail boundaries.
3. Run `tools/test_lanczos_packed.py` from outside the job directory against the
   saved original baseline. It covers full single-copy and two-copy streaming,
   resident mode, both checkpoint directions, exact measured-pair counts, and
   the absence of transfer messages for resident runs. MPI checkpoint paths
   still require a separate MPI build/run; they have not been validated here.
4. On an otherwise idle GPU, repeat C190 windows with `single_copy=1
   max_gpu_mem=9500`, identical VBITS/block/kernel settings, and column-only,
   column-plus-row, and all-three storage options. Compare cached/uncached
   row16 at identical residency. Alternate controls and candidates; include
   complete iteration time, upload bytes, staging/resident allocation, and
   startup time. Test the 5090 and other GPUs separately before changing their
   recommendations.

All three storage flags remain opt-in. The complete-row-slab experiment above
remains a separate future project; this follow-up does not implement it.

### Review follow-up validation results

Clean CPU, cached CUDA, and uncached CUDA builds passed at VBITS=128. Build
artifacts and logs are under `bench/codex-review-20261005/{cpu,cached,uncached}/`
and `bench/codex-review-20261005/build-*.log`. The uncached build compiles SpMV
with `-DSPMV_CACHE_ROW_GROUPS=0`; its source is otherwise identical.

- `tools/test_bench_la.py` passed both bare-name and absolute-path cases with
  a conflicting executable on PATH and a directory containing spaces.
- Host codec tests passed with optimization and strict aliasing enabled for
  VBITS 64, 128 and 256, with the cache enabled and disabled.
- All six GPU oracle runs passed all four storage formats, three gather modes,
  both products, XOR accumulation and five matrix shapes. Compute Sanitizer
  memcheck on the cached VBITS=128 test reported zero errors.
- The cached VBITS=128 integration suite passed from `/tmp`, using its own
  input and polynomial. All six full solves returned dependencies independently
  verified against the original synthetic matrix: new single-copy streaming,
  original baseline, new restart, original restart, new two-copy streaming,
  and new resident mode. The old checkpoint at dimension 40,059 resumed with
  the new binary; its checkpoint at 60,051 resumed in the old binary. Transfer
  summaries contained the exact expected number of completed pairs, and the
  resident run emitted none. Artifacts: `packed-validation-ro5lkykd/` under
  the review directory; summary: `integration-cached-gpu.log`.
- A CPU VBITS=128 full synthetic solve passed the independent dependency check
  (`cpu-validation-6xvqdqym/`, `integration-cpu.log`). No MPI compiler is installed
  here, so MPI build/runtime validation remains outstanding.

An initial sandboxed integration attempt stopped at CUDA initialization before
any GPU computation. Repeating with direct GPU access produced the passing
results above; the failed attempt is retained in `integration-cached.log`.

For the sm_120 512-item col24/row16 kernels, static resource inspection reports
40 versus 38 registers for cached versus uncached segscan, and 26 versus 20
for scatter; neither variant spills to local memory. Warpmerge uses 40 in both.
These counts explain a possible tradeoff but do not establish a speed result.

### C190 review follow-up measurements

RTX 5070, VBITS=128, `single_copy=1 max_gpu_mem=9500`, default block size
100,176,501, `-t 8`, 30-second warmup and 180-second timing windows. Each run
completed its integrity check and checkpoint shutdown with status OK. These
remain short-window measurements, not full C190 solves.

| Storage options | Row-group cache | dims/s | ms/iteration | streamed blocks/slots | uploaded MiB/iteration | measured transfer pairs |
|---|---|---:|---:|---:|---:|---:|
| `col24=1` | irrelevant to raw rows | 611.1 | 208.2 | 10/2 | 6120.901 | 1034 |
| `col24=1 row16=1 host_check=1` | enabled | 534.1 | 238.2 | 6/2 | 2688.854 | 898 |
| `col24=1 row16=1 host_check=1` | disabled | 525.5 | 242.1 | 6/2 | 2688.854 | 894 |
| `col24=1 row16=1` (evening) | enabled | 511.9 | 248.5 | 8/2 | 4033.282 | 860 |
| `col24=1` (evening control) | irrelevant to raw rows | 613.6 | 207.4 | 10/2 | 6120.901 | 1031 |

The cached all-three sample is 1.6% faster than its uncached counterpart. One
pair with a difference this small does not establish a repeatable speedup.
Both have identical block placement and copy counts. All 24 raw-row kernels
also have identical instructions between these two builds, as checked by
disassembly. The two column-only controls agree within 0.4%. Column-only
remains the fastest configuration measured here, including the evening
comparison: 613.6 versus 511.9 dims/s with column-plus-row packing (19.9%
higher throughput). Column packing saves about 2.24 GiB of matrix storage;
column-plus-row packing saves 3.33 GiB. The all-three representation saves
about 3.70 GiB in matrix/vector storage and trades speed for capacity on this
workload. These savings describe the total storage footprint; the planner
spends freed space on more resident blocks, so peak allocated VRAM need not fall.

Load-to-first-progress times were 79, 78, 152, 92 and 75 seconds respectively.
Initial reported VRAM use was 2269, 2391, 2044, 3183 and 2448 MiB. The last two
runs started at 22:34 and 22:39, roughly ten hours after the first three.
Those differences and the single cached/uncached pair limit conclusions about
small performance changes, even though the column-only controls agree closely.
All five runs passed their checkpoint integrity checks; the independent
correctness checks above also passed. These measurements apply to this
5070/C190 setup, not to an untested 5090 or another matrix.

Artifacts are in `bench/review-col24-control-a-20261005-120422/`,
`bench/review-all3-cached-20261005-121236/`,
`bench/review-all3-uncached-20261005-122118/`,
`bench/review-col24-row16-cached-20261005-223420/`, and
`bench/review-col24-control-b-20261005-223954/`; wrapper summaries are
`bench/codex-review-20261005/bench-*.log`, and rows are in `bench/results.tsv`.
The final benchmark exited at 22:44:43 local time on October 5. No test or
benchmark launched by this review remains running.

All planned local CPU/single-GPU correctness checks and these five benchmark
windows are complete. MPI and measurements on other GPU models remain outside
the validated results here. The tested review executable is
`bench/codex-review-20261005/cached/msieve`; its code matches the working tree.
The root `./msieve` still contains the pre-review implementation.

For speed on this setup, retain `single_copy=1 col24=1` and the tested memory
budget. Add `row16=1` and/or `host_check=1` when their additional capacity is
useful; row-group caching has not closed the speed gap. These formats are
chosen while loading and preparing the matrix for the GPU. The existing
`.mat` remains reusable with `skip_matbuild=1`.

## 2026-10-06: Profiling col24 on C190, two rejected forward kernels, review fixes

Setup for every GPU run below unless stated: RTX 5070 under WSL, the C190
matrix (25,152,405 dims, 2.40B sparse nonzeros after packing 128 dense rows),
VBITS=128, `single_copy=1 max_gpu_mem=9500 col24=1 -t 8`, default block_nnz
100,176,501 (24 blocks; 10 streamed through 2 buffers of 383 MB, 6120.9 MiB
uploaded per iteration). Profiling used the October 5 review build,
`bench/codex-review-20261005/cached/msieve`.

### What the October 4-5 storage results mean

Column packing changed which resource limits the run. Copy rates per
iteration, from the transfer counters (the baseline from the October 4
planner replay):

| Storage | Uploaded/iteration | ms/iteration | Copy rate |
|---|---:|---:|---:|
| none (Oct 4 baseline) | ~11.0 GB | 268.6 | ~41 GB/s, close to the 44-48 GB/s the link reaches |
| col24 | 6.42 GB | 208 | ~31 GB/s |
| col24 + row16 + host_check | 2.82 GB | 238 | ~12 GB/s |

The baseline was copy-bound; with col24 the copies have slack and the SpMV
kernels set the pace (confirmed by the nsys trace below). row16 and
host_check only save memory and uploads, so they could not help this run, and
row16's decoding landed on the critical path: the October 5 noon pair (all
three options vs col24 alone) was ~30 ms/iteration (~15%) slower. They pay
only when a run is still copy-bound after col24. In practice col24 at the
safer 9500 MB cap (611-640 dims/s) already matches the old binary's
full-budget production run (594.5).

Under WSL, CUDA reported 11003-11015 MB free in every run regardless of what
else held VRAM (2.0-3.2 GB per nvidia-smi), so max_gpu_mem never adapts to
other programs. The October 6 nsys run started with 3157 MiB in use by other
programs and still ran at full speed, so the slow October 5 evening
col24+row16 window (511.9, 3183 MiB in use) was probably row16 itself, not
WDDM paging.

### nsys: where a col24 iteration goes

`bench/nsys-col24-20261006-111923/` (`col24.nsys-rep`, `col24.sqlite`):
30 s capture starting 200 s after launch, 145 iterations, 206.4 ms/iteration,
about 617 dims/s (matches the 611-614 October 5 windows). Integrity check
passed at the SIGTERM.

| Part | ms/iteration | Share | Launches |
|---|---:|---:|---:|
| forward SpMV (segscan) | 104.3 | 50.5% | 24 |
| transpose SpMV (scatter) | 82.3 | 39.9% | 24 |
| inner_prod (NxB * BxB) | 9.6 | 4.7% | 4.6 |
| outer_prod (BxN * NxB) | 7.3 | 3.5% | 3 |
| memset + mask | 2.0 | 1.0% | 4.6 |
| **GPU busy** | **205.5** | **99.6%** | |

- The compute stream was never idle waiting for a block copy (0.0 ms) and
  idle 0.9 ms/iteration for the host. The copy engine was busy 125.5 ms per
  iteration (61%) at 51.1 GB/s for 16 large uploads: the copies are hidden.
- Kernels overlapping a copy averaged 4.5% (segscan) and 9% (scatter) longer,
  but the wider middle blocks overlap copies more often, so streaming's
  hidden cost is a few percent at most.
- Forward time per block is flat at 4.2-4.6 ms across block widths of
  0.77M-1.15M columns; transpose time grows with width, 3.2 to 3.6 ms.
- Vector operations are 9% at VBITS=128 (vs ~19% at VBITS=256 on C170), so
  fusing them would gain ~2% here.

An ncu attempt that collected nothing (see below) still gave the longest
col24 measurement: 1,459,296 dims in 38m58s, 624 dims/s, 203.9 ms/iteration,
integrity check passed (`bench/ncu-col24-20261006-120244/`).

### ncu: forward vs transpose on the same block

`bench/ncu-col24-20261006-125449/` (`turnaround.ncu-repz`, `raw.csv`). ncu
needs GPU performance counters; under WSL it fails with ERR_NVGPUCTRPERM
until NVIDIA Control Panel > Desktop > Enable Developer Settings, then
Developer > Manage GPU Performance Counters > allow all users, followed by
`wsl --shutdown`. Profiled the last two forward and first two transpose
launches of a steady-state iteration (`--kernel-name regex:"segscan|scatter"
--launch-skip 262 --launch-count 4 --set full`). The first forward launch ran
at 1.45 GHz and is excluded; the rest ran at 2.58 GHz. Block 23 (98.9M
nonzeros, 766K columns) in both directions:

| | Forward (segscan) | Transpose (scatter) |
|---|---:|---:|
| time | 4.12 ms | 3.28 ms |
| DRAM read / write | 0.76 / 0.32 GB | 0.75 / 0.003 GB |
| DRAM bandwidth | 262 GB/s (~40%) | 229 GB/s (~35%) |
| L2 throughput | 51% | 64% |
| L2 / L1 hit rate | 81% / 46% | 81% / 48% |
| instructions | 935M | 453M |
| issue slots active | 48% | 30% |
| load/store pipe | 68% | 32% |
| long-scoreboard share of stalls | 62.5% | 80.5% |
| registers, achieved occupancy | 38, 93% | 20, 87% |

- Neither kernel is bandwidth-bound; both are latency-bound on scattered
  memory accesses at high occupancy.
- The forward's rewrite of the full 25M-row output vector adds 0.32 GB of
  DRAM writes per block, but DRAM stays at 40%, so it is not the limiter. A
  forward kernel that walks two blocks per pass (to halve those rewrites)
  would target the wrong cost; dropped.
- The forward runs 2.06x the transpose's instructions for the same nonzeros,
  from its segmented scan.
- At VBITS=128 each random gather uses 14.2 of the 32 bytes fetched.

### Two forward kernels tried and rejected

Opt-in variants, both passing the CPU oracle at VBITS 64/128/256, a
compute_75 PTX build (the pre-sm_80 path) and memcheck:
- `spmv_kernel=red`: the scatter kernel's shape for the forward product.
  Each group XORs x[col] straight into y[row] with an atomic, no scan,
  SEGSCAN_UNROLL gathers issued ahead.
- `spmv_kernel=ballot`: segscan with the row starts marked by one ballot, so
  the scan shuffles only values; group 0 settles the carry from its own row,
  and on sm_80+ `__reduce_max_sync` skips levels longer than the step's
  longest run. About 12 shuffle instructions per step instead of 17.

Same binary for all runs, ABCCBA order, 30 s warmup + 180 s windows, all
integrity checks OK (rows in `bench/results.tsv`):

| Forward kernel | Run 1 | Run 2 | Average dims/s | ms/iteration |
|---|---:|---:|---:|---:|
| segscan (current) | 629.1 | 640.2 | 634.7 | 198.8-202.3 |
| ballot | 628.2 | 627.9 | 628.0 | 202.6-202.7 |
| red | 481.7 | 482.1 | 481.9 | 263.9-264.1 |

- red is 24% slower: about four atomics per step hit each output row, and L2
  applies same-address atomics one after another.
- ballot ties: the ballot, clz and max-reduce instructions cost what the
  saved shuffles did. Shuffle count is not the limiter.
- The forward's random accesses are gathers whose results the scan must wait
  for; the transpose's random accesses are atomics it never waits on. That
  matches the ncu stall profile and neither variant changes it.
- Both were removed from the tree; the source, build and logs are in
  `bench/claude-fwd-20261006/`. Untried: `-DSEGSCAN_UNROLL=8` (more gathers in
  flight, at the cost of registers).
- The segscan controls (629-640) came out above the October 5 col24 windows
  (611-614) with the same kernel code; a fresh build with the current
  compiler and less VRAM held by other programs (2.2-2.3 GB) can't be
  separated.

Conclusion for this card and matrix: nothing is idle, copies are hidden, and
the two SpMV kernels are latency-bound near what this layout allows. The
remaining identified upside is the forward product (at most ~0.8 ms x 24
blocks, ~10%, if it matched the transpose) and ~2% from vector-op fusion.
Matrix reordering remains the only large lever.

VBITS=256 was not run on C190: at the 9500 MB cap its vectors take ~5.4 GB,
its default blocks are half the size (twice the row-pointer arrays), and most
of the matrix would stream, copy-bound again. On C189 before col24, VBITS=256
got 685 dims/s against 787 for VBITS=128.

### Review follow-up (October 6 /code-review)

No high-severity findings. Fixed:
1. A stop signal could fail the integrity check spuriously. The check is only
   valid against a check vector more than three iterations old, but a
   SIGINT/SIGTERM forced it at once: in iterations 1-3 (v0 still the
   starting vector), within three iterations of a `-ncr` restart (v0 is the
   vnext of the checkpoint iteration), and, by the same rule, within three
   iterations of a periodic check or the dump-interval calibration (not
   reproduced separately). msieve printed "error: corrupt state" and exited
   -1 instead of checkpointing. In non-MPI builds
   the stop now waits until the check vector is old enough (at most four
   iterations), then checks, checkpoints and halts as before; MPI still stops
   at once without a checkpoint. `tools/test_lanczos_interrupt.py` (CPU
   builds, 600k-row synthetic job, `bench/claude-fixes-20261006/early-stop/`):
   the old build failed both the start and the restart cases; the new one
   halted at iteration 4 and at restart+4 with clean checkpoints, and the
   deferred checkpoint resumed to dependencies verified against the original
   matrix.
2. The GPU oracle now covers a block 2^24 columns wide with indices at the
   top of the range and with a 0xff high byte, the `spmv_col_bytes` 24-bit
   cutoff (2^24 columns: 3 bytes; 2^24+1: 4), and the autotuner's size
   computation. A deliberately broken device decoder (dropping bit 23 on the
   GPU only) passed the host codec checks and failed the new case.
3. CLAUDE.md no longer carries run status; it keeps the guidance and points
   here.
4. Cleanups: row16 payload and total-size arithmetic and the 24-bit cutoff
   live in `spmv_format.h` (`spmv_row16_payload_bytes`,
   `spmv_row16_bytes_from_last`, `spmv_col_bytes`); the autotuner copies one
   header word to size the array; `col_region_bytes` replaces the duplicated
   padding formula and a block's bytes are derived from its column and row
   regions; `check_copyin`/`check_copyout` are only built without MPI;
   `check_upload` exists only in CUDA builds (CPU builds use v0 directly).

Validation: CPU and CUDA (sm_120, VBITS=128) builds in
`bench/claude-fixes-20261006/` give exactly the review builds' warnings (95 and
99); the oracle passes at VBITS 64/128/256; the interrupt test passes. MPI is
still neither built nor run.

Left open from the review: every SpMV kernel is compiled for all four storage
formats (48 instances instead of 12, for every architecture); row16 planning
scans all rows once per forward block on one thread and packing rebuilds the
same pointer array; `bench_la.sh -b` hashes only the executable, so
`CUDA_SINGLE_BINARY=0` builds would load the current directory's
`spmv_engine.so`; col24 or row16 with an old `spmvlib=` engine exits even
when no block would end up packed.

The root `./msieve` (built 2026-10-05 05:54) predates the October 5 review
follow-up and these fixes; rebuild before production runs.

## Idea: smaller matrices from filtering

Every lever above makes the same matrix cheaper to store or multiply.
Filtering decides the matrix itself, so it acts on both costs at once.
Lanczos runs about N/VBITS iterations for an N-dimension matrix, each
touching every nonzero, so time scales roughly with N x W (W = total
nonzeros). Memory is the vectors (about 7 x N x VBITS/8 bytes on one GPU,
plus the dense rows) and the matrix (3-4 bytes per nonzero, plus a
row-pointer array per block). Two knobs:
- Target density trades dimensions for weight: higher density gives fewer
  dimensions (smaller vectors, fewer iterations) but more nonzeros per
  column (bigger matrix, slower iterations).
- More relations (oversieving) shrinks both, at the cost of sieving time.

What we have measured, all single copy on the RTX 5070:

| Job | TD | Dims | Sparse nonzeros | dims/s | Full ETA |
|---|---:|---:|---:|---:|---:|
| C170 (VBITS=256, resident) | 90 | 13.17M | 1.13B | 1602 | 2h16m |
| C170 | 110 | 12.31M | | 1635 | 2h05m |
| C189 (VBITS=128, streaming) | 90 | 23.9M | 2.06B | 774 | 8h34m |
| C189 | 110 | 22.3M | 2.32B | 753 | 8h14m |
| C189 | 130 | 21.2M | 2.57B | 700 | 8h24m |

TD 110 was fastest both times, but only by 4-8%. The simple N x W model
would have picked TD 90 for C189 (49.2 vs 51.7 vs 54.5 x 10^15); it misses
that while streaming, fewer dimensions also mean smaller vectors and more
of the matrix resident. The best density depends on the card, and with
multi-density filtering plus `all_matbuild=1` each candidate is one
`bench_la.sh -d N` window away.

Not yet explored:
- How much oversieving shrinks the matrix at a given size. The NFS@Home
  pastebin logs scraped in `~/code/paste-scraper` (`data/results.csv`,
  about 4,770 jobs with digits, SNFS/GNFS, unique relations, matrix rows,
  matrix MB and target density) could fit matrix rows against relations at
  fixed digits.
- Whether a different merge gives smaller or lighter matrices from the same
  relations, e.g. comparing msieve's filtering with CADO-NFS's on one job.
  Unverified idea.

## 2026-10-07: C191 col24 in both storage modes, TD 110 vs 130

C191 from a `target_density=110,130` filter: TD 110 is 26,691,588 dims with
2.56B sparse nonzeros, TD 130 is 25,320,748 dims with 2.84B. RTX 5070, WSL,
VBITS=128, `max_gpu_mem=9500 -t 8`, rebuilt root `./msieve` (October 6 fixes).
Each window is traced with nsys from launch: 30 s after the first progress
line, the last 70 s of iterations are analyzed, then SIGTERM. All eight
passed the integrity check. Scripts: `bench/claude-c191-scripts/`; runs and
`summary.json` files: `bench/c191-*`.

| TD 110 config | dims/s | ms/iter | Uploaded GB/iter | Copy rate | GPU busy ms/iter |
|---|---:|---:|---:|---:|---:|
| single copy | 399.8 | 318.4 | 14.04 | 49.8 GB/s | 221.2 |
| single copy, col24 (first) | 490.2 | 259.7 | 8.18 | 41.6 GB/s | 236.2 |
| single copy, col24 (repeat, 32 min later) | 550.4 | 231.3 | 8.18 | 51.2 GB/s | 229.9 |
| two-copy (default 160M blocks) | 273.0 | 466.0 | 19.56 | 48.1 GB/s | 266.2 |
| two-copy, col24 | 318.3 | 399.9 | 13.66 | 40.8 GB/s | 293.5 |
| two-copy, 400M blocks | 211.7 | 601.0 | 21.24 | 47.1 GB/s | 547.7 |
| two-copy, 400M blocks, col24 | 234.1 | 543.5 | 13.78 | 46.7 GB/s | 512.5 |

- The host-to-GPU copy rate varied from 41 to 51 GB/s between windows with
  identical settings, which moved the single-copy col24 result by 12%
  (490.2 vs 550.4). The slow windows were the first one after the filter
  wrote the matrices and one at 09:10; a single-thread `ecm -pp1` was
  running throughout (its stage 2 is memory-heavy). Treat single dims/s
  comparisons of streamed runs as +-10% unless the copy rates match.
- End to end, col24 helped every mode: single copy +23% to +38% depending on
  which col24 window, two-copy +17% (its col24 window had the slow copies),
  400M blocks +11%. Single copy + col24 is the configuration for this card.
- Kernel time alone (per-block medians): col24 makes segscan slower (single
  copy forward +7%, the copy-free blocks +5-8%; two-copy default forward
  +13%, transpose +10%), leaves the scatter unchanged (within 1%), and makes
  warpmerge at 400M blocks faster (forward -7%, transpose -6%). Every
  warpmerge launch overlapped a copy and the raw run copied more, so part of
  that gain may be less interference: call it 0-7%. Blocks of 400M are
  ~16 nonzeros per row with an input window larger than L2, the closest
  this card gets to Greg's A100 regime.
- Density, back to back at equal copy rates: TD 130 521.1 dims/s (244.2
  ms/iter, 5.2 GB streamed) vs TD 110 550.4 (231.3 ms/iter, 4.7 GB
  streamed): full ETAs 13h30m vs 13h28m, a tie. TD 110 left the GPU busy
  229.9 of 231.3 ms; TD 130 237.4 of 244.2, slightly copy-limited.

### VBITS 128 vs 192 vs 256 on the same C191 matrix

VBITS=192 had never run on the GPU. Before the C191 windows it passed the
SpMV oracle (all formats, both directions, six shapes including the 2^24
column case) and full synthetic GPU solves with streaming (single copy with
col24+row16+host_check, and two-copy with col24), dependencies verified
against the original matrix; VBITS=256 passed the same solves. Builds:
`bench/claude-v192-20261007/`, `bench/claude-v256-20261007/` (same sources as
the root build). TD 110, single copy, `max_gpu_mem=9500 col24=1`, default
block sizes, run back to back with the 128 control first and last; copy
rates were 49.9-50.5 GB/s in every window, all integrity checks passed.

| VBITS | Options | Blocks (nnz each) | Streamed | dims/s | ms/iter | GPU busy | Uploaded GB/iter |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | col24 (first) | 26 (100.7M) | 12, 4.7 GB | **540.1** | 235.5 | 233.7 | 8.18 |
| 128 | col24 (last) | 26 | 12, 4.7 GB | **542.8** | 234.4 | 233.2 | 8.18 |
| 192 | col24 | 39 (64.7M) | 26, 7.3 GB | 491.0 | 389.4 | 382.1 | 14.29 |
| 192 | col24 row16 host_check | 39 | 19, 4.5 GB | 433.4 | 441.2 | 439.5 | 8.49 |
| 256 | col24 | 46 (53.4M) | 37, 9.3 GB | 498.3 | 512.1 | 489.5 | 18.56 |
| 256 | col24 row16 host_check | 46 | 30, 6.0 GB | 477.9 | 534.3 | 531.8 | 11.95 |

Kernel time per dimension (ms/iteration divided by dims/iteration):

| VBITS | forward SpMV | transpose SpMV | vector ops |
|---:|---:|---:|---:|
| 128 | 0.94 | 0.74 | 0.16 |
| 192 | 1.03 | 0.77 | 0.20 |
| 256 | 0.95 | 0.74 | 0.24 |

- VBITS=128 won: 8% ahead of 256 and 10% ahead of 192. In every run the
  GPU was busy at least 96% of the iteration, so copies were hidden at all
  three widths; this is a kernel comparison.
- 256's full-sector gathers did not pay here. Its default single-copy
  blocks are half the size (the L2/3 window holds half as many 32-byte
  entries), so each iteration walks all 26.7M rows 46 times instead of 26,
  and that per-block row work cancelled the gather savings: SpMV per
  dimension came out equal to 128. Its vector operations cost ~45% more
  per dimension. Larger blocks don't rescue it on this card: the September
  C170 sweep lost 20% going from 48M to 64M blocks at VBITS=256.
- 192 has the gather cost of 128 per dimension (24-byte entries straddle
  32-byte sectors half the time) plus 39 blocks' row work, so its forward
  product is the slowest per dimension.
- row16 and host_check only slowed both wider widths: their copies were
  already hidden, so only row16's decoding showed.
- For this 12 GB card at ~26M dimensions: VBITS=128, single copy, col24.
  Recommended for the C191 run: TD 110 with that configuration (projected
  13h42m at 541.5 dims/s). A smaller C168 is next, which may leave room to
  try 512 against 256 and 128 with the matrix fully resident.

### C191 production run and the WSL freezes (October 7-8)

Run: TD 130 (25,320,748 dims, 2.84B sparse nonzeros), VBITS=128,
`single_copy=1 col24=1 -t 8`, root `./msieve` with the October 6 fixes.
Lanczos needed four starts; three ended early, none for a code reason:

| Segment | Budget | Streamed | Dims covered | Rate | End |
|---|---|---|---|---:|---|
| 13:06-18:45 | `max_gpu_mem=10000` | 11 of 25, 4.8 GB | 0 -> 10,050,127 (checkpoint 18:15) | ~543 dims/s | froze 18:45 |
| 20:02-21:10 | `max_gpu_mem=9500` | 12 of 25, 5.2 GB | -> 10,560,000 (checkpoint 20:18) | ~535 dims/s | froze 21:10 at 12,209,064 |
| 23:05-23:11 | default (free - 1536 MB) | 12 of 25 | -> 10,675,634 | ~330 dims/s | Ctrl-C (Windows GPU scheduling was off) |
| 23:15-06:41 | default | 12 of 25 | -> 25,320,746 | **546.9 dims/s** | finished: 28 dependencies |

The final segment uploaded 8,655.9 MiB in 20 copies per iteration
(BLanczosTime 26,931 s for that segment). Pure Lanczos time at ~545 dims/s
was about 12.9 hours; the freezes stretched it to about 17.6 hours of wall
time. TD 130 at the 9500 MB benchmark cap measured 521.1; the production rate
was ~5% higher.

The freezes:
- Both times msieve's main thread spun at ~91% CPU, about 70% of it in the
  kernel (the CUDA driver polling `/dev/dxg`), while the GPU sat idle at
  6%, 14 W, P8 with its 11.5 GB still allocated. Ctrl-C cannot stop it in
  that state (the thread never returns from the CUDA call); `kill -9`, then
  `-ncr` from the last checkpoint.
- Windows logged nvlddmkm Event ID 153 ("Error occurred on GPUID: 100")
  at 6:45 PM and 9:10 PM, matching the two freezes: the NVIDIA driver reset
  the GPU, and the WSL process was left waiting for work that no longer
  existed. Windows apps recover from such a reset, so nothing visibly
  crashed. The WSL kernel log showed nothing relevant.
- Not the 2-second TDR: `TdrDelay` and `TdrDdiDelay` were already 60.
  Not checkpointing: the freezes came ~30 and ~51 minutes after a
  checkpoint. The user's GPU-ECM has hung the same way.
- Changes made: WSL updated (kernel 6.6.87 -> 6.18.40), displays no longer
  turn off (a blank screensaver instead), hardware-accelerated GPU
  scheduling briefly turned off. With GPU scheduling off, the same layout
  ran ~40% slower (~330 dims/s at 99-100% SM utilization, ~180 W instead of
  the 250 W limit); turning it back on restored full speed. With displays
  kept on, the final 7h26m segment ran through the night without a freeze.
  That points at display power-off as the trigger, from one night only.
- The ~4.5 GB/s GPU-to-host PCIe traffic `nvidia-smi dmon` shows during
  streaming is normal: read requests for the copy engine's host reads,
  about a tenth of the 25-55 GB/s inbound. It is not WDDM paging.
- With `-nc3` on the same command line, Ctrl-C during Lanczos goes straight
  on to the square root, which failed harmlessly here ("read_cycles can't
  open dependency file"). A stale `.dep` from another matrix would be used.

Square root: dependencies 1-8 all reached the GCD ("GCD is 1" or "GCD is
N"), so they were valid squares that split N trivially; dependency 9 gave
p68 * p123 at 08:38 on October 8:
22677242890382744370618828743254770649640173390940296665233655349901 *
952651576690228008247786495474124549513052821401333567352348350806986203843055980770347807873061452273616671405424898966583

## 2026-10-08: C168 with the matrix on the card (density, kernels, VBITS)

The basement C168 matrices from the filtering experiment
(`bench/c168-quality-m0fm80c8/`): TD 70 9,595,945 dims and 646M nonzeros,
TD 90 8,777,560 / 747M, TD 110 8,224,520 / 844M, plus the `merge_cost=net`
TD 70 (9,584,209 dims) and TD 90 (8,770,757). All fit on the 12 GB 5070, so
nothing streamed: these are kernel and vector-op comparisons. nsys-traced
windows, 20 s warmup + last 55 s analyzed, single copy unless noted, VBITS
128 unless noted; each window passed the integrity check. Builds: root
`./msieve` (128), `bench/claude-v{64,512}-20261008/`, `bench/claude-v{192,256}-20261007/`
(same sources). VBITS 512 and 64 first passed the SpMV oracle and full
synthetic GPU solves with verified dependencies. Scripts:
`bench/claude-c168-scripts/`; runs: `bench/c168-gpu-*`. The C191 square root
ran on the CPU throughout; one v128 window with 5 ms/iteration of idle GPU
(1869.9 dims/s, probably host contention) is excluded.

### The automatic kernel choice picks the slower kernel

When these runs were made, `spmv_kernel=auto` used segscan below 8
nonzeros per row per block (`SEGSCAN_MAX_ROW_MEAN`) and warpmerge above
(the cutoff is now 28, see below). On this card segscan won every case
measured, 7 to 24 nonzeros per row:

| Case (TD 90 unless noted) | nnz/row/block | warpmerge | segscan | segscan gain |
|---|---:|---:|---:|---:|
| VBITS 128 single, TD 70 | 6.8 | 1866.6 (forced) | 2587.9 | +39% |
| VBITS 128 single | 9.4 | 1984.7 (auto) | 2529.8 | +27% |
| VBITS 128 single, TD 110 | 12.0 | 1940.7 (auto) | 2419.9 | +25% |
| VBITS 128 two-copy | 14.6 | 1744.4 (auto) | 2097.4 | +20% |
| VBITS 256 two-copy | 14.6 | 1783.4 (auto) | 1907.6 | +7% |
| VBITS 64 single | 20.3 | 1507.2 (auto) | 1687.7 | +12% |
| VBITS 512 two-copy | 24.0 | 1498.4 (auto) | 1775.7 | +19% |

dims/s, all col24. Earlier data: on C170 streaming at VBITS 256 (September)
segscan was +2% at 9.7 and tied at 19.4; on Greg's A100 blocks (29-57 per
row) forced segscan was 2% slower. The streamed C190/C191 production runs
had 4-5 nonzeros per row, so auto already chose segscan there. The threshold
was raised to 28 on October 8: every 5070 case above stays on segscan and
the A100's ~29-per-row forward blocks stay on warpmerge. The cutoff now lives
in `cub/spmv_engine.h`, and msieve logs how many blocks it expects on each
kernel ("SpMV kernel auto: segscan for N of M blocks ..."), so runs before
and after the change can be told apart in their logs. Any density or VBITS
comparison at this size needs the kernel pinned (`spmv_kernel=segscan`),
or it measures the threshold: with auto, TD 70 looked 17% faster than TD 90.

### Density: TD 110 fastest, against the N x W proxy

| | TD 70 | TD 90 | TD 110 |
|---|---:|---:|---:|
| VBITS 128, segscan, col24: dims/s | 2587.9 | 2529.8 | 2419.9 |
| full Lanczos | 61.8 min | 57.8 min | 56.6 min |
| VBITS 256, col24 (auto = segscan): dims/s | 2593.8 | 2662.0 | 2592.4 |
| full Lanczos | 61.7 min | 55.0 min | 52.9 min |

The rows x sparse-nonzeros proxy ranked them the other way (TD 70 lowest by
6-12%). Measured, TD 110 took 8-14% less Lanczos time than TD 70 and 2-4%
less than TD 90, in line with
the C189 and C191 measurements: costs that scale with rows (vector
operations, per-block row work) dominate more than the proxy assumes.

`merge_cost=net` showed no measurable effect: TD 70 -0.9% full time
(2608.1 vs 2587.9 dims/s, 0.12% fewer dims), TD 90 +0.5% (auto kernel),
both within the ~1% run-to-run spread. The option was removed the same
day (see below).

### VBITS and storage with the matrix on the card (TD 90)

| VBITS | single copy | two-copy | notes |
|---:|---:|---:|---|
| 64 | 1687.7 (segscan) | 1414.4 (auto) | 5 blocks of 179M |
| 128 | 2613.6 raw / 2529.8 col24 (segscan) | 2097.4 (segscan) | |
| 192 | 2483.5 | | |
| 256 | **2660.2 raw / 2662.0 col24** | 1907.6 (segscan) | 17 blocks of 38M |
| 512 | 1789.8 | 1775.7 (segscan) | vector ops 124 of 286 ms/iteration |

- VBITS 256 single copy was fastest: 55.0 min for TD 90 against 56.0 min
  for VBITS 128 without col24. The C191 result (128 ahead) was a streaming
  result; with everything on the card the full-sector gathers win
  narrowly.
- 512 loses ~33%: its vector operations cost 124 ms per iteration, about
  as much as the whole SpMV.
- With segscan in both, single copy beat two copies by 21% at VBITS 128
  (col24 in both) and 40% at 256, and tied at 512. The VBITS 64 pair is
  not comparable: its two-copy run used warpmerge.
- col24 on the card: 3.2% slower at VBITS 128 (segscan forward +7.7%:
  23.9 vs 22.2 ms), unchanged at 256. Use it when the matrix streams
  (+20-38% on C190/C191), not when it fits.
- Best measured for this C168: VBITS 256, single copy, TD 110, about 53 min.

## 2026-10-08: Filtering experiments tried and removed

Another model tried two ways to get smaller matrices from the same
relations, on the basement C168 (artifacts and matrices in
`bench/c168-quality-m0fm80c8/`):
- `merge_cost=net`: order the full merge by estimated net nonzero growth,
  (k-2)(w-2)+1 for an ideal in k relation sets whose lightest has weight w,
  instead of the Markowitz count (k-1)(w-1). It cut rows x sparse nonzeros
  by 0.47% at TD 70 and 0.36% at TD 90; on the GPU the full Lanczos time
  moved -0.9% and +0.5%, within noise.
- `post_sge`: after the matrix build, eliminate low-degree sparse rows over
  GF(2), carrying the substitutions into the dense rows and cycle lists. It
  cut the proxy by at most 0.074% (TD 70, fill limit 64), 0.007% at TD 110:
  msieve's merge already removes nearly everything that is cheap to remove.

Both were removed before being committed. The October 8 code review had also
found problems that would have needed fixing (`-nc1` with `post_sge`
rewriting the `.cyc` that matched the `.mat`, negative tests passing on old
log text, no `reduce_matrix`-style cleanup of the eliminated matrix). The
code, tests and documentation are archived in
`bench/c168-quality-m0fm80c8/removed-code/`.

The useful result from that work was the density comparison it set up: the
rows x nonzeros proxy ranked TD 70 best, and the GPU measurements above
ranked it worst. Measure densities on the GPU, with the kernel pinned or the
threshold fixed, rather than trusting the proxy.

## 2026-10-10: Synthetic matrices for a Lanczos tune

Goal: a tune (`tools/tune_la.sh`) that times settings on any card without
real job files, so other people can send comparable results and msieve can
later pick settings from them. It needs synthetic matrices that the GPU code
treats like real ones. `tools/synthmat.c` measures a real `.mat`
(`profile`), generates matrices from a profile (`gen`), and prints the
per-block row-segment histogram the SpMV kernels see (`blockstats`).

What a real matrix file looks like (C168, TD 70/90/110, 8.2-9.6M columns):
- 104 dense rows (bit-packed in each column) at about 50% density.
- Sparse rows sorted by weight, heaviest first (`reduce_matrix` permutes
  them; 99.9% of neighbors are in order). The heaviest sparse row is in
  8.7/10.8/12.7% of columns at TD 70/90/110, which is proportional to the
  column weight. The top 392 sparse rows (what VBITS 256 removes or packs)
  hold 14.7% of the nonzeros, and the top 10,000 hold 36%.
- Column weights spread from 6 to 400-600. `reduce_matrix` sorts columns by
  weight and then swaps every other one end to end
  (`lanczos_pre.c`, "make heavy columns alternate with light columns"), so
  the file alternates 6, 270, 6, 269, ...

The generator gives every row its profiled weight. Each row picks
that many distinct columns with probability proportional to each column's
target weight, where targets follow the profiled distribution, sorted and
alternated the same way. That is all it models, apart from a few columns
that come out below the lightest profiled weight and are topped up. At
other densities the lightest row and column weights stay at 3 and 6, as in
the real C168 TD 70, 90 and 110 matrices (until 2026-10-11 they scaled with
density, giving columns of weight 4 and rows of weight 1 at TD 70). At the profiled size, the
per-block row-segment histograms (single copy at VBITS 256 and 128, two
copies, 38M/82M/134M-nonzero blocks) matched the real TD 90 matrix within
0.1 percentage points in every bin. Real NFS matrices behave like rows
placed in columns at random in proportion to column weight. The shipped
profile, `tools/la_profile_td110.txt` (22 KB), is from the C168 TD 110 matrix.

### GPU check at the profiled size

Same binaries (HEAD, VBITS 128 and 256), real and synthetic back to back,
nsys traces, RTX 5070, `-t 4`, with a CPU job of the user's running
throughout. dims/s:

| TD 90 setting | VBITS 256 real / synthetic | VBITS 128 real / synthetic |
|---|---|---|
| single copy | 2505.5 / 2524.3 | 2496.5 / 2489.9 |
| two copies | 1760.0 / 1743.0 | 2093.5 / 2087.5 |
| single, warpmerge forced | 1567.3 / 1578.6 | 1783.7 / 1784.7 |
| single + col24, streamed (max_gpu_mem) | 2478.5 / 2477.9 | 2065.3 / 2061.8 |
| two + col24, streamed (max_gpu_mem) | 1766.2 / 1764.1 | 1759.1 / 1755.4 |

TD 70 and TD 110, single copy, VBITS 256: 2470.0 / 2505.2 and
2481.0 / 2478.7. The worst gap is 1.4% (TD 70), and the rest are within 1%.
Forward, transpose and vector-op times match within about 1 ms, the
streamed runs chose the same blocks and uploaded the same GB per iteration,
and forcing warpmerge cost the same on both (37% at 256, 29% at 128).

### Scaling to other sizes

Stretching the whole profile by rank fraction was wrong. A synthetic
C191-sized matrix (26.7M columns, 103.82/col, VBITS 128) came out with
99.6 nonzeros per column after msieve removed the heavy rows, against the
real C191's 96.0. The stretch spreads the heavy rows thin, while real heavy
rows are the same small-prime ideals at any size and keep their share of
each column. As a result, every synthetic run was 4-8% slower than the
recorded C191 numbers, with 3-11% more uploads. Recorded heads:

| Matrix | Columns | Heaviest sparse rows removed | Share of nonzeros |
|---|---:|---|---:|
| C168 TD 110 | 8.2M | top 136 / top 392 | 8.8% / 14.7% |
| C191 TD 110 | 26.7M | top 136 (VBITS 128) | 7.53% (0.86x) |
| C199 TD 130 | 41.6M | top 392 (VBITS 256) | 12.83% (0.87x) |

The generator now keeps the first 1% of profiled rows at their absolute
rank and column share, times (size ratio)^-0.1. It stretches the rest over
the remaining rows and rescales it to the target weight. That predicts 7.9%
for the C191 head and 12.5% for the C199 head. At the profiled size nothing
changes. Rerun against the Oct 7 C191 records:

| VBITS 128, max_gpu_mem=9500 | real C191 | synthetic |
|---|---|---|
| sparse part after packing | 96.00/col | 95.66/col |
| single + col24 | 12 of 26 blocks streamed, 8.18 GB/iter, GPU busy 230-234 ms, 540-550 dims/s (490 in a slow-copy window) | 12 of 26, 8.15 GB/iter, 240 ms, 521 |
| single copy | 14.04 GB/iter, 221 ms, 399.8 | 14.00 GB/iter, 227 ms, 397.9 |
| two copies | 19.56 GB/iter, 266 ms, 273.0 (48.1 GB/s copies) | 19.57 GB/iter, 265 ms, 291.5 (51.5 GB/s) |

Streaming plans and upload volumes now match, kernel times are within 2-3%,
and what remains of the dims/s gap follows copy rates, which varied ±10%
between sessions on Oct 7 as well. The head model rests on two aggregate
shares. The next check is the C199 on a 5090 (815 dims/s real at 41.6M,
TD 130, VBITS 256, `single_copy=1 block_nnz=139000000`).

Density depends on TD, not size. Sparse nonzeros per column in the matrix
file, from the C168, the C191 and the C199 (whose `all_matbuild` built all
four densities):

| TD | C168 (8-10M) | C191 (26.7M) | C199 (42-53M) |
|---|---:|---:|---:|
| 70 | 68.0 | - | 68.5 |
| 90 | 86.0 | - | 86.7 |
| 110 | 103.6 | 103.8 | 104.4 |
| 130 | - | - | 121.8 |

The 4,786 archived NFS@Home logs agree: median 104-106 per column in every
size bucket from 5M to 35M since 2022, with TD 110-130 the common settings.
The tune therefore scales one profile to each size at about 104 per column,
and `SIZE@W` covers other densities.

### The tune script

`tools/tune_la.sh` builds VBITS 128 and 256 for the detected GPU and
generates each size in turn (default 7, 14, 25, 35 and 45M columns,
deleted after use). It times single copy, single + col24, two copies and
two + col24 with `bench_la.sh` (warmup 30 s, window 90 s, SIGTERM with the
integrity check). Sizes beyond the machine's RAM (about 1.5x the matrix
file) or disk are skipped. If msieve logs that it cannot stream the matrix,
the run is killed and recorded as NOFIT, because the fallback of loading
everything would page GPU memory under WDDM. The script writes
`results.tsv` and a per-size ranked `report.txt`. A full default run is
about 2.5 hours per machine. Generation took 35-45 s for C168 sizes and
220-250 s at 26.7M (14 GB RAM).

## 2026-10-11: First tune reports, and a C193 set up from real-matrix timings

### Default tune on three cards

Best setting per size from `tools/tune_la.sh` with defaults (dims/s):

| Size | RTX 5070 (12 GB, WSL) | RTX 5090 (32 GB) | RTX 3060 (12 GB) |
|---|---|---|---|
| 7M | VBITS 256 single: 3234 | VBITS 256 two + col24: 8885 | VBITS 256 two + col24: 654 |
| 14M | VBITS 256 single: 1404 | VBITS 256 two + col24: 4046 | VBITS 256 two + col24: 306 |
| 25M | VBITS 128 single + col24: 647 | VBITS 256 single + col24: 1971 | VBITS 256 single + col24: 104 (VBITS 128 two + col24: 101) |
| 35M | VBITS 256 single + col24: 324 | VBITS 256 single + col24: 1164 (VBITS 128 two + col24: 1136) | failed |
| 45M | VBITS 128 single + col24: 185 | VBITS 128 two + col24: 791 (VBITS 256 single + col24: 780) | failed |

- While the matrix fits on the card, two copies win on the 5090 and the
  3060 (single copy 7-8% and 25-30% behind), but single copy wins on the
  5070. No single VBITS or setting is best on all three cards.
- Once matrices stream, single + col24 usually leads, and which VBITS wins
  changes with size and card.
- On the 5070, VBITS 256 cannot fit its vectors at 45M (9.6 GB), and two
  copies at VBITS 256 is NOFIT from 35M (the row floor makes 420M-nonzero
  blocks too big to stage).
- The 3060 failed nearly every run at 35M and 45M, and VBITS 256 two copies
  at 25M, because it ran out of host memory. Its container has a 30 GiB
  cgroup limit. `memory.peak` reached that limit and `memory.events`
  counted 12 kills for 13 failed runs. Other jobs were also running there,
  and the tune saw 12.3 GB available at the start. No failed run logged an
  error. Each stopped either while packing streamed blocks into pinned
  host memory (last line "leaving 256 MB of GPU memory free while
  streaming") or just after "commencing Lanczos iteration". The tune's RAM
  estimate (1.5x the matrix file) leaves out the pinned host copy of
  streamed blocks. On a 12 GB card that adds 7-20 GB at 35M; the 5070's
  35M logs show 13 GB (VBITS 128 single) to 27 GB (VBITS 128 two) on top of
  msieve's own 9.4 GB. Since Oct 12 the tune also checks the cgroup limit,
  records kernel kills as NORAM (from `memory.events`, else
  `/proc/vmstat`), and records each run's peak resident memory and the host
  memory it took. It also stops runs that would not fit before they pin
  anything (calibration below).
- C199 check on the 5090 (Oct 12, `-s 41.646178@121.79 -v 256 -c single
  -A "c199=single_copy=1 block_nnz=139000000"`): the synthetic matrix with
  the real run's settings gave 818.9 dims/s against the real C199's 815
  (+0.5%). msieve's default single-copy block size gave 826.3, so the
  hand-picked `block_nnz` gained nothing. Together with the C191 recheck,
  this means the size scaling of the heavy rows holds from 8M to 41.6M.
  The streaming plan matched too: 4 of 32 blocks streamed, as in the real
  run (2,756 MB through two 689 MB buffers). The default block size
  streamed 6 of 40 (3,511 MB) and was still 1% faster.

### C193: choosing density and VBITS on the real matrices

193-digit GNFS (degree 5), 521.6M unique relations, filtered with
`target_density=90,110,130 all_matbuild=1`. Filtering plus the three
matrix builds took 5h06m, of which the builds took 74 min. Each matrix was then timed with
`bench_la.sh` (warmup 30 s, window 90 s, `-t 8`) using the tune's
current-source VBITS 128 and 256 builds, with ECM running on the CPU and
the default 1.5 GB WDDM margin. All six runs passed the integrity check.

| TD | Dims | Sparse nonzeros/col | VBITS 128 single + col24 | VBITS 256 single + col24 |
|---|---:|---:|---|---|
| 90 | 31.88M | 86.1 | 455.9 dims/s, 19h25m | 398.2 dims/s, 22h14m |
| 110 | 29.82M | 103.8 | 446.7 dims/s, 18h32m | 444.3 dims/s, 18h38m |
| 130 | 28.31M | 121.1 | 422.5 dims/s, 18h36m | 413.1 dims/s, 19h02m |

- TD 110 and 130 tie within noise; TD 90 has the fastest iterations, but
  its 2M extra dimensions cost about an hour.
- Upload volume does not decide VBITS. At TD 90, VBITS 256 uploaded 12%
  less per dimension (78.0 vs 88.7 MiB) and was still 13% slower, so both
  are partly kernel-bound at this size. An automatic choice should use
  measured rates, not upload volume.
- The synthetic tune brackets the result: VBITS 128 single + col24 gave
  647 dims/s at 25M and 311 at 35M, against 446.7 here at 29.8M. A
  synthetic matrix of exactly this size was not run.

Production run: TD 110, VBITS 128, single copy + col24, `-t 8`, root
`./msieve` rebuilt at 15:01 on Oct 11. 17 of 29 blocks streamed, with
11,855.8 MiB uploaded per iteration, identical to the benchmark. Lanczos
ran from 15:04:50 to 09:45:02: 18h40m12s, 443.6 dims/s, 0.7% below the
benchmark's 446.7 (it predicted 18h32m). BLanczosTime was 67,373 s over
234,356 iterations, with 30 dependencies. The square root took 21m45s
and found p70 x p123. Elapsed for `-nc2`/`-nc3` was 19h04m39s.

## 2026-10-12: Host memory of streamed runs (tune calibration)

Synthetic 25M and 35M matrices (103.6 nonzeros per column), RTX 5070 with
80 GB of RAM and the 1.5 GB WDDM margin, short windows
(`tools/tune_la.sh -s "25 35" -c "single single-col24 two" -A
"single-6g=single_copy=1 max_gpu_mem=6000" -T 20 -W 10`). Peak is msieve's
VmHWM. Pinned memory counts there: it matched the drop in available RAM
within 0.2-2 GB. "Extra" is the peak minus the matrix and the streamed
blocks. All memory figures are in MB.

| Run | Matrix | Needed on GPU | Streamed | Peak | Extra | Extra per column / (VBITS/8) |
|---|---:|---:|---:|---:|---:|---:|
| 25M VBITS 128 single | 11,023 | 11,434 | 6,140 | 21,066 | 3,902 | 10.2 |
| 25M VBITS 128 single + col24 | 11,023 | 9,159 | 3,815 | 18,732 | 3,893 | 10.2 |
| 25M VBITS 128 two | 11,023 | 21,407 | 16,813 | 31,741 | 3,904 | 10.2 |
| 25M VBITS 128 single, 6 GB cap | 11,023 | 11,434 | 9,478 | 24,412 | 3,910 | 10.3 |
| 25M VBITS 256 single | 11,023 | 12,961 | 10,296 | 28,714 | 7,394 | 9.7 |
| 25M VBITS 256 single + col24 | 11,023 | 10,816 | 8,104 | 26,506 | 7,378 | 9.7 |
| 25M VBITS 256 two | 11,023 | 18,717 | 18,021 | 36,384 | 7,339 | 9.6 |
| 35M VBITS 128 single | 15,432 | 17,379 | 13,182 | 34,068 | 5,453 | 10.2 |
| 35M VBITS 128 single + col24 | 15,432 | 14,186 | 9,913 | 30,788 | 5,442 | 10.2 |
| 35M VBITS 128 two | 15,432 | 30,015 | 27,145 | 48,013 | 5,435 | 10.2 |
| 35M VBITS 128 single, 6 GB cap | 15,432 | 17,379 | 16,795 | 37,694 | 5,466 | 10.2 |
| 35M VBITS 256 single | 15,432 | 18,305 | 18,213 | 43,926 | 10,280 | 9.6 |
| 35M VBITS 256 single + col24 | 15,432 | 15,287 | 15,195 | 40,852 | 10,224 | 9.6 |

"Matrix" is the first "matrix is ... (N MB)" line, about 1.065x the file.
At the planning step ("GPU memory:"), the three NOFIT runs held the matrix
plus only 1.3-2.5 GB. The pinned streamed blocks and most of the host
vectors come after that line.

So the peak is the matrix, plus the streamed blocks, plus 9.6-10.2 bytes
per column per VBITS/8, independent of size and setting. The streamed
amount follows from the planning line: what is needed, minus the budget
(the space for blocks less the streaming margin, 1.5 GB under WDDM,
256 MB natively, none with `max_gpu_mem`), plus the staging buffers. The
tune estimates staging as three buffers of 1.1x (block_nnz x 4 + columns x
4) bytes, and the host vectors as 10.5 bytes per column per VBITS/8. On
all 13 runs above, that ran 0.5-1.4 GB above the measured peak.

`tools/tune_la.sh` now reads that line as soon as msieve logs it. A run
whose estimated peak plus 2 GB exceeds the RAM available when it started
is stopped before it pins anything, and recorded as NORAM with the
breakdown. A run whose layout cannot fit on the card at all is still
NOFIT. Tested with fake programs replaying real planning logs: in a 12 GB
memory-limited scope, the 25M VBITS 128 single run (estimate 21.6 GB) was
stopped, and with no limit it ran.

On the 3060's 30 GiB container, the estimate would have stopped the runs
that were killed. For example, 25M VBITS 256 two copies comes to about
36 GB, and 35M VBITS 128 single to about 34 GB. With the 2 GB margin it
would also have stopped 25M VBITS 128 two copies (about 29.7 GB
estimated), which got through there, presumably with little to spare.

## 2026-10-13: C207 on the 5070 (density, row16 + host_check), and an RTX 3080 tune

### C207: three densities on the 12 GB 5070

207-digit GNFS (degree 5), 925.1M unique relations, filtered with
`target_density=90,110,130 all_matbuild=1` (filtering 4h09m, builds about an
hour). The job itself goes to the 5090. These timings are for comparison.
`bench_la.sh -W 30 -T 90 -- -t 8` with the root `./msieve` (VBITS 128, built
Oct 11 15:01, sha256 f25a6b6c...), single copy + col24. VBITS 256 cannot fit
its vectors on this card at these sizes (10-11.4 GB). All runs passed the
integrity check.

| TD | Dims | Sparse nonzeros/col (after packing) | Blocks streamed | MiB uploaded/iter | ms/iter | dims/s | Full Lanczos |
|---|---:|---:|---|---:|---:|---:|---:|
| 90 | 53.31M | 79.9 | 37 of 40 | 35,578 | 805 | 158.0 | 93h43m |
| 110 | 49.49M | 96.3 | 43 of 48 | 38,935 | 858 | 148.4 | 92h39m |
| 130 | 46.77M | 112.3 | 40 of 45 | 39,031 | 851 | 149.4 | 86h55m |

- All three were copy-bound at 43-45 GiB/s, the pinned-copy ceiling of this
  link, so iteration time followed the bytes uploaded and TD 130 won on
  having the fewest dimensions. As on the C193, the lowest density lost.
- At VBITS 128, TD 90 hit the single-copy row floor (block_nnz 106.6M, the
  L2 fit would be about 84M) and logged the "only saving GPU memory" note.
- The first TD 110 run measured 48.3 dims/s while another session ran a GPU
  lattice-sieve benchmark on the same card. The 148.4 above is a rerun with
  nothing else on the GPU. TD 90's window overlapped by at most 3 s, and
  TD 130's by none. Row 73 of `bench/results.tsv` is the invalid run.

### row16 and host_check pay off once col24 is still copy-bound

TD 130 again, same binary and window:

| Settings (single copy) | Blocks streamed | Needed on GPU | MiB uploaded/iter | Copy rate | dims/s | Full Lanczos |
|---|---|---:|---:|---:|---:|---:|
| col24 | 40 of 45 | 23,146 MB | 39,031 | 44.8 GiB/s | 149.4 | 86h55m |
| col24 + row16 | 39 of 45 | 19,258 MB | 31,607 | 40.9 GiB/s | 168.7 (+12.9%) | 77h01m |
| col24 + row16 + host_check | 37 of 45 | 19,258 MB | 29,891 | 39.2 GiB/s | 170.8 (+14.3%) | 76h04m |

- row16 shrank the 45 row-pointer arrays (one full array per block, about
  187 MB each at 46.8M rows) and saved 3.9 GB of the matrix, cutting uploads
  by 19%. Its slower decoding cost less than the copies it saved. On the
  C190 (25.2M, 2026-10-04 and 10-05 sections), col24 alone had already
  hidden the copies, and row16 only made the kernels slower.
- host_check moved the check vector to host memory, so 714 MB more of the
  matrix stayed on the card. Uploads fell another 5.4%, but the time only
  fell 1.2%, about the run-to-run spread. The copy rate dropped from 44.8 to
  39.2 GiB/s as uploads fell, so the kernels probably show through now, and
  removing more bytes gains less.
- The tune's default settings leave out row16, so they cannot find this.
  It matters for matrices that still stream heavily after col24.

### C207 TD 130 on the 5090

The job itself runs on the rented 5090 (32 GB, native Linux, 256 MB
streaming margin). `bench_la.sh -T 90`, default warmup and threads, binaries
built there from the current source at VBITS 128 and 256. All runs passed
the integrity check.

| Settings | dims/s | Full Lanczos | ms/iter | Streamed | MiB uploaded/iter |
|---|---:|---:|---:|---:|---:|
| VBITS 256 single + col24 | **686.4** | **18h55m** | 371.8 | 3,980 MB | 5,970 (12 uploads) |
| VBITS 128 single + col24 | 639.0 | 20h19m | 199.1 | none | - |
| VBITS 128 two + col24 | 487.2 | 26h40m | 261.2 | 12,987 MB | 12,987 (11 uploads) |

- VBITS 256 won by 7.4%. The C199 margin was 13-15%, but that VBITS 128 run
  had no col24 and different block sizes.
- VBITS 128 single + col24 fit on the card: 19.1 GB of blocks, 25.4 GB peak
  VRAM. At 199 ms/iteration it scales from the C199's resident VBITS 128
  (179 ms at 5.02B nonzeros) by the 12% more nonzeros almost exactly.
- VBITS 256 streamed about 4 GB at 16 GiB/s, so it is not copy-bound, and
  row16 would only add decoding cost. Per iteration it was 1.19x the C199's
  (313 ms at 815 dims/s), a little more than the nonzero ratio.
- Two copies at VBITS 128 streamed 13 GB at 48.6 GiB/s (the PCIe 5 limit)
  and lost by 24%.
- The pre-run estimate from the C199 (700-730 dims/s for VBITS 256 single +
  col24) came out 2-6% high.

### RTX 3080 default tune (someone else's, WSL)

RTX 3080 10 GB (sm_86, 5 MB L2), driver 595.95, WDDM, 763 MiB in use at the
start, i7-12700K, 32.6 GB MemTotal in WSL (31.8 GB available). nvidia-smi
reports a maximum PCIe link of **gen 3** x16. Both the 3080 and the 12700K
support gen 4, and the local 5070 reports its gen 5 correctly under WSL,
so the board slot, BIOS or a riser probably limits the link. Defaults: sizes
7-45M, VBITS 128 and 256, 90 s window, 30 s warmup, `-t 8`. dims/s:

| Size | VBITS 256 two + col24 | VBITS 256 two | VBITS 256 single + col24 | VBITS 256 single | VBITS 128 single | VBITS 128 single + col24 | VBITS 128 two + col24 | VBITS 128 two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7M | **1093.7** | 1048.0 | 920.9 | 921.4 | 1010.0 | 941.5 | 815.3 | 801.2 |
| 14M | **488.7** | 356.3 | 368.5 | 353.3 | 328.7 | 344.7 | 332.6 | 229.2 |
| 25M | NOFIT | NOFIT | NORAM | NORAM | 82.0 | **106.8** | NORAM | NORAM |
| 35M, 45M | | | | | | | | |

At 35M and 45M every run was NORAM or NOFIT.

- Two copies win while the matrix fits, as on the 5090 and 3060. Only the
  5070 prefers single copy there. At 7M, VBITS 256 two + col24 beat single
  by 19%.
- At 14M, col24 made two copies 37% faster (356.3 to 488.7). Without it,
  two copies most likely streamed: the run took 19.9 GB of host RAM against
  16.8 GB with col24. This answers the open question from the September 30
  C178 run (14.15M, single copy at the row floor, 446 dims/s): two copies +
  col24 would probably have been about a third faster there (its density
  was lower, 73 per column against the tune's 104).
- At 25M only VBITS 128 single copy ran, and col24 was 30% faster. That run
  took 22.5 GB of host RAM. Less the matrix (about 11.0 GB) and the host
  vectors (about 4.1 GB from the October 12 calibration), roughly 7.4 GB
  streamed. At the C207's 1.9-2 uploads per streamed byte, that is about
  14 GB in each 1.19 s iteration, about 11-12 GB/s: what a gen 3 x16 link
  carries. So the 25M runs were probably copy-bound by the link (inferred,
  no transfer counters in the report). If the September 30 C178 ran on this
  machine, its ~8 GB/s copies were closer to the link's limit than the
  "well under PCIe 4.0" noted there.
- 35M and 45M were all NORAM or NOFIT. WSL takes half of Windows' RAM by
  default, so this machine probably has 64 GB, and `memory=` in
  `.wslconfig` could give WSL enough for 35M. On the 5070, 35M VBITS 128
  single + col24 peaked at 30.8 GB. A 10 GB card streams 2-3 GB more, and
  the tune keeps 2 GB to spare.
- In the report, a NORAM row's host RAM is only what loaded before the
  stop. The estimate that triggered the stop is only in `runs.log`.
