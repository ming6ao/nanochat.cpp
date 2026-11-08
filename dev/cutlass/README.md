# dev/cutlass - the CUTLASS bring-up probe

Phase 0 of the CUTLASS reuse in [docs/kernels.md](../../docs/kernels.md) Part 3.
The probe answers the three questions that stand between this tree and a Hopper
mainloop borrowed from NVIDIA CUTLASS. **It measures no rate.** Every row is
compile-time data, and the binary launches no kernel, so it runs on the host
with no card.

```bash
tools/nanochat build //dev/cutlass:cutlass_probe
tools/nanochat run t0-cpu -- ./bazel-bin/dev/cutlass/cutlass_probe
cuobjdump -sass bazel-bin/dev/cutlass/_objs/cutlass_probe_lib/cutlass_probe.pic.o \
  | grep -oE 'UTMALDG[^ ]*|UTMASTG[^ ]*|HGMMA[^ ]*'
```

## 1. The three questions, and the answers

| Question | Answer | How it is decided |
|---|---|---|
| Does the vendored header tree compile here, at CUDA 12.0 and C++20? | Yes | `tools/nanochat build //dev/cutlass:cutlass_probe` |
| Does a `cuda_library` that states `archs = ["sm_90a"]` build beside the command-line `sm_75`? | Yes | The two targets below disagree on architecture, and one build of both succeeds |
| Which tile candidates fit the Hopper window, and what stage count does the builder choose? | Section 3 | The report's first table |

The second answer is the one that changes the build file. The tree emits
`sm_75` on the command line and the Hopper legs emit `sm_90`. A per-target
`archs` means the adapter states `sm_90a` on itself, and **no `.bazelrc` line and
no command-line flag changes**. A build of `//...` on the Turing host keeps
working untouched, which is what every agent's inner loop needs.

The split is why there are two targets. `:cutlass_probe_lib` is the only target
in the tree that names an architecture on itself. `:cutlass_probe` is a binary at
the command line's architecture, and it links the library. A build of both is the
proof.

## 2. What the object contains

`cuobjdump` over the library reports `arch = sm_90a` and 12 kernels. The report
names 13 rows, and two of them are the same type: the 128x256x64 tile at an fp32
store is the first fp32 case. Their instructions are the seven mechanisms of
[docs/kernels.md](../../docs/kernels.md) Part 3 section 4, in one build:

| Instruction | Count | Mechanism |
|---|---:|---|
| `UTMALDG.3D` | 113 | The Tensor Memory Accelerator load, one thread for each tile |
| `UTMALDG.3D.MULTICAST` | 30 | The cluster multicast |
| `UTMASTG.3D` | 96 | The bulk epilogue store |
| `UTMACMDFLUSH` | 96 | The TMA command path |
| `UTMACCTL.PF` | 48 | The TMA prefetch |
| `HGMMA.64` | 108 | The warpgroup matrix multiply-accumulate |
| `LDSM.16.M88.4` | 64 | The register-source operand path |
| `USETMAXREG.TRY_ALLOC.CTAPOOL`, `.DEALLOC.CTAPOOL` | 24 | The register reallocation between warpgroups |

The mangled names carry the rest: `MainloopSm90TmaGmmaWarpSpecialized`,
`GMMA::MMA_64x256x16_F32BF16BF16_SS`, `SM90_TMA_LOAD`,
`SM90_TMA_LOAD_MULTICAST`, `SM90_TMA_STORE`, `Sm90TmaWarpSpecialized`, and
`PersistentScheduler`.

## 3. The tile table

One block of 384 threads: one producer warp and two consumer warpgroups. The
window is 227 KiB of the 228 KiB the multiprocessor holds.

| Tile | Cluster | Stages | Window | Threads |
|---|---|---:|---:|---:|
| 128x256x64 | 2x1 | 3 | 215,040 B | 384 |
| 128x256x32 | 2x1 | 6 | 215,040 B | 384 |
| 128x128x64 | 2x1 | 5 | 231,424 B | 384 |
| 128x128x128 | 2x1 | 2 | 198,656 B | 384 |
| 256x128x64 | 2x1 | 3 | 215,040 B | 384 |

Every row fits. **The schedule takes one block for each multiprocessor**, against
the two that the hand tile of `backends/cuda/gemm/` keeps. That is the standard
Hopper trade and it is the reason a deeper pipeline pays here.

**A tile narrower than 128 in M does not exist on this schedule.** The first build
of this probe held a 64 by 256 candidate. The compiler rejected it:

> Cooperative kernel requires Tile Size to be greater than or equal to 128 along
> the M-dimension. (`include/cutlass/gemm/kernel/`
> `sm90_gemm_tma_warpspecialized_cooperative.hpp`, line 374.)

Two consumer warpgroups split the M extent, so the floor is one warpgroup tile.

## 4. The eight seam cases

`GemmParams` needs the cross product of two transposes and two store types. Every
one of the eight compiles and emits a kernel.

| Case | Store | Window |
|---|---|---:|
| plain A, plain B | `ComputeType` | 230,400 B |
| transposed A, plain B | `ComputeType` | 230,400 B |
| plain A, transposed B | `ComputeType` | 230,400 B |
| transposed A, transposed B | `ComputeType` | 230,400 B |
| plain A, plain B | fp32 | 215,040 B |
| transposed A, plain B | fp32 | 215,040 B |
| plain A, transposed B | fp32 | 215,040 B |
| transposed A, transposed B | fp32 | 215,040 B |

**The probe asserts the layout tags.** CUTLASS names a stride over the
modes `[M, K, L]` for A and `[N, K, L]` for B, so its `RowMajor` tag means
mode-1 contiguous for A but mode-0 contiguous for B. The seam states each operand
form in terms of which axis is contiguous, so the probe holds one `static_assert`
for each operand:

| Seam form | Tag | Stride | Assert |
|---|---|---|---|
| A plain, k contiguous | `RowMajor` | `(lda, 1)` | `is_k_major_A` |
| A transposed, m contiguous | `ColumnMajor` | `(1, lda)` | `not is_k_major_A` |
| B plain, n contiguous | `RowMajor` | `(1, ldb)` | `not is_k_major_B` |
| B transposed, k contiguous | `ColumnMajor` | `(ldb, 1)` | `is_k_major_B` |

A tag that disagreed with the caller's transposes would compile and multiply the
wrong operands. The first draft of this probe did exactly that for B, and the
assert is what caught it.

## 5. Two defects the probe found

Both would have cost a card run in the adapter.

1. **The `64x256x64` tile does not exist** on the cooperative schedule. Section 3.
2. **A name-only instantiation compiles no kernel.** A probe that writes
   `sizeof(Kernel::SharedStorage)` instantiates the class and skips the mainloop,
   so the object holds header data and no SASS. The first build of this probe
   passed with an object that held no kernel at all. The fix is
   `&cutlass::device_kernel<Kernel>`, which odr-uses the body. `cuobjdump
   -sass` is the gate, and the report's `kernel` column states it per row.

## 6. What is not here

- A rate. No kernel launches. `backends/cuda/gemm/gemm_benchmark.cc` is the rate
  gate, and the adapter that gives it a CUTLASS path is P1.
- An emulated leg. The emulation engine maps no TMA and no WGMMA
  ([docs/simulator.md](../../docs/simulator.md) section 9.2), so this probe and
  every kernel it instantiates are card-only.
- The fp8, stream-K, and grouped tiers. The probe's own compile confirmed the fp8
  mainloop and the persistent and stream-K schedulers instantiate; the grouped
  path needs its own problem shape and scheduler, and it is P4.

The vendor pin, its licence, and the reason the rule fetches the tree instead of
tracking it are in [third_party/cutlass.bzl](../../third_party/cutlass.bzl) and
[docs/build.md](../../docs/build.md).
