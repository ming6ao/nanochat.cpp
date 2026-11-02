# R2: one efficient Muon step

Status: proposed. The optimizer is the largest single phase of a training step
on an H100. Its device arithmetic runs about 300 times slower than the model
GEMMs beside it in the same step.

## 1. Goal

Make one Muon step cost about what its arithmetic costs. Today it is 331 ms at
the 8-layer shape and 2989 ms at the 20-layer shape on an H100, at roughly
1 TFLOP/s. The target is 4 ms and at least 100 TFLOP/s at the 8-layer shape.

## 2. What the tree holds

| Part | Path | What it does |
|---|---|---|
| Driver | `src/optim.cc` | `NanochatOptimizer::Step` groups matrices by shape, gathers each group's gradients device to device, calls `MuonUpdate` once per group, calls `AdamWUpdate` once per group, and reads the gradient norm back to the host once per step. |
| Kernel | `backends/cuda/kernels/muon.cu` | Batched over the group: momentum, Frobenius norm, row equilibration, then the Polar Express loop. |
| GEMM | `backends/cuda/gemm.cu` | `cublasGemmEx` and `cublasGemmStridedBatchedEx` with `CUBLAS_GEMM_DEFAULT`. |
| Reference | `nanochat/optim.py:111-150` | One `@torch.compile(dynamic=False, fullgraph=True)` function. Its own comment: "All in one compiled graph to eliminate Python overhead between ops." The orthogonalization runs in bf16, `X = g.bfloat16()`, so its GEMMs use tensor cores. |

The tree already mirrors the reference's algorithm. It has the Polar Express
coefficients, the row equilibration, and the tall and wide branches. What it
does not have is the rate.

## 3. The evidence

From the phase timer on the H100, with the classifier and the norms fused:

| Shape | Step | Optimizer | Share |
|---|---|---|---|
| 8 layers, hidden 512, sequence 256, batch 16 | 535 ms | 331 ms | 62 percent |
| 20 layers, hidden 1280, sequence 1024, batch 16 | 9037 ms | 2989 ms | 33 percent |

Four facts follow.

1. At the 8-layer shape the orthogonalization is about 343 GFLOPs per step. At
   331 ms that is 1 TFLOP/s, which is 0.1 percent of the card's bf16 peak and
   1.5 percent of its fp32 peak.
2. The model's own GEMMs do 1 TFLOP in about 3 ms on the same card in the same
   step, so the optimizer path is roughly 300 times less efficient than the path
   next to it.
3. The optimizer is 4.4 times slower on the H100 than on the local 1080 Ti. A
   card 87 times faster at bf16 cannot be slower at the same arithmetic, so the
   cost is not the arithmetic.
4. The step does 1.34 TFLOPs of device work at the 8-layer shape and the report
   divides by 1.03, because `EstimateFlopsPerToken` counts the model's matmuls
   and not the optimizer's.

## 4. Decisions

| # | Decision |
|---|---|
| R2-D1 | Keep the seam signature. `MuonUpdate` stays the one entry point, so the CPU backend and every parity fixture keep working. |
| R2-D2 | Run the orthogonalization in the compute dtype with fp32 accumulation, as the reference does. The master weights stay fp32. |
| R2-D3 | One launch per Polar Express iteration across every shape group, not one call per group. The groups differ in shape, so this needs a segmented or grouped GEMM rather than one batched call. |
| R2-D4 | Cache the `cublasLt` plan per (rows, cols, batch, mode). The heuristic query is not the cost; the algorithm choice is. |
| R2-D5 | Remove the per-step blocking copy. Fold the norm into the clip on the device and read one scalar per step at most. |
| R2-D6 | Report the optimizer's arithmetic beside the model's, so a utilization number states what it divides by. |

## 5. Work rows

| # | Work | Gate |
|---|---|---|
| R2.1 | A benchmark of the Polar Express GEMM shapes alone: `[20, 1280, 1280]` batched, five iterations, three GEMMs each | the benchmark reports a rate, and it is the number every later row moves |
| R2.2 | Compute dtype operands and a cached `cublasLt` plan | the benchmark rate rises, and `//src:optim_test` and `//src:optimizer_state_test` pass |
| R2.3 | One launch per iteration across the groups | the benchmark rate holds, and the step's `opt_ms` falls |
| R2.4 | No blocking copy per step | `sync_ms` and `opt_ms` both fall, and the loss trace is unchanged |
| R2.5 | Count the optimizer FLOPs in the report | the report names both numbers, and no gate reads the sum |
| R2.6 | Re-measure the split on the H100 at the 20-layer shape | `opt_ms` at or below 10 ms |

## 6. Why this is the second limit and not the first

The rotary leg (R1) costs 6 s of the 9.04 s step and the optimizer costs 3 s.
R1 is the larger single removal, so it goes first. After both, the step models at
roughly 100 to 400 ms, which is 10 to 40 percent MFU at the 20-layer shape. That
is where 30 percent becomes reachable, and R3 of the plan, the device smear,
moves it further.

## 7. Risks

- bf16 orthogonalization changes the trajectory. The reference does it in bf16,
  so the fixture that defines parity is a bf16 reference trajectory. A fixture
  recorded from a fp32 path would fail, and that would be a fixture problem, not
  a kernel problem.
- A grouped or segmented GEMM is a new kernel, not a tuning flag. If the tree
  cannot express it, R2-D3 becomes one call per group and the rate depends on
  each group being large enough.
- Plan caching must key on the batch count as well as the matrix shape.
- The optimizer state buffers are already sized for the widest group. A grouped
  GEMM must not outgrow them.

## 8. Non-goals

- Changing the optimizer algorithm, the momentum schedule, or the learning rate
  schedule.
- Multi-device sharding of the optimizer state.
- An fp8 orthogonalization path.
