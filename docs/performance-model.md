# The performance model

Status: implemented for phases M0 to M2. The counter table, the precision
assignment, the device limit table, and the estimator are in the tree, and
`tools/nanochat roofline` prints the roofline report. The attainment record in
section 10 still awaits the first in-tree measurement (phase M3).

The tree has a throughput calculation for the MFU line, and a
standalone estimator in `.agents/skills/model-fit/scripts/model_fit.py`. This
document turns the second one into a project artifact, and binds it to the
device limit table and to the benchmark report.

The model answers one question without a GPU: what is the fastest possible step
at this shape, and what limits it?

- [performance.md](performance.md) — the measurement protocol.
- [simulator.md](simulator.md) — the S0 simulator and its limits.
- [kernels.md](kernels.md) — the kernel inventory the model counts.
- [fp8-training.md](fp8-training.md) — the fp8 tier and the layer set.

## 1. Goal and scope

**Goal.** Report the bound on the step time from the device limits, the
operation that binds, and the distance from the measured step to the bound.

**In scope.**

- One structural counter table that lists the arithmetic, the bytes, and the
  launch count of each operation.
- One precision assignment, so each operation uses its own peak rate.
- One device limit table: the peak rate per precision, and the peak bandwidth.
- One roofline estimator that uses the peak limits directly.
- A work-ratio check that reports redundant issued work.
- An attainment record that compares the bound with the measured step.

**Out of scope.**

- A fitted correction factor. The model uses the device peaks, and nothing else.
- A cycle-accurate model.
- A full memory-system simulation.
- A model of the data loader and of the checkpoint write.

## 2. Why the simulator is not enough

The S0 simulator proves values. It does not prove time.

| Engine | Proves | Gives time |
|---|---|---|
| Reference engine | The graph arithmetic | No |
| API interposer | The call shape | No |
| Emulation engine | The device-code arithmetic | No |
| Performance model | The step-time bound | Yes |

The reference engine runs on the CPU at the CPU rate. The emulation engine runs
each CUDA thread as a host fiber, at the host rate. Neither rate is the device
rate.

`docs/simulator.md` section 14 marks a functional PTX interpreter as a non-goal.
A cycle-accurate model is a large project with a small return. The device peaks
give a useful bound at a small cost.

## 3. The inputs

The model has three inputs. Each one has one owner and one stored form.

| Input | Source | Stored in |
|---|---|---|
| The operation counters | The model configuration | A counter struct in `src/train.cc` |
| The precision assignment | The fp8 rule and the layer set | `docs/fp8-training.md` |
| The device limits | The device profile table | `include/nanochat/device_profile.h` |
| The launch cost | A device property | The device profile table |

The model holds no fitted value. A number that comes from a measurement appears
in the attainment record only, and never in the estimator.

## 4. The operation counter table

The tree counts the total arithmetic today. `EstimateFlopsPerToken` in
`src/train.cc` returns one number. The model needs the parts, each with its
precision.

Add one struct beside `NumMatmulParams` and `CountParams`:

```cpp
struct FlopCounts {
  double gemm_forward;      // 2 * params * tokens
  double gemm_backward;     // 4 * params * tokens
  double attention_forward;
  double attention_backward;
  double norm_and_pointwise;
  double classifier;
  double embedding;
};
```

Fill it from the same terms that `EstimateFlopsPerToken` uses. Keep
`EstimateFlopsPerToken` as the sum of the fields, so one source of truth
remains. `include/nanochat/mfu.h` owns the declaration.

Each row of the counter table also carries three non-arithmetic fields:

- the precision, for the rate selection;
- the operand bytes, for the memory-bound case;
- the launch count, for the launch overhead.

| Operation | Arithmetic | Precision | Operand bytes | Launch count |
|---|---|---|---|---|
| `c_q`, `c_k`, `c_v`, `c_proj` | `2 * p * t` forward, `4 * p * t` backward | The assigned GEMM precision | The activations and the weights | One per call |
| `mlp.c_fc`, `mlp.c_proj` | Same form | The assigned GEMM precision | Same | One per call |
| the language-model head | Same form | The assigned GEMM precision | Same | One per call |
| the quantization cast | A row pass | Not applicable | 5 bytes per element | One per operand |
| the layout copies | A copy | Not applicable | The operand size | One per copy |
| the attention scores | `4 * h * d * w` forward over the window `w` | The attention precision | The score matrix | One per tile group |
| the attention output | `4 * h * d * w` forward | The attention precision | The probability matrix | One per tile group |
| the attention backward | `8 * h * d * w` | The attention precision | The score and probability matrices | One per tile group |
| the norms and the pointwise steps | A row pass | The compute precision | The row buffers | One per call |
| the classifier | A row pass over the vocabulary | fp32 | The logits | One per call |
| the embedding | A gather and a scatter | The compute precision | The table rows | One per call |
| the optimizer | A stateful pass | fp32 | Three buffers | One per tensor |

Here `p` is the matrix parameter count, `t` the token count, `h` the head
count, `d` the head width, and `w` the effective window.

## 5. The precision assignment

A single build precision is not enough. The reference nanochat converts only
some of the matrix multiplies, and it keeps the rest in bf16 or fp32. The model
must select a rate per operation.

### 5.1 The rule from the reference

The reference applies `--fp8` through a module filter. A Linear layer converts
when all three conditions hold:

1. it is a matrix multiply;
2. both dimensions are divisible by 16;
3. the smaller dimension is at least 128.

The reference then uses two fp8 formats. The forward operands use E4M3. The
gradient operand uses E5M2, which has a wider range. The accumulator and the
output stay in the compute type.

Each converted matrix multiply becomes three fp8 GEMMs:

| GEMM | Operand A | Operand B | Accumulator | Output |
|---|---|---|---|---|
| Forward | The activation, E4M3 | The weight, E4M3 | fp32 | bf16 |
| Input gradient | The output gradient, E5M2 | The weight, E4M3 | fp32 | bf16 |
| Weight gradient | The output gradient, E5M2 | The activation, E4M3 | fp32 | bf16 |

The reference computes the forward with a fast accumulation mode. The two
backward GEMMs use the full fp32 accumulation.

### 5.2 The assignment at depth 24

| Operation | Precision | Reason |
|---|---|---|
| `c_q`, `c_k`, `c_v`, `c_proj` | fp8 | A matrix, and both dimensions are 1536 |
| `mlp.c_fc` | fp8 | A matrix, 1536 by 6144 |
| `mlp.c_proj` | fp8 | A matrix, 6144 by 1536 |
| `lm_head` | fp8 | A matrix, 1536 by 32768 |
| `ve_gate` | bf16 | The output width is 12, which is below 128 |
| `smear_gate` | bf16 | The output width is 24, which is below 128 |
| the embeddings | bf16 | A gather, not a matrix multiply |
| the attention | bf16 | The flash attention engine runs bf16 |
| the norms and RoPE | bf16 | A row kernel |
| the softmax and the saved statistics | fp32 | A row reduction with an exponential |
| the classifier softcap | bf16 | A row kernel |
| the loss | fp32 | The loss must stay fp32 |
| the optimizer | fp32 | The state must stay fp32 |
| the gradient reduction | The gradient type | The native plan reduces in fp32 |

At depth 24 the eligible matrices hold 729.8 million parameters. The two gates
hold 1,752 parameters. So the assignment is simple: the matrix multiplies run
in fp8, and everything else runs in bf16 or fp32.

### 5.3 The two costs that come with fp8

fp8 is not free. Two terms appear that do not exist in a bf16 run.

**The quantization cast.** Each operand must be measured, scaled, and cast
before the GEMM. The amax pass reads the operand, and the cast pass reads it
again and writes one byte per element. A fused kernel needs about five bytes of
traffic per element. An unfused sequence of framework calls needs more.

**The layout copies.** The fp8 GEMM engine requires the first operand
row-major and the second operand column-major. A tensor that is already in the
wrong layout needs a physical copy.

The reference performs three copies per backward call. This project can avoid
most of them: store the weight in the layout the GEMM wants, once, and use that
layout for every step. This is a design requirement, not a tuning detail.

## 6. The device limit table

`DeviceProfile` carries the FLOP rates today. It carries no bandwidth and no
fp8 rate. Add these two fields:

```cpp
// Dense (non-sparse) peak rates and the peak HBM bandwidth.
double peak_fp32_flops = 0.0;   // exists
double peak_fp16_flops = 0.0;   // exists
double peak_bf16_flops = 0.0;   // exists
double peak_fp8_flops = 0.0;    // new
double peak_bandwidth_bytes_per_second = 0.0;  // new
```

Fill the fp8 rate for Hopper and Ada. Fill the bandwidth for every entry.

| Device | fp32 | fp16 and bf16 | fp8 | Bandwidth | bf16 ridge point |
|---|---:|---:|---:|---:|---:|
| h200 | 67.0 TFLOP/s | 989 TFLOP/s | 1,979 TFLOP/s | 4.8 TB/s | 206 FLOP/byte |
| h100 | 66.9 TFLOP/s | 989 TFLOP/s | 1,979 TFLOP/s | 3.35 TB/s | 295 FLOP/byte |
| a100 | 19.5 TFLOP/s | 312 TFLOP/s | 0 | 2.0 TB/s | 156 FLOP/byte |
| t4 | 8.1 TFLOP/s | 65 TFLOP/s | 0 | 320 GB/s | 203 FLOP/byte |
| gtx1080ti | 11.34 TFLOP/s | 0.177 TFLOP/s | 0 | 484 GB/s | none |

The ridge point is the rate divided by the bandwidth. It decides which limit
binds. Section 7 uses it.

Two functions return a device peak today, and they disagree by design.

| Function | Source | Use |
|---|---|---|
| `PeakFlopsForDevice` | A string pattern table in `src/train.cc` | The reference MFU line |
| `PeakFlopsForCompute` | `src/device_profile.cc`, selected by the build precision | The project MFU line |

The model selects the rate from the precision assignment of section 5, not from
the build precision alone. Extend `PeakFlopsForCompute` with an explicit
precision argument. The consistency test `//tests:device_profile_test` holds
the two tables together. Extend it with the fp8 and bandwidth rows.

The table is dense and non-sparse. State that in the table, because a vendor
sheet often quotes the sparse rate.

## 7. The estimator

The estimator is the roofline. Each operation has two limits, and the bound is
the larger of the two.

```text
compute_time = flops / peak_rate[precision]
memory_time  = bytes / peak_bandwidth
time         = max(compute_time, memory_time) + launch_time
bound        = sum(time) + sync_time
```

| Term | Formula | Source |
|---|---|---|
| `compute_time` | `flops / peak_rate[precision]` | The counter table and the limit table |
| `memory_time` | `bytes / peak_bandwidth` | The counter table and the limit table |
| `launch_time` | `launches * per_launch_us` | The device limit table |
| `sync_time` | `2 * (n - 1) / n * payload / link_bandwidth` | The topology |

The bound is an upper limit. A measured step below the bound is normal. The gap
is the finding, and section 9 shows how to read it.

### 7.1 The roofline

The roofline decides which limit binds. The arithmetic intensity is the
arithmetic divided by the bytes.

At the depth-24 shape on the H100, in bf16:

| Operation | Intensity (FLOP/byte) | Limit |
|---|---:|---|
| `c_q`, `c_k`, `c_v` (`32768 x 1536 x 1536`) | 750 | Compute |
| `mlp.c_fc` (`32768 x 6144 x 1536`) | 1,184 | Compute |
| `mlp.c_proj` (`32768 x 1536 x 6144`) | 1,184 | Compute |
| the language-model head (`32768 x 32768 x 1536`) | 1,404 | Compute |
| the attention scores (`2048 x 2048 x 128`) | 114 | Memory |
| the attention output (`2048 x 128 x 2048`) | 114 | Memory |
| the quantization cast | Near 0 | Memory |
| the layout copies | 0 | Memory |

The bf16 ridge point on the H100 is 295 FLOP/byte. `1979e12 / 3.35e12` gives
591 FLOP/byte in fp8. The matrix multiplies stay above both ridge points even
after the rate doubles, so they remain compute bound.

Two families sit far below the ridge point: the dense attention, and the
quantization cast. Both are memory bound, and both use the bf16 or fp8 rate
only in name. This is the most useful result of the model.

### 7.2 The bound at the depth-24 shape

Take 24 layers, 12 heads, head width 128, sequence 2048, `SSSL`, and 5.84
billion tokens on eight H100 cards.

| Term | bf16 everywhere | The reference fp8 | The C++ fp8 target |
|---|---:|---:|---:|
| The matrix multiplies | 54 min | 27 min | 27 min |
| The quantization cast, fused | — | 13 min | 13 min |
| The fp8 layout copies | — | 5 min | About 0 |
| The attention, bf16 | 5 min | 5 min | 5 min |
| The gradient sync | 2 min | 2 min | 2 min |
| **Subtotal** | **61 min** | **52 min** | **47 min** |
| The row kernels and the optimizer | Memory bound, from the counter table | Same | Same |

Three results follow.

1. **fp8 is worth about 9 minutes of the bound at this shape.** The GEMM term
   falls by 27 minutes. The cast and the copies give 18 minutes back. The gain
   is real, and it is not a factor of two.
2. **The cast must be one fused kernel per operand.** At five bytes per element
   the cast costs 13 minutes. At sixteen bytes it costs 42 minutes, and at
   thirty-two bytes it costs 84 minutes. An unfused cast makes fp8 slower than
   bf16.
3. **The layout is a design decision.** The reference pays about 5 minutes for
   per-step copies. Store the weight in the required layout once, and that term
   disappears.

The fp32 fused attention tile would change the attention row from 5 minutes to
120 or 240 minutes. See
[attention-tensor-core.md](attention-tensor-core.md).

The nanochat reference measured 99 minutes against a bound of about 52 minutes.
The fraction is about 0.53. The remaining 47 minutes are the row kernels, the
launch gaps, the imperfect overlap, and the achieved GEMM rate.

## 8. The work-ratio check

The benchmark report already carries two fields:

```json
{"name": "attention_fwd:d8_s512", "ms": 12.3, "gflops": 45.6, "work_ratio": 2.0}
```

- `gflops` is the rate against the necessary arithmetic.
- `work_ratio` is the issued work divided by the necessary work.

Extend the same idea to the model. One operation can be correct and still issue
too much work. Report the work ratio for each family, against the necessary
count from the counter table.

`docs/performance.md` section 3.1 gives the precedent. The earlier attention
tile recomputed the query-key product once per owned output dimension. The
issued work therefore grew with the square of the head width. The work ratio
shows that fault without a device.

For the fp8 path, the work ratio covers one more case: the number of passes
over an operand. Four passes over an activation to measure, scale, cast, and
copy it is three passes more than the minimum.

## 9. The reference points

A bound alone does not say whether a kernel is good. Compare the measured rate
with the bound for the same family.

| Reference | Measured | Bound | Fraction |
|---|---:|---:|---:|
| The `lm_head` GEMM, `M=16384 N=32768 K=768`, fp32, T4, in PyTorch | 92.9 ms | 8.1 TFLOP/s | 1.09 |
| The same GEMM in nanochat.cpp, before the call fix | 229.1 ms | 8.1 TFLOP/s | 0.44 |
| The full d12 forward, `B=16 T=1024`, fp32, T4 | 679 ms | 8.1 TFLOP/s | — |
| The attention forward, `d8_s512` window 128, fp32, GTX 1080 Ti | 1.087 ms | 11.34 TFLOP/s | — |
| The depth-24 run, eight H100 cards, fp8 | 99 min | about 52 min | 0.53 |

Two lessons come from the table.

1. **A fraction above 1.0 means the table peak is wrong.** The first row is
   8.8 TFLOP/s measured against an 8.1 TFLOP/s table value. The table rate is
   likely a base-clock value. Fix the table, and do not absorb the fault into a
   factor.
2. **The fraction is a diagnostic, not an input.** The 0.44 row is a real
   defect that the fix in `backends/cuda/gemm.cu` removed. The model points at
   the family; the measurement confirms it.

The d24 fraction of 0.53 is the target to beat. Each of the four gap items in
section 7.2 is a separate investigation.

## 10. The attainment record

Keep one table. Update it after each measurement.

| Date | Revision | Device | Build | Precision | Configuration | Bound | Measured | Fraction | Gap cause |
|---|---|---|---|---|---|---|---|---|---|
| — | reference `a825e63` | 8 x H100 | fp8 | fp8 | d24, batch 16, grad-accum 4, 5.84e9 tokens | 52 min | 99 min | 0.53 | The reference PyTorch run, not this tree (docs/node-h100.md). The in-tree measurement is pending. |
| — | this tree | 8 x H100 | fp8 | fp8 | d24, batch 16, grad-accum 4, 5.84e9 tokens | 53 min | pending | — | The bound includes the row kernels; the estimator prints it with `tools/nanochat roofline`. |

The record is the audit trail. A bound without a measurement is a limit, and
the document must say so.

## 11. Deliverables and owners

| Deliverable | Path | Owner |
|---|---|---|
| The counter struct, with the precision per row | `include/nanochat/mfu.h`, `src/train.cc` | Harness |
| The precision assignment | `src/performance_model.cc`, from `docs/fp8-training.md` | Harness |
| The fp8 rate and the bandwidth column | `include/nanochat/device_profile.h`, `src/device_profile.cc` | Architect |
| The explicit-precision rate selection | `src/train.cc` | Harness |
| The estimator | `src/performance_model.cc` | Harness |
| The roofline report | The estimator, printed with the step summary; `tools/nanochat roofline` | Harness |
| The consistency test | `tests/device_profile_test.cc` | Oracle |
| The attainment record | `docs/performance-model.md` | Architect |
| This document | `docs/performance-model.md` | Architect |

## 12. Validation

1. The counter struct sums to `EstimateFlopsPerToken` for every committed
   configuration. Add this as a unit test.
2. The precision assignment matches the rule in section 5.1 for every committed
   configuration. Add this as a unit test.
3. The fp8 rate and the bandwidth rows pass `//tests:device_profile_test`.
4. The roofline labels the dense attention and the cast as memory bound, and the
   matrix multiplies as compute bound.
5. The estimator reproduces the bound in section 7.2 for the depth-24 shape.
6. The measured attention rows from phase A0 of
   [attention-tensor-core.md](attention-tensor-core.md) appear in section 10 with
   their fractions.

Checks 1 to 5 run in `//tests:performance_model_test`; check 3 also runs in
`//tests:device_profile_test`. Check 6 needs the A0 measurement.

## 13. Phases

| Phase | Work | Gate | GPU |
|---|---|---|---|
| M0 | The counter struct, with the precision per row, and the sum test | The test passes for every committed configuration | no |
| M1 | The fp8 rate, the bandwidth column, and the explicit-precision rate selection | The consistency test passes | no |
| M2 | The estimator and the roofline report | The bound in section 7.2 is reproduced | no |
| M3 | The attainment record, and the first depth-24 bound | The table in section 10 holds one row with a measurement | no |

## 14. Status by section

| Section | Status |
|---|---|
| 4 The counter table | Implemented (`CountFlopsPerToken`, `FlopCounts`) |
| 5 The precision assignment | Implemented (`AssignMatrixPrecision`) |
| 6 The device limit table | Implemented (fp8 rate and bandwidth in `DeviceProfile`) |
| 7 The estimator | Implemented (`src/performance_model.cc`) |
| 8 The work-ratio check | Benchmark side only; the model rows carry the binding limit |
| 9 The reference points | Reference data, not a computed output |
| 13 Phases | M0 to M2 complete; M3 awaits a measurement |

## 15. Risks and non-goals

- **The bound is an upper limit, not a prediction.** A run below the bound is
  normal. Section 7.2 shows how to split the gap.
- **A table peak may be wrong.** Section 9 records one such case. Check the
  table before you judge a kernel.
- **Measurement cannot find a missing counter term.** If the counter table
  omits the cast traffic, the bound is wrong by 13 minutes. Review the counter
  table against `docs/kernels.md`.
- **The precision assignment can drift from the code.** Keep the assignment in
  one function, and unit-test it against the layer set in
  [fp8-training.md](fp8-training.md).
- **The bandwidth peak is hard to reach.** The memory term is a bound, like
  the compute term.
- **The launch cost is a device property.** Record it in the table. Do not set
  it per shape.
- **Not a goal:** a fitted correction, a cycle-accurate model, a cache model,
  and a full memory simulation.

## 16. Open questions

1. The counter struct lives beside `NumMatmulParams` in `src/train.cc`. The
   declaration is in `include/nanochat/mfu.h`.
2. The estimator lives in the C library, in `src/performance_model.cc`.
3. The report goes to the training log, and `roofline_main` also writes the
   `nanochat.roofline.v1` JSON.
4. The launch cost lives in `DeviceProfile::launch_us`.
5. The C++ fp8 path stores the weight transposed. The estimator models the
   reference copies behind `ModelPlan::transposed_weights`.

## See also

- [performance.md](performance.md) — the measurement protocol and the anchors.
- [simulator.md](simulator.md) — the S0 engines and their limits.
- [kernels.md](kernels.md) — the kernel inventory.
- [fp8-training.md](fp8-training.md) — the fp8 rule and the layer set.
- [attention-tensor-core.md](attention-tensor-core.md) — the option decision the
  model gates.
