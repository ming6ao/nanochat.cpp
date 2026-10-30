# backends/cuda

The NVIDIA device axis: the device/memory half of the frozen seam
(`nanochat/kernels.h`), the capability declaration, the two reductions of the
primitive tier, and the backend's fused-override registrations. The per-family
compute kernels live in [`kernels/`](kernels/README.md); the collectives live in
[`collective/`](collective) and belong to the Collective workstream.

The layering rule is [DESIGN.md](../../DESIGN.md) §2.1: nothing here includes a
model header, and above the seam nothing includes a CUDA header.

## `caps.json` — the single source of the declaration

`caps.json` is the authority for what this backend offers. It is read, not
maintained twice: `src/device_profile.cc` derives the profile table in
`nanochat/device_profile.h` from it, and `device.cu` reads that derived entry
through `FindDeviceProfile` + `CapsFromProfile` instead of keeping a second list
(`docs/model-and-device-bringup.md` §6, §9 rule 1).

Shape, one object per device this backend supports:

| Field | Meaning |
|---|---|
| `format`, `capability_vocabulary` | `nanochat.caps/1`, and the
  `spec/versions.md` vocabulary number the `features` strings speak |
| `backend`, `vendor` | the axis and the silicon vendor |
| `devices[]` | one entry per target, keyed by `name` — the lowercased join key
  `FindDeviceProfile` and `PeakFlopsForDevice` match |
| `devices[].compute_major` / `compute_minor` / `total_memory_bytes` /
  `has_tensor_cores` / `supports_bf16` / `supports_fp8` / `peak_*_flops` /
  `peak_bandwidth_bytes_per_second` / `warp_size` / `max_threads_per_block` /
  `max_shared_memory_per_block` / `launch_us` | the profile facts, one field per
  `DeviceProfile` field so the derivation is a direct mapping |
| `devices[].is_device` / `devices[].seam` | this backend is a device, and it
  implements `kKernelSeamVersion` 2 |
| `devices[].utilities`, `devices[].primitives` | the entry points this backend
  supplies, by tier (`docs/model-and-device-bringup.md` §7.2) — the declaration
  that makes `primitive_tier_complete` auditable instead of asserted |
| `devices[].pointwise_codes`, `devices[].reduce_ops` | the closed code lists of
  `nanochat/kernels.h` that the implementation covers: all nine pointwise codes
  and all four combine ops |
| `devices[].attention` | the variants (`full`, `sliding`) and the tile size, in
  query rows per block — the "one tile size" the tier requires |
| `devices[].stateful` | the two stateful entries |
| `devices[].features` | the capability bits this backend honours **on that
  device**, as `spec/capability_bits.md` lexical names, one string per bit |

Two rules govern `features`:

1. **A name is a vocabulary name, verbatim.** `backends/cuda/caps_names.h` maps
   each one onto its `kFeature*` constant in `include/nanochat/tensor.h`, in one
   table with a compile-time distinctness check. A name that table does not hold
   is a hard error (`ParseCapabilityName` returns false, `caps_test.cc` fails by
   name, `RequireNamedBits` aborts the first `GetCaps()`), never a dropped row.
   There are fourteen of them today: vocabulary 1 has fifteen, and
   `override_value_gate` is missing because the frozen header has no constant for
   it. Declaring a name with no constant would promise the planner a bit it can
   never see.
2. **A device declares only what it can honour.** `bf16` appears only on the
   devices whose silicon has bf16 arithmetic (sm_80 and later) and the fp8 pair
   only where the E4M3/E5M2 tensor core exists (sm_89 and later, Hopper 9.0).
   The primitive tier, the two stateful entries, and the three registered
   override bits appear on every entry, because they are promises about this
   backend's code and not about a card. `spec/capability_bits.md` §4: a device
   that offers no fp8 path declares neither fp8 bit.

## The capability mask

`kernels::GetCaps()` fills `Caps::feature_mask` from two inputs and no third one:

```text
mask = (caps.json-derived profile entry, minus its precision bits)
     | (the four precision bits cudaDeviceProp reports)
```

- The **declaration** supplies the architecture (`gqa`, `rope`,
  `sliding_window`, `value_gate`), tier (`primitive_tier_complete`), stateful
  (`stateful_adamw`, `stateful_muon`) and override (`override_*`) bits.
- The **silicon** supplies the precision bits (`fp32` always, `bf16` from
  sm_80, the fp8 pair from sm_89 when the fp8 path is built), because a profile
  written for the fastest member of a family must not make a Pascal card claim
  bf16.
- `supports_fp32`, `supports_bf16`, `supports_fp8` and `Supports(DType)` are
  read back off the mask, so the two spellings of one answer cannot disagree.
  `has_tensor_cores` stays a plain boolean: it is a fact about the part, not a
  promise the backend makes (`spec/capability_bits.md` §5).

Until the caps.json derivation has regenerated `src/device_profile.cc`, the
profile entries carry no mask, so the mask holds only the silicon's precision
bits and the tier, stateful and override bits are absent — a device that offers
nothing the planner can read, which is a named rejection rather than a wrong
answer. `caps_gpu_test` prints exactly that state (with the list of bits the
derivation must supply) and asserts the other direction, that the mask never
claims a bit `caps.json` does not declare, so the gap is visible and can never
silently widen.

`device_name` keeps the driver's spelling verbatim. It is the join key
`FindDeviceProfile` normalizes (case and whitespace), so `GetCaps().device_name`
and the profile entry named `gtx1080ti` are one device, and `caps_test.cc`
asserts exactly that.

## What is implemented here

| File | Contents |
|---|---|
| `device.cu` / `device.h` | `Alloc`, `Free`, `Memcpy`, `Memset`, `Synchronize`, `GetCaps`, the stream/launch helpers, and the two mask helpers (`DeclaredFeatureMask`, `PrecisionFeatureBits`) |
| `gemm.cu` | `Gemm`, `GemmWgrad`, `QuantizeFp8`, `GemmScaledFp8` (cuBLAS and cublasLt) |
| `row_reduce.cu` | `RowReduce` — the per-segment coalesced scan |
| `span_reduce.cu` | `SpanReduce` — the device-wide ragged reduction over a list the device reads itself |
| `fused_overrides.cc` | `RegisteredFusedOverrides` — the fused families the plan may choose |
| `caps_names.h` / `caps_names.cc` | the vocabulary-name to capability-bit table |

The two reductions are separate translation units on purpose: a per-segment
scan and a device-wide ragged reduction are different algorithms with different
correctness risks, and a device may run them on different units
(`docs/model-and-device-bringup.md` §7.2, `spec/decisions.md` D-001).

`caps.json` is exported as the filegroup `//backends/cuda:caps_json`, the same
label name `//backends/cpu` uses, for the derivation and for `caps_gpu_test`.

## Fused overrides

A fused override is a single-kernel replacement for one composition; the
composition defines the semantics and the override is an optimization
(`DESIGN.md` §3.1, `docs/model-and-device-bringup.md` §7.4 condition 1). The
registration names a composition, a family, and a capability bit — never a graph,
a block index, or a kernel symbol, which is what stops a backend from adding
topology through the back door.

Five fused families exist in [`kernels/`](kernels/README.md). Three are
registered:

| Composition | Family | Capability bit |
|---|---|---|
| `rms_norm` | `rms_norm` (with the fused residual variant) | `kFeatureFusedResidualRmsNorm` |
| `qk_prep` | `qk_prep` | `kFeatureFusedQkPrep` |
| `classifier` | `classifier` | `kFeatureFusedClassifier` |

Two are not, and the reason is the vocabulary rather than the code:

- `value_gate` — vocabulary bit 15 is `override_value_gate` and
  `include/nanochat/tensor.h` has no constant for it, so no device could ever set
  the bit and the registration would be dead code that looks live
  (`spec/capability_bits.md` §5). The follow-up is one header constant plus one
  row in `caps_names.h`.
- `grad_norm` — vocabulary 1 has no override bit at all: `GradNorm` is a
  composition with no fused counterpart (`docs/model-and-device-bringup.md`
  §7.3, `spec/capability_bits.md` §5).

`attention` is not registered either: `kFeatureFusedAttention` exists in the
header (the v2 candidate of `DESIGN.md` §3.1) and has no vocabulary name, so no
caps file declares it.

Every family, registered or not, carries a trace comparison at the T1 tiny
shapes in `fused_override_trace_test.cc`: the fused forward, its saved slots,
and its backward against the composition's. That test holds the registration
table and the coverage table to each other, so a family cannot be promoted
without its gate and cannot keep a registration after losing one. It is the gate
`docs/model-and-device-bringup.md` §7.6 requires before a fused family may be
removed.

## Tests

All of them are T1 GPU correctness, tagged `gpu`, and aggregated into
`//backends/cuda:all` so `//:all_tests` needs no special case:

| Target | Covers |
|---|---|
| `cuda_runtime_gpu_test` | the runtime and the GEMM entry points |
| `row_reduce_gpu_test` | `RowReduce`: every combine op, the offset/stride descriptor, and a segment wider than one block |
| `span_reduce_gpu_test` | `SpanReduce`: ragged spans, empty spans, the dot form, the scale, `accumulate`, and a chunked launch |
| `caps_gpu_test` | `caps.json` against the reported mask, the name table, and the registrations |
| `fused_override_trace_gpu_test` | one override-versus-composition trace per fused family |

Run them through the entry point, never directly: `tools/nanochat test --gpu
//backends/cuda:all`.
