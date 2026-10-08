#ifndef NANOCHAT_TESTS_NUMERICS_TRACE_H_
#define NANOCHAT_TESTS_NUMERICS_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "tests/oracle_fixture.h"

// The staged numeric trace of docs/numerics-integration.md: one 10-layer model,
// four phases (training, evaluation, greedy generation, and the SFT and RL
// arithmetic), recorded as a flat list of named records, so one committed
// golden file can be replayed on the CPU reference backend, under
// `--config=sim`, and on the Pascal GPU.
//
// Container. The trace reuses the little-endian record layout that
// `tests/oracle_fixture.h` reads, under its own 8-byte magic `NANONUM1`. The
// layout is unchanged, so Phase 0 added one magic string and no record
// sequence: `[magic][u32 version][u32 count]`, then one
// `[u16 name_len][name][u8 dtype][u8 ndim][u64 shape...][payload]` per record.
// The two 64-bit hashes travel as `kInt64` bit patterns, and the strings travel
// as `kUInt8` byte payloads.
//
// Record names. The header and the run parameters start with `header/`,
// `config/`, `opt/`, or `sched/`. The phases add `train/`, `eval/`, `gen/`,
// `sft/`, and `rl/`. A parameter record is `train/step<N>/param_l2/<name>`,
// `train/step<N>/param_hash/<name>`, or `train/step<N>/grad_hash/<name>`, with
// the model's own parameter names from `Model::params()`.
//
// Determinism. Every value comes from the model, the seed, and the committed
// tokenizer; nothing reads the clock or an entropy source. The trace pins the
// host thread count to one (`PinSingleThreaded`), so the golden file does not
// depend on the sandbox thread budget. The hashes are valid only for the exact
// legs: they are meaningless on the GPU leg, where the atomic scatter-adds
// change the bytes by design.
namespace nanochat {
namespace numerics {

// The fixture magic of the numeric trace container.
inline constexpr char kTraceMagic[] = "NANONUM1";

// The tolerance table of docs/numerics-integration.md section 4.
// `kLossRtol` covers every scalar that is not a parameter norm: the train loss
// and gradient norm, the evaluation bits per byte and losses, the SFT
// arithmetic, and the RL arithmetic. `kParamL2Rtol` covers the parameter L2
// norms. `kMarginAtol` compares the recorded logit margins, which are
// differences of two logits and can sit near zero. `kMinMargin` is the tie rule
// of section 6.4: every recorded margin must exceed it, or a one-unit roundoff
// difference can flip a greedy identifier. The measured maxima are in the
// section 4 table.
inline constexpr double kLossRtol = 1e-5;
inline constexpr double kParamL2Rtol = 1e-4;
inline constexpr double kMarginAtol = 1e-5;
inline constexpr double kMinMargin = 1e-5;

// One record of the trace, in the container's encoding.
struct Record {
  std::string name;
  oracle::DType dtype = oracle::DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> payload;
};

Record Fp32Record(const std::string& name, float value);
Record Fp32VectorRecord(const std::string& name, const float* values,
                        std::int64_t count);
Record Int32Record(const std::string& name, std::int32_t value);
Record Int32VectorRecord(const std::string& name, const int* values,
                         std::int64_t count);
Record Int64Record(const std::string& name, std::int64_t value);
Record BytesRecord(const std::string& name, const std::string& text);

// FNV-1a over the raw bytes, 64 bit. The hash is a bit-pattern checksum of the
// staged host bytes, so it is exact on the CPU legs and skipped on the GPU leg.
std::uint64_t HashBytes(const void* data, std::size_t bytes);

// The compile-time identity of the running build. `BackendName` is "cpu" for
// the reference backend -- including the simulator leg, which shares the code
// path -- and "cuda" for the device backend. `PrecisionName` is the selected
// storage precision, and `BuildName` joins the two.
std::string BackendName();
std::string PrecisionName();
std::string BuildName();

// Serializes records into the little-endian container that `Fixture::Parse`
// reads. The order is the vector order.
std::vector<std::uint8_t> SerializeTrace(const std::vector<Record>& records);

// The reference configuration of docs/numerics-integration.md section 6.2.
Config ReferenceConfig();

// The fixed shape of the trace. The defaults are the reference values; a caller
// changes them only to build a smaller trace.
struct TraceShape {
  std::uint64_t seed = 42;
  int steps = 5;        // optimizer steps in the train phase
  int batches = 6;      // synthetic pretraining batches (one per trace step)
  int eval_steps = 2;   // evaluation batches
  int gen_prompt = 16;  // prompt tokens of the generation phase
  int gen_tokens = 16;  // greedy tokens of the generation phase
  int rl_samples = 4;   // rollout rows of the RL phase
  int rl_prompt = 16;   // prompt tokens of the RL rollout
  int rl_tokens = 32;   // greedy tokens of the RL rollout
  int rl_num_passes = 2;
  int rl_examples_per_rank = 4;
};

const TraceShape& DefaultShape();

// Observed quantities of one trace run, for the report.
struct TraceStats {
  int records = 0;
  int parameters = 0;
  double min_gen_margin = 0.0;
  double min_eval_margin = 0.0;
  double min_rl_margin = 0.0;
  double eval_bpb = 0.0;
  double sft_loss = 0.0;
  double rl_loss = 0.0;
  int gen_path_mismatch = 0;
  int rl_path_mismatch = 0;
  std::vector<double> train_loss;
};

// Runs every phase once and returns the records in a fixed order. The tokenizer
// path supplies the document loader of the evaluation phase, so it must name a
// readable `NCTOKEN1` artifact. `stats` may be null.
std::vector<Record> RunTrace(const Config& config, const TraceShape& shape,
                             const std::string& tokenizer_path,
                             TraceStats* stats);

// One comparison group: the records that share a tolerance.
struct GroupStat {
  std::string name;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  int count = 0;
};

// The verdict of one comparison.
struct TraceComparison {
  int failures = 0;
  int compared = 0;
  int skipped = 0;
  std::vector<GroupStat> groups;
};

// Compares `got` against the golden records. `exact` requires identical bytes
// (the CPU and simulator legs); otherwise the float records use the tolerance
// table, the integer records compare exactly, and the hash records are skipped
// (the GPU leg). Failures are written to stderr with the record name.
TraceComparison CompareTraces(const std::vector<Record>& got,
                              const oracle::Fixture& golden, bool exact,
                              const std::string& label);

// Pins the host thread count to one before the first kernel call, so the CPU
// trace does not depend on the sandbox thread budget. Call it first in main().
void PinSingleThreaded();

}  // namespace numerics
}  // namespace nanochat

#endif  // NANOCHAT_TESTS_NUMERICS_TRACE_H_
