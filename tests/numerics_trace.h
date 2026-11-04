#ifndef NANOCHAT_TESTS_NUMERICS_TRACE_H_
#define NANOCHAT_TESTS_NUMERICS_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "tests/oracle_fixture.h"

// The staged numeric trace of docs/testing.md: one 10-layer model,
// four phases (training, evaluation, greedy generation, and the SFT and RL
// arithmetic), recorded as a flat list of named records, so one committed
// fixture can be replayed on the CPU reference backend, under `--config=sim`,
// and on the Pascal GPU.
//
// The fixture holds the **exact class** only: configuration, inputs, token ids,
// masks, greedy identifiers, argmax, and valid-target counts. Every one of them
// is an integer or a fixed constant, so two arithmetic implementations either
// agree byte for byte or one of them is wrong. Floating-point results are not
// records: their contract is the per-stage parity tests of
// docs/testing.md, which own a bound each.
//
// Container. The trace reuses the little-endian record layout that
// `src/container.h` owns, under its own 8-byte magic `NANONUM1`:
// `[magic][u32 version][u32 count]`, then one
// `[u16 name_len][name][u8 dtype][u8 ndim][u64 shape...][payload]` per record.
// The strings travel as `kUInt8` byte payloads.
//
// Record names. The header and the run parameters start with `header/`,
// `config/`, `opt/`, or `sched/`. The phases add `eval/`, `gen/`, `sft/`, and
// `rl/`. In the committed fixture every name carries a precision prefix, so one
// file holds all three sections: `fp32/gen/ids`, `bf16/gen/ids`, and
// `fp8/gen/ids`. A build compares its own section.
//
// Determinism. Every value comes from the model, the seed, and the committed
// tokenizer; nothing reads the clock or an entropy source. The trace pins the
// host thread count to one (`PinSingleThreaded`), so the fixture does not
// depend on the sandbox thread budget.
namespace nanochat {
namespace numerics {

// The fixture magic of the numeric trace container.
inline constexpr char kTraceMagic[] = "NANONUM1";

// The tie rule of docs/testing.md Part 2, section 6.4: every logit margin
// the run reports must exceed this value, or a one-unit roundoff difference can
// flip a greedy identifier. The margin is a diagnostic, not a compared record,
// so this is the only numeric threshold in the gate.
inline constexpr double kMinMargin = 1e-5;

// One record of the trace, in the container's encoding.
struct Record {
  std::string name;
  oracle::DType dtype = oracle::DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> payload;
};

Record Fp32Record(const std::string& name, float value);
Record Int32Record(const std::string& name, std::int32_t value);
Record Int32VectorRecord(const std::string& name, const int* values,
                         std::int64_t count);
Record Int64Record(const std::string& name, std::int64_t value);
Record BytesRecord(const std::string& name, const std::string& text);

// FNV-1a over the raw bytes, 64 bit. The emitter uses it to print a checksum of
// the container it writes.
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

// The reference configuration of docs/testing.md Part 2, section 6.2.
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

// The verdict of one comparison.
struct TraceComparison {
  int failures = 0;
  int compared = 0;
};

// The three sections a committed fixture carries, in file order.
const std::vector<std::string>& TraceSections();

// The prefix a record name carries in the fixture: "<precision>/".
std::string SectionPrefix();

// Compares `got` (unprefixed names) against the section of the golden that
// belongs to the running build. The comparison is byte for byte: a difference
// is a failure, and there is no tolerance mode. The schema check requires all
// three sections to exist and to carry the same record names in the same order,
// so a section cannot silently lose a record.
TraceComparison CompareTraces(const std::vector<Record>& got,
                              const oracle::Fixture& golden,
                              const std::string& label);

// Pins the host thread count to one before the first kernel call, so the CPU
// trace does not depend on the sandbox thread budget. Call it first in main().
void PinSingleThreaded();

}  // namespace numerics
}  // namespace nanochat

#endif  // NANOCHAT_TESTS_NUMERICS_TRACE_H_
