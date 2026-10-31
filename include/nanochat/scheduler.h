#ifndef NANOCHAT_SCHEDULER_H_
#define NANOCHAT_SCHEDULER_H_

// Learning-rate, Muon-momentum, and weight-decay schedules (docs/model.md).
// The schedule is split out from the optimizer so the training loop and the
// optimizer agree on exactly one source of truth. Vendor-free.

namespace nanochat {

struct SchedulerConfig {
  int num_iterations = 0;
  int warmup_steps = 0;
  // How much of the run is warmup, as a fraction of `progress` rather than as a
  // step count (design decision D1 of docs/sft-parity-plan.md). Only
  // `LrMultiplierForProgress` reads it: the pretraining entry point
  // `LrMultiplier(int step)` keeps `warmup_steps`, because a pretraining run
  // knows its total before it starts and an SFT run may stop at the end of its
  // data. The default of 0 is today's behavior: the pretraining schedule never
  // warms up through this field, and neither does an SFT run pinned with
  // `warmup_ratio = 0` (section 5.2 of the plan).
  float warmup_ratio = 0.0f;
  float warmdown_ratio = 0.65f;  // fraction of the run spent warming down
  float final_lr_frac = 0.0f;    // LR multiplier at the end of warmdown
  float weight_decay_base = 0.0f;

  float muon_momentum_warmup_steps = 400.0f;
  float muon_momentum_start = 0.85f;
  float muon_momentum_peak = 0.97f;
  float muon_momentum_final = 0.90f;
};

class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config) : config_(config) {}

  // The pretraining learning-rate multiplier: linear warmup over
  // `warmup_steps`, constant, then linear warmdown over the last
  // `warmdown_ratio` of the run, ending at `final_lr_frac` (mirrors nanochat's
  // `get_lr_multiplier`). `step` is 1-based and the schedule is evaluated at
  // `step - 1`, because the reference loop counter is 0-based. That convention
  // does not change (design decision D2 of docs/sft-parity-plan.md), so this
  // entry point and `MuonMomentum(int step)` keep their indexing and the SFT
  // driver restarts its counter at 1 after the warm start.
  //
  // The pretraining driver (`src/train.cc`) is the caller that uses this one.
  // The SFT driver (`src/sft_main.cc`) uses `LrMultiplierForProgress` instead
  // (design decision D1), because a finetune's schedule follows the fraction of
  // the run that happened, not a step total fixed before the run started.
  float LrMultiplier(int step) const;

  // The SFT learning-rate multiplier, indexed by `progress`, the fraction of
  // the run that has happened so far, in [0, 1]. This is the body of
  // `scripts/chat_sft.py:310-317` and it has exactly three branches:
  //
  //   if (progress < warmup_ratio)
  //     return (progress + 1e-8) / warmup_ratio;
  //   if (progress <= 1.0 - warmdown_ratio)
  //     return 1.0;
  //   decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio;
  //   return 1.0 + decay * (final_lr_frac - 1.0);
  //
  // The `1e-8` keeps the warmup branch finite at `progress == 0`. The second
  // branch is inclusive at its right endpoint, so the constant segment ends at
  // `1.0 - warmdown_ratio`, and `progress == 1.0` returns `final_lr_frac`
  // exactly. A `warmup_ratio` of 0 disables the first branch (the comparison is
  // false for every non-negative `progress`), and an implementation keeps an
  // explicit `warmup_ratio > 0` guard so no input can divide by zero.
  //
  // The caller feeds the running maximum of the reference `approx_progress`
  // (section 5.2 of docs/sft-parity-plan.md), so `progress` is monotone
  // non-decreasing within one run. `src/optim.cc` implements this beside
  // `LrMultiplier`, and nothing else defines the curve.
  float LrMultiplierForProgress(float progress) const;

  // Muon momentum: warm up from `muon_momentum_start` to
  // `muon_momentum_peak`, hold, then decay to `muon_momentum_final` during
  // warmdown.
  float MuonMomentum(int step) const;

  // Cosine weight decay to zero over the run.
  float WeightDecay(int step) const;

  const SchedulerConfig& config() const { return config_; }

 private:
  int WarmdownIters() const;

  SchedulerConfig config_;
};

}  // namespace nanochat

#endif  // NANOCHAT_SCHEDULER_H_
