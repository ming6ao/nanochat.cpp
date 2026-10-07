#ifndef NANOCHAT_SCHEDULER_H_
#define NANOCHAT_SCHEDULER_H_

// Learning-rate, Muon-momentum, and weight-decay schedules (docs/model.md).
// The schedule is split out from the optimizer so the training loop and the
// optimizer agree on exactly one source of truth. Vendor-free.

namespace nanochat {

struct SchedulerConfig {
  int num_iterations = 0;
  int warmup_steps = 0;
  float warmdown_ratio = 0.65f;  // fraction of the run spent warming down
  float final_lr_frac = 0.0f;    // LR multiplier at the end of warmdown
  float weight_decay_base = 0.0f;

  float muon_momentum_warmup_steps = 400.0f;
  float muon_momentum_start = 0.85f;
  float muon_momentum_peak = 0.97f;
  float muon_momentum_final = 0.90f;

  // ANVIL's fast-rail beta (docs/optimizer-anvil-design.md): linear warmup from
  // `rail_beta_min` to `rail_beta_max`, flat, then a linear cooldown over the
  // last `rail_beta_cooldown_steps`.
  float rail_beta_warmup_steps = 240.0f;
  float rail_beta_cooldown_steps = 50.0f;
  float rail_beta_min = 0.85f;
  float rail_beta_max = 0.93f;
};

class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config) : config_(config) {}

  // Linear warmup, constant, then linear warmdown (mirrors nanochat's
  // `get_lr_multiplier`).
  float LrMultiplier(int step) const;

  // Muon momentum: warm up from `muon_momentum_start` to
  // `muon_momentum_peak`, hold, then decay to `muon_momentum_final` during
  // warmdown.
  float MuonMomentum(int step) const;

  // Cosine weight decay to zero over the run.
  float WeightDecay(int step) const;

  // ANVIL's fast-rail beta and Nesterov lookahead: warm up from
  // `rail_beta_min` to `rail_beta_max`, hold, then cool down over the last
  // `rail_beta_cooldown_steps` of the run.
  float RailBeta(int step) const;

  const SchedulerConfig& config() const { return config_; }

 private:
  int WarmdownIters() const;

  SchedulerConfig config_;
};

}  // namespace nanochat

#endif  // NANOCHAT_SCHEDULER_H_
