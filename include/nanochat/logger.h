#ifndef NANOCHAT_LOGGER_H_
#define NANOCHAT_LOGGER_H_

#include <memory>
#include <string>

// Training logger: one structured record per step to stdout and, optionally, a
// run file. Kept separate from the compute path so logging never lands on the
// critical path (docs/model.md).

namespace nanochat {

struct LogRecord {
  int step = 0;
  float loss = 0.0f;
  float lr = 0.0f;
  float grad_norm = 0.0f;
  float tokens_per_second = 0.0f;
  float mfu = 0.0f;  // model FLOPs utilization, 0..1

  // The phase split of one optimizer step, in milliseconds
  // (docs/distributed-t4-plan.md phase B1). `step_ms` is the training time:
  // the data fetch, the forward, the backward, the sync, and the optimizer.
  // `eval_ms` is separate, so a periodic evaluation does not distort the
  // reported throughput. A phase reads 0 when it did not run on this step.
  float step_ms = 0.0f;
  float data_ms = 0.0f;
  float forward_ms = 0.0f;
  float backward_ms = 0.0f;
  float sync_ms = 0.0f;
  float optim_ms = 0.0f;
  float eval_ms = 0.0f;
};

class Logger {
 public:
  // Logs to stdout only when `path` is empty.
  Logger();
  explicit Logger(const std::string& path);
  ~Logger();

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void Log(const LogRecord& record);
  void Info(const std::string& message);
  void Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nanochat

#endif  // NANOCHAT_LOGGER_H_
