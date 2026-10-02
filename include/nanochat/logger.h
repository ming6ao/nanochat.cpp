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
