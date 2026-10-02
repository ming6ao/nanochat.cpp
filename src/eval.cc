// Forward-only bits-per-byte evaluation (mirrors nanochat's `evaluate_bpb`).
//
// The metric is vocabulary-size independent: sum the per-token cross-entropy
// (nats) over the batches, sum the number of source bytes the target tokens
// represent, and divide. Special tokens and masked ids (byte length 0) are
// skipped. When the loader has no byte table, every counted token contributes
// one byte, so the result reduces to mean-nats / ln(2).

#include "nanochat/model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "model_impl.h"
#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"
#include "ops.h"

namespace nanochat {

float EvalBpb(Model* model, DataLoader* loader, int steps) {
  if (model == nullptr || loader == nullptr || steps <= 0) {
    return std::numeric_limits<float>::infinity();
  }
  const int batch = loader->batch();
  const int seq = loader->seq();
  if (batch <= 0 || seq <= 0) {
    return std::numeric_limits<float>::infinity();
  }

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));

  int byte_vocab = 0;
  const std::uint8_t* byte_table = loader->token_bytes(&byte_vocab);

  double total_nats = 0.0;
  std::int64_t total_bytes = 0;
  for (int s = 0; s < steps; ++s) {
    if (!loader->Next(tokens.data(), targets.data())) break;
    model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    auto* impl = static_cast<TrainModel*>(model);
    const ComputeType* losses = impl->losses();
    if (losses == nullptr) break;
    // Stage the per-token losses once; the CUDA backend keeps them in device
    // memory.
    std::vector<ComputeType> losses_host(static_cast<std::size_t>(rows));
    kernels::Memcpy(losses_host.data(), losses,
                    static_cast<std::size_t>(rows) * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
    for (std::int64_t i = 0; i < rows; ++i) {
      const int target = targets[static_cast<std::size_t>(i)];
      if (target < 0) continue;  // ignore index
      std::int64_t bytes = 1;
      if (byte_table != nullptr) {
        bytes = target < byte_vocab ? byte_table[target] : 0;
      }
      if (bytes <= 0) continue;
      total_nats +=
          static_cast<double>(AsF(losses_host[static_cast<std::size_t>(i)]));
      total_bytes += bytes;
    }
  }

  if (total_bytes == 0) return std::numeric_limits<float>::infinity();
  const double denominator = std::log(2.0) * static_cast<double>(total_bytes);
  return static_cast<float>(total_nats / denominator);
}

}  // namespace nanochat
