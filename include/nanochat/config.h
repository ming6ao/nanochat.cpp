#ifndef NANOCHAT_CONFIG_H_
#define NANOCHAT_CONFIG_H_

#include <cmath>
#include <string>

// Model hyperparameters and their derived dimensions. Configuration checks
// live here so every entry point uses the same rules (docs/model.md).

namespace nanochat {

struct Config {
  int num_layers = 12;
  int num_heads = 6;     // query heads
  int num_kv_heads = 6;  // key/value heads (group-query attention)
  int hidden_dim = 768;
  int seq_len = 2048;
  int vocab_size = 32768;
  int padded_vocab_size = 32768;  // rounded up for GEMM/classifier efficiency
  float rope_base = 100000.0f;
  std::string window_pattern = "SSSL";  // tiled across layers; last is always L
  // Value embeddings (docs/distributed-design.md section 8). The default is the
  // current model; the Track 3 baseline turns them off.
  bool value_embedding = true;

  // Per-head width. `hidden_dim` must be divisible by `num_heads`.
  int head_dim() const { return hidden_dim / num_heads; }

  // Width of the query / key / value projections.
  int query_dim() const { return num_heads * head_dim(); }
  int kv_dim() const { return num_kv_heads * head_dim(); }

  // relu^2 MLP expansion (docs/model.md).
  int mlp_dim() const { return 4 * hidden_dim; }

  // Value embeddings are present on alternating layers, and always on the last
  // layer (mirrors nanochat's `has_ve`). The field `value_embedding` disables
  // them on every layer when false.
  bool has_value_embedding(int layer) const {
    if (!value_embedding) return false;
    return layer % 2 == (num_layers - 1) % 2;
  }

  // Sliding-window length for short (`S`) layers: a quarter of the context,
  // rounded up to a 128-token attention tile. Mirrors nanochat's
  // `_compute_window_sizes`.
  int short_window() const { return ((seq_len / 4 + 127) / 128) * 128; }

  // Left extent of the attention window for `layer`; the right extent is always
  // 0 (causal). Returns `-1` for the final layer and for `L` entries in the
  // pattern (full context).
  int window_left(int layer) const {
    if (layer >= num_layers - 1) return -1;
    if (window_pattern.empty()) return -1;
    const char kind = window_pattern[layer % window_pattern.size()];
    return kind == 'S' ? short_window() : -1;
  }

  int window_right() const { return 0; }

  // Number of KV positions the attention kernel must reserve for a sequence of
  // `tokens`, capped by the window.
  int EffectiveWindow(int layer, int tokens) const {
    const int left = window_left(layer);
    return left < 0 ? tokens : (left < tokens ? left : tokens);
  }

  // Checks the model dimensions before the graph allocates its parameters.
  // Empty window patterns mean full-context attention at every layer.
  bool Validate(std::string* error = nullptr) const {
    if (error != nullptr) error->clear();
    const auto fail = [error](const char* message) {
      if (error != nullptr) *error = message;
      return false;
    };
    if (num_layers <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        hidden_dim <= 0 || seq_len <= 0 || vocab_size <= 0 ||
        padded_vocab_size <= 0) {
      return fail("model dimensions and vocabulary sizes must be positive");
    }
    if (hidden_dim < 24) {
      return fail(
          "hidden_dim must be at least 24 for the smear and value "
          "gates");
    }
    if (hidden_dim % num_heads != 0) {
      return fail("hidden_dim must be divisible by num_heads");
    }
    if (num_heads % num_kv_heads != 0) {
      return fail("num_heads must be divisible by num_kv_heads");
    }
    if (head_dim() % 2 != 0) {
      return fail("head_dim must be even for rotary embeddings");
    }
    if (padded_vocab_size < vocab_size) {
      return fail("padded_vocab_size must be >= vocab_size");
    }
    if (!(rope_base > 0.0f) || !std::isfinite(rope_base)) {
      return fail("rope_base must be finite and positive");
    }
    for (char kind : window_pattern) {
      if (kind != 'S' && kind != 'L') {
        return fail("window_pattern can contain only 'S' and 'L'");
      }
    }
    return true;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_CONFIG_H_
