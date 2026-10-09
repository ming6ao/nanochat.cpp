#ifndef NANOCHAT_SRC_GENERATE_H_
#define NANOCHAT_SRC_GENERATE_H_

#include <vector>

#include "nanochat/model.h"

// Tool forcing in the generation loop (docs/parity.md item P1,
// docs/post-training.md section 5.1). The batched engine in `generate.cc` is
// tokenizer-agnostic: it never decodes text, so it cannot recognize a
// `<|python_start|>` block on its own. The forcing seam is therefore a
// caller-supplied hook that watches the emitted token stream and injects the
// tool-output ids.
//
// The reference `nanochat.engine.Engine.generate` yields one token column at a
// time and keeps a per-row queue of forced tokens; a forced token is written
// with mask 0, like a prompt id, and is fed back through the KV cache. This
// header exposes the same mechanism on the batched engine without teaching the
// engine about the tokenizer; a future C ABI binding can hand it to the Python
// driver (`python/nanochat_cpp/chat.py`), which today steps the model itself.

namespace nanochat {

// The per-token tool-forcing hook. The engine calls `OnToken` once for every
// emitted token, in emission order. `sampled` is true for a token the model
// sampled and false for a token this hook forced earlier. `row` indexes the
// output rows. The hook appends forced ids to `*forced`; the engine drains
// them one per step, feeds each through the KV cache exactly like a sampled
// token, and records it with mask 0, so the cache position and the token count
// stay exact. `OnToken` returns false to stop the row after the current token.
// A hook must be prepared to be called again with the forced ids it returned.
class GenerateHook {
 public:
  virtual ~GenerateHook() = default;
  virtual bool OnToken(int row, int token, bool sampled,
                       std::vector<int>* forced) = 0;
};

// `GenerateBatch` with an optional forcing hook. When `hook` is null the rows
// and masks are byte-identical to a `GenerateBatch` call. When `hook` is
// non-null it may inject forced tokens after any emitted token; a forced token
// is written with mask 0 and is fed to the KV cache exactly like a sampled
// token, so the continuation sampled after a forced block matches a run that
// decoded the same ids by hand. `out` is resized to `params.num_samples` rows.
void GenerateBatchWithHook(Model* model, const int* prompt, int prompt_len,
                           const GenerateParams& params, GenerateHook* hook,
                           std::vector<GeneratedSequence>* out);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_GENERATE_H_
