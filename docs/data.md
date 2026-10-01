# Data, checkpoint, oracle

- **Data**: pre-tokenized `.bin` shards — header (magic, version, `ntok`) plus a
  `uint16`/`uint32` token stream. The training loop does a `read()`. No
  tokenizer, parquet, pyarrow, or numpy at runtime.
- **Checkpoint**: a self-describing tensor container (name, shape, dtype). Not a
  framework `state_dict`.
- **Oracle**: `debug_state.bin`, produced offline by nanochat's `gpt.py`.
  `tests/oracle_test.cc` compares logits, loss, and gradients, then runs a few
  optimizer steps and matches losses. Per-backend tolerances (fp32 CUDA ~1e-5;
  fp16 Turing looser).

The oracle fixture is consumed by the test harness described in
[testing.md](testing.md).
