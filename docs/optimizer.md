# Optimizer

`Optimizer` holds the exact parameter grouping of nanochat's `setup_optimizer`:

- **AdamW groups**: lm_head, token embedding, value embeddings, resid/x0/smear/
  backout scalars.
- **Muon groups**: matrix parameters, grouped by shape and stacked.

Schedules (LR multiplier, Muon momentum, weight decay) live in `Scheduler`.
`AdamWUpdate` and `MuonUpdate` are the kernels; Muon's Polar Express iterations
use batched cuBLAS GEMMs.

The kernels themselves are declared in [kernels.md](kernels.md).
