# Topology and the gradient sync

The engine models data parallel training. Every device holds the full model
and processes its own micro-batch. The devices all-reduce their gradients once
per optimizer step.

## What the topology changes

The topology changes one cost: the gradient reduction. It does not change the
per-device memory, because data parallel replicates the model.

The reduction moves the gradient buffers between the devices. A large model
sends more bytes. A slow link sends the bytes slower. Both lower the scaling
efficiency.

## The interconnect table

The bandwidth is per rank and unidirectional, in gigabytes per second
(1e9 bytes per second).

| Interconnect | Bandwidth | Typical use |
|---|---:|---|
| nvlink1 | 160 GB/s | P100, V100 |
| nvlink2 | 300 GB/s | V100, A100 pairs |
| nvlink3 | 600 GB/s | A100, H100 |
| nvlink4 | 900 GB/s | H100, H200 |
| pcie3 | 16 GB/s | Turing, Pascal |
| pcie4 | 32 GB/s | Ampere |
| pcie5 | 64 GB/s | Hopper, Blackwell |
| shared-memory | 20 GB/s | Two cards on one host, no peer access |
| ethernet-100g | 12.5 GB/s | A small cluster |
| ethernet-400g | 50 GB/s | A modern cluster |
| infiniband-ndr | 50 GB/s | A training cluster |

The two Kaggle T4 cards share a host and have no NVLink. Use `pcie3`, or
`shared-memory` if a measurement shows that peer access is off.

## The ring all-reduce model

A ring all-reduce moves this many bytes on each rank:

```text
moved = 2 * (world_size - 1) / world_size * payload_bytes
```

The payload is the parameter count times the gradient element size. The time
is:

```text
sync_seconds = (1 - overlap) * (moved / bandwidth + latency_us * 1e-6)
```

`overlap` is the fraction of the reduction that hides behind the backward
pass. nanochat.cpp reduces after the accumulation loop today, so the default
is 0.0. Gradient bucketing and a second stream raise it.

## The scaling efficiency

```text
efficiency = group_tokens_per_second / (world_size * one_device_rate)
```

The engine computes the one-device rate from the same model without the
reduction. The gap is the reduction.

The sync fraction follows `1 / tokens_per_step`. A long sequence or a large
micro-batch amortizes the reduction. A small micro-batch does not.

## What the model omits

- The optimizer step time. It is a memory-bound pass over the parameters.
- The data loader. A starved device shows up as idle time, not as a slow link.
- The kernel launch overhead. A small micro-batch pays it more often.
- The network congestion and the route. Use `--bandwidth-gbps` with a measured
  value.

The design is in `docs/distributed-design.md`. The execution plan is in
`docs/distributed-t4-plan.md`.
