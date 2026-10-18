#!/usr/bin/env python3
"""Generate grouped offline fixtures: api, numerics."""

from __future__ import annotations

import sys
import textwrap

_COMMANDS = ('api', 'numerics')
_USAGE = "usage: dump_runtime_fixtures.py {api|numerics} [options]"
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_USAGE)
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    _command = sys.argv[1]
    if _command not in _COMMANDS:
        print("unknown fixture kind: " + _command, file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)
else:
    _command = None

if _command == 'api':
    __doc__ = textwrap.dedent("""\
        Generate the Python-API fixture, ``tests/data/api_fixture.bin``.

        The in-process Python API must reproduce the C++ trainer and the C++
        evaluator. This script records the reference values for that check. It runs
        the committed CPU binaries through ``tools/nanochat``:

        * ``train_main`` trains a tiny model on the committed document set
          ``src/parquet/testdata/text.parquet`` with the committed tokenizer
          ``tests/data/loader_tokenizer.nctoken``, and writes a checkpoint;
        * ``eval_main`` loads that checkpoint and prints the bits-per-byte result.

        The script records the model config, the optimizer config, the scheduler
        config, the seed, the data paths, the per-step loss, and the bits-per-byte
        result. The container is the little-endian ``NANOEVL1`` record format owned by
        ``python/nanochat_cpp/eval_fixture.py``. Every record name starts with ``api/``
        so the generic record reader stays separate from the semantic evaluation
        reader.

        The fixture is deterministic. A fixed seed and one tokenizer thread remove the
        run-to-run variation. Two runs give the same file.

        The parameter trajectory and the synthetic optimizer check already live in
        ``tests/data/train_parity.bin``. This fixture records only the document-mode
        values, and it names that file in the ``api/data/train_parity`` record. The
        optimizer and scheduler values below copy the committed train-parity fixture,
        so the two files describe the same update rule.

        Run the generator from the repository root::

            tools/nanochat build //src:train_main //src:eval_main
            python3 tools/dump_runtime_fixtures.py api --out tests/data/api_fixture.bin

        Read the file with ``nanochat_cpp.eval_fixture.read`` and index the records by
        name. Do not call ``records_to_fixture``: the ``api/`` records are not the
        evaluation schema.

    """)
    import argparse
    import os
    import re
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # The repository root holds the `tools/` entry point and the `python/` package.
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "python"))

    from nanochat_cpp import eval_fixture as ef  # noqa: E402
    from nanochat_cpp.data import read_nctoken1  # noqa: E402

    # The fixture schema version, recorded as `api/version`.
    API_FIXTURE_VERSION = 1

    # A tiny, structurally complete model. It copies the committed train-parity
    # architecture. Two layers use grouped-query attention and a sliding-window
    # pattern. The tokenizer vocabulary has 487 tokens. The padded vocabulary is
    # the next multiple of 64.
    DEFAULT_SEED = 1234
    DEFAULT_LAYERS = 2
    DEFAULT_HEADS = 8
    DEFAULT_KV_HEADS = 2
    DEFAULT_HIDDEN = 32
    DEFAULT_SEQ = 8
    DEFAULT_BATCH = 2
    DEFAULT_GRAD_ACCUM = 1
    DEFAULT_STEPS = 4
    DEFAULT_EVAL_STEPS = 3
    DEFAULT_WINDOW = "SL"
    #: One encode worker keeps the token stream deterministic.
    DEFAULT_TOKENIZER_THREADS = 1
    #: The committed default encoded-document buffer.
    DEFAULT_DOCUMENT_BUFFER = 1000

    # The optimizer and scheduler values of the committed train-parity fixture.
    # tests/data/train_parity.bin already covers these values.
    OPTIMIZER = {
        "unembedding_lr": 0.004,
        "embedding_lr": 0.2,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
        "weight_decay": 0.0,
        "clip": 0.0,
        "adam_eps": 1e-4,
        "muon_ns_steps": 5,
        "muon_beta2": 0.9,
    }
    SCHEDULER = {
        "warmup_steps": 2,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.1,
        "weight_decay_base": 0.05,
        "muon_momentum_warmup_steps": 4.0,
        "muon_momentum_start": 0.85,
        "muon_momentum_peak": 0.97,
        "muon_momentum_final": 0.90,
    }

    # The recorded reference for the parameter trajectory.
    TRAIN_PARITY_PATH = "tests/data/train_parity.bin"

    #: One ``step NNNNNN | loss X | ...`` line from the training log.
    _STEP_LOSS = re.compile(
        r"^step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.eE+-]+)\s+\|")
    #: The training summary line.
    _FINAL_LOSS = re.compile(
        r"^train_main: final loss (?P<loss>[0-9.eE+-]+) at step (?P<step>\d+)",
        re.MULTILINE)
    #: One ``eval_main: <split> bpb X`` line from the evaluator.
    _BPB = re.compile(r"^eval_main: (?P<split>train|val) bpb (?P<bpb>[0-9.eE+-]+)",
                      re.MULTILINE)


    def _i32(values) -> bytes:
        values = [int(value) for value in values]
        return struct.pack("<" + "i" * len(values), *values)


    def _i64(values) -> bytes:
        values = [int(value) for value in values]
        return struct.pack("<" + "q" * len(values), *values)


    def _f32(values) -> bytes:
        values = [float(value) for value in values]
        return struct.pack("<" + "f" * len(values), *values)


    def _bytes_record(name: str, text: str) -> ef.Record:
        """A UTF-8 string record, used for the paths."""
        payload = text.encode("utf-8")
        return ef.Record(name, ef.DTYPE_UINT8, (len(payload),), payload)


    def _int_record(name: str, value: int) -> ef.Record:
        return ef.Record(name, ef.DTYPE_INT32, (1,), _i32([value]))


    def _float_record(name: str, value: float) -> ef.Record:
        return ef.Record(name, ef.DTYPE_FP32, (1,), _f32([value]))


    def _vector_record(name: str, values) -> ef.Record:
        values = [float(value) for value in values]
        return ef.Record(name, ef.DTYPE_FP32, (len(values),), _f32(values))


    def _padded_vocab(vocab_size: int, multiple: int = 64) -> int:
        return ((vocab_size + multiple - 1) // multiple) * multiple


    def _relative(repo_root: Path, path: Path) -> str:
        """A stable repository-relative path, or the absolute path as a fallback."""
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()


    def vocab_size(tokenizer_path: Path) -> int:
        """The token count of an ``NCTOKEN1`` artifact."""
        artifact = read_nctoken1(tokenizer_path)
        return 256 + len(artifact.merge_pairs) + len(artifact.special_tokens)


    def run_nanochat(repo_root: Path, arguments) -> subprocess.CompletedProcess:
        """Run one command through the single entry point and capture its output."""
        command = [str(repo_root / "tools" / "nanochat"), *arguments]
        result = subprocess.run(command, cwd=str(repo_root), capture_output=True,
                                text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(
                f"command failed ({result.returncode}): " + " ".join(command))
        return result


    def build_binaries(repo_root: Path, build: bool) -> None:
        if build:
            run_nanochat(repo_root, ["build", "//src:train_main", "//src:eval_main"])


    def binary_path(repo_root: Path, name: str) -> Path:
        path = repo_root / "bazel-bin" / "src" / name
        if not path.is_file():
            raise SystemExit(
                f"missing built binary {path}; run "
                "tools/nanochat build //src:train_main //src:eval_main")
        return path


    def model_flags(args: argparse.Namespace, vocab: int) -> list[str]:
        return [
            "--layers", str(args.layers),
            "--heads", str(args.heads),
            "--kv-heads", str(args.kv_heads),
            "--hidden", str(args.hidden),
            "--seq", str(args.seq),
            "--vocab", str(vocab),
            "--padded-vocab", str(_padded_vocab(vocab)),
            "--window-pattern", args.window,
        ]


    def optimizer_flags() -> list[str]:
        flags: list[str] = []
        for key in ("unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr",
                    "weight_decay", "clip", "adam_eps", "muon_ns_steps",
                    "muon_beta2"):
            flags += ["--" + key.replace("_", "-"), repr(OPTIMIZER[key])]
        return flags


    def scheduler_flags() -> list[str]:
        flags: list[str] = []
        for key in ("warmup_steps", "warmdown_ratio", "final_lr_frac",
                    "weight_decay_base", "muon_momentum_warmup_steps",
                    "muon_momentum_start", "muon_momentum_peak",
                    "muon_momentum_final"):
            flags += ["--" + key.replace("_", "-"), repr(SCHEDULER[key])]
        return flags


    def run_train(repo_root: Path, train_main: Path, parquet: Path,
                  tokenizer: Path, args: argparse.Namespace, vocab: int,
                  checkpoint: Path) -> tuple[list[tuple[int, float]], int, float]:
        """Train the tiny model and return the per-step losses and the final loss."""
        train_args = [
            "--train-parquet", str(parquet),
            "--tokenizer", str(tokenizer),
            "--tokenizer-threads", str(args.tokenizer_threads),
            "--buffer-docs", str(args.document_buffer),
            *model_flags(args, vocab),
            *optimizer_flags(),
            *scheduler_flags(),
            "--batch", str(args.batch),
            "--grad-accum", str(args.grad_accum),
            "--num-iterations", str(args.steps),
            "--log-every", "1",
            "--save-every", "0",
            "--eval-every", "0",
            "--seed", str(args.seed),
            "--checkpoint", str(checkpoint),
        ]
        result = run_nanochat(repo_root, ["run", "t0-cpu", "--", str(train_main),
                                          *train_args])
        steps: list[tuple[int, float]] = []
        for line in result.stdout.splitlines():
            step_match = _STEP_LOSS.match(line)
            if step_match:
                steps.append((int(step_match.group("step")),
                              float(step_match.group("loss"))))
        steps.sort()
        if len(steps) != args.steps:
            raise SystemExit(
                f"expected {args.steps} loss lines, found {len(steps)}")
        final_match = _FINAL_LOSS.search(result.stdout)
        if final_match is None:
            raise SystemExit("the training output has no final-loss line")
        final_step = int(final_match.group("step"))
        final_loss = float(final_match.group("loss"))
        return steps, final_step, final_loss


    def run_eval(repo_root: Path, eval_main: Path, parquet: Path,
                 tokenizer: Path, args: argparse.Namespace, vocab: int,
                 checkpoint: Path) -> dict[str, float]:
        """Evaluate the checkpoint and return the bits-per-byte for each split."""
        eval_args = [
            "--train-parquet", str(parquet),
            "--val-parquet", str(parquet),
            "--tokenizer", str(tokenizer),
            "--tokenizer-threads", str(args.tokenizer_threads),
            "--buffer-docs", str(args.document_buffer),
            *model_flags(args, vocab),
            "--batch", str(args.batch),
            "--steps", str(args.eval_steps),
            "--model", str(checkpoint),
            "--seed", str(args.seed),
        ]
        result = run_nanochat(repo_root, ["run", "t0-cpu", "--", str(eval_main),
                                          *eval_args])
        values: dict[str, float] = {}
        for line in result.stdout.splitlines():
            match = _BPB.match(line)
            if match:
                values[match.group("split")] = float(match.group("bpb"))
        missing = [split for split in ("train", "val") if split not in values]
        if missing:
            raise SystemExit(f"the evaluator output has no bpb for: {missing}")
        return values


    def build_records(args: argparse.Namespace, vocab: int,
                      steps: list[tuple[int, float]], final_step: int,
                      final_loss: float, bpb: dict[str, float],
                      train_parquet: str, val_parquet: str,
                      tokenizer: str) -> list[ef.Record]:
        records: list[ef.Record] = []
        # Header and identity.
        records.append(_int_record("api/version", API_FIXTURE_VERSION))
        records.append(ef.Record("api/seed", ef.DTYPE_INT64, (1,),
                                 _i64([args.seed])))
        # Data paths, repository-relative so the file is portable.
        records.append(_bytes_record("api/data/train_parquet", train_parquet))
        records.append(_bytes_record("api/data/val_parquet", val_parquet))
        records.append(_bytes_record("api/data/tokenizer", tokenizer))
        records.append(_bytes_record("api/data/train_parity", TRAIN_PARITY_PATH))
        # Model config.
        records.append(_int_record("api/config/layers", args.layers))
        records.append(_int_record("api/config/heads", args.heads))
        records.append(_int_record("api/config/kv_heads", args.kv_heads))
        records.append(_int_record("api/config/hidden", args.hidden))
        records.append(_int_record("api/config/seq", args.seq))
        records.append(_int_record("api/config/vocab", vocab))
        records.append(_int_record("api/config/padded_vocab",
                                   _padded_vocab(vocab)))
        records.append(_int_record("api/config/batch", args.batch))
        records.append(_int_record("api/config/grad_accum", args.grad_accum))
        records.append(_int_record("api/config/steps", args.steps))
        records.append(_bytes_record("api/config/window_pattern", args.window))
        records.append(_int_record("api/config/tokenizer_threads",
                                   args.tokenizer_threads))
        records.append(_int_record("api/config/document_buffer",
                                   args.document_buffer))
        # Optimizer config (the values of tests/data/train_parity.bin).
        for key in ("unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr",
                    "weight_decay", "clip", "adam_eps", "muon_beta2"):
            records.append(_float_record(f"api/config/opt/{key}",
                                         OPTIMIZER[key]))
        records.append(_int_record("api/config/opt/muon_ns_steps",
                                   OPTIMIZER["muon_ns_steps"]))
        # Scheduler config (the values of tests/data/train_parity.bin).
        records.append(_int_record("api/config/sched/warmup_steps",
                                   SCHEDULER["warmup_steps"]))
        for key in ("warmdown_ratio", "final_lr_frac", "weight_decay_base",
                    "muon_momentum_warmup_steps", "muon_momentum_start",
                    "muon_momentum_peak", "muon_momentum_final"):
            records.append(_float_record(f"api/config/sched/{key}",
                                         SCHEDULER[key]))
        # Training results.
        records.append(_int_record("api/train/steps", args.steps))
        records.append(_vector_record("api/train/loss",
                                      [loss for _, loss in steps]))
        records.append(_int_record("api/train/final_step", final_step))
        records.append(_float_record("api/train/final_loss", final_loss))
        # Evaluation results.
        records.append(_int_record("api/eval/steps", args.eval_steps))
        records.append(_float_record("api/eval/train/bpb", bpb["train"]))
        records.append(_float_record("api/eval/val/bpb", bpb["val"]))
        return records


    def parse_args(argv) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", default="tests/data/api_fixture.bin",
                            help="output fixture path")
        parser.add_argument("--parquet",
                            default="src/parquet/testdata/text.parquet",
                            help="the committed document set")
        parser.add_argument("--tokenizer",
                            default="tests/data/loader_tokenizer.nctoken",
                            help="the committed NCTOKEN1 artifact")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
        parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
        parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
        parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
        parser.add_argument("--window", default=DEFAULT_WINDOW)
        parser.add_argument("--tokenizer-threads", type=int,
                            default=DEFAULT_TOKENIZER_THREADS)
        parser.add_argument("--document-buffer", type=int,
                            default=DEFAULT_DOCUMENT_BUFFER)
        parser.add_argument("--no-build", action="store_true",
                            help="use the current bazel-bin binaries")
        return parser.parse_args(argv)


    def main(argv) -> int:
        args = parse_args(argv)
        # The fixture is the CPU reference. Keep the shared GPU free.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        repo_root = _REPO_ROOT
        parquet = (repo_root / args.parquet).resolve()
        tokenizer = (repo_root / args.tokenizer).resolve()
        out = Path(args.out or DEFAULT_OUT_BY_PRECISION[args.precision])
        if not out.is_absolute():
            out = repo_root / out
        for path, what in ((parquet, "parquet"), (tokenizer, "tokenizer")):
            if not path.is_file():
                raise SystemExit(f"missing {what} file: {path}")

        vocab = vocab_size(tokenizer)
        build_binaries(repo_root, build=not args.no_build)
        train_main = binary_path(repo_root, "train_main")
        eval_main = binary_path(repo_root, "eval_main")

        with tempfile.TemporaryDirectory(prefix="api_fixture_") as workdir:
            checkpoint = Path(workdir) / "model.nchkpt"
            steps, final_step, final_loss = run_train(
                repo_root, train_main, parquet, tokenizer, args, vocab, checkpoint)
            bpb = run_eval(repo_root, eval_main, parquet, tokenizer, args, vocab,
                           checkpoint)

        train_parquet = _relative(repo_root, parquet)
        tokenizer_path = _relative(repo_root, tokenizer)
        records = build_records(args, vocab, steps, final_step, final_loss, bpb,
                                train_parquet, train_parquet, tokenizer_path)
        ef.write(out, records)
        print(f"wrote {out} ({out.stat().st_size} bytes, "
              f"layers={args.layers}, heads={args.heads}, "
              f"kv_heads={args.kv_heads}, hidden={args.hidden}, seq={args.seq}, "
              f"vocab={vocab}, steps={args.steps}, seed={args.seed})")
        print("per-step loss: " + ", ".join(f"{loss:.6f}" for _, loss in steps))
        print("final loss: %.6f" % final_loss)
        print("bpb: train %.6f val %.6f" % (bpb["train"], bpb["val"]))
        return 0
    _selected_main = main

if _command == 'numerics':
    __doc__ = textwrap.dedent("""\
        Generate the staged numeric golden trace. The default output is
        ``tests/data/numerics_golden_10l.bin`` at fp32; ``--precision fp8`` writes
        ``tests/data/numerics_golden_10l_fp8.bin`` from the fp8 CPU build.

        The trace of ``docs/numerics-integration.md`` is one flat list of named records
        in the little-endian container that ``tests/oracle_fixture.h`` reads, under the
        magic ``NANONUM1``. The C++ emitter owns the exact bytes, because it computes
        the values; this script builds the emitter, runs it once through
        ``tools/nanochat`` under the ``t0-cpu`` sandbox profile, verifies the container,
        and moves the file into place. The emitter pins the host thread count to one, so
        the file does not depend on the sandbox thread budget.

        The fixture is the fp32 CPU reference of section 6.2: 10 layers, 6 heads, 3 KV
        heads, hidden dimension 384, sequence length 256, batch 4, vocabulary 512, the
        window pattern ``SSSL``, value embeddings on, seed 42, and five optimizer steps.
        The trace holds the training trajectory (loss, gradient norm, and the L2 norm
        and byte hash of every parameter and gradient at every step), the evaluation
        section (bits per byte, per-batch loss, logits hash, greedy argmax identifiers),
        the greedy generation section (identifiers, mask, and logit margins), the SFT
        arithmetic, and the RL arithmetic.

        Run the generator from the repository root::

            python3 tools/dump_runtime_fixtures.py numerics

        Then read the file with ``tests/numerics_trace_test.cc`` (the CPU and simulator
        legs compare byte for byte; the GPU leg compares within the tolerance table) or
        with ``read_records`` below.

    """)
    import argparse
    import hashlib
    import os
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # The repository root holds the `tools/` entry point and the emitter binary.
    _REPO_ROOT = Path(__file__).resolve().parents[1]

    # The fixture magic and version of the container.
    MAGIC = b"NANONUM1"
    VERSION = 1

    #: One element size per dtype code (`tests/oracle_fixture.h`).
    _ELEM_SIZE = {0: 4, 1: 4, 2: 8, 3: 8, 4: 1}

    # The fp32 golden keeps the original name. Each other precision has its own
    # file, because a bf16 or fp8 build cannot reproduce the fp32 golden
    # (docs/precision.md section 6).
    DEFAULT_OUT_BY_PRECISION = {
        "fp32": "tests/data/numerics_golden_10l.bin",
        "bf16": "tests/data/numerics_golden_10l_bf16.bin",
        "fp8": "tests/data/numerics_golden_10l_fp8.bin",
    }
    DEFAULT_TOKENIZER = "tests/data/loader_tokenizer.nctoken"
    EMIT_TARGET = "//tests:numerics_trace_emit"
    EMIT_BINARY = "bazel-bin/tests/numerics_trace_emit"


    def read_records(path: Path) -> list[tuple[str, int, tuple, bytes]]:
        """Reads the flat record container: name, dtype, shape, payload."""
        data = path.read_bytes()
        if data[:8] != MAGIC:
            raise SystemExit(f"{path}: bad magic {data[:8]!r}")
        version, count = struct.unpack_from("<II", data, 8)
        if version != VERSION:
            raise SystemExit(f"{path}: unsupported version {version}")
        offset = 16
        records = []
        for _ in range(count):
            (name_len,) = struct.unpack_from("<H", data, offset)
            offset += 2
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            dtype = data[offset]
            ndim = data[offset + 1]
            offset += 2
            shape = struct.unpack_from("<" + "q" * ndim, data, offset) if ndim else ()
            offset += 8 * ndim
            numel = 1
            for extent in shape:
                numel *= extent
            nbytes = numel * _ELEM_SIZE[dtype]
            payload = data[offset:offset + nbytes]
            offset += nbytes
            records.append((name, dtype, shape, payload))
        if offset != len(data):
            raise SystemExit(f"{path}: {len(data) - offset} trailing bytes")
        return records


    def scalar(values: dict[str, tuple], name: str) -> float:
        dtype, _shape, payload = values[name]
        if dtype == 0:
            return struct.unpack("<f", payload)[0]
        if dtype == 1:
            return struct.unpack("<i", payload)[0]
        if dtype == 3:
            return struct.unpack("<q", payload)[0]
        raise SystemExit(f"{name}: not a scalar record (dtype {dtype})")


    def run_nanochat(repo_root: Path, arguments) -> subprocess.CompletedProcess:
        """Runs one command through the single entry point and captures its output."""
        command = [str(repo_root / "tools" / "nanochat"), *arguments]
        result = subprocess.run(command, cwd=str(repo_root), capture_output=True,
                                text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(
                f"command failed ({result.returncode}): " + " ".join(command))
        return result


    def summarize(records, out: Path, expected_precision: str) -> None:
        """Prints the fixture summary and the checks a reviewer needs."""
        values = {name: (dtype, shape, payload)
                  for name, dtype, shape, payload in records}
        backend = values["header/backend"][2].decode()
        precision = values["header/precision"][2].decode()
        steps = int(scalar(values, "header/steps"))
        print(f"wrote {out} ({out.stat().st_size} bytes, {len(records)} records, "
              f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")
        print(f"identity: backend {backend}, precision {precision}, "
              f"seed {int(scalar(values, 'header/seed'))}, "
              f"parameters {int(scalar(values, 'config/parameters'))}")
        losses = [scalar(values, f"train/step{step}/loss") for step in range(steps + 1)]
        print("train loss: " + ", ".join(f"{loss:.6f}" for loss in losses))
        print("train gradient norm: %.6f" % scalar(values, "train/step1/grad_norm"))
        print("eval: bpb %.6f, loss %.6f, min logit margin %.6g"
              % (scalar(values, "eval/bpb"), scalar(values, "eval/loss/0"),
                 scalar(values, "eval/margin_min")))
        print("generation: min logit margin %.6g"
              % scalar(values, "gen/margin_min"))
        print("sft: loss %.6f, gradient norm %.6f, valid targets %d"
              % (scalar(values, "sft/loss"), scalar(values, "sft/grad_norm"),
                 int(scalar(values, "sft/valid_targets"))))
        print("rl: loss %.6g, gradient norm %.6g, mean reward %.6f, "
              "min logit margin %.6g"
              % (scalar(values, "rl/loss"), scalar(values, "rl/grad_norm"),
                 scalar(values, "rl/mean_reward"), scalar(values, "rl/margin_min")))
        for row in range(4):
            print("rl row %d: reward %.6f, advantage %.6g"
                  % (row, scalar(values, f"rl/reward/{row}"),
                     scalar(values, f"rl/advantage/{row}")))
        if backend != "cpu" or precision != expected_precision:
            raise SystemExit("the golden trace must come from the "
                             + expected_precision + " CPU build")


    def parse_args(argv) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", default=None,
                            help="output fixture path (default: the precision's name)")
        parser.add_argument("--precision", default="fp32",
                            choices=("fp32", "bf16", "fp8"),
                            help="the build precision of the golden")
        parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                            help="the committed NCTOKEN1 artifact of the eval phase")
        parser.add_argument("--no-build", action="store_true",
                            help="use the current bazel-bin emitter")
        return parser.parse_args(argv)


    def main(argv) -> int:
        args = parse_args(argv)
        # The fixture is the CPU reference. Keep the shared GPU free.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        repo_root = _REPO_ROOT
        out = Path(args.out)
        if not out.is_absolute():
            out = repo_root / out
        tokenizer = Path(args.tokenizer)
        if not tokenizer.is_absolute():
            tokenizer = repo_root / tokenizer
        if not tokenizer.is_file():
            raise SystemExit(f"missing tokenizer file: {tokenizer}")
        if not args.no_build:
            config = [] if args.precision == "fp32" else ["--config=" + args.precision]
            run_nanochat(repo_root, ["build", *config, EMIT_TARGET])
        emitter = repo_root / EMIT_BINARY
        if not emitter.is_file():
            raise SystemExit(f"missing emitter {emitter}; run "
                             f"tools/nanochat build {EMIT_TARGET}")

        with tempfile.TemporaryDirectory(prefix="numerics_golden_") as workdir:
            staged = Path(workdir) / "numerics_golden_10l.bin"
            result = run_nanochat(repo_root, [
                "run", "t0-cpu", "--", str(emitter),
                "--out", str(staged),
                "--tokenizer", str(tokenizer),
            ])
            sys.stdout.write(result.stdout)
            records = read_records(staged)
            out.parent.mkdir(parents=True, exist_ok=True)
            # The temporary directory may live on another filesystem, so stage
            # beside the target and replace inside one directory.
            beside = out.parent / (out.name + ".tmp")
            beside.write_bytes(staged.read_bytes())
            os.replace(beside, out)

        # Re-read the committed path, so the summary describes the file on disk.
        summarize(read_records(out), out, args.precision)
        return 0
    _selected_main = main

if __name__ == "__main__":
    raise SystemExit(_selected_main(sys.argv[2:]))
