"""``python -m nanochat_cpp.chat_eval``: a nanochat-compatible chat evaluation CLI.

Mirrors ``scripts/chat_eval.py`` through the process seam described in
``docs/eval.md`` section 5: Python owns the task datasets, prompt rendering,
answer extraction, code execution, and every metric decision, while the C++
binaries own the forward pass.

Two evaluation paths:

* **Categorical** (ARC-Easy, ARC-Challenge, MMLU) -- render the multiple-choice
  prompt with :func:`nanochat_cpp.tasks.render_mc`, tokenize it, cache the
  answer-letter token ids (asserting each is a single token), and call
  ``score_main`` (``ScoreBatch``) with the final prompt position as the focus
  and the letter ids as the focus set. The bridge argmaxes the focused logits
  and never moves a full ``(B, T, V)`` tensor.
* **Generative** (GSM8K, HumanEval) -- render the completion prompt, call
  ``generate_main`` (``GenerateBatch``), decode the completions in the bridge,
  then apply the reference criterion: the number after ``####`` for GSM8K, and
  for HumanEval the first code block plus the test harness run through the
  reference ``nanochat.execution.execute_code`` (fresh interpreter, rlimits,
  scrubbed environment, timeout) inside the resource sandbox.

When all five tasks ran, ChatCORE is the mean centered accuracy with baselines
ARC/MMLU ``0.25`` and GSM8K/HumanEval ``0.0``; the categorical subset is
reported separately, mirroring the post-training loop.

Example::

    tools/nanochat eval -- tools/nanochat_cpp chat_eval -i sft \\
        -a ARC-Easy,GSM8K --max-problems 20
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

from . import base_eval, eval_fixture, reference, tasks

# ---------------------------------------------------------------------------
# Task inventory and baselines (mirrors scripts/chat_eval.py)
# ---------------------------------------------------------------------------

ALL_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")
CATEGORICAL_TASKS = frozenset({"ARC-Easy", "ARC-Challenge", "MMLU"})
BASELINE_ACCURACIES = {
    "ARC-Easy": 0.25,       # multiple choice 1 of 4
    "ARC-Challenge": 0.25,  # multiple choice 1 of 4
    "MMLU": 0.25,           # multiple choice 1 of 4
    "GSM8K": 0.0,           # open-ended
    "HumanEval": 0.0,       # open-ended
}


def build_task(task_name: str) -> tasks.Task:
    """Construct the reference task object for ``task_name``.

    The subset/split choices match ``scripts/chat_eval.py`` exactly.
    """
    if task_name == "HumanEval":
        return tasks.HumanEval()
    if task_name == "MMLU":
        return tasks.MMLU(subset="all", split="test")
    if task_name == "ARC-Easy":
        return tasks.ARC(subset="ARC-Easy", split="test")
    if task_name == "ARC-Challenge":
        return tasks.ARC(subset="ARC-Challenge", split="test")
    if task_name == "GSM8K":
        return tasks.GSM8K(subset="main", split="test")
    raise SystemExit(f"unknown task: {task_name}")


def _parse_task_names(text: str | None) -> list[str]:
    """Split ``-a`` on ``|`` (reference) or ``,`` (convenience)."""
    if text is None:
        return list(ALL_TASKS)
    names = [name.strip() for name in re.split(r"[|,]", text) if name.strip()]
    invalid = [name for name in names if name not in ALL_TASKS]
    if invalid:
        raise SystemExit(f"unknown task(s) {invalid}; valid: {list(ALL_TASKS)}")
    return names


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _special_token_id(tokenizer, name: str) -> int:
    """The id of a special token, or ``-1`` when the tokenizer lacks it."""
    try:
        return tokenizer.encode_special(name)
    except Exception:  # noqa: BLE001 - older tokenizers may lack a token
        return -1


def _letter_token_ids(letters, tokenizer, cache: dict) -> list[int]:
    """Token ids of the answer letters, cached and asserted single-token.

    Mirrors the reference's ``letter_to_id_cache``: each letter must encode to
    exactly one id, and repeated letters (``A``/``B``/``C``/``D``) are only
    encoded once.
    """
    ids = []
    for letter in letters:
        if letter not in cache:
            encoded = tokenizer.encode(letter)
            assert len(encoded) == 1, "Each letter must be a single token"
            cache[letter] = encoded[0]
        ids.append(cache[letter])
    return ids


def _chunk_problems(problems, batch_size: int, max_tokens: int):
    """Yield problem batches bounded by ``batch_size`` and the padded budget.

    ``ScoreBatch`` materialises a ``batch * seq * padded_vocab`` logits buffer
    on the host, so the padded token count ``rows * seq`` is capped at
    ``max_tokens``. The decision is per problem, so the grouping only changes
    the order of independent forwards.
    """
    batch = []
    current_max = 0
    for item in problems:
        length = item[0].length
        new_max = max(current_max, length)
        if batch and (len(batch) + 1) * new_max > max_tokens:
            yield batch
            batch = []
            current_max = 0
            new_max = length
        if batch and len(batch) >= batch_size:
            yield batch
            batch = []
            current_max = 0
            new_max = length
        batch.append(item)
        current_max = new_max
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Generative evaluation (mirrors run_generative_eval)
# ---------------------------------------------------------------------------


def _generate(runners, spec, prompt_ids, max_tokens: int, num_samples: int,
              temperature: float, top_k: int, stop_id: int, seed: int,
              out_path: Path) -> list[list[int]]:
    """Tokenize-free generation: write the prompt ids and run ``generate_main``.

    The binary returns the effective prompt followed by the generated ids
    (excluding the terminal token), so the caller strips the prompt prefix.
    """
    prompt_file = out_path.with_suffix(".prompt")
    prompt_file.write_text(" ".join(str(i) for i in prompt_ids),
                           encoding="utf-8")
    arguments = [
        "--prompt-file", str(prompt_file),
        "--max-tokens", str(max_tokens),
        "--num-samples", str(num_samples),
        "--temperature", repr(float(temperature)),
        "--top-k", str(top_k),
        "--stop-id", str(stop_id),
        "--bos-id", "-1",
        "--seed", str(seed),
        "--out", str(out_path),
        "--model", str(spec.path),
    ] + spec.model_flags()
    runners.run("generate_main", arguments)
    return base_eval._parse_generation_file(out_path)


def run_generative_eval(task_object, tokenizer, spec, runners, num_samples,
                        max_new_tokens, temperature, top_k, max_problems,
                        work_dir: Path, seed: int, tag: str) -> float:
    """One problem at a time: sample, decode, and score (reference loop)."""
    num_problems = (len(task_object) if max_problems is None
                    else min(len(task_object), max_problems))
    assistant_end = _special_token_id(tokenizer, "<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()

    num_passed, total = 0, 0
    for i in range(num_problems):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        out_path = work_dir / f"gen_{tag}_{i}.txt"
        rows = _generate(runners, spec, encoded_prompt, max_new_tokens,
                         num_samples, temperature, top_k, assistant_end, seed,
                         out_path)
        prefix_length = len(encoded_prompt)
        completions = []
        for row in rows:
            generated = list(row[prefix_length:])
            # The reference engine also completes a row on a sampled BOS; the
            # C++ primitive stops only on the configured terminal id, so trim
            # the suffix the reference would never have produced.
            if bos in generated:
                generated = generated[:generated.index(bos)]
            completions.append(tokenizer.decode(generated))
        outcomes = [task_object.evaluate(conversation, completion)
                    for completion in completions]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)
        print(f"\r{tag} | {num_passed}/{total} "
              f"({100 * num_passed / total:.2f}%)", end="", flush=True)
    print()
    if total == 0:
        return float("nan")
    average = num_passed / total
    print(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# ---------------------------------------------------------------------------
# Categorical evaluation (mirrors run_categorical_eval)
# ---------------------------------------------------------------------------


def run_categorical_eval(task_object, tokenizer, spec, runners, batch_size,
                         max_problems, work_dir: Path, score_max_tokens: int,
                         tag: str) -> float:
    """Batch problems, score the focused answer letters, and argmax."""
    bos = tokenizer.get_bos_token_id()
    num_problems = (len(task_object) if max_problems is None
                    else min(len(task_object), max_problems))

    # A problem carries its ScoreBatch case, the conversation for evaluation,
    # and the ordered answer letters the focus ids align with.
    problems = []
    letter_to_id_cache: dict[str, int] = {}
    for i in range(num_problems):
        conversation = task_object[i]
        letters = conversation["letters"]
        letter_ids = _letter_token_ids(letters, tokenizer, letter_to_id_cache)
        prompt_ids = tokenizer.render_for_completion(conversation)
        case = eval_fixture.EvalCase(
            tokens=tuple(prompt_ids),
            start=0,
            end=0,
            focus_position=len(prompt_ids) - 1,
            focus_ids=tuple(letter_ids),
        )
        problems.append((case, conversation, letters))

    num_passed, total = 0, 0
    for chunk_index, batch in enumerate(
            _chunk_problems(problems, batch_size, score_max_tokens)):
        cases = [item[0] for item in batch]
        results = base_eval._score_cases(runners, spec, cases, bos, work_dir,
                                         f"{tag}_{chunk_index}")
        for (case, conversation, letters), result in zip(batch, results):
            focus_logits = list(result.focus_logits)
            assert len(focus_logits) == len(letters), (
                f"focus logits {len(focus_logits)} != letters {len(letters)}")
            best = max(range(len(focus_logits)),
                       key=lambda index: focus_logits[index])
            predicted_letter = letters[best]
            num_passed += int(task_object.evaluate(conversation,
                                                   predicted_letter))
            total += 1

    if total == 0:
        return float("nan")
    average = num_passed / total
    print(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# ---------------------------------------------------------------------------
# Dispatch and ChatCORE (mirrors run_chat_eval and the __main__ block)
# ---------------------------------------------------------------------------


def run_chat_eval(task_name, tokenizer, spec, runners, batch_size=1,
                  num_samples=1, max_new_tokens=512, temperature=0.0,
                  top_k=50, max_problems=None, work_dir: Path | None = None,
                  score_max_tokens: int = 4096, seed: int = 42) -> float:
    """Evaluate one task and return its accuracy."""
    if work_dir is None:
        raise SystemExit("run_chat_eval needs a work directory")
    task_object = build_task(task_name)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", task_name).strip("_").lower()
    if task_object.eval_type == "generative":
        return run_generative_eval(
            task_object, tokenizer, spec, runners, num_samples,
            max_new_tokens, temperature, top_k, max_problems, work_dir, seed,
            tag)
    if task_object.eval_type == "categorical":
        return run_categorical_eval(
            task_object, tokenizer, spec, runners, batch_size, max_problems,
            work_dir, score_max_tokens, tag)
    raise ValueError(
        f"Unsupported task evaluation type: {task_object.eval_type}")


def _centered_mean(results: dict, task_names) -> float:
    """Mean centered accuracy with the shared baselines (reference formula)."""
    names = list(task_names)
    total = 0.0
    for name in names:
        baseline = BASELINE_ACCURACIES[name]
        total += (results[name] - baseline) / (1.0 - baseline)
    return total / len(names)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.chat_eval",
        description="Chat evaluation (nanochat CLI): ARC, MMLU, GSM8K, "
                    "HumanEval, ChatCORE")
    # Task selection (mirrors scripts/chat_eval.py).
    parser.add_argument("-i", "--source", type=str, required=True,
                        help="source of the model: base|sft|rl")
    parser.add_argument("-a", "--task-name", type=str, default=None,
                        help="task name(s), separated by | or comma. Default = all")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="batch size for categorical evaluation")
    parser.add_argument("-g", "--model-tag", type=str, default=None,
                        help="model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None,
                        help="step to load")
    parser.add_argument("-x", "--max-problems", "--max-examples",
                        dest="max_problems", type=int, default=None,
                        help="max problems to evaluate")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    # Checkpoint selection (mirrors nanochat.checkpoint_manager).
    parser.add_argument("--model", type=str, default=None,
                        help="explicit .pt or NCHKPT01 checkpoint path")
    parser.add_argument("--meta", type=str, default=None,
                        help="explicit model metadata JSON sidecar")
    # Model config overrides (normally read from the metadata sidecar).
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--padded-vocab", type=int, default=None)
    parser.add_argument("--window-pattern", type=str, default=None)
    parser.add_argument("--rope-base", type=float, default=None)
    # Bridge-specific.
    parser.add_argument("--base-dir", type=str, default=None,
                        help="nanochat base directory override")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default=None,
                        help="C++ backend for the evaluation binaries")
    parser.add_argument("--profile", type=str, default="t2-parity",
                        help="sandbox profile for a launch outside the sandbox")
    parser.add_argument("--no-build", action="store_true",
                        help="skip the Bazel build and use the current binaries")
    parser.add_argument("--score-max-tokens", type=int, default=4096,
                        help="max padded tokens (rows * seq) per score_main "
                             "call; bounds the focused-logits workspace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned tasks and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_names = _parse_task_names(args.task_name)

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = str(Path(args.base_dir).resolve())

    if args.dry_run:
        print("[nanochat_cpp] chat eval dry run")
        print("[nanochat_cpp] tasks: " + ", ".join(task_names))
        print(f"[nanochat_cpp] source={args.source} model_tag={args.model_tag} "
              f"step={args.step} max_problems={args.max_problems}")
        print(f"[nanochat_cpp] backend={args.backend or 'autodetect'} "
              f"profile={args.profile}")
        return 0

    backend = args.backend or base_eval._autodetect_backend(args.device_type)
    cuda = backend == "cuda"
    repo_root = reference.repository_root()
    runners = base_eval.Runners(repo_root, cuda, args.profile,
                                build=not args.no_build)
    spec = base_eval.resolve_model(args)
    tokenizer = base_eval.load_tokenizer()
    if tokenizer.get_vocab_size() != int(spec.config["vocab_size"]):
        raise SystemExit(
            f"tokenizer vocab {tokenizer.get_vocab_size()} does not match "
            f"model vocab {spec.config['vocab_size']}")

    print(f"[nanochat_cpp] checkpoint: {spec.path}")
    print(f"[nanochat_cpp] source={spec.source} tag={spec.model_tag} "
          f"step={spec.step} backend={backend}")

    results: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="nanochat_chat_eval_") as tmp:
        work_dir = Path(tmp)
        for task_name in task_names:
            accuracy = run_chat_eval(
                task_name, tokenizer, spec, runners,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
                work_dir=work_dir,
                score_max_tokens=args.score_max_tokens,
                seed=args.seed,
            )
            results[task_name] = accuracy
            print(f"{task_name} accuracy: {100 * accuracy:.2f}%")

    # ChatCORE when every task ran; the categorical subset whenever its three
    # tasks ran (the post-training loop logs both).
    if all(task_name in results for task_name in ALL_TASKS):
        chatcore = _centered_mean(results, ALL_TASKS)
        print(f"ChatCORE metric: {chatcore:.4f}")
    if all(task_name in results for task_name in CATEGORICAL_TASKS):
        chatcore_cat = _centered_mean(results, sorted(CATEGORICAL_TASKS))
        print(f"ChatCORE_cat metric: {chatcore_cat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
