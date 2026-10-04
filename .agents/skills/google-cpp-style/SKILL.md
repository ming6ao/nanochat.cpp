---
name: google-cpp-style
description: Apply the Google C++ Style Guide and the nanochat.cpp conventions to C++ and CUDA changes. Use when you edit a header, a source file, or a CUDA kernel, or before a commit or merge.
---

# Google C++ Style for nanochat.cpp

The project follows the [Google C++ Style
Guide](https://google.github.io/styleguide/cppguide.html). `CONTRIBUTING.md`
holds the project naming table. `.clang-format` is the machine-readable form.

## Run the gate

Run this command after every change:

```bash
tools/nanochat lint
```

It runs `clang-format` in check mode and the text checks. It exits non-zero on
a violation. To fix formatting, run `clang-format -i <file>`.

The bundled script `scripts/check.sh` calls the same gate.

## Rules the gate enforces

- Keep every line to 80 columns. Keep the tree `clang-format` clean.
- Do not write `using namespace`. Use a using-declaration or a namespace alias.
- Include a project header with a path from the repository root, for example
  `"src/ops.h"`. Do not rely on a bare file name.
- Use `static_cast`, not a C-style cast.
- Use ASCII characters in comments.

## Naming

- Types and functions: `PascalCase`.
- Variables and accessors: `snake_case`.
- Class members: `snake_case_` with a trailing underscore.
- Constants and enum values: `kPascalCase`.
- File names: `lower_snake.cc` and `lower_snake.h`.
- Macros: `NANOCHAT_*`.

## Project decisions

- Production code does not use exceptions or run-time type information. The
  tests are the current exception; see `docs/testing.md`.
- `src/data.cc` keeps one raw `new` because `TokenShard` has a private
  constructor. The comment records the reason.

## Read more

- `CONTRIBUTING.md` — the naming table and the Definition of Done.
- `docs/testing.md` — the test tiers and the Definition of Done.
- `docs/build.md` — the file layout.
