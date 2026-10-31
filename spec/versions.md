# Contract versions

The contract has four independent version numbers. Each one covers a different
surface, each one moves on its own schedule, and none of them implies another.
A reader who needs to know whether two artifacts agree reads the number of the
contract that governs the thing they disagree about.

This file is the authority for the numbers and for the rules that move them.
The design is
[docs/model-and-device-bringup.md §10.3](../docs/model-and-device-bringup.md);
this file records the values and the change log, and it is where a contract
extension is written down.

`spec/` holds prose and data and no code. The entry-point inventory lives in
[docs/kernels.md](../docs/kernels.md) and is not restated here or in any other
file in this directory: this directory names contracts, `docs/` describes the
tree.

Related material:

- [capability_bits.md](capability_bits.md) — the capability vocabulary (one of
  the four contracts below).
- [block_kinds.md](block_kinds.md) — the block kind names, part of the spec
  format contract.
- [pointwise_ops.md](pointwise_ops.md) — the op-code table and the declared
  accuracy bounds, part of the kernel seam contract.
- [decisions.md](decisions.md) — why each of these numbers moved or did not.
- [../docs/checkpoint-format.md](../docs/checkpoint-format.md) — the
  `NCHKPT01` reader policy this file copies.

## 1. The four contracts

| Contract | The version word lives in | Authority for the values | What it covers |
|---|---|---|---|
| Spec format | `spec_version` in `models/<name>/spec.json`; the field list is [docs/model-and-device-bringup.md §8](../docs/model-and-device-bringup.md) | this file | The closed `ModelSpec` field list, the block kind names of [block_kinds.md](block_kinds.md), and the meaning of each field |
| Capability vocabulary | [capability_bits.md](capability_bits.md) (the version row at the top of that file) | this file | The bit names the two axes declare and negotiate. The numeric bit *values* live in `include/nanochat/tensor.h` and are not a contract: a program never carries them across a build |
| Kernel seam | The header comment preamble of `include/nanochat/kernels.h` | this file | The entry points, their parameter blocks, the closed `PointwiseOp` code list, and the declared accuracy bound per code in [pointwise_ops.md](pointwise_ops.md) |
| Fixture format | The `version` word of the record header, written through `src/record_file.h` and read by `tests/oracle_fixture.h` | this file | The record encoding (name, dtype, shape, payload), the record header layout, and the meaning of a record name. It does not cover the set of magic words: a magic names a container, and the version word describes the bytes inside one container |

The four are independent on purpose. Adding a block kind moves the spec format
and leaves the seam alone. Adding an op code moves the seam and leaves the spec
format alone. A model package pinned at spec format 2 runs on a device pinned
at seam 1 without either side learning the other's number: the two vocabularies
meet only through [capability_bits.md](capability_bits.md).

## 2. Current versions

| Contract | Current | Previous | Reader acceptance |
|---|---|---|---|
| Spec format | 1 | — (no earlier number was ever written) | A reader accepts spec format 1 and refuses any other number with a named error. There is no previous version to accept yet, because `models/nanochat-default/spec.json` is the first file in the format |
| Capability vocabulary | 1 | — (the boolean fields on `Caps` carried no number) | A backend declares vocabulary 1 in its `caps.json`. A declaration with a number this build does not know is rejected by name, not by guessing |
| Kernel seam | 1 | 0 | A backend accepts seam 1 and seam 0. Seam 1 is the numbered header; seam 0 is the unnumbered header that preceded it, whose entry-point set is the inventory in [docs/kernels.md](../docs/kernels.md) before this campaign's addition |
| Fixture format | 1 | — (version 1 is the only number ever written) | A reader accepts fixture format 1. The first bump makes 1 the previous version and the acceptance set becomes {2, 1} |

**The seam version this campaign bumps is 1.** The header preamble of
`include/nanochat/kernels.h` states the version, and the one bump covers the
whole contract extension rather than one bump per edit. The required preamble
statement is:

```text
Seam version: 1
Reads: 1, 0
```

The two new reduction entry points, the `PointwiseParams` block, the four new
op codes, and the accuracy-bound obligation of
[pointwise_ops.md](pointwise_ops.md) are all seam-1 content. A reader of seam 0
is the header before this campaign; a backend that still implements only seam 0
entry points is a seam-0 backend and must not claim seam 1. Nothing in the tree
declares seam 0 after this campaign, so the acceptance set is stated for the
benefit of an out-of-tree backend and of a downstream fork, exactly as
[docs/checkpoint-format.md](../docs/checkpoint-format.md) keeps accepting
`NCHKPT01` version 0 after every in-tree writer moved to version 1.

## 3. The rules

Four rules govern every one of the four contracts. They are the rules of
[docs/model-and-device-bringup.md §10.3](../docs/model-and-device-bringup.md)
written out for a reviewer to cite.

### 3.1 Additions are free

A new field, a new block kind, a new bit, a new entry point, a new op code, or a
new magic word costs no deprecation release and no migration. It costs one
thing: the new name must have a stated meaning that the existing names cannot
express, and a reader must be able to see the name and refuse it by name when it
does not implement it.

An addition raises the version number (rule 4) but breaks nothing, because
forward compatibility is refused on purpose. A reader that meets a spec field it
does not know, a block kind it does not implement, a bit it does not define, or
an op code it does not compile rejects it with the name in the message. It never
drops the unknown field, never rounds the unknown kind to a default, and never
treats an unknown code as the nearest one it has: a silence is worse than a
rejection, because a silence produces a run-time lie.

### 3.2 A removal needs one deprecation release

To retire a name — a block kind, a bit, an op code, an entry point, a fixture
record, or a field:

1. The release that announces it keeps the name working, marks it deprecated in
   the authority file, names its replacement, and raises the contract version.
2. The next release removes it and raises the version again.

During the announcing release readers accept both the current version and the
previous one (rule 4), so a file, a backend, or a fixture produced before the
announcement still reads. A removal that skips the announcement release is a
breaking change to every model package and every backend at once, and it is
rejected in review for that reason alone.

A retired name is never reused for a different meaning. It stays burnt, like a
retired syscall number.

### 3.3 A meaning change needs a new name

If the computation behind a name changes, the name changes. A block kind that
computes something else is a new block kind. An op code with a different
arithmetic is a new op code. A capability bit with a different obligation is a
new bit. A field whose units or interpretation change is a new field.

The reason is the golden trace. A fixture, a tolerance, and a plan all name a
meaning. Re-pointing a name at a new meaning silently invalidates every fixture
that used it, and it turns a numeric regression into an argument about which
side is wrong. A new name makes the old fixtures refuse to load rather than
compare against the wrong arithmetic.

The old name then follows rule 3.2 to its retirement, or it stays in place with
its original meaning, which is what a compatibility shim looks like here.

### 3.4 A version rises only with a change a reader can detect

A version number is not a commit counter. It moves when the input space of a
reader changed in a way the reader can observe, and it does not move otherwise.
A re-worded comment, a typo fix, a reordered table, a file move, or a new
paragraph of prose moves nothing. A new name, a retired name, a changed shape, a
changed meaning, or a changed numeric obligation moves the number of the
contract that owns it.

## 4. What a reader can detect, per contract

| Contract | Rises on | Does not rise on |
|---|---|---|
| Spec format | A field appears; a field retires; a field's meaning or units change (which also needs a new name, rule 3.3); a block kind name appears or retires | A change to the *value* of a field inside one model package, which is that model's data and not the format; a tolerance value, which is the model's numeric contract; a re-wording in [block_kinds.md](block_kinds.md) that leaves every meaning identical |
| Capability vocabulary | A bit appears or retires; a bit's obligation changes (new name, rule 3.3) | A bit *value* changing in `tensor.h`, because no program carries a value across a build and the mask is never persisted; a `caps.json` declaring fewer bits than before, which is that device's declaration |
| Kernel seam | An entry point appears, changes its parameter block, or retires; an op code appears or retires; a declared accuracy bound in [pointwise_ops.md](pointwise_ops.md) changes | A comment change; a backend's implementation change that honours the same signature and bound; a per-device choice of tile size, which no caller can see |
| Fixture format | The record header layout changes; the record encoding changes; a record name's meaning changes; a magic word's container gains a field | A new magic word for a new container, because the version word describes the bytes inside one container and not the set of containers; new records added to an existing container; a regenerated payload whose values moved inside tolerance |

Two consequences worth stating because reviewers ask:

- A loosening of an accuracy bound or of a tolerance is a detectable change, so
  it rises the version of the contract that owns it, and a tolerance loosening
  additionally needs the contract-owner review of
  [docs/model-and-device-bringup.md §11](../docs/model-and-device-bringup.md).
  A loosening that moves no version number and cites no review is the
  self-certification the isolation gate exists to prevent.
- A tightening of an accuracy bound is free (rule 3.1) and still rises the
  version, because a reader can detect it: a tighter bound is what will fail a
  device that used to pass, so the number must move for the failure to be
  diagnosable.

## 5. The precedent: `NCHKPT01`

[docs/checkpoint-format.md](../docs/checkpoint-format.md) already runs this
policy on a real container, and every one of the four contracts above copies it:

> One format. Magic `NCHKPT01`. Writers write version 1; readers accept version
> 0 and version 1.

> The `version` word is the only difference between the writers; they all use
> the one container.

> `NCHKPT01` is the only container the project writes, and the `version` word
> says which layout inside it a file carries […] Reads accept both versions.

Three properties carry over unchanged:

- **Writers write the current version and only the current version.** There is
  no compatibility writer and no version-selecting flag. A model package writes
  the current spec format, a backend declares the current vocabulary, and the
  fixture generator writes the current fixture format.
- **Readers accept the current version and the previous one.** The acceptance
  set is written down beside the number, in the row of §2, and it shrinks from
  the bottom one release after each bump. This is what makes rule 3.2's
  deprecation window real rather than ceremonial.
- **The number is a word in the data, not in a build setting.** A compiler flag
  cannot tell a reviewer which contract a fixture was produced under. The seam
  version is the one exception, and it is in a comment in the header precisely
  so that it is read by a human and not by a preprocessor.

## 6. Change log

One row per version movement. A row names the phase of
[docs/model-and-device-bringup.md §16](../docs/model-and-device-bringup.md), the
contract, the movement, the change, and the decision record in
[decisions.md](decisions.md) that justifies it. A movement with no record is a
movement that was not reviewed.

| Phase | Contract | Movement | Change | Record |
|---|---|---|---|---|
| before P0 | Fixture format | — → 1 | The record encoding, its version word, and the four magic words (`NANOORC1`, `NANOEVL1`, `NANOTOK1`, `NANONUM1`) | none: predates this campaign |
| P0 | Kernel seam | 0 → 1 | `RowReduce` and `SpanReduce` added; `PointwiseForward`/`PointwiseBackward` take a `PointwiseParams` block instead of a scalar argument list; `kExp`, `kRsqrt`, `kLog`, `kRowScale` added to the closed `PointwiseOp`; the accuracy bound per code becomes a seam obligation | D-001, D-002, D-003 |
| P0 | Capability vocabulary | — → 1 | The fifteen bit names of [capability_bits.md](capability_bits.md) replace the ad-hoc booleans on `Caps` | §9 rules 1 to 3: the mask, the declaration, the named rejection |
| P2 | Spec format | — → 1 | The first `models/nanochat-default/spec.json`: the closed `ModelSpec` field list (§8) and the block kind names of [block_kinds.md](block_kinds.md) | §8, and D-009 for where a kind's implementation lives |
| — | Fixture format | 1 (unchanged) | This campaign changes no fixture byte. If a reader change needs a bump, the Oracle node implements the current-and-previous acceptance and reports the row for §2 and §6 rather than editing this directory | — |

The seam row is deliberately one row, not two. §10.3 of the design puts one
number on the whole seam, so the reductions and the pointwise change land under
one bump; a second bump for the second half of the same contract change would
tell a backend author to do two migrations where there is one.

## 7. What this file does not do

- It does not list entry points. That inventory is
  [docs/kernels.md](../docs/kernels.md); a second copy here is the drift
  [docs/model-and-device-bringup.md §17.1](../docs/model-and-device-bringup.md)
  warns about.
- It does not hold the bit meanings. Those are in
  [capability_bits.md](capability_bits.md), one table per bit, because a bit's
  obligation is read per bit and not in a version table.
- It does not hold the decision rationale. Each movement above cites a record in
  [decisions.md](decisions.md), and the record holds the question, the options,
  the evidence, and the phase.
- It does not name a model or a device. The contract does not know either axis
  (`docs/model-and-device-bringup.md` §4, Rule 1).
