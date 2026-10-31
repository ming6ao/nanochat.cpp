// src/fields_test.cc -- the field visitor of `nanochat/fields.h`
// (docs/checkpoint-metadata-and-examples-plan.md stage A.1).
//
// This is the port of `gemma.cpp` `io/fields_test.cc`, and it keeps the case
// table of the upstream suite: the round trip, new code reading old data, old
// code reading new data, an invalid float, an invalid string, back-to-back
// writes, the vector cap, and enum validity.
//
// The cross-version cases use the `OldFields`/`NewFields` trick: two structs
// share a prefix of fields and the newer one appends fields last, which is the
// only change the format allows (rule 1 in `nanochat/checkpoint_metadata.h`).
// The old reader then counts the fields it did not see and the new reader
// counts the words it did not read, and neither is an error.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "nanochat/checkpoint_metadata.h"
#include "nanochat/fields.h"
#include "src/recipe.h"

namespace {

using nanochat::CheckpointMetadata;
using nanochat::ComputePrecision;
using nanochat::ConfigBlock;
using nanochat::IFields;
using nanochat::IFieldsVisitor;
using nanochat::Recipe;
using nanochat::Stage;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

// --- The cross-version structs -------------------------------------------
//
// `OldFields` is the older code's view of the block. `NewFields` appends
// `u64` and `notes` last, which is the one change the format allows: no field
// is renamed, reordered, or removed.

struct OldFields : public IFields {
  const char* Name() const override { return "OldFields"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(u32);
    visitor(i32);
    visitor(text);
    visitor(flag);
  }

  std::uint32_t u32 = 7;
  std::int32_t i32 = -9;
  std::string text = "old";
  bool flag = true;
};

struct NewFields : public IFields {
  const char* Name() const override { return "NewFields"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(u32);
    visitor(i32);
    visitor(text);
    visitor(flag);
    // New in this version, and therefore last.
    visitor(u64);
    visitor(notes);
  }

  std::uint32_t u32 = 0;
  std::int32_t i32 = 0;
  std::string text;
  bool flag = false;
  std::uint64_t u64 = 0;
  std::vector<std::string> notes;
};

// --- One-field structs, so a test can hand the reader one crafted word ------

struct U32Field : public IFields {
  const char* Name() const override { return "U32Field"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  std::uint32_t value = 0;
};

struct FloatField : public IFields {
  const char* Name() const override { return "FloatField"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  float value = 0.0f;
};

struct BoolField : public IFields {
  const char* Name() const override { return "BoolField"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  bool value = false;
};

struct StringField : public IFields {
  const char* Name() const override { return "StringField"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  std::string value;
};

struct VectorField : public IFields {
  const char* Name() const override { return "VectorField"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  std::vector<std::uint32_t> value;
};

// An enum the visitor can range-check. `EnumValid` is found by
// argument-dependent lookup, so it lives in the enum's namespace, which is the
// rule `nanochat/fields.h` states for every enum a block carries.
enum class Shape {
  kSquare = 0,
  kCircle = 1,
};

bool EnumValid(Shape shape) {
  return shape == Shape::kSquare || shape == Shape::kCircle;
}

struct EnumField : public IFields {
  const char* Name() const override { return "EnumField"; }
  void VisitFields(IFieldsVisitor& visitor) override { visitor(value); }
  Shape value = Shape::kSquare;
};

// --- Nested blocks ------------------------------------------------------

// A sized nested block: one size word and then its payload.
struct Inner : public IFields {
  const char* Name() const override { return "Inner"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(value);
    visitor(text);
  }

  std::uint32_t value = 3;
  std::string text = "inner";
};

struct Outer : public IFields {
  const char* Name() const override { return "Outer"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(x);
    visitor(y);
    visitor(inner);  // Appended last, so an older reader sees `x` and `y`.
  }

  std::uint32_t x = 1;
  std::uint32_t y = 2;
  Inner inner;
};

// The older view of `Outer`: no nested block, and nothing after it.
struct OuterOld : public IFields {
  const char* Name() const override { return "OuterOld"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(x);
    visitor(y);
  }

  std::uint32_t x = 0;
  std::uint32_t y = 0;
};

// The older view of the metadata block: the fields that existed before
// `tokenizer_id`, `git_revision`, `created_utc`, and `notes` were appended. The
// config block is one nested block, so this view skips a changed config whole
// instead of misreading it.
struct OldMetadataView : public IFields {
  const char* Name() const override { return "OldMetadataView"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    if (fields == nullptr) return;
    visitor(fields->schema);
    visitor(fields->stage);
    visitor(fields->step);
    visitor(fields->world_size);
    visitor(fields->precision);
    ConfigBlock config_block(fields->config);
    visitor(config_block);
  }

  CheckpointMetadata* fields = nullptr;
};

// --- Cases --------------------------------------------------------------

void TestRoundTrip() {
  NewFields written;
  written.u32 = 0xdeadbeefu;
  written.i32 = -123456;
  written.text = "hello, fields";
  written.flag = false;
  // Both words are nonzero, so the low-word-first order is checked.
  written.u64 = 0x123456789ull;
  written.notes = {"alpha", "beta gamma"};

  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("round trip: Write produced no words");
    return;
  }

  NewFields read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("round trip: Read failed");
  Expect(result.missing_fields == 0, "round trip: missing_fields is not zero");
  Expect(result.extra_u32 == 0, "round trip: extra_u32 is not zero");
  Expect(result.pos == words.size(),
         "round trip: pos did not consume the block");
  Expect(read.u32 == written.u32, "round trip: u32 changed");
  Expect(read.i32 == written.i32, "round trip: i32 changed");
  Expect(read.text == written.text, "round trip: text changed");
  Expect(read.flag == written.flag, "round trip: bool changed");
  Expect(read.u64 == written.u64, "round trip: u64 changed");
  Expect(read.notes == written.notes, "round trip: notes changed");
}

// New code reading old data: the fields the data does not carry keep their
// member initializer and are counted, which is valid (design decision D5).
void TestNewCodeOldData() {
  OldFields written;
  written.u32 = 42;
  written.i32 = -7;
  written.text = "older";
  written.flag = false;
  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("new code, old data: Write produced no words");
    return;
  }

  NewFields read;
  read.u64 = 99;             // An initializer that must survive the read.
  read.notes = {"untouched"};
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("new code, old data: Read failed");
  Expect(result.missing_fields == 2,
         "new code, old data: missing_fields is not 2");
  Expect(result.extra_u32 == 0, "new code, old data: extra_u32 is not zero");
  Expect(read.u32 == written.u32, "new code, old data: u32 changed");
  Expect(read.i32 == written.i32, "new code, old data: i32 changed");
  Expect(read.text == written.text, "new code, old data: text changed");
  Expect(read.flag == written.flag, "new code, old data: bool changed");
  Expect(read.u64 == 99, "new code, old data: the u64 initializer was lost");
  Expect(read.notes == std::vector<std::string>({"untouched"}),
         "new code, old data: the notes initializer was lost");
}

// Old code reading new data: the trailing words are counted and skipped, which
// is valid (design decision D6).
void TestOldCodeNewData() {
  NewFields written;
  written.u32 = 5;
  written.i32 = -5;
  written.text = "newer";
  written.flag = true;
  written.u64 = 0x100000001ull;
  written.notes = {"x"};
  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("old code, new data: Write produced no words");
    return;
  }

  OldFields read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("old code, new data: Read failed");
  Expect(result.missing_fields == 0,
         "old code, new data: missing_fields is not zero");
  // u64 takes two words, and notes takes the count word plus one word per
  // string: one count word, plus "x" in one word.
  Expect(result.extra_u32 == 5, "old code, new data: extra_u32 is not 5");
  Expect(read.u32 == written.u32, "old code, new data: u32 changed");
  Expect(read.i32 == written.i32, "old code, new data: i32 changed");
  Expect(read.text == written.text, "old code, new data: text changed");
  Expect(read.flag == written.flag, "old code, new data: bool changed");
}

// A NaN or an infinity is a bad value: the read fails and nothing is loaded
// (design decision D7).
void TestInvalidFloat() {
  FloatField written;
  written.value = 1.5f;
  const std::vector<std::uint32_t> good = written.Write();
  if (good.size() != 1) {
    Fail("invalid float: the block is not one word");
    return;
  }
  const std::uint32_t patterns[] = {
      0x7fc00000u,  // quiet NaN
      0x7f800000u,  // +infinity
      0xff800000u,  // -infinity
  };
  for (std::uint32_t pattern : patterns) {
    const std::vector<std::uint32_t> bad = {pattern};
    FloatField read;
    read.value = 2.5f;
    const IFields::ReadResult result = read.Read(bad, 0);
    Expect(result.pos == 0, "invalid float: the read did not fail");
    Expect(read.value == 2.5f, "invalid float: the member was overwritten");
  }
  // The writer refuses to produce such a word in the first place.
  FloatField not_a_number;
  not_a_number.value = std::numeric_limits<float>::quiet_NaN();
  Expect(not_a_number.Write().empty(),
         "invalid float: the writer accepted a NaN");
}

// A string longer than the cap, and a string whose characters the block does
// not carry, are both hard errors (rule 3).
void TestInvalidString() {
  const std::uint32_t cap_chars = nanochat::kMaxStringWords * 4;
  {
    // One character past the cap.
    const std::vector<std::uint32_t> words = {cap_chars + 1};
    StringField read;
    Expect(read.Read(words, 0).pos == 0,
           "invalid string: a length above the cap was accepted");
  }
  {
    // The length word promises two words of characters and the block holds
    // none.
    const std::vector<std::uint32_t> words = {8};
    StringField read;
    Expect(read.Read(words, 0).pos == 0,
           "invalid string: a truncated string was accepted");
  }
  {
    // The cap itself is representable, so it round-trips.
    StringField written;
    written.value.assign(cap_chars, 'a');
    const std::vector<std::uint32_t> words = written.Write();
    Expect(words.size() ==
           static_cast<std::size_t>(nanochat::kMaxStringWords) + 1,
           "invalid string: the cap does not fit its words");
    StringField read;
    Expect(read.Read(words, 0).pos != 0,
           "invalid string: a string at the cap did not load");
    Expect(read.value == written.value,
           "invalid string: a string at the cap changed");
  }
  {
    // One character past the cap cannot be written.
    StringField written;
    written.value.assign(cap_chars + 1, 'a');
    Expect(written.Write().empty(),
           "invalid string: the writer accepted a string above the cap");
  }
}

// Two blocks append back to back, and one stream reads them in order.
void TestMultipleWrite() {
  OldFields first;
  first.u32 = 11;
  first.i32 = -11;
  first.text = "first";
  first.flag = true;
  OldFields second;
  second.u32 = 22;
  second.i32 = -22;
  second.text = "second";
  second.flag = false;

  std::vector<std::uint32_t> storage;
  if (!first.AppendTo(storage) || !second.AppendTo(storage)) {
    Fail("multiple write: AppendTo failed");
    return;
  }
  const std::size_t first_words = first.Write().size();
  Expect(storage.size() == first_words + second.Write().size(),
         "multiple write: the two blocks are not back to back");

  OldFields read_first;
  const IFields::ReadResult first_result = read_first.Read(storage, 0);
  if (first_result.pos == 0) {
    Fail("multiple write: the first block did not load");
    return;
  }
  Expect(read_first.u32 == first.u32 && read_first.text == first.text,
         "multiple write: the first block changed");

  OldFields read_second;
  const IFields::ReadResult second_result =
      read_second.Read(storage, first_result.pos);
  if (second_result.pos == 0) {
    Fail("multiple write: the second block did not load");
    return;
  }
  Expect(read_second.u32 == second.u32 && read_second.text == second.text,
         "multiple write: the second block changed");
  Expect(second_result.pos == storage.size(),
         "multiple write: the second read did not reach the end");
}

// The vector cap holds in both directions (rule 3).
void TestVectorCap() {
  const std::uint32_t cap = nanochat::kMaxVectorEntries;
  {
    VectorField written;
    written.value.assign(cap, 0);
    written.value[cap - 1] = 9;
    const std::vector<std::uint32_t> words = written.Write();
    Expect(words.size() == static_cast<std::size_t>(cap) + 1,
           "vector cap: the cap does not fit its words");
    VectorField read;
    Expect(read.Read(words, 0).pos != 0,
           "vector cap: a vector at the cap did not load");
    Expect(read.value.size() == cap, "vector cap: the size changed");
    Expect(!read.value.empty() && read.value[cap - 1] == 9,
           "vector cap: an element changed");
  }
  {
    VectorField written;
    written.value.assign(cap + 1, 0);
    std::vector<std::uint32_t> storage;
    Expect(!written.AppendTo(storage),
           "vector cap: the writer accepted a vector above the cap");
    Expect(storage.empty(), "vector cap: a failed append left a partial block");
  }
  {
    const std::vector<std::uint32_t> words = {cap + 1};
    VectorField read;
    Expect(read.Read(words, 0).pos == 0,
           "vector cap: a count above the cap was accepted");
  }
}

// An enum is a word, and only a value `EnumValid` knows is accepted (rule 2).
void TestEnumValidity() {
  EnumField written;
  written.value = Shape::kCircle;
  const std::vector<std::uint32_t> words = written.Write();
  if (words.size() != 1) {
    Fail("enum: the block is not one word");
    return;
  }
  EnumField read;
  Expect(read.Read(words, 0).pos != 0, "enum: a valid value did not load");
  Expect(read.value == Shape::kCircle, "enum: the value changed");

  const std::vector<std::uint32_t> unknown = {5};
  EnumField invalid;
  invalid.value = Shape::kSquare;
  Expect(invalid.Read(unknown, 0).pos == 0,
         "enum: an unknown value was accepted");
  Expect(invalid.value == Shape::kSquare,
         "enum: the member was overwritten by a bad value");
}

// A bool accepts only 0 and 1.
void TestBoolValidity() {
  const std::vector<std::uint32_t> truthy = {1};
  BoolField read;
  Expect(read.Read(truthy, 0).pos != 0, "bool: the value 1 did not load");
  Expect(read.value, "bool: the value 1 did not set the field");

  const std::vector<std::uint32_t> invalid = {2};
  BoolField bad;
  Expect(bad.Read(invalid, 0).pos == 0, "bool: the value 2 was accepted");
}

// A nested block carries its size, so an older reader skips it whole.
void TestNestedBlock() {
  Outer written;
  written.x = 100;
  written.y = 200;
  written.inner.value = 300;
  written.inner.text = "inner";
  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("nested block: Write produced no words");
    return;
  }

  Outer read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("nested block: Read failed");
  Expect(result.missing_fields == 0 && result.extra_u32 == 0,
         "nested block: a counter is not zero");
  Expect(read.x == 100 && read.y == 200, "nested block: a scalar changed");
  Expect(read.inner.value == 300 && read.inner.text == "inner",
         "nested block: the payload changed");

  // The older code sees the same prefix and counts the block it does not know:
  // the size word plus the four payload words (the number and "inner", which
  // needs a length word and two words of characters).
  OuterOld old;
  const IFields::ReadResult old_result = old.Read(words, 0);
  if (old_result.pos == 0) Fail("nested block: the old view failed");
  Expect(old_result.extra_u32 == 5,
         "nested block: the old view did not count the block");
  Expect(old.x == 100 && old.y == 200, "nested block: the old view changed");

  // A block that claims more words than remain is a corrupted stream.
  const std::vector<std::uint32_t> too_big = {1, 2, 9, 0};
  Outer bad;
  Expect(bad.Read(too_big, 0).pos == 0,
         "nested block: an oversized block was accepted");
}

// The real metadata block round-trips, and an older view of it reads the
// prefix and counts the tail (docs/checkpoint-metadata-and-examples-plan.md
// sections 3 and 4).
void TestMetadataBlock() {
  CheckpointMetadata written;
  written.stage = Stage::kRl;
  written.step = 0x123456789ull;
  written.world_size = 8;
  written.precision = ComputePrecision::kFp8;
  written.config.num_layers = 4;
  written.config.window_pattern = "SL";
  written.tokenizer_id = "nanochat-32k";
  written.git_revision = "abcdef0";
  written.notes = {"rl", "from base"};

  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("metadata: Write produced no words");
    return;
  }
  Expect(words.size() <= nanochat::kMaxBlockWords,
         "metadata: the block is above the cap");

  CheckpointMetadata read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("metadata: Read failed");
  Expect(result.missing_fields == 0 && result.extra_u32 == 0,
         "metadata: a counter is not zero");
  Expect(read.schema == written.schema, "metadata: schema changed");
  Expect(read.stage == written.stage, "metadata: stage changed");
  Expect(read.step == written.step, "metadata: step changed");
  Expect(read.world_size == written.world_size, "metadata: world_size changed");
  Expect(read.precision == written.precision, "metadata: precision changed");
  Expect(read.tokenizer_id == written.tokenizer_id,
         "metadata: tokenizer_id changed");
  Expect(read.git_revision == written.git_revision,
         "metadata: git_revision changed");
  Expect(read.notes == written.notes, "metadata: notes changed");

  // The config block travels as a sized nested block of its own, so the older
  // view below reads the six leading fields and skips the rest whole.
  CheckpointMetadata old_view;
  OldMetadataView old;
  old.fields = &old_view;
  const IFields::ReadResult old_result = old.Read(words, 0);
  if (old_result.pos == 0) Fail("metadata: the old view failed");
  Expect(old_result.extra_u32 != 0,
         "metadata: the old view did not count the tail");
  Expect(old_view.stage == written.stage && old_view.step == written.step,
         "metadata: the old view changed a shared field");
}

// The `Recipe` block (design decision D7 of docs/sft-parity-plan.md): the field
// order is the format, the block is visited last, and every member round-trips.
void TestRecipeBlock() {
  CheckpointMetadata written;
  written.stage = Stage::kSft;
  written.step = 8;
  written.world_size = 4;
  written.config.num_layers = 2;
  written.config.hidden_dim = 32;

  Recipe& recipe = written.recipe;
  recipe.device_batch_size = 2;
  recipe.total_batch_size = 128;
  recipe.embedding_lr = 0.3f;
  recipe.unembedding_lr = 0.004f;
  recipe.matrix_lr = 0.02f;
  recipe.scalar_lr = 0.5f;
  recipe.adam_eps = 1e-10f;
  recipe.init_lr_frac = 0.8f;
  recipe.warmup_ratio = 0.0f;
  recipe.warmdown_ratio = 0.5f;
  recipe.final_lr_frac = 0.0f;
  recipe.muon_momentum_steps = 300.0f;
  recipe.muon_momentum_start = 0.85f;
  recipe.muon_momentum_peak = 0.95f;
  recipe.muon_momentum_final = 0.95f;
  recipe.mmlu_epochs = 3;
  recipe.gsm8k_epochs = 4;

  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("recipe: Write produced no words");
    return;
  }

  CheckpointMetadata read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("recipe: Read failed");
  Expect(result.missing_fields == 0, "recipe: missing_fields is not zero");
  Expect(result.extra_u32 == 0, "recipe: extra_u32 is not zero");
  Expect(result.pos == words.size(), "recipe: pos did not consume the block");

  Expect(read.recipe.device_batch_size == recipe.device_batch_size,
         "recipe: device_batch_size changed");
  Expect(read.recipe.total_batch_size == recipe.total_batch_size,
         "recipe: total_batch_size changed");
  Expect(read.recipe.embedding_lr == recipe.embedding_lr,
         "recipe: embedding_lr changed");
  Expect(read.recipe.unembedding_lr == recipe.unembedding_lr,
         "recipe: unembedding_lr changed");
  Expect(read.recipe.matrix_lr == recipe.matrix_lr,
         "recipe: matrix_lr changed");
  Expect(read.recipe.scalar_lr == recipe.scalar_lr,
         "recipe: scalar_lr changed");
  Expect(read.recipe.adam_eps == recipe.adam_eps, "recipe: adam_eps changed");
  Expect(read.recipe.init_lr_frac == recipe.init_lr_frac,
         "recipe: init_lr_frac changed");
  Expect(read.recipe.warmup_ratio == recipe.warmup_ratio,
         "recipe: warmup_ratio changed");
  Expect(read.recipe.warmdown_ratio == recipe.warmdown_ratio,
         "recipe: warmdown_ratio changed");
  Expect(read.recipe.final_lr_frac == recipe.final_lr_frac,
         "recipe: final_lr_frac changed");
  Expect(read.recipe.muon_momentum_steps == recipe.muon_momentum_steps,
         "recipe: muon_momentum_steps changed");
  Expect(read.recipe.muon_momentum_start == recipe.muon_momentum_start,
         "recipe: muon_momentum_start changed");
  Expect(read.recipe.muon_momentum_peak == recipe.muon_momentum_peak,
         "recipe: muon_momentum_peak changed");
  Expect(read.recipe.muon_momentum_final == recipe.muon_momentum_final,
         "recipe: muon_momentum_final changed");
  Expect(read.recipe.mmlu_epochs == recipe.mmlu_epochs,
         "recipe: mmlu_epochs changed");
  Expect(read.recipe.gsm8k_epochs == recipe.gsm8k_epochs,
         "recipe: gsm8k_epochs changed");

  // Rule 1: the block is visited last, so its words are the tail of the stream:
  // one size word and then the member words, which are the words a bare
  // `Recipe` writes.
  const std::vector<std::uint32_t> recipe_words = recipe.Write();
  Expect(words.size() > recipe_words.size(),
         "recipe: the block is not in the stream");
  const std::size_t begin = words.size() - recipe_words.size() - 1;
  Expect(words[begin] == recipe_words.size(),
         "recipe: the size word is not the block length");
  Expect(std::equal(recipe_words.begin(), recipe_words.end(),
                    words.begin() + static_cast<std::ptrdiff_t>(begin + 1)),
         "recipe: the block is not the last field of the metadata block");
}

// The file predates the recipe block: the reader counts the one field it did not
// see and keeps every `Recipe` initializer (design decision D7).
void TestRecipeBlockMissing() {
  CheckpointMetadata written;
  written.stage = Stage::kSft;
  written.step = 3;
  written.tokenizer_id = "nanochat-32k";
  written.notes = {"pre-recipe"};
  const std::vector<std::uint32_t> full = written.Write();
  if (full.empty()) {
    Fail("recipe missing: Write produced no words");
    return;
  }
  // The block a build older than D7 wrote is the same stream without the recipe
  // block: its size word and its member words.
  const std::size_t recipe_words = Recipe().Write().size() + 1;
  Expect(full.size() > recipe_words,
         "recipe missing: the stream is shorter than the block");
  const std::vector<std::uint32_t> older(
      full.begin(), full.end() - static_cast<std::ptrdiff_t>(recipe_words));

  // A reader that knows the block reports it as one missing field and touches
  // nothing else. The sentinel members prove the read did not resize or zero
  // the defaults away.
  CheckpointMetadata read;
  read.recipe.device_batch_size = -1;
  read.recipe.total_batch_size = -1;
  read.recipe.embedding_lr = -1.0f;
  read.recipe.gsm8k_epochs = -1;
  const IFields::ReadResult result = read.Read(older, 0);
  if (result.pos == 0) Fail("recipe missing: Read failed");
  Expect(result.missing_fields == 1,
         "recipe missing: missing_fields is not 1 (the block)");
  Expect(result.extra_u32 == 0, "recipe missing: extra_u32 is not zero");
  Expect(read.stage == written.stage && read.step == written.step,
         "recipe missing: a leading field changed");
  Expect(read.tokenizer_id == written.tokenizer_id,
         "recipe missing: tokenizer_id changed");
  Expect(read.notes == written.notes, "recipe missing: notes changed");
  Expect(read.recipe.device_batch_size == -1 &&
             read.recipe.total_batch_size == -1 &&
             read.recipe.embedding_lr == -1.0f && read.recipe.gsm8k_epochs == -1,
         "recipe missing: the block defaults were overwritten");

  // With no sentinel the same read yields exactly the member initializers the
  // header documents, which is what a finetune inherits from such a file.
  CheckpointMetadata defaults;
  const IFields::ReadResult default_result = defaults.Read(older, 0);
  if (default_result.pos == 0) Fail("recipe missing: the default read failed");
  const Recipe expected;
  Expect(defaults.recipe.device_batch_size == expected.device_batch_size &&
             defaults.recipe.total_batch_size == expected.total_batch_size &&
             defaults.recipe.embedding_lr == expected.embedding_lr &&
             defaults.recipe.unembedding_lr == expected.unembedding_lr &&
             defaults.recipe.matrix_lr == expected.matrix_lr &&
             defaults.recipe.scalar_lr == expected.scalar_lr &&
             defaults.recipe.init_lr_frac == expected.init_lr_frac &&
             defaults.recipe.warmdown_ratio == expected.warmdown_ratio &&
             defaults.recipe.muon_momentum_steps ==
                 expected.muon_momentum_steps &&
             defaults.recipe.mmlu_epochs == expected.mmlu_epochs &&
             defaults.recipe.gsm8k_epochs == expected.gsm8k_epochs,
         "recipe missing: the member defaults changed");
}

// A file newer than this build: three trailing words after the recipe block are
// counted and skipped, and the block still reads (rule 1, design decision D6).
void TestRecipeExtraTrailingWords() {
  CheckpointMetadata written;
  written.stage = Stage::kSft;
  written.recipe.init_lr_frac = 0.8f;
  written.recipe.total_batch_size = 64;
  std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("recipe extra: Write produced no words");
    return;
  }
  words.push_back(0xfeedfaceu);
  words.push_back(7);
  words.push_back(0x0badf00du);

  CheckpointMetadata read;
  const IFields::ReadResult result = read.Read(words, 0);
  if (result.pos == 0) Fail("recipe extra: Read failed");
  Expect(result.missing_fields == 0, "recipe extra: missing_fields is not zero");
  Expect(result.extra_u32 == 3, "recipe extra: extra_u32 is not 3");
  Expect(result.pos == words.size() - 3,
         "recipe extra: pos did not stop at the first extra word");
  Expect(read.stage == written.stage, "recipe extra: stage changed");
  Expect(read.recipe.init_lr_frac == 0.8f,
         "recipe extra: init_lr_frac changed");
  Expect(read.recipe.total_batch_size == 64,
         "recipe extra: total_batch_size changed");
}

// A malformed block is a hard error: no partial load and no silently adopted
// rate (rule 5, design decision D7).
void TestRecipeBlockMalformed() {
  CheckpointMetadata written;
  written.stage = Stage::kSft;
  written.recipe.init_lr_frac = 0.8f;
  const std::vector<std::uint32_t> words = written.Write();
  if (words.empty()) {
    Fail("recipe malformed: Write produced no words");
    return;
  }
  const std::size_t recipe_words = Recipe().Write().size();
  const std::size_t size_word = words.size() - recipe_words - 1;

  {
    // The block claims more words than the enclosing block holds.
    std::vector<std::uint32_t> bad = words;
    bad[size_word] = static_cast<std::uint32_t>(recipe_words + 8);
    CheckpointMetadata read;
    Expect(read.Read(bad, 0).pos == 0,
           "recipe malformed: an oversized block was accepted");
  }
  {
    // The block is cut short, so the words the size word promises are absent.
    std::vector<std::uint32_t> bad(
        words.begin(), words.end() - static_cast<std::ptrdiff_t>(6));
    CheckpointMetadata read;
    Expect(read.Read(bad, 0).pos == 0,
           "recipe malformed: a truncated block was accepted");
  }
  {
    // A rate the format cannot carry is a hard error, wherever it sits in the
    // block.
    std::vector<std::uint32_t> bad = words;
    bad[size_word + 3] = 0x7fc00000u;  // a quiet NaN rate
    CheckpointMetadata read;
    read.recipe.init_lr_frac = 0.5f;
    const IFields::ReadResult result = read.Read(bad, 0);
    Expect(result.pos == 0, "recipe malformed: a NaN rate was accepted");
    Expect(read.recipe.init_lr_frac == 0.5f,
           "recipe malformed: a failed read overwrote a member");
  }
}

// The recipe a run can start from (design decision D7). The word stream only
// carries the values; `RecipeUsable` is the half that reads like a schedule, and
// the SFT driver refuses to inherit from a block it rejects.
void TestRecipeUsable() {
  std::string error;
  Recipe recipe;  // Every default is a recipe the build itself writes.
  if (!nanochat::RecipeUsable(recipe, &error)) {
    Fail("recipe usable: the defaults were rejected: " + error);
  }
  // A recorded batch shape is usable, and so is an absent one.
  recipe.device_batch_size = 2;
  recipe.total_batch_size = 2 * 32;
  if (!nanochat::RecipeUsable(recipe, &error)) {
    Fail("recipe usable: a recorded batch shape was rejected: " + error);
  }

  const struct {
    const char* what;
    void (*mutate)(Recipe*);
  } kRejected[] = {
      {"half a batch shape",
       [](Recipe* r) {
         r->device_batch_size = 2;
         r->total_batch_size = 0;
       }},
      {"the other half of a batch shape",
       [](Recipe* r) {
         r->device_batch_size = 0;
         r->total_batch_size = 64;
       }},
      {"a negative batch size",
       [](Recipe* r) {
         r->device_batch_size = -2;
         r->total_batch_size = -64;
       }},
      {"a negative rate", [](Recipe* r) { r->matrix_lr = -0.02f; }},
      {"a negative epsilon", [](Recipe* r) { r->adam_eps = -1e-10f; }},
      {"a negative momentum ramp",
       [](Recipe* r) { r->muon_momentum_steps = -300.0f; }},
      {"a negative mixture weight", [](Recipe* r) { r->gsm8k_epochs = -4; }},
      {"a warmdown ratio above one",
       [](Recipe* r) { r->warmdown_ratio = 1.5f; }},
      {"a negative init_lr_frac", [](Recipe* r) { r->init_lr_frac = -0.8f; }},
      {"a momentum above one",
       [](Recipe* r) { r->muon_momentum_peak = 1.2f; }},
  };
  for (const auto& tc : kRejected) {
    Recipe bad;
    tc.mutate(&bad);
    error.clear();
    if (nanochat::RecipeUsable(bad, &error)) {
      Fail(std::string("recipe usable: ") + tc.what + " was accepted");
    }
    if (error.empty()) {
      Fail(std::string("recipe usable: ") + tc.what + " gave no message");
    }
  }
}

}  // namespace

int main() {
  TestRoundTrip();
  TestNewCodeOldData();
  TestOldCodeNewData();
  TestInvalidFloat();
  TestInvalidString();
  TestMultipleWrite();
  TestVectorCap();
  TestEnumValidity();
  TestBoolValidity();
  TestNestedBlock();
  TestMetadataBlock();
  TestRecipeBlock();
  TestRecipeBlockMissing();
  TestRecipeExtraTrailingWords();
  TestRecipeBlockMalformed();
  TestRecipeUsable();
  if (g_failures != 0) {
    std::fprintf(stderr, "fields_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("fields_test: ok\n");
  return 0;
}
