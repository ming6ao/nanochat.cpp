// The self-describing checkpoint container and the document-mode `DataLoader`
// (docs/model.md, docs/parquet-native.md).
//
// Everything here is host-side and vendor-free; it only reads and writes
// little-endian bytes. The loader reads parquet documents and tokenizes them
// during the run, so no pre-tokenized shard is needed.
//
// `LoadTokenizer` lives in the `tokenizer` library
// (src/tokenizer/tokenizer.cc); this file does not define it.

#include "nanochat/data.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/tokenizer.h"

namespace nanochat {
namespace {

// --- Little-endian stream helpers ------------------------------------------

void WriteU16(std::ostream& out, std::uint16_t value) {
  const unsigned char bytes[2] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteU32(std::ostream& out, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu),
      static_cast<unsigned char>((value >> 16) & 0xffu),
      static_cast<unsigned char>((value >> 24) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteU64(std::ostream& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    const unsigned char byte =
        static_cast<unsigned char>((value >> (8 * i)) & 0xffu);
    out.write(reinterpret_cast<const char*>(&byte), 1);
  }
}

void WriteI64(std::ostream& out, std::int64_t value) {
  WriteU64(out, static_cast<std::uint64_t>(value));
}

bool ReadBytes(std::istream& in, void* out, std::size_t count) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(count));
  return static_cast<std::size_t>(in.gcount()) == count;
}

bool ReadU16(std::istream& in, std::uint16_t* value) {
  unsigned char bytes[2];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  *value = static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
  return true;
}

bool ReadU32(std::istream& in, std::uint32_t* value) {
  unsigned char bytes[4];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  *value = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
  return true;
}

bool ReadU64(std::istream& in, std::uint64_t* value) {
  unsigned char bytes[8];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  std::uint64_t result = 0;
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  *value = result;
  return true;
}

bool ReadI64(std::istream& in, std::int64_t* value) {
  std::uint64_t raw = 0;
  if (!ReadU64(in, &raw)) return false;
  *value = static_cast<std::int64_t>(raw);
  return true;
}

// Writes one self-describing record: name, dtype, shape, and raw payload.
// Parameter records and optimizer-state records share this layout.
void WriteRecord(std::ostream& out, const TensorRecord& record) {
  const std::size_t name_len = record.name.size();
  const std::size_t clipped = std::min<std::size_t>(name_len, 0xffffu);
  WriteU16(out, static_cast<std::uint16_t>(clipped));
  out.write(record.name.data(), static_cast<std::streamsize>(clipped));
  WriteU32(out, static_cast<std::uint32_t>(record.dtype));
  WriteU32(out, static_cast<std::uint32_t>(record.shape.size()));
  for (std::int64_t dim : record.shape) WriteI64(out, dim);
  WriteU64(out, static_cast<std::uint64_t>(record.data.size()));
  if (!record.data.empty()) {
    out.write(reinterpret_cast<const char*>(record.data.data()),
              static_cast<std::streamsize>(record.data.size()));
  }
}

// Reads one self-describing record. On failure it writes a short reason to
// `error` and returns false.
bool ReadRecord(std::istream& in, TensorRecord* record, std::string* error) {
  auto bad = [&](const char* reason) {
    if (error != nullptr) *error = reason;
    return false;
  };
  std::uint16_t name_len = 0;
  if (!ReadU16(in, &name_len)) return bad("truncated record name");
  record->name.resize(name_len);
  if (name_len > 0 && !ReadBytes(in, record->name.data(), name_len)) {
    return bad("truncated record name");
  }
  std::uint32_t dtype = 0;
  std::uint32_t rank = 0;
  if (!ReadU32(in, &dtype) || !ReadU32(in, &rank)) {
    return bad("truncated record header");
  }
  if (dtype > static_cast<std::uint32_t>(DType::kFp16)) {
    return bad("unknown checkpoint dtype");
  }
  if (rank > 8) return bad("checkpoint rank is too large");
  record->dtype = static_cast<DType>(dtype);
  record->shape.resize(rank);
  for (std::uint32_t d = 0; d < rank; ++d) {
    if (!ReadI64(in, &record->shape[d])) return bad("truncated record shape");
    if (record->shape[d] < 0) return bad("negative checkpoint dimension");
  }
  std::uint64_t data_bytes = 0;
  if (!ReadU64(in, &data_bytes)) return bad("truncated record payload size");
  record->data.resize(static_cast<std::size_t>(data_bytes));
  if (data_bytes > 0 && !ReadBytes(in, record->data.data(), data_bytes)) {
    return bad("truncated record payload");
  }
  return true;
}

constexpr char kCheckpointMagic[8] = {'N', 'C', 'H', 'K', 'P', 'T', '0', '1'};

}  // namespace

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

bool Checkpoint::Load(const std::string& path, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };

  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open checkpoint: " + path);

  char magic[8];
  if (!ReadBytes(in, magic, sizeof(magic)) ||
      std::memcmp(magic, kCheckpointMagic, sizeof(magic)) != 0) {
    return fail("bad checkpoint magic: " + path);
  }
  std::uint32_t version = 0;
  std::uint32_t count = 0;
  if (!ReadU32(in, &version) || !ReadU32(in, &count)) {
    return fail("truncated checkpoint header: " + path);
  }
  if (version != 1) return fail("unsupported checkpoint version: " + path);

  std::vector<TensorRecord> records;
  records.reserve(count);
  for (std::uint32_t r = 0; r < count; ++r) {
    TensorRecord record;
    std::string reason;
    if (!ReadRecord(in, &record, &reason)) {
      return fail("checkpoint " + reason + ": " + path);
    }
    records.push_back(std::move(record));
  }

  // The optimizer-state section is optional (docs/post-training.md section 7).
  // A parameter-only file ends after the parameter records; a file with
  // optimizer state carries a second count and its records. This keeps every
  // parameter-only NCHKPT01 file loadable.
  std::vector<TensorRecord> optimizer_state;
  if (in.peek() != std::char_traits<char>::eof()) {
    std::uint32_t state_count = 0;
    if (!ReadU32(in, &state_count)) {
      return fail("truncated checkpoint optimizer-state header: " + path);
    }
    optimizer_state.reserve(state_count);
    for (std::uint32_t r = 0; r < state_count; ++r) {
      TensorRecord record;
      std::string reason;
      if (!ReadRecord(in, &record, &reason)) {
        return fail("checkpoint optimizer state " + reason + ": " + path);
      }
      optimizer_state.push_back(std::move(record));
    }
  }

  tensors_ = std::move(records);
  optimizer_state_ = std::move(optimizer_state);
  return true;
}

bool Checkpoint::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;

  out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
  WriteU32(out, 1);
  WriteU32(out, static_cast<std::uint32_t>(tensors_.size()));
  for (const TensorRecord& record : tensors_) WriteRecord(out, record);
  // The optimizer-state section follows the parameters (docs/post-training.md
  // section 7). A parameter-only save writes a zero count and stays loadable.
  WriteU32(out, static_cast<std::uint32_t>(optimizer_state_.size()));
  for (const TensorRecord& record : optimizer_state_) WriteRecord(out, record);
  return static_cast<bool>(out);
}

const TensorRecord* Checkpoint::Find(std::string_view name) const {
  for (const TensorRecord& record : tensors_) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

TensorRecord* Checkpoint::Find(std::string_view name) {
  for (TensorRecord& record : tensors_) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

void Checkpoint::Add(TensorRecord record) {
  tensors_.push_back(std::move(record));
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------
//
// Document mode (docs/parquet-native.md). A producer thread reads documents,
// tokenizes them with the reference encoder, and pushes them into a bounded
// deque. The consumer packs every row with the reference BOS-aligned best-fit
// rule: refill to `document_buffer`, take the largest document that fits, and
// crop the shortest document when none fits. `tokens` is the row and `targets`
// is its one-position shift.

struct DataLoader::Impl {
  DocumentSourceFactory source_factory;
  const Tokenizer* tokenizer = nullptr;
  int tokenizer_threads = 1;
  std::size_t document_buffer = 1000;
  std::uint64_t seed = 42;

  // The data cursor (docs/distributed-design.md section 11). `epoch` counts the
  // `Reset` calls. The resume cursor is the encoded `documents` buffer plus the
  // source position, so `State()` can snapshot it consistently.
  std::int64_t epoch = 0;

  std::thread producer;
  std::mutex mutex;
  // The producer holds this gate across a source read, its encode, and the
  // push. `State` locks the gate too, so a snapshot never catches a read that
  // has not reached the buffer yet.
  std::mutex producer_gate;
  std::condition_variable space_available;
  std::condition_variable data_available;
  std::deque<std::vector<int>> documents;
  std::unique_ptr<DocumentSource> source;
  bool producer_done = false;
  bool stop = false;
  std::string producer_error;

  ~Impl() { StopProducer(); }

  // Encodes one document batch. `tokenizer_threads` workers keep the producer
  // ahead of the training step. The result order matches `batch`.
  void EncodeBatch(const std::vector<std::string>& batch,
                   std::vector<std::vector<int>>* out) {
    const std::size_t count = batch.size();
    if (count == 0) return;
    const int bos = tokenizer != nullptr ? tokenizer->bos_id() : -1;
    if (tokenizer_threads <= 1 || count == 1) {
      for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = tokenizer->Encode(batch[i], bos);
      }
      return;
    }
    std::atomic<std::size_t> next{0};
    auto worker = [&]() {
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= count) break;
        (*out)[i] = tokenizer->Encode(batch[i], bos);
      }
    };
    std::vector<std::thread> pool;
    const int workers =
        std::min<int>(tokenizer_threads, static_cast<int>(count));
    for (int t = 1; t < workers; ++t) pool.emplace_back(worker);
    worker();
    for (std::thread& thread : pool) thread.join();
  }

  void ProducerLoop() {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        space_available.wait(lock, [this] {
          return stop || documents.size() < document_buffer;
        });
        if (stop) break;
      }
      // The gate spans the read, the encode, and the push. `State` takes the
      // same gate, so the snapshot never sees a read that is not yet in the
      // buffer (docs/distributed-design.md section 11).
      std::lock_guard<std::mutex> gate(producer_gate);
      std::vector<std::string> batch;
      std::string error;
      const bool ok = source != nullptr && source->Next(&batch, &error);
      if (!ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) producer_error = error;
        producer_done = true;
        data_available.notify_all();
        break;
      }
      if (batch.empty()) continue;
      std::vector<std::vector<int>> encoded(batch.size());
      EncodeBatch(batch, &encoded);
      std::lock_guard<std::mutex> lock(mutex);
      if (stop) break;
      for (std::vector<int>& document : encoded) {
        documents.push_back(std::move(document));
      }
      data_available.notify_all();
    }
  }

  void StartProducer() {
    std::string error;
    source = source_factory != nullptr ? source_factory(&error) : nullptr;
    if (source == nullptr) {
      producer_done = true;
      producer_error =
          error.empty() ? "cannot open the document source" : error;
      return;
    }
    producer_done = false;
    stop = false;
    producer_error.clear();
    producer = std::thread([this]() { ProducerLoop(); });
  }

  void StopProducer() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
      documents.clear();
      space_available.notify_all();
      data_available.notify_all();
    }
    if (producer.joinable()) producer.join();
  }

  bool NextDocuments(int batch, int seq, int* tokens, int* targets,
                     std::string* error) {
    if (source == nullptr) {
      if (error != nullptr) *error = producer_error;
      return false;
    }
    const std::size_t capacity = static_cast<std::size_t>(seq) + 1;
    std::vector<int> row(capacity);
    std::unique_lock<std::mutex> lock(mutex);
    for (int r = 0; r < batch; ++r) {
      std::size_t pos = 0;
      while (pos < capacity) {
        // Refill to `document_buffer` before each placement, the same wait as
        // the reference `while len(doc_buffer) < buffer_size`. The producer
        // pulls whole source batches, so both sides see the same buffer.
        data_available.wait(lock, [this] {
          return documents.size() >= document_buffer || producer_done || stop;
        });
        if (documents.empty()) {
          if (producer_done && error != nullptr && !producer_error.empty()) {
            *error = producer_error;
          }
          return false;
        }
        const std::size_t remaining = capacity - pos;
        // Pick the largest document that fits entirely.
        std::size_t best = documents.size();
        std::size_t best_len = 0;
        for (std::size_t i = 0; i < documents.size(); ++i) {
          const std::size_t length = documents[i].size();
          if (length <= remaining && length > best_len) {
            best = i;
            best_len = length;
          }
        }
        if (best < documents.size()) {
          std::copy(documents[best].begin(), documents[best].end(),
                    row.begin() + static_cast<std::ptrdiff_t>(pos));
          pos += best_len;
          documents.erase(documents.begin() +
                          static_cast<std::ptrdiff_t>(best));
        } else {
          // Crop the shortest document to fill the row exactly.
          std::size_t shortest = 0;
          for (std::size_t i = 1; i < documents.size(); ++i) {
            if (documents[i].size() < documents[shortest].size()) shortest = i;
          }
          const std::size_t take =
              std::min(remaining, documents[shortest].size());
          std::copy(
              documents[shortest].begin(),
              documents[shortest].begin() + static_cast<std::ptrdiff_t>(take),
              row.begin() + static_cast<std::ptrdiff_t>(pos));
          pos += take;
          documents.erase(documents.begin() +
                          static_cast<std::ptrdiff_t>(shortest));
        }
        space_available.notify_all();
      }
      for (int j = 0; j < seq; ++j) {
        const std::size_t base =
            static_cast<std::size_t>(r) * static_cast<std::size_t>(seq);
        tokens[base + static_cast<std::size_t>(j)] =
            row[static_cast<std::size_t>(j)];
        targets[base + static_cast<std::size_t>(j)] =
            row[static_cast<std::size_t>(j) + 1];
      }
    }
    return true;
  }

  // A consistent snapshot of the cursor
  // (docs/distributed-design.md section 11). The gate keeps the producer from
  // reading a batch that is not yet in the buffer, so the restored buffer and
  // position replay the packing exactly.
  DataLoaderState Snapshot() {
    DataLoaderState state;
    std::lock_guard<std::mutex> gate(producer_gate);
    std::lock_guard<std::mutex> lock(mutex);
    state.epoch = epoch;
    state.source_position = source != nullptr ? source->position() : 0;
    state.documents.assign(documents.begin(), documents.end());
    return state;
  }

  void ResetDocuments() {
    StopProducer();
    {
      std::lock_guard<std::mutex> lock(mutex);
      documents.clear();
      producer_done = false;
      producer_error.clear();
      stop = false;
      // A new epoch restarts the cursor
      // (docs/distributed-design.md section 11).
      ++epoch;
    }
    StartProducer();
  }
};

DataLoader::DataLoader(DocumentSourceFactory source_factory,
                       const Tokenizer* tokenizer, int batch, int seq,
                       std::uint64_t seed, int tokenizer_threads,
                       std::size_t document_buffer,
                       const DataLoaderState* restore)
    : impl_(std::make_unique<Impl>()), batch_(batch), seq_(seq) {
  impl_->source_factory = std::move(source_factory);
  impl_->tokenizer = tokenizer;
  impl_->tokenizer_threads = tokenizer_threads > 0 ? tokenizer_threads : 1;
  impl_->document_buffer = document_buffer == 0 ? 1 : document_buffer;
  impl_->seed = seed;
  if (restore != nullptr) {
    // Prime the epoch and the encoded buffer before the producer starts. The
    // caller built `source_factory` to seek to `source_position`
    // (docs/distributed-design.md section 11).
    impl_->epoch = restore->epoch;
    impl_->documents.assign(restore->documents.begin(),
                            restore->documents.end());
  }
  impl_->StartProducer();
}

DataLoader::~DataLoader() = default;

bool DataLoader::Next(int* tokens, int* targets) {
  if (tokens == nullptr || targets == nullptr || impl_ == nullptr) return false;
  if (batch_ <= 0 || seq_ <= 0) return false;
  std::string error;
  return impl_->NextDocuments(batch_, seq_, tokens, targets, &error);
}

void DataLoader::Reset() {
  if (impl_ == nullptr) return;
  impl_->ResetDocuments();
}

DataLoaderState DataLoader::State() const {
  if (impl_ == nullptr) return DataLoaderState();
  return impl_->Snapshot();
}

const std::uint8_t* DataLoader::token_bytes(int* vocab_size) const {
  if (impl_ == nullptr || impl_->tokenizer == nullptr) {
    if (vocab_size != nullptr) *vocab_size = 0;
    return nullptr;
  }
  if (vocab_size != nullptr) *vocab_size = impl_->tokenizer->vocab_size();
  return impl_->tokenizer->TokenBytes();
}

// ---------------------------------------------------------------------------
// Document sharding (docs/distributed-design.md section 7)
// ---------------------------------------------------------------------------

namespace {

// Filters one document stream by rank. The wrapper counts every document it
// reads from `source_`, so rank `rank_` keeps document `i` only when
// `i % world_size_ == rank_`. The count is deterministic from the start of the
// stream, so a `Reset` (a new epoch) reproduces the same partition.
//
// `begin_` is the global source position to resume at
// (docs/distributed-design.md section 11). The wrapper reads and drops every
// document before `begin_`, then continues the stride filter. The wrapper also
// reports `position()`, the count of documents read from the underlying stream,
// which is the resumable cursor.
class ShardedDocumentSource final : public DocumentSource {
 public:
  ShardedDocumentSource(std::unique_ptr<DocumentSource> source, int rank,
                        int world_size, std::int64_t begin = 0)
      : source_(std::move(source)),
        rank_(rank),
        world_size_(world_size),
        begin_(begin) {}

  bool Next(std::vector<std::string>* documents, std::string* error) override {
    documents->clear();
    std::vector<std::string> batch;
    while (true) {
      batch.clear();
      if (!source_->Next(&batch, error)) return !documents->empty();
      for (std::string& document : batch) {
        if (index_ < begin_) {
          ++index_;
          continue;
        }
        const bool mine = (index_ % world_size_) == rank_;
        ++index_;
        if (mine) documents->push_back(std::move(document));
      }
      if (!documents->empty()) return true;
    }
  }

  std::int64_t position() const override { return index_; }

 private:
  std::unique_ptr<DocumentSource> source_;
  int rank_ = 0;
  int world_size_ = 1;
  std::int64_t begin_ = 0;
  std::int64_t index_ = 0;
};

}  // namespace

DocumentSourceFactory ShardDocumentSourceFactory(DocumentSourceFactory factory,
                                                 int rank, int world_size,
                                                 std::int64_t start,
                                                 int start_on_creation) {
  auto creation = std::make_shared<int>(0);
  return
      [factory = std::move(factory), rank, world_size, start, start_on_creation,
       creation](std::string* error) -> std::unique_ptr<DocumentSource> {
        std::unique_ptr<DocumentSource> source = factory(error);
        if (source == nullptr) return nullptr;
        // The stored epoch selects the creation that carries the resume offset.
        // An earlier creation begins at 0, so the earlier epochs are skipped
        // without a full replay (docs/distributed-design.md section 11).
        const std::int64_t begin = (*creation == start_on_creation) ? start : 0;
        ++*creation;
        return std::make_unique<ShardedDocumentSource>(std::move(source), rank,
                                                       world_size, begin);
      };
}

}  // namespace nanochat
