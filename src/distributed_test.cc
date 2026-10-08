// src/distributed_test.cc -- the backend-free gradient-sync host reference.
//
// Runs two ranks and checks two things (docs/distributed-design.md sections 4
// and 7):
//   1. `AllReduceSum` returns the elementwise sum on every rank. The two ranks
//      run in separate processes over loopback TCP, and the result is checked
//      against an independently computed reference buffer.
//   2. The stride document sharding sends each document to exactly one rank,
//      and a world size of 1 leaves the stream unchanged.

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/sandbox.h"
#include "src/distributed.h"
#include "src/ops.h"

namespace {

using nanochat::ComputeType;
using nanochat::DistributedConfig;
using nanochat::DocumentSource;
using nanochat::DocumentSourceFactory;
using nanochat::GradientSync;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::vector<ComputeType> MakeBuffer(const std::vector<float>& values) {
  std::vector<ComputeType> buffer(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    buffer[i] = nanochat::ToC(values[i]);
  }
  return buffer;
}

// Binds a socket to an ephemeral loopback port, reads the port back, and
// closes the socket. The two ranks then use that port. The short window before
// the server binds again is acceptable for a test.
int FindFreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    return -1;
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    ::close(fd);
    return -1;
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

// Joins the group as `rank` and reduces `start` once. Returns true when the
// reduced buffer equals `expected`.
bool RunRank(int rank, int port, const std::vector<ComputeType>& start,
             const std::vector<float>& expected) {
  DistributedConfig config;
  config.rank = rank;
  config.world_size = 2;
  config.master = "127.0.0.1";
  config.port = port;
  std::unique_ptr<GradientSync> sync = nanochat::CreateGradientSync(config);
  if (sync == nullptr) return false;
  if (sync->rank() != rank || sync->world_size() != 2) return false;
  std::vector<ComputeType> buffer = start;
  sync->AllReduceSum(buffer.data(), static_cast<std::int64_t>(buffer.size()));
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    if (std::fabs(nanochat::AsF(buffer[i]) - expected[i]) > 1e-4f) return false;
  }
  return true;
}

// Two processes reduce two different buffers; both must see the sum.
void TestTwoRankAllReduce() {
  const std::vector<float> rank0 = {1.5f, -2.25f, 3.0f, 0.125f, -4.0f};
  const std::vector<float> rank1 = {0.5f, 2.25f, -1.0f, 0.875f, 4.5f};
  std::vector<float> expected(rank0.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = rank0[i] + rank1[i];
  }

  const int port = FindFreePort();
  if (port <= 0) {
    Fail("cannot find a free port");
    return;
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    Fail("fork failed");
    return;
  }
  if (pid == 0) {
    // Child: rank 1.
    const bool ok = RunRank(1, port, MakeBuffer(rank1), expected);
    std::_Exit(ok ? 0 : 1);
  }

  // Parent: rank 0.
  if (!RunRank(0, port, MakeBuffer(rank0), expected)) {
    Fail("rank 0 all-reduce sum mismatch");
  }
  int status = 0;
  ::waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    Fail("rank 1 all-reduce sum mismatch");
  }
}

// A fixed in-memory document source that yields each document once.
class ListSource final : public DocumentSource {
 public:
  explicit ListSource(std::vector<std::string> documents)
      : documents_(std::move(documents)) {}

  bool Next(std::vector<std::string>* documents, std::string* error) override {
    (void)error;
    if (index_ >= documents_.size()) return false;
    documents->clear();
    const std::size_t end = index_ + kBatch;
    for (; index_ < end && index_ < documents_.size(); ++index_) {
      documents->push_back(documents_[index_]);
    }
    return true;
  }

 private:
  static constexpr std::size_t kBatch = 3;
  std::vector<std::string> documents_;
  std::size_t index_ = 0;
};

std::vector<std::string> Collect(const DocumentSourceFactory& factory) {
  std::vector<std::string> out;
  std::unique_ptr<DocumentSource> source = factory(nullptr);
  if (source == nullptr) return out;
  std::vector<std::string> batch;
  while (source->Next(&batch, nullptr)) {
    for (std::string& document : batch) out.push_back(std::move(document));
  }
  return out;
}

int DocumentIndex(const std::string& name) { return std::stoi(name.substr(3)); }

void TestDocumentSharding() {
  std::vector<std::string> documents;
  for (int i = 0; i < 10; ++i) {
    documents.push_back("doc" + std::to_string(i));
  }
  const DocumentSourceFactory base =
      [documents](std::string*) -> std::unique_ptr<DocumentSource> {
    return std::make_unique<ListSource>(documents);
  };

  const std::vector<std::string> got0 =
      Collect(nanochat::ShardDocumentSourceFactory(base, 0, 2));
  const std::vector<std::string> got1 =
      Collect(nanochat::ShardDocumentSourceFactory(base, 1, 2));
  if (got0.size() != 5 || got1.size() != 5) {
    Fail("sharding: a rank did not receive half the documents");
  }
  for (const std::string& name : got0) {
    if (DocumentIndex(name) % 2 != 0) Fail("rank 0 saw an odd document");
  }
  for (const std::string& name : got1) {
    if (DocumentIndex(name) % 2 != 1) Fail("rank 1 saw an even document");
  }
  std::vector<std::string> union_all = got0;
  union_all.insert(union_all.end(), got1.begin(), got1.end());
  std::sort(union_all.begin(), union_all.end());
  std::vector<std::string> want = documents;
  std::sort(want.begin(), want.end());
  if (union_all != want)
    Fail("sharding: the ranks do not cover every document");

  // A world size of 1 leaves the stream unchanged.
  const std::vector<std::string> single =
      Collect(nanochat::ShardDocumentSourceFactory(base, 0, 1));
  if (single != documents) {
    Fail("sharding: world size 1 changed the document stream");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestDocumentSharding();
  TestTwoRankAllReduce();
  if (g_failures != 0) {
    std::fprintf(stderr, "distributed_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("distributed_test: ok\n");
  return 0;
}
