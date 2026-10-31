// src/plan_main.cc -- print the device-and-graph plan for a probe spec.
//
// Why a binary and not a log line. The plan is a decision a user has to be able
// to *see* before a model loads (`docs/model-and-device-bringup.md` section 9.1
// rule 3): `tools/nanochat doctor` runs this and shows which implementation each
// composition resolves to on this device, and why the ones that were declined
// lost. The plan is pure, so the probe asks for one without a device present, a
// checkpoint, or a tokenizer.
//
// This file includes the public headers only - `nanochat/plan.h`,
// `nanochat/model_spec.h` and `nanochat/tensor.h` - so it cannot depend on a
// graph, a backend, or a workspace. The one thing it knows that a report does
// not is how to spell the two outputs: text for a terminal, JSON for a tool.
//
// Note the name: *this* plan is the device-and-graph plan built by `BuildPlan`.
// `python/nanochat_cpp/plan.py` is the training-hyperparameter planner and the
// two are unrelated.
//
// Usage:
//   plan_main [--spec=<name>] [--spec-path=<file>] [--graph=<name>]
//             [--json] [--text] [--quiet]
//
// Without `--spec`/`--spec-path` the probe uses the built-in default spec, which
// is the one `Model::Create(config)` would use.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model_spec.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"

namespace {

// The probe spec's path, from the Bazel runfiles first and the source tree
// second, so the binary works under `bazel run`, under a test, and by hand.
std::string Locate(const std::string& relative) {
  std::vector<std::string> candidates;
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.push_back(std::string(src_dir) + "/" + workspace + "/" +
                           relative);
    }
    candidates.push_back(std::string(src_dir) + "/_main/" + relative);
  }
  if (const char* runfiles = std::getenv("RUNFILES_DIR")) {
    candidates.push_back(std::string(runfiles) + "/_main/" + relative);
  }
  candidates.push_back(relative);
  candidates.push_back("../" + relative);
  candidates.push_back("../../" + relative);
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate);
    if (in.good()) return candidate;
  }
  return std::string();
}

std::string EscapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buffer;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

// A human-readable name of the plan's outcome, for the text report.
const char* Outcome(const nanochat::Plan& plan) {
  return plan.ok ? "ok" : "rejected";
}

void PrintText(const nanochat::Plan& plan, const std::string& why_not,
               const nanochat::Caps& caps) {
  std::printf("plan: spec \"%s\", graph %s, device %d (%s)\n",
              plan.spec_name.c_str(), nanochat::GraphKindName(plan.graph),
              plan.device_index,
              caps.device_name != nullptr ? caps.device_name : "(unknown)");
  std::printf("resolution: %s\n", Outcome(plan));
  if (!plan.ok) {
    std::printf("%s", why_not.c_str());
    for (const nanochat::PlanGap& gap : plan.gaps) {
      std::printf("  %s: %s\n",
                  nanochat::CompositionName(gap.composition),
                  gap.reason.c_str());
    }
    return;
  }
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    std::printf("  %-12s %s%s\n",
                nanochat::CompositionName(choice.composition),
                choice.implementation.c_str(), choice.fused ? " (fused)" : "");
    for (const nanochat::DeclinedImpl& declined : choice.declined) {
      std::printf("               declined %s: %s\n",
                  declined.implementation.c_str(), declined.reason.c_str());
    }
  }
}

void PrintJson(const nanochat::Plan& plan, const std::string& why_not) {
  std::string out = "{\n";
  out += "  \"ok\": ";
  out += plan.ok ? "true" : "false";
  out += ",\n  \"spec\": \"" + EscapeJson(plan.spec_name) + "\",\n";
  out += "  \"graph\": \"" +
         EscapeJson(nanochat::GraphKindName(plan.graph)) + "\",\n";
  out += "  \"device_index\": " + std::to_string(plan.device_index) + ",\n";
  out += "  \"choices\": [";
  for (std::size_t i = 0; i < plan.choices.size(); ++i) {
    const nanochat::CompositionChoice& choice = plan.choices[i];
    out += i == 0 ? "\n" : ",\n";
    out += "    {\"composition\": \"" +
           EscapeJson(nanochat::CompositionName(choice.composition)) +
           "\", \"implementation\": \"" + EscapeJson(choice.implementation) +
           "\", \"fused\": ";
    out += choice.fused ? "true" : "false";
    out += ", \"declined\": [";
    for (std::size_t j = 0; j < choice.declined.size(); ++j) {
      if (j != 0) out += ", ";
      out += "{\"implementation\": \"" +
             EscapeJson(choice.declined[j].implementation) +
             "\", \"reason\": \"" + EscapeJson(choice.declined[j].reason) +
             "\"}";
    }
    out += "]}";
  }
  out += plan.choices.empty() ? "],\n" : "\n  ],\n";
  out += "  \"gaps\": [";
  for (std::size_t i = 0; i < plan.gaps.size(); ++i) {
    if (i != 0) out += ", ";
    out += "{\"composition\": \"" +
           EscapeJson(nanochat::CompositionName(plan.gaps[i].composition)) +
           "\", \"reason\": \"" + EscapeJson(plan.gaps[i].reason) + "\"}";
  }
  out += "],\n";
  out += "  \"why_not\": \"" + EscapeJson(why_not) + "\"\n";
  out += "}\n";
  std::printf("%s", out.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string spec_name;
  std::string spec_path;
  std::string graph = "train";
  bool want_json = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      want_json = true;
    } else if (arg.rfind("--spec=", 0) == 0) {
      spec_name = arg.substr(7);
    } else if (arg.rfind("--spec-path=", 0) == 0) {
      spec_path = arg.substr(12);
    } else if (arg.rfind("--graph=", 0) == 0) {
      graph = arg.substr(8);
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "usage: plan_main [--spec=<name>] [--spec-path=<file>]\n"
          "                 [--graph=train|prefill|decode|eval|generate]\n"
          "                 [--json]\n"
          "Prints the device-and-graph plan for a probe spec. The plan is\n"
          "built by nanochat::BuildPlan (nanochat/plan.h) and is unrelated to\n"
          "python/nanochat_cpp/plan.py.\n");
      return 0;
    } else {
      std::fprintf(stderr, "plan_main: unknown argument \"%s\"\n", arg.c_str());
      return 2;
    }
  }

  nanochat::GraphKind graph_kind = nanochat::GraphKind::kTrain;
  const nanochat::GraphKind all_graphs[] = {
      nanochat::GraphKind::kTrain, nanochat::GraphKind::kPrefill,
      nanochat::GraphKind::kDecode, nanochat::GraphKind::kEval,
      nanochat::GraphKind::kGenerate};
  bool graph_known = false;
  for (nanochat::GraphKind candidate : all_graphs) {
    if (graph == nanochat::GraphKindName(candidate)) {
      graph_kind = candidate;
      graph_known = true;
    }
  }
  if (!graph_known) {
    std::fprintf(stderr,
                 "plan_main: unknown graph \"%s\" (train, prefill, decode, "
                 "eval, generate)\n",
                 graph.c_str());
    return 2;
  }

  // The probe spec: a path, a package name, or the built-in default.
  nanochat::ModelSpec spec;
  std::string why;
  if (!spec_path.empty()) {
    if (!nanochat::ParseModelSpec(spec_path, &spec, &why)) {
      std::fprintf(stderr, "plan_main: %s\n", why.c_str());
      return 1;
    }
  } else if (!spec_name.empty()) {
    const std::string located = Locate(nanochat::ModelSpecPath(spec_name));
    if (located.empty()) {
      std::fprintf(stderr, "plan_main: no spec for model \"%s\" (%s is not "
                           "readable)\n",
                   spec_name.c_str(),
                   nanochat::ModelSpecPath(spec_name).c_str());
      return 1;
    }
    if (!nanochat::ParseModelSpec(located, &spec, &why)) {
      std::fprintf(stderr, "plan_main: %s\n", why.c_str());
      return 1;
    }
  } else {
    const std::string located =
        Locate(nanochat::ModelSpecPath("nanochat-default"));
    if (!located.empty() &&
        nanochat::ParseModelSpec(located, &spec, &why)) {
      // The flagship package's own description, which is the one this tree
      // ships.
    } else {
      spec = nanochat::DefaultModelSpec();
    }
  }

  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = graph_kind;
  request.device_index = 0;
  std::string why_not;
  const nanochat::Plan plan = nanochat::BuildPlan(request, &why_not);
  const nanochat::Caps caps = nanochat::kernels::GetCaps();

  if (want_json) {
    PrintJson(plan, why_not);
  } else {
    PrintText(plan, why_not, caps);
  }
  // The exit status is the plan's own answer, so a `doctor` check can gate on
  // it: 0 when every composition is placed, 1 when one is not.
  return plan.ok ? 0 : 1;
}
