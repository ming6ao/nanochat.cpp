# nanochat.cpp — optional non-Bazel fallback build. See docs/build.md.
#
# Bazel is the primary driver. This Makefile mirrors the same source layout for
# an environment where rules_cuda is unavailable; it autodetects nvcc and the
# architecture in the spirit of llm.c. It currently builds only the P0
# toolchain spike. The real kernel families are added as the layout fills in.
#
#   make                 build the spike and its GPU test
#   make check           build, then run the GPU test through tools/nanochat
#   make clean
#
# The GPU test must run under the sandbox, so `make check` goes through
# tools/nanochat gpu -- rather than invoking the binary directly.

NVCC      ?= nvcc
HOSTCXX   ?= g++-12
PRECISION ?= fp32
CUDA_ARCH ?= $(shell \
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
  head -n1 | tr -d '.' | sed 's/^/sm_/')
CUDA_ARCH := $(if $(CUDA_ARCH),$(CUDA_ARCH),sm_61)
COMPUTE   := $(patsubst sm_%,compute_%,$(CUDA_ARCH))

ifeq ($(PRECISION),fp16)
PRECISION_DEFINE := -DNANOCHAT_PRECISION_FP16
else
PRECISION_DEFINE := -DNANOCHAT_PRECISION_FP32
endif

BUILD_DIR := build
INCLUDES  := -Iinclude -I.
CXXFLAGS  ?= -std=c++20 -Wall -Wextra -O2 $(PRECISION_DEFINE)
NVCCFLAGS ?= -std=c++20 -O2 -ccbin $(HOSTCXX) $(PRECISION_DEFINE) \
             -gencode arch=$(COMPUTE),code=$(CUDA_ARCH) \
             -Xcompiler -Wall -Xcompiler -Wextra

.PHONY: all check clean

all: $(BUILD_DIR)/pascal_spike_gpu_test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/pascal_spike.o: dev/kernels/pascal_spike.cu \
		dev/kernels/pascal_spike.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/pascal_spike_test.o: dev/kernels/pascal_spike_test.cc \
		dev/kernels/pascal_spike.h | $(BUILD_DIR)
	$(HOSTCXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/pascal_spike_gpu_test: $(BUILD_DIR)/pascal_spike.o \
		$(BUILD_DIR)/pascal_spike_test.o
	$(HOSTCXX) $(CXXFLAGS) $^ -o $@ -L/usr/lib/x86_64-linux-gnu -lcudart

check: all
	tools/nanochat gpu --profile t1-gpu -- $(BUILD_DIR)/pascal_spike_gpu_test

clean:
	rm -rf $(BUILD_DIR)
