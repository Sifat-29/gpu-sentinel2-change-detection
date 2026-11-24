# Build system for change_detect (GPU satellite time-series change detection).
#
# Quick reference (run `make help` for the list of targets):
#   make            build the GPU program when nvcc exists, else the CPU one
#   make gpu        bin/change_detect       (nvcc + g++, includes CPU backends)
#   make cpu        bin/change_detect_cpu   (g++ only, no CUDA needed)
#   make test       build and run the GPU unit tests
#   make test-cpu   build and run the CPU-only unit tests
#
# Every variable below can be overridden on the command line, e.g.
#   make gpu CUDA_ARCH=sm_75 VERBOSE_PTXAS=1
#   make cpu BUILD_DIR=/tmp/b BIN_DIR=/tmp/bin
#
# Sources are found by wildcard (src/*.cc, src/cpu/*.cc, src/cuda/*.cu,
# tests/*.cc, tests/*.cu), so adding a file never needs a Makefile change.

# --- Tools and options (all overridable) -----------------------------------
CXX ?= g++
NVCC ?= nvcc

# nvcc 12.0 officially supports gcc <= 12: use g++-12 as host compiler for
# CUDA code when it is installed. Set NVCC_CCBIN= (empty) to use nvcc's
# default host compiler.
ifeq ($(origin NVCC_CCBIN),undefined)
NVCC_CCBIN := $(if $(shell command -v g++-12 2>/dev/null),g++-12,)
endif

# GPU architecture: sm_<compute capability of GPU 0>, else sm_75 when the
# query fails (for example on a machine without a GPU).
ifeq ($(origin CUDA_ARCH),undefined)
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. \r')
CUDA_ARCH := sm_$(if $(DETECTED_CC),$(DETECTED_CC),75)
endif

BUILD_DIR ?= build
BIN_DIR ?= bin
VERBOSE_PTXAS ?= 0
PYTHON ?= .venv/bin/python
FIGURE_ARGS ?=

# --- Flags -----------------------------------------------------------------
# -ffp-contract=off makes CPU FP32 results reproducible across thread counts
# and builds (no fused multiply-add differences).
CXXFLAGS ?= -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra \
            -pthread -Isrc -Ilib
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo -Isrc -Ilib \
             -Xcompiler -O3,-march=native,-Wall,-Wextra,-pthread
ifneq ($(strip $(NVCC_CCBIN)),)
NVCCFLAGS += -ccbin $(NVCC_CCBIN)
endif
ifeq ($(VERBOSE_PTXAS),1)
NVCCFLAGS += --ptxas-options=-v
endif
GPU_DEFINE := -DCD_WITH_CUDA
# GPU programs link cuFFT (co-registration check); CPU builds do not need it.
GPU_LIBS := -lcufft
# The program reports the git commit of the source tree it was built from,
# wherever the binary is placed (see GitShortCommit in src/system_info.cc).
SOURCE_DIR_DEFINE := -DCD_SOURCE_DIR='"$(CURDIR)"'

# --- Sources ----------------------------------------------------------------
STUB_SRC := src/cuda/gpu_stub.cc
CORE_CC := $(filter-out src/main.cc,$(wildcard src/*.cc)) \
           $(wildcard src/cpu/*.cc)
CUDA_CU := $(wildcard src/cuda/*.cu)
TEST_CC := $(wildcard tests/*.cc)
TEST_CU := $(wildcard tests/*.cu)

# The stub is always linked in CPU builds; GPU builds link it only while the
# real backend (src/cuda/gpu_backend.cu) does not exist.
ifeq ($(wildcard src/cuda/gpu_backend.cu),)
GPU_STUB := $(STUB_SRC)
else
GPU_STUB :=
endif

CPU_CORE_OBJS := $(patsubst %.cc,$(BUILD_DIR)/cpu/%.o,$(CORE_CC) $(STUB_SRC))
CPU_MAIN_OBJ := $(BUILD_DIR)/cpu/src/main.o
CPU_TEST_OBJS := $(patsubst %.cc,$(BUILD_DIR)/cpu/%.o,$(TEST_CC))

GPU_CORE_OBJS := $(patsubst %.cc,$(BUILD_DIR)/gpu/%.o,$(CORE_CC) $(GPU_STUB)) \
                 $(patsubst %.cu,$(BUILD_DIR)/gpu/%.o,$(CUDA_CU))
GPU_MAIN_OBJ := $(BUILD_DIR)/gpu/src/main.o
GPU_TEST_OBJS := $(patsubst %.cc,$(BUILD_DIR)/gpu/%.o,$(TEST_CC)) \
                 $(patsubst %.cu,$(BUILD_DIR)/gpu/%.o,$(TEST_CU))

PROGRAM_GPU := $(BIN_DIR)/change_detect
PROGRAM_CPU := $(BIN_DIR)/change_detect_cpu
TESTS_GPU := $(BIN_DIR)/run_tests
TESTS_CPU := $(BIN_DIR)/run_tests_cpu

ALL_OBJS := $(CPU_CORE_OBJS) $(CPU_MAIN_OBJ) $(CPU_TEST_OBJS) \
            $(GPU_CORE_OBJS) $(GPU_MAIN_OBJ) $(GPU_TEST_OBJS)

# Default target: the GPU program when nvcc is available, else the CPU one.
ifeq ($(shell command -v $(NVCC) 2>/dev/null),)
DEFAULT_TARGET := cpu
else
DEFAULT_TARGET := gpu
endif

.PHONY: all gpu cpu test test-cpu smoke lint format proof figures bench venv \
        clean help

all: $(DEFAULT_TARGET) ## Build the GPU program if nvcc exists, else the CPU one

gpu: $(PROGRAM_GPU) ## Build bin/change_detect (CUDA + CPU backends)

cpu: $(PROGRAM_CPU) ## Build bin/change_detect_cpu (CPU backends only, no nvcc)

test: $(TESTS_GPU) ## Build and run the GPU unit-test runner
	@$(abspath $(TESTS_GPU))

test-cpu: $(TESTS_CPU) ## Build and run the CPU-only unit-test runner
	@$(abspath $(TESTS_CPU))

smoke: ## Run the synthetic end-to-end smoke test (scripts/smoke.sh)
	bash scripts/smoke.sh

lint: ## Run clang-format and cpplint checks (prints LINT: PASS or FAIL)
	bash scripts/lint.sh

format: ## Reformat all C++/CUDA sources in place with clang-format
	find src tests \( -name '*.cc' -o -name '*.h' -o -name '*.cu' \
	  -o -name '*.cuh' \) -print0 | xargs -0 clang-format --style=file -i

proof: ## Build proof_of_execution.tar.gz (scripts/make_proof.sh)
	bash scripts/make_proof.sh

# Arguments are passed through FIGURE_ARGS, for example:
#   make figures FIGURE_ARGS="--result out --cube <manifest.txt> --out figs"
figures: ## Render figures (pass FIGURE_ARGS="--result D --cube M --out D")
	$(PYTHON) scripts/render_figures.py $(FIGURE_ARGS)

bench: ## Run the benchmark suite (scripts/run_benchmarks.sh)
	bash scripts/run_benchmarks.sh

venv: ## Create .venv and install the Python helper dependencies
	bash scripts/setup_python.sh

clean: ## Remove the build directory and built binaries
	rm -rf $(BUILD_DIR)
	rm -f $(PROGRAM_GPU) $(PROGRAM_CPU) $(TESTS_GPU) $(TESTS_CPU)

help: ## List the targets
	@echo "Targets (variables: CUDA_ARCH=$(CUDA_ARCH) NVCC_CCBIN=$(NVCC_CCBIN)"
	@echo "BUILD_DIR=$(BUILD_DIR) BIN_DIR=$(BIN_DIR) VERBOSE_PTXAS=$(VERBOSE_PTXAS)):"
	@grep -h -E '^[a-zA-Z_-]+:.*## ' $(firstword $(MAKEFILE_LIST)) | \
	  sed -E 's/^([a-zA-Z_-]+):.*## (.*)$$/  \1|\2/' | \
	  awk -F'|' '{ printf "  %-10s %s\n", $$1, $$2 }'

# --- Link rules ---------------------------------------------------------------
$(PROGRAM_CPU): $(CPU_CORE_OBJS) $(CPU_MAIN_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ -pthread

$(TESTS_CPU): $(CPU_CORE_OBJS) $(CPU_TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ -pthread

# GPU programs are linked with nvcc so that the CUDA runtime is found.
$(PROGRAM_GPU): $(GPU_CORE_OBJS) $(GPU_MAIN_OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GPU_LIBS)

$(TESTS_GPU): $(GPU_CORE_OBJS) $(GPU_TEST_OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GPU_LIBS)

# --- Compile rules (automatic header dependencies through -MMD -MP) ----------
$(BUILD_DIR)/cpu/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/gpu/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(GPU_DEFINE) -MMD -MP -c $< -o $@

$(BUILD_DIR)/gpu/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(GPU_DEFINE) -MMD -MP -c $< -o $@

$(BUILD_DIR)/cpu/src/system_info.o $(BUILD_DIR)/gpu/src/system_info.o: \
    CXXFLAGS += $(SOURCE_DIR_DEFINE)

-include $(ALL_OBJS:.o=.d)
