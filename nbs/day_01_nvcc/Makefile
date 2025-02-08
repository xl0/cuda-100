# Compiler and flags
NVCC        := nvcc
CUDA_ARCH   := -arch=sm_60 -gencode=arch=compute_60,code=sm_60 \
               -gencode=arch=compute_70,code=sm_70 \
               -gencode=arch=compute_75,code=sm_75

# Build flags
NVCC_FLAGS  := -std=c++14 -O3 $(CUDA_ARCH)
WARN_FLAGS  := -Xcompiler -Wall,-Wextra
DEBUG_FLAGS := -g -G -lineinfo

# Build directories
BUILD_DIR   := build
SRC_DIR     := src

# Default target name
TARGET      := hello

# Source files
SRCS        := $(wildcard $(SRC_DIR)/*.cu)
OBJS        := $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Build targets
all: release

debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: $(BUILD_DIR)/$(TARGET)

release: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(WARN_FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(WARN_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all debug release clean