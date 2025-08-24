# -------------------------
# Config
# -------------------------
# Override on the CLI like: just BUILD_TYPE=Release build
BUILD_DIR    := "cmake-build"
BUILD_TYPE   := "Release"        # Debug | Release | RelWithDebInfo | MinSizeRel
GENERATOR    := ""             # e.g., -G Ninja
CMAKE_FLAGS  := "-DCMAKE_BUILD_TYPE={{BUILD_TYPE}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# If you set an install prefix, uncomment the next line (useful for pybind11 modules):
# INSTALL_PREFIX ?= "{{pwd()}}/.local"
# INSTALL_FLAG   := "-DCMAKE_INSTALL_PREFIX={{INSTALL_PREFIX}}"

# -------------------------
# Default
# -------------------------
default: configure build test

# -------------------------
# CMake lifecycle
# -------------------------
configure:
    cmake -S . -B {{BUILD_DIR}} {{GENERATOR}} {{CMAKE_FLAGS}}

build: configure
    cmake --build {{BUILD_DIR}} -j

install: build
    cmake --install {{BUILD_DIR}}

test: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure

clean:
    rm -rf {{BUILD_DIR}}

# -------------------------
# Run binaries (examples)
# -------------------------
# Add more run-* recipes as we add crates
run-kalman:
    {{BUILD_DIR}}/crates/kf_linear/kf_demo

# -------------------------
# Linting & formatting
# -------------------------
lint: build
    clang-tidy crates/*/src/*.cpp -p {{BUILD_DIR}} -- -std=c++17

fmt:
    find crates -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec clang-format -i {} +

fmt-check:
    find crates -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec clang-format --dry-run --Werror {} +

