# Inference Engine

A compact, high-performance C++ inference engine for neural networks.

## Overview

- Small, embeddable runtime for executing ONNX-style graphs
- Memory-efficient allocators (arena/bump allocator) and flexible `Allocator` interface
- Support for quantized INT8 and floating-point workloads
- Minimal dependencies; built with CMake and GoogleTest for unit tests

## Requirements

- Linux, macOS, or Windows
- CMake 3.18+ (older versions may work)
- A C++17-capable compiler (GCC, Clang, MSVC)
- Optional: ONNX model support requires an ONNX parser at build time

## Build

From the repository root:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)
```

This produces binaries in `build/bin/` such as `infer_example` and `onnx_inference`.

## Run examples

- Simple inference example (uses a built-in or sample model):

```bash
./build/bin/infer_example
```

- Run ONNX model (`onnx_inference` to be built and ONNX support to be enabled):

```bash
./build/bin/onnx_inference path/to/model.onnx
```

## Tests

From the `build` directory you can run the unit tests (GoogleTest):

```bash
ctest --output-on-failure -j4
# or
./build/bin/test_tensor
```

## Project layout (brief)

- `include/` — Public headers (core, graph, memory)
- `src/` — Implementation files
- `examples/` — Small example programs (`simple_inference.cpp`, `onnx_inference.cpp`)
- `tests/` — Unit tests
- `tools/` — Developer tools (e.g., `onnx_inspect`)

## Contributing

1. Fork the repository and create a feature branch.
2. Run tests and ensure new code has unit tests.
3. Open a PR with a clear description and changelog entry.

## License

This project is provided under the MIT license. See the `LICENSE` file for details.

---

