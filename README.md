# Inference Engine

A high-performance inference engine for neural networks written in C++.

## Features

- **High Performance**: Scalar, SIMD (AVX2), and Multi-threaded kernels
- **Quantization Support**: INT8 quantization for reduced memory and faster inference
- **ONNX Support**: Load and execute ONNX models
- **Optimizations**: Layer fusion and constant folding
- **Cross-platform**: Linux, Windows, and macOS support

## Project Structure

```
inference_engine/
├── include/
│   └── inference_engine/
│       ├── core/
│       │   ├── tensor.h
│       │   ├── dtype.h
│       │   ├── shape.h
│       │   └── common.h
│       ├── memory/
│       │   ├── arena.h
│       │   ├── allocator.h
│       │   └── buffer.h
│       └── graph/
│           ├── graph.h
│           ├── node.h
│           ├── value.h
│           ├── operator.h
│           └── attributes.h
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   ├── dtype.cpp
│   │   ├── shape.cpp
│   │   └── common.cpp
│   ├── memory/
│   │   ├── arena.cpp
│   │   ├── allocator.cpp
│   │   └── buffer.cpp
│   └── graph/
│       ├── graph.cpp
│       ├── node.cpp
│       ├── value.cpp
│       ├── operator.cpp
│       └── attributes.cpp
├── tests/
│   ├── core/
│   │   ├── test_tensor.cpp
│   │   ├── test_shape.cpp
│   │   └── test_dtype.cpp
│   ├── memory/
│   │   ├── test_arena.cpp
│   │   └── test_allocator.cpp
│   └── graph/
│       ├── test_graph.cpp
│       └── test_node.cpp
├── CMakeLists.txt
└── README.md
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running Examples

```bash
./infer_example
./onnx_inference
./benchmark
```

## Running Tests

```bash
cd build
ctest
```

or run individual tests:

```bash
./test_tensor
./test_shape
./test_dtype
./test_arena
./test_allocator
./test_graph
./test_node
```

## Dependencies

- CMake 3.21+
- C++17 compatible compiler
- OpenMP (for multi-threading)
- Google Test (for unit tests, optional)

## CPU Feature Detection

The engine automatically detects and uses available CPU features:
- AVX2 for vectorized operations
- AVX-512 (if available)
- NEON on ARM platforms

## Quantization

Reduce model size and improve inference speed with INT8 quantization:

```bash
./quantize.py model.onnx -o model_int8.onnx
```

## Performance Notes

- Scalar implementation: Baseline performance
- SIMD implementation: ~8x faster on modern CPUs with AVX2
- Multi-threaded: Scales linearly with number of threads

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All tests pass
- New features include tests

## License

Apache License 2.0
