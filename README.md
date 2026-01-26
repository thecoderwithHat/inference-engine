# Inference Engine

A high-performance inference engine for neural networks written in C++.

## Features

- **High Performance**: Scalar, SIMD (AVX2), and Multi-threaded kernels
- **Quantization Support**: INT8 quantization for reduced memory and faster inference
- **ONNX Support**: Load and execute ONNX models
- **Optimizations**: Layer fusion and constant folding
- **Cross-platform**: Linux, Windows, and macOS support


Core Component Files
include/inference_engine/core/common.h
Purpose: Common definitions, macros, platform detection, and utilities used throughout the codebase.
Contents:

Platform detection macros (x86, ARM, Windows, Linux)
Compiler-specific attributes (alignment, inline, restrict)
Assertion macros for debug vs release
Common constants (alignment values, max dimensions)
Status/error code enumerations
Logging macros
Utility macros for array sizes, min/max

include/inference_engine/core/dtype.h
Purpose: Data type system for all tensor operations.
Contents:

DataType enumeration (Float32, Float16, Int8, UInt8, Int32, Int64, Bool)
Size lookup function (bytes per element)
Alignment requirements per type
String conversion for debugging
Type trait helpers (is_floating_point, is_integer, etc.)
Quantization parameter structures for INT8 types

src/core/dtype.cpp
Implementation of dtype.h functions.

include/inference_engine/core/shape.h
Purpose: Tensor shape representation and manipulation.
Contents:

Shape class holding dimensions vector
Rank accessor (number of dimensions)
Dimension accessor by index
Total element count calculation
Shape equality comparison
Shape broadcasting logic
Utility functions (squeeze, unsqueeze, reshape validation)
Stride calculation from shape

src/core/shape.cpp
Implementation of shape manipulation logic.

include/inference_engine/core/tensor.h
Purpose: Main tensor abstraction - the workhorse of your engine.
Contents:

Tensor class with metadata (shape, dtype, strides)
Pointer to data buffer
Memory ownership flag
Allocation from allocator
Shape and dtype accessors
Data pointer accessors (typed and void*)
Element count and byte size calculations
Contiguity checks
View creation (slice, reshape without copy)
Debug utilities (print shape, validate)
Quantization parameters if dtype is INT8

src/core/tensor.cpp
Implementation of tensor operations and utilities.

Memory Management Files
include/inference_engine/memory/buffer.h
Purpose: Raw memory buffer abstraction with ownership semantics.
Contents:

Buffer class wrapping void* data
Size and alignment tracking
Ownership flag (owned vs borrowed)
Allocate and deallocate methods
Copy and move semantics
Alignment validation
Debug guards (canary values for overflow detection)

src/memory/buffer.cpp
Implementation of buffer allocation and management.

include/inference_engine/memory/arena.h
Purpose: Fast bump allocator for inference workloads.
Contents:

Arena class with large pre-allocated buffer
Current offset tracking
Allocate method (bump pointer, return aligned address)
Reset method (zero offset without freeing)
Capacity and used space queries
Alignment enforcement
Optional memory pool statistics (allocations, peak usage)
Thread safety considerations (mark if thread-local or locked)

src/memory/arena.cpp
Implementation of arena allocation logic.

include/inference_engine/memory/allocator.h
Purpose: High-level allocator interface that can use different backends.
Contents:

Abstract Allocator interface
allocate() and deallocate() virtual methods
ArenaAllocator implementation using Arena
SystemAllocator implementation using malloc/free
Allocator factory methods
Memory alignment parameters
Allocation tracking for debugging

src/memory/allocator.cpp
Implementation of various allocator strategies.

Graph IR Files
include/inference_engine/graph/attributes.h
Purpose: Operation attribute storage (compile-time parameters).
Contents:

Attribute variant type (int, float, string, array of these)
AttributeMap class (string key to attribute value)
Type-safe get/set methods
Existence checks
Serialization helpers for debugging
Common attribute names as constants

src/graph/attributes.cpp
Implementation of attribute storage and retrieval.

include/inference_engine/graph/value.h
Purpose: Represents a value in the graph (abstract tensor reference).
Contents:

Value class with unique identifier
Shape and dtype information
Producer node reference (which op creates this value)
Consumer node list (which ops use this value)
Name for debugging
Actual tensor pointer (nullptr during graph construction, filled during execution)
Quantization info if needed

src/graph/value.cpp
Implementation of value tracking and relationships.

include/inference_engine/graph/operator.h
Purpose: Base class for all operations.
Contents:

Abstract Operator base class
Operation type string
Virtual execute() method (pure virtual)
Input and output value lists
Attribute map reference
Validation method (check shapes, dtypes)
Memory requirement estimation
Clone/copy for graph optimization

src/graph/operator.cpp
Implementation of operator base class utilities.

include/inference_engine/graph/node.h
Purpose: Graph node wrapping an operator instance.
Contents:

Node class with unique ID and name
Operator instance pointer
Input value references
Output value references
Parent graph reference
Topological order index (set during sort)
Execution state flags for scheduling
Debug information

src/graph/node.cpp
Implementation of node management.

include/inference_engine/graph/graph.h
Purpose: Complete computational graph representation.
Contents:

Graph class holding all nodes and values
Node addition and removal
Value creation and registration
Input value list (graph inputs)
Output value list (graph outputs)
Topological sort method
Validation (check for cycles, dangling references)
Memory planning method (analyze lifetimes)
Graph-level attributes (model name, version)
Optimization pass application

src/graph/graph.cpp
Implementation of graph construction and analysis algorithms (topological sort, validation, etc.).

Test Files
tests/core/test_tensor.cpp
Tests for tensor functionality:

Creation with different shapes and dtypes
Memory layout verification
Stride calculations
View creation
Shape manipulation
Data access patterns
Quantization parameter handling

tests/core/test_shape.cpp
Tests for shape operations:

Broadcasting rules
Reshape validation
Dimension manipulation
Element count calculations
Edge cases (scalar, zero dimensions)

tests/core/test_dtype.cpp
Tests for data type system:

Size calculations
Alignment requirements
Type conversions
Quantization parameter structures

tests/memory/test_arena.cpp
Tests for arena allocator:

Basic allocation and reset
Alignment guarantees
Capacity limits
Multiple allocation patterns
Memory reuse after reset
Peak usage tracking

tests/memory/test_allocator.cpp
Tests for allocator interface:

Different backend implementations
Allocation failures
Thread safety if applicable
Memory leak detection

tests/graph/test_graph.cpp
Tests for graph operations:

Node addition
Edge creation
Topological sort correctness
Cycle detection
Input/output marking

tests/graph/test_node.cpp
Tests for node functionality:

Operator attachment
Input/output connection
Attribute handling


Build Configuration
CMakeLists.txt
Purpose: Build system configuration.
Contents:

Project declaration and versioning
C++ standard requirement (C++17 minimum)
Compiler flags (optimization, warnings, SIMD)
Include directories
Library target definitions
Test target with Google Test
Installation rules
Optional: build type configurations (Debug/Release)

README.md
Purpose: Project documentation and setup instructions.
Contents:

Project overview
Build instructions
Dependency requirements
Architecture overview
Usage examples
Development roadmap
Contributing guidelines


Additional Supporting Files
include/inference_engine/core/status.h (optional but recommended)
Purpose: Error handling and status codes.
Contents:

Status enumeration (Success, InvalidArgument, OutOfMemory, etc.)
Status class with error messages
Success checking macros
Error propagation helpers

include/inference_engine/core/logger.h (optional but recommended)
Purpose: Logging infrastructure.
Contents:

Log level enumeration
Logging macros (LOG_INFO, LOG_ERROR, etc.)
Logger backend interface
Console logger implementation


Key Design Principles Reflected in Files
Separation of interface and implementation: All headers in include/, all implementations in src/. This keeps compilation fast and interfaces clean.
Logical grouping: Core abstractions, memory management, and graph IR are separate concerns with their own directories.
Testing isolation: Each component has corresponding test files, enabling test-driven development.
Minimal dependencies: Core files don't depend on graph files. Memory files don't depend on graph. This creates clear layering.
Extensibility points: Operator and Allocator are abstract base classes, making it easy to add new implementations without modifying existing code.
This file structure gives you a solid foundation that's maintainable, testable, and ready to scale as you add kernels, schedulers, and frontends in later phases.