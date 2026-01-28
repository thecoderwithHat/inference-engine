cmake_minimum_required(VERSION 3.21)
project(inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SIMD "Enable AVX2" ON)
option(ENABLE_MT   "Enable Multithreading" ON)
option(BUILD_TESTS "Build unit tests" ON)

# Core library
add_library(infer_engine
    # Core components
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp
    ${CMAKE_SOURCE_DIR}/src/core/shape.cpp
    ${CMAKE_SOURCE_DIR}/src/core/common.cpp
    ${CMAKE_SOURCE_DIR}/src/core/model.cpp
    
    # Memory components
    ${CMAKE_SOURCE_DIR}/src/memory/arena.cpp
    ${CMAKE_SOURCE_DIR}/src/memory/allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/memory/buffer.cpp
    
    # Graph components
    ${CMAKE_SOURCE_DIR}/src/graph/graph.cpp
    ${CMAKE_SOURCE_DIR}/src/graph/node.cpp
    ${CMAKE_SOURCE_DIR}/src/graph/value.cpp
    ${CMAKE_SOURCE_DIR}/src/graph/operator.cpp
    ${CMAKE_SOURCE_DIR}/src/graph/attributes.cpp
)

target_include_directories(infer_engine PUBLIC include)

# Enable AVX2
if (ENABLE_SIMD)
    if (MSVC)
        target_compile_options(infer_engine PRIVATE /arch:AVX2)
    else()
        target_compile_options(infer_engine PRIVATE -mavx2)
    endif()
endif()

# Enable multithreading
if (ENABLE_MT)
    find_package(OpenMP REQUIRED)
    target_link_libraries(infer_engine PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(infer_engine PRIVATE ENABLE_MT)
endif()

# Examples
add_executable(infer_example ${CMAKE_SOURCE_DIR}/examples/simple_inference.cpp)
target_link_libraries(infer_example PRIVATE infer_engine)

add_executable(onnx_inference ${CMAKE_SOURCE_DIR}/examples/onnx_inference.cpp)
target_link_libraries(onnx_inference PRIVATE infer_engine)

add_executable(benchmark ${CMAKE_SOURCE_DIR}/examples/benchmark.cpp)
target_link_libraries(benchmark PRIVATE infer_engine)

# Tools
add_executable(onnx_inspect ${CMAKE_SOURCE_DIR}/tools/onnx_inspect.cpp)

# Tests
if (BUILD_TESTS)
    enable_testing()

    # Always build at least one framework-free smoke test so `ctest` works
    # even when GoogleTest isn't available on the system.
    # add_executable(test_dtype_quant_smoke ${CMAKE_SOURCE_DIR}/tests/test_dtype_quant_smoke.cpp)
    # target_link_libraries(test_dtype_quant_smoke PRIVATE infer_engine)
    # add_test(NAME DTypeQuantSmoke COMMAND test_dtype_quant_smoke)

    add_executable(test_dtype ${CMAKE_SOURCE_DIR}/tests/core/test_dtype.cpp)
    target_link_libraries(test_dtype PRIVATE infer_engine)
    add_test(NAME DType COMMAND test_dtype)

    add_executable(test_quant ${CMAKE_SOURCE_DIR}/tests/test_quant.cpp)
    target_link_libraries(test_quant PRIVATE infer_engine)
    add_test(NAME Quant COMMAND test_quant)

    # Framework-free tests (do not require GoogleTest)
    add_executable(test_tensor ${CMAKE_SOURCE_DIR}/tests/core/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE infer_engine)
    add_test(NAME TensorTests COMMAND test_tensor)

    add_executable(test_shape ${CMAKE_SOURCE_DIR}/tests/core/test_shape.cpp)
    target_link_libraries(test_shape PRIVATE infer_engine)
    add_test(NAME ShapeTests COMMAND test_shape)
    
    find_package(GTest QUIET)
    
    if (GTEST_FOUND)
        add_executable(test_arena ${CMAKE_SOURCE_DIR}/tests/memory/test_arena.cpp)
        target_link_libraries(test_arena PRIVATE infer_engine GTest::gtest_main)
        add_test(NAME ArenaTests COMMAND test_arena)
        
        add_executable(test_allocator ${CMAKE_SOURCE_DIR}/tests/memory/test_allocator.cpp)
        target_link_libraries(test_allocator PRIVATE infer_engine GTest::gtest_main)
        add_test(NAME AllocatorTests COMMAND test_allocator)
        
        add_executable(test_graph ${CMAKE_SOURCE_DIR}/tests/graph/test_graph.cpp)
        target_link_libraries(test_graph PRIVATE infer_engine GTest::gtest_main)
        add_test(NAME GraphTests COMMAND test_graph)
        
        add_executable(test_node ${CMAKE_SOURCE_DIR}/tests/graph/test_node.cpp)
        target_link_libraries(test_node PRIVATE infer_engine GTest::gtest_main)
        add_test(NAME NodeTests COMMAND test_node)
    else()
        message(STATUS "Google Test not found. Skipping unit tests.")
    endif()
endif()
