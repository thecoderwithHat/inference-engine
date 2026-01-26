#ifndef INFERENCE_ENGINE_CORE_COMMON_H_
#define INFERENCE_ENGINE_CORE_COMMON_H_

/*
 * Common definitions for the inference engine core.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

namespace inference_engine {
namespace core {

/* -------------------- Platform detection -------------------- */
#if defined(_WIN32) || defined(_WIN64)
#  define IE_PLATFORM_WINDOWS 1
#else
#  define IE_PLATFORM_WINDOWS 0
#endif

#if defined(__APPLE__) && defined(__MACH__)
#  define IE_PLATFORM_APPLE 1
#else
#  define IE_PLATFORM_APPLE 0
#endif

#if defined(__linux__) || defined(__linux)
#  define IE_PLATFORM_LINUX 1
#else
#  define IE_PLATFORM_LINUX 0
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#  define IE_ARCH_X86 1
#else
#  define IE_ARCH_X86 0
#endif

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#  define IE_ARCH_ARM 1
#else
#  define IE_ARCH_ARM 0
#endif

/* -------------------- Compiler detection and attributes -------------------- */
#if defined(_MSC_VER)
#  define IE_COMPILER_MSVC 1
#else
#  define IE_COMPILER_MSVC 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define IE_COMPILER_GCC 1
#else
#  define IE_COMPILER_GCC 0
#endif

#if defined(__clang__)
#  define IE_COMPILER_CLANG 1
#else
#  define IE_COMPILER_CLANG 0
#endif

#if IE_COMPILER_MSVC
#  define IE_INLINE __inline
#  define IE_FORCE_INLINE __forceinline
#  define IE_NOINLINE __declspec(noinline)
#else
#  define IE_INLINE inline
#  define IE_FORCE_INLINE inline __attribute__((always_inline))
#  define IE_NOINLINE __attribute__((noinline))
#endif

#if IE_COMPILER_MSVC
#  define IE_ALIGNAS(x) __declspec(align(x))
#else
#  define IE_ALIGNAS(x) alignas(x)
#endif

#if IE_COMPILER_MSVC
#  define IE_RESTRICT __restrict
#else
#  if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#    define IE_RESTRICT restrict
#  else
#    define IE_RESTRICT __restrict__
#  endif
#endif

#if (IE_COMPILER_GCC || IE_COMPILER_CLANG)
#  define IE_LIKELY(x)   (__builtin_expect(!!(x), 1))
#  define IE_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#  define IE_UNUSED __attribute__((unused))
#  define IE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#  define IE_LIKELY(x)   (x)
#  define IE_UNLIKELY(x) (x)
#  define IE_UNUSED
#  if IE_COMPILER_MSVC
#    define IE_DEPRECATED(msg) __declspec(deprecated(msg))
#  else
#    define IE_DEPRECATED(msg)
#  endif
#endif

/* -------------------- Assertions -------------------- */
#ifndef NDEBUG
#  define INF_ENGINE_ASSERT(cond) \
     do { \
       if (!(cond)) { \
         std::fprintf(stderr, "Assertion failed: %s (%s:%d) in %s\n", #cond, __FILE__, __LINE__, __func__); \
         std::abort(); \
       } \
     } while (0)
#else
#  define INF_ENGINE_ASSERT(cond) ((void)0)
#endif

#define INF_ENGINE_CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::fprintf(stderr, "Check failed: %s (%s:%d) in %s\n", #cond, __FILE__, __LINE__, __func__); \
      std::abort(); \
    } \
  } while (0)

/* UNIMPLEMENTED helper */
#define INF_ENGINE_UNIMPLEMENTED() \
  do { \
    std::fprintf(stderr, "Unimplemented code path: %s (%s:%d) in %s\n", __func__, __FILE__, __LINE__, __func__); \
    std::abort(); \
  } while (0)

/* -------------------- Common constants -------------------- */
static constexpr std::size_t INF_ENGINE_DEFAULT_ALIGNMENT = 64u;
static constexpr std::size_t INF_ENGINE_MIN_ALIGNMENT = 8u;
static constexpr int INF_ENGINE_MAX_DIMS = 8;
static constexpr std::size_t INF_ENGINE_CACHE_LINE_SIZE = 64u;

/* -------------------- Status / Error codes -------------------- */
enum class StatusCode : int {
    OK = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_MEMORY = 2,
    NOT_IMPLEMENTED = 3,
    RUNTIME_ERROR = 4,
    NETWORK_NOT_FOUND = 5,
    MODEL_MISMATCH = 6,
    TIMEOUT = 7,
    UNKNOWN = -1
};

inline const char* status_code_to_string(StatusCode code) noexcept {
    switch (code) {
    case StatusCode::OK: return "OK";
    case StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case StatusCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case StatusCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case StatusCode::RUNTIME_ERROR: return "RUNTIME_ERROR";
    case StatusCode::NETWORK_NOT_FOUND: return "NETWORK_NOT_FOUND";
    case StatusCode::MODEL_MISMATCH: return "MODEL_MISMATCH";
    case StatusCode::TIMEOUT: return "TIMEOUT";
    default: return "UNKNOWN";
    }
}

/* -------------------- Logging -------------------- */
#ifndef INF_ENGINE_ENABLE_LOGS
#  define INF_ENGINE_ENABLE_LOGS 1
#endif

#if INF_ENGINE_ENABLE_LOGS
#  define IE_LOG_INTERNAL(level, fmt, ...) \
     do { std::fprintf(stderr, "[" level "] %s:%d %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); } while (0)
#  define IE_LOG_DEBUG(fmt, ...) IE_LOG_INTERNAL("DEBUG", fmt, ##__VA_ARGS__)
#  define IE_LOG_INFO(fmt, ...)  IE_LOG_INTERNAL("INFO",  fmt, ##__VA_ARGS__)
#  define IE_LOG_WARN(fmt, ...)  IE_LOG_INTERNAL("WARN",  fmt, ##__VA_ARGS__)
#  define IE_LOG_ERROR(fmt, ...) IE_LOG_INTERNAL("ERROR", fmt, ##__VA_ARGS__)
#else
#  define IE_LOG_DEBUG(fmt, ...) ((void)0)
#  define IE_LOG_INFO(fmt, ...)  ((void)0)
#  define IE_LOG_WARN(fmt, ...)  ((void)0)
#  define IE_LOG_ERROR(fmt, ...) ((void)0)
#endif

/* -------------------- Utility macros / functions -------------------- */
#define INF_ENGINE_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#ifndef INF_ENGINE_MIN
#  define INF_ENGINE_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef INF_ENGINE_MAX
#  define INF_ENGINE_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef INF_ENGINE_CLAMP
#  define INF_ENGINE_CLAMP(x, lo, hi) ( (x) < (lo) ? (lo) : ( (x) > (hi) ? (hi) : (x) ) )
#endif

#define INF_ENGINE_UNUSED_PARAM(x) (void)(x)

static inline void* inf_engine_align_ptr(void* ptr, std::size_t align) noexcept {
    std::uintptr_t p = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t a = (static_cast<std::uintptr_t>(align) - 1u);
    return reinterpret_cast<void*>((p + a) & ~a);
}

static inline std::size_t inf_engine_align_size(std::size_t size, std::size_t align) noexcept {
    const std::size_t a = (align - 1u);
    return (size + a) & ~a;
}

} // namespace core
} // namespace inference_engine

#endif // INFERENCE_ENGINE_CORE_COMMON_H_