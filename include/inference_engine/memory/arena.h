#pragma once

#include <memory>
#include <cstddef>

namespace infer {

class Arena {
public:
    explicit Arena(size_t capacity);
    ~Arena();
    
    void* allocate(size_t size);
    void reset();

private:
    size_t capacity_;
    size_t used_;
    std::unique_ptr<char[]> buffer_;
};

} // namespace infer
