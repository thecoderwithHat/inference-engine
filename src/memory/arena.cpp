#include "inference_engine/memory/arena.h"
#include <cstring>

namespace infer {

Arena::Arena(size_t capacity) : capacity_(capacity), used_(0), buffer_(std::make_unique<char[]>(capacity)) {
}

Arena::~Arena() = default;

void* Arena::allocate(size_t size) {
    if (used_ + size > capacity_) {
        return nullptr;
    }
    void* ptr = buffer_.get() + used_;
    used_ += size;
    return ptr;
}

void Arena::reset() {
    used_ = 0;
}

} // namespace infer
