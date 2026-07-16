#include "tests/allocation_counter.hpp"

#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace {

bool tracking_allocations = false;
std::size_t num_allocations = 0;

void record_allocation() {
  if (tracking_allocations) {
    ++num_allocations;
  }
}

auto allocate(const std::size_t size) -> void * {
  record_allocation();
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc();
}

auto allocate_aligned(const std::size_t size, const std::size_t alignment)
    -> void * {
  record_allocation();
#ifdef _WIN32
  if (void *memory = _aligned_malloc(size == 0 ? 1 : size, alignment)) {
    return memory;
  }
#else
  void *memory = nullptr;
  if (posix_memalign(&memory, alignment, size == 0 ? 1 : size) == 0) {
    return memory;
  }
#endif
  throw std::bad_alloc();
}

void free_aligned(void *memory) noexcept {
#ifdef _WIN32
  _aligned_free(memory);
#else
  std::free(memory);
#endif
}

} // namespace

void *operator new(const std::size_t size) { return allocate(size); }

void *operator new[](const std::size_t size) { return allocate(size); }

void operator delete(void *memory) noexcept { std::free(memory); }

void operator delete[](void *memory) noexcept { std::free(memory); }

void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

void *operator new(const std::size_t size, const std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
}

void *operator new[](const std::size_t size, const std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<std::size_t>(alignment));
}

void operator delete(void *memory, std::align_val_t) noexcept {
  free_aligned(memory);
}

void operator delete[](void *memory, std::align_val_t) noexcept {
  free_aligned(memory);
}

void operator delete(void *memory, std::size_t, std::align_val_t) noexcept {
  free_aligned(memory);
}

void operator delete[](void *memory, std::size_t, std::align_val_t) noexcept {
  free_aligned(memory);
}

namespace allocation_counter {

void start() {
  num_allocations = 0;
  tracking_allocations = true;
}

auto stop() -> std::size_t {
  tracking_allocations = false;
  return num_allocations;
}

} // namespace allocation_counter
