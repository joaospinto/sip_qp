#pragma once

#include <cstddef>

namespace allocation_counter {

void start();
auto stop() -> std::size_t;

} // namespace allocation_counter
