#pragma once

#include "sip_qp/ordering.hpp"

namespace sip::qp {

auto solve(const Input &input, const Settings &settings, Workspace &workspace)
    -> Output;

} // namespace sip::qp
