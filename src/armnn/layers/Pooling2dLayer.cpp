//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Pooling2dLayer.hpp"


using namespace armnnUtils;

namespace armnn
{

Pooling2dLayer::Pooling2dLayer(const Pooling2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pooling2d, param, name)
{
}


} // namespace armnn
