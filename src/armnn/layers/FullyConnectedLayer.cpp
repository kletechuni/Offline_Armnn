//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FullyConnectedLayer.hpp"


namespace armnn
{

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::FullyConnected, param, name)
{
}


} // namespace armnn
