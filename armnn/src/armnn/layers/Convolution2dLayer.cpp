//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Convolution2dLayer.hpp"
#include "InternalTypes.hpp"



namespace armnn
{

Convolution2dLayer::Convolution2dLayer(const Convolution2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Convolution2d, param, name)
{
}


} // namespace armnn
