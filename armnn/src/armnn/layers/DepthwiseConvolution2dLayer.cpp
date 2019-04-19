//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DepthwiseConvolution2dLayer.hpp"


using namespace armnnUtils;

namespace armnn
{

DepthwiseConvolution2dLayer::DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param,
                                                         const char* name)
    : LayerWithParameters(1, 1, LayerType::DepthwiseConvolution2d, param, name)
{
}


} // namespace armnn
