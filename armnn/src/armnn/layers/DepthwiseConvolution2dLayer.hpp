//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

/// This layer represents a depthwise convolution 2d operation.
class DepthwiseConvolution2dLayer : public LayerWithParameters<DepthwiseConvolution2dDescriptor>
{
public:
    /// A unique pointer to store Weight values.
    ConstTensor m_Weight;
    /// A unique pointer to store Bias values.
    ConstTensor m_Bias;



    /// Constructor to create a DepthwiseConvolution2dLayer.
    /// @param [in] param DepthwiseConvolution2dDescriptor to configure the depthwise convolution2d.
    /// @param [in] name Optional name for the layer.
    DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param, const char* name);

    /// Default destructor
    ~DepthwiseConvolution2dLayer() = default;

};

} // namespace
