//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"
#include "InternalTypes.hpp"
namespace armnn
{

class ScopedCpuTensorHandle;

/// This layer represents a convolution 2d operation.
class Convolution2dLayer : public LayerWithParameters<Convolution2dDescriptor>
{
public:
    /// A unique pointer to store Weight values.
    ConstTensor m_Weight;
    /// A unique pointer to store Bias values.
    ConstTensor m_Bias;

 

    /// Constructor to create a Convolution2dLayer.
    /// @param [in] param Convolution2dDescriptor to configure the convolution2d operation.
    /// @param [in] name Optional name for the layer.
    Convolution2dLayer(const Convolution2dDescriptor& param, const char* name);

    /// Default destructor
    ~Convolution2dLayer() = default;

   
};

} // namespace
