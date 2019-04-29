//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"
#include<Layer.hpp>
namespace armnn
{



/// This layer represents a fully connected operation.
class FullyConnectedLayer : public LayerWithParameters<FullyConnectedDescriptor>
{
public:
    /// A unique pointer to store Weight values.
    ConstTensor m_Weight;
    /// A unique pointer to store Bias values.
    ConstTensor m_Bias;



    /// Constructor to create a FullyConnectedLayer.
    /// @param [in] param FullyConnectedDescriptor to configure the fully connected operation.
    /// @param [in] name Optional name for the layer.
    FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name);

    /// Default destructor
    ~FullyConnectedLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    //  ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace
