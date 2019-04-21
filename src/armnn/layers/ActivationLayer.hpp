//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{
/// This layer represents an activation operation with the specified activation function.
class ActivationLayer : public LayerWithParameters<ActivationDescriptor>
{
public:
  


protected:
    /// Constructor to create an ActivationLayer.
    /// @param [in] param ActivationDescriptor to configure the activation operation.
    /// @param [in] name Optional name for the layer.
    ActivationLayer(const ActivationDescriptor &param, const char* name);

    /// Default destructor
    ~ActivationLayer() = default;
};

} // namespace
