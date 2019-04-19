//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a softmax operation.
class SoftmaxLayer : public LayerWithParameters<SoftmaxDescriptor>
{
public:
  
protected:
    /// Constructor to create a SoftmaxLayer.
    /// @param [in] param SoftmaxDescriptor to configure the softmax operation.
    /// @param [in] name Optional name for the layer.
    SoftmaxLayer(const SoftmaxDescriptor& param, const char* name);

    /// Default destructor
    ~SoftmaxLayer() = default;
};

} // namespace
