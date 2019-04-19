//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a pooling 2d operation.
class Pooling2dLayer : public LayerWithParameters<Pooling2dDescriptor>
{
public:


protected:
    /// Constructor to create a Pooling2dLayer.
    /// @param [in] param Pooling2dDescriptor to configure the pooling2d operation.
    /// @param [in] name Optional name for the layer.
    Pooling2dLayer(const Pooling2dDescriptor& param, const char* name);

    /// Default destructor
    ~Pooling2dLayer() = default;
};

} // namespace
