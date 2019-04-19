//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a normalization operation.
class NormalizationLayer : public LayerWithParameters<NormalizationDescriptor>
{
public:

protected:
    /// Constructor to create a NormalizationLayer.
    /// @param [in] param NormalizationDescriptor to configure the normalization operation.
    /// @param [in] name Optional name for the layer.
    NormalizationLayer(const NormalizationDescriptor& param, const char* name);

    /// Default destructor
    ~NormalizationLayer() = default;
};

} // namespace
