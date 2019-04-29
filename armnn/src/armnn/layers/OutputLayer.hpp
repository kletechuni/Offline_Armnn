//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

/// A layer user-provided data can be bound to (e.g. inputs, outputs).
class OutputLayer : public BindableLayer
{
public:


    /// Constructor to create an OutputLayer.
    /// @param id The layer binding id number.
    /// @param name Optional name for the layer.
    OutputLayer(LayerBindingId id, const char* name);

    /// Default destructor
    ~OutputLayer() = default;
};

} // namespace
