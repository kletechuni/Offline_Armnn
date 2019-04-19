//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

/// A layer user-provided data can be bound to (e.g. inputs, outputs).
class InputLayer : public BindableLayer
{
public:
  
protected:
    /// Constructor to create an InputLayer.
    /// @param id The layer binding id number.
    /// @param name Optional name for the layer.
    InputLayer(LayerBindingId id, const char* name);

    /// Default destructor
    ~InputLayer() = default;
};

} // namespace
