//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InputLayer.hpp"

#include "InternalTypes.hpp"



namespace armnn
{

InputLayer::InputLayer(LayerBindingId id, const char* name)
    : BindableLayer(0, 1, LayerType::Input, name, id)
{
}

} // namespace
