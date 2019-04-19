//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OutputLayer.hpp"


namespace armnn
{

OutputLayer::OutputLayer(LayerBindingId id, const char* name)
    : BindableLayer(1, 0, LayerType::Output, name, id)
{
}


} // namespace armnn
