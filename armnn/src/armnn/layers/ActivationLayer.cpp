//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ActivationLayer.hpp"
#include "InternalTypes.hpp"




namespace armnn
{

ActivationLayer::ActivationLayer(const ActivationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Activation, param, name)
{
}


} // namespace armnn
