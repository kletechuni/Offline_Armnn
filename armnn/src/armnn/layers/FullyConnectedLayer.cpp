//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FullyConnectedLayer.hpp"
#include "Layer.hpp"
#include "InternalTypes.hpp"
namespace armnn
{

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::FullyConnected, param, name)
{
}

// ConstantTensors FullyConnectedLayer::GetConstantTensorsByRef()
// {
//     return {m_Weight,m_Bias};
// }

} // namespace armnn
