//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NormalizationLayer.hpp"

#include "InternalTypes.hpp"

namespace armnn
{

NormalizationLayer::NormalizationLayer(const NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Normalization, param, name)
{
}


} // namespace armnn
