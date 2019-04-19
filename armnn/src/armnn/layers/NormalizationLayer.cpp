//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

NormalizationLayer::NormalizationLayer(const NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Normalization, param, name)
{
}


} // namespace armnn
