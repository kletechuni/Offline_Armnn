//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SoftmaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

SoftmaxLayer::SoftmaxLayer(const SoftmaxDescriptor &param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Softmax, param, name)
{
}



} // namespace armnn
