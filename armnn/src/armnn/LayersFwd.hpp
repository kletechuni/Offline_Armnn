//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once


#include "layers/ActivationLayer.hpp"
#include "layers/Convolution2dLayer.hpp"
#include "layers/DepthwiseConvolution2dLayer.hpp"
#include "layers/FullyConnectedLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/NormalizationLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/Pooling2dLayer.hpp"
#include "layers/SoftmaxLayer.hpp"

namespace armnn
{

template <LayerType Type>
struct LayerTypeOfImpl;

template <LayerType Type>
using LayerTypeOf = typename LayerTypeOfImpl<Type>::Type;

template <typename T>
constexpr LayerType LayerEnumOf(const T* = nullptr);

#define DECLARE_LAYER_IMPL(_, LayerName)                     \
    class LayerName##Layer;                                  \
    template <>                                              \
    struct LayerTypeOfImpl<LayerType::_##LayerName>          \
    {                                                        \
        using Type = LayerName##Layer;                       \
    };                                                       \
    template <>                                              \
    constexpr LayerType LayerEnumOf(const LayerName##Layer*) \
    {                                                        \
        return LayerType::_##LayerName;                      \
    }

#define DECLARE_LAYER(LayerName) DECLARE_LAYER_IMPL(, LayerName)

DECLARE_LAYER(Activation)
DECLARE_LAYER(Convolution2d)
DECLARE_LAYER(DepthwiseConvolution2d)
DECLARE_LAYER(FullyConnected)
DECLARE_LAYER(Input)
DECLARE_LAYER(Normalization)
DECLARE_LAYER(Pooling2d)
DECLARE_LAYER(Output)
DECLARE_LAYER(Softmax)

}
