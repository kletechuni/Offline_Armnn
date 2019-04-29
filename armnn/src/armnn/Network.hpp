//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include "LayersFwd.hpp"
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <armnn/INetwork.hpp>

#include <string>
#include <vector>
#include <memory>

#include "Layer.hpp"

namespace armnn
{

/// Private implementation of INetwork.
class Network final : public INetwork
{
public:
    Network();
    ~Network();

    // const Graph& GetGraph() const { return *m_Graph; }

    // Status PrintGraph() override;
    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name=nullptr) override;
    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name=nullptr) override;


    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) override;

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) override;

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor&                      weights,
        const char*                             name = nullptr) override;

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor&                      weights,
        const ConstTensor&                      biases,
        const char*                             name = nullptr) override;

   

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) override;

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) override;


    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr) override;


private:
    IConnectableLayer* AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);

    IConnectableLayer* AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);

    IConnectableLayer* AddDepthwiseConvolution2dLayerImpl(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);
   
    //std::unique_ptr<Graph> m_Graph;
    std::vector<Layer*> m_Graph;
};



} // namespace armnn
