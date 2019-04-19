//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
//#include "Graph.hpp"
#include "Layer.hpp"


#include <fcntl.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/cast.hpp>

namespace armnn
{

armnn::INetwork* INetwork::CreateRaw()
{
    return new Network();
}

armnn::INetworkPtr INetwork::Create()
{
    return INetworkPtr(CreateRaw(), &INetwork::Destroy);
}

void INetwork::Destroy(INetwork* network)
{
    delete boost::polymorphic_downcast<Network*>(network);
}

// Status Network::PrintGraph()
// {
//     m_Graph->Print();
//     return Status::Success;
// }



Network::Network()
: m_Graph(std::make_unique<Graph>())
{
}

Network::~Network()
{
}

IConnectableLayer* Network::AddInputLayer(LayerBindingId id, const char* name)
{
    //return m_Graph->AddLayer<InputLayer>(id, name);
    Layer l=new InputLayer(id,name);
    m_Graph.push_back(l);
    return l;
}


IConnectableLayer* Network::AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                       const ConstTensor& weights,
                                                       const ConstTensor* biases,
                                                       const char* name)
{
    if (fullyConnectedDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddFullyConnectedLayer: biases cannot be NULL");
    }

    //const auto layer = m_Graph->AddLayer<FullyConnectedLayer>(fullyConnectedDescriptor, name);
    Layer layer=new FullyConnectedLayer(fullyConnectedDescriptor,name);
    layer->m_Weight = weights;

    if (fullyConnectedDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = *biases;
    }
    m_Graph.push_back(layer);
    return layer;
} 

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const char* name)
{
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, nullptr, name);
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const ConstTensor& biases,
                                                   const char* name)
{
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, &biases, name);
}

IConnectableLayer* Network::AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
                                                      const ConstTensor& weights,
                                                      const ConstTensor* biases,
                                                      const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddConvolution2dLayer: biases cannot be NULL");
    }

    //const auto layer = m_Graph->AddLayer<Convolution2dLayer>(convolution2dDescriptor, name);
    Layer layer = new Convolution2dLayer(convolution2dDescriptor,name);
    layer->m_Weight = weights;

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = *biases;
    }
    layer.push_back(layer);
    return layer;
}

IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const char* name)
{
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, nullptr, name);
}
IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const ConstTensor& biases,
                                                  const char* name)
{
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, &biases, name);
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayerImpl(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const ConstTensor* biases,
    const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddDepthwiseConvolution2dLayer: biases cannot be NULL");
    }

    //const auto layer = m_Graph->AddLayer<DepthwiseConvolution2dLayer>(convolution2dDescriptor, name);
    Layer layer = new DepthwiseConvolution2dLayer(convolution2dDescriptor,name);
    layer->m_Weight = weights;

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = *biases;
    }
    m_Graph.push_back(layer);

    return layer;
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const char* name)
{
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, nullptr, name);
}
IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const ConstTensor& biases,
    const char* name)
{
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, &biases, name);
}


IConnectableLayer* Network::AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
    const char* name)
{
   // return m_Graph->AddLayer<Pooling2dLayer>(pooling2dDescriptor, name);
   Layer layer = new Pooling2dLayer(pooling2dDescriptor,name);
   return layer;
}

IConnectableLayer* Network::AddActivationLayer(const ActivationDescriptor& activationDescriptor,
    const char* name)
{
    //return m_Graph->AddLayer<ActivationLayer>(activationDescriptor, name);
    Layer layer=new ActivationLayer(activationDescriptor,name);
    m_Graph.push_back(layer);
}

IConnectableLayer* Network::AddNormalizationLayer(const NormalizationDescriptor&
normalizationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<NormalizationLayer>(normalizationDescriptor, name);
}

IConnectableLayer* Network::AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SoftmaxLayer>(softmaxDescriptor, name);
}




IConnectableLayer* Network::AddOutputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<OutputLayer>(id, name);
}


// void Network::Accept(ILayerVisitor& visitor) const
// {
//     for (auto layer : GetGraph())
//     {
//         layer->Accept(visitor);
//     };
// }

OptimizedNetwork::OptimizedNetwork(std::unique_ptr<Graph> graph)
    : m_Graph(std::move(graph))
{
}

OptimizedNetwork::~OptimizedNetwork()
{
}

} // namespace armnn
