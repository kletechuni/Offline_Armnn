//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/NetworkFwd.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/TensorFwd.hpp>


#include <armnn/Types.hpp>

#include <memory>
#include <vector>

namespace armnn
{
    /// @brief An input connection slot for a layer.
/// The input slot can be connected to an output slot of the preceding layer in the graph.
/// Only one connection to the input slot is allowed.
class IInputSlot
{
public:
    virtual const IOutputSlot* GetConnection() const = 0;
    virtual IOutputSlot* GetConnection() = 0;

protected:
   /// Not user deletable.
    ~IInputSlot() {}
};

/// @brief An output connection slot for a layer.
/// The output slot may be connected to 1 or more input slots of subsequent layers in the graph.
class IOutputSlot
{
public:
    virtual unsigned int GetNumConnections() const = 0;
    virtual const IInputSlot* GetConnection(unsigned int index) const = 0;
    virtual IInputSlot* GetConnection(unsigned int index) = 0;

    virtual void SetTensorInfo(const TensorInfo& tensorInfo) = 0;
    virtual const TensorInfo& GetTensorInfo() const = 0;
    virtual bool IsTensorInfoSet() const = 0;

    virtual int Connect(IInputSlot& destination) = 0;
    virtual void Disconnect(IInputSlot& slot) = 0;

    virtual unsigned int CalculateIndexOnOwner() const = 0;

    virtual LayerGuid GetOwningLayerGuid() const = 0;

protected:
    /// Not user deletable.
    ~IOutputSlot() {}
};


/// @brief Interface for a layer that is connectable to other layers via InputSlots and OutputSlots.
class IConnectableLayer
{
public:
    virtual const char* GetName() const = 0;

    virtual unsigned int GetNumInputSlots() const = 0;
    virtual unsigned int GetNumOutputSlots() const = 0;

    virtual const IInputSlot& GetInputSlot(unsigned int index) const = 0;
    virtual IInputSlot& GetInputSlot(unsigned int index) = 0;

    virtual const IOutputSlot& GetOutputSlot(unsigned int index) const = 0;
    virtual IOutputSlot& GetOutputSlot(unsigned int index) = 0;

    virtual std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const = 0;

    virtual LayerGuid GetGuid() const = 0;

    
protected:
      /// Objects are not deletable via the handle
    ~IConnectableLayer() {}
};

using INetworkPtr = std::unique_ptr<INetwork, void(*)(INetwork* network)>;

/*  vinay karnam
class INetwork
{
public:
    static INetwork* CreateRaw();
    static INetworkPtr Create();
    static void Destroy(INetwork* network);

    protected:
    ~INetwork() {}

};
*/


/// Main network class which provides the interface for building up a neural network.
/// This object is subsequently required by the IRuntime::Load() method.
class INetwork
{
public:
    static INetwork* CreateRaw();
    static INetworkPtr Create();
    static void Destroy(INetwork* network);

    //virtual Status PrintGraph() = 0;

    /// Adds an input layer to the network.
    /// @param id - User generated id to uniquely identify a particular input. The same id needs to be specified.
    /// when passing the inputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name = nullptr) = 0;

    /// Adds a 2D convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - (Optional) Tensor for the bias data. Must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    /// Adds a 2D depthwise convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D depthwise convolution layer.
    /// @param weights - Tensor for the weights data. Expected format: [1, outputChannels, height, width].
    /// @param biases (Optional) - Tensor for the bias data. Must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    /// Adds a fully connected layer to the network.
    /// @param fullyConnectedDescriptor - Description of the fully connected layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - (Optional) Tensor for the bias data.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    
    /// Adds a pooling layer to the network.
    /// @param pooling2dDescriptor - Pooling2dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr) = 0;

    /// Adds an activation layer to the network.
    /// @param activationDescriptor - ActivationDescriptor to configure the activation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr) = 0;

    /// Adds a normalization layer to the network.
    /// @param normalizationDescriptor - NormalizationDescriptor to configure the normalization.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr) = 0;

    /// Adds a softmax layer to the network.
    /// @param softmaxDescriptor - SoftmaxDescriptor to configure the softmax.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr) = 0;

    
protected:
    ~INetwork() {}
};


}