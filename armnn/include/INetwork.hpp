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

/// Main network class which provides the interface for building up a neural network.
/// This object is subsequently required by the IRuntime::Load() method.

class INetwork
{
public:
    static INetwork* CreateRaw();
    static INetworkPtr Create();
    static void Destroy(INetwork* network);

    protected:
    ~INetwork() {}

};

}