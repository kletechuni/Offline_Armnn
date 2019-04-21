//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Layer.hpp"




#include <boost/cast.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>


#include <numeric>

namespace armnn
{

void InputSlot::Insert(Layer& layer)
{
    BOOST_ASSERT(layer.GetNumOutputSlots() == 1);

    OutputSlot* const prevSlot = GetConnectedOutputSlot();

    if (prevSlot != nullptr)
    {
        // Disconnects parent from this.
        prevSlot->Disconnect(*this);

        // Connects inserted layer to parent.
        BOOST_ASSERT(layer.GetNumInputSlots() == 1);
        prevSlot->Connect(layer.GetInputSlot(0));

        // Sets tensor info for inserted layer.
        const TensorInfo& tensorInfo = prevSlot->GetTensorInfo();
        .SetTensorInfo(tensorInfo);
    }

    // Connects inserted layer to this.
    layer.GetOutputSlot(0).Connect(*this);
}

const InputSlot* OutputSlot::GetConnection(unsigned int index) const
{
    ValidateConnectionIndex(index);
    return m_Connections[index];
}

InputSlot* OutputSlot::GetConnection(unsigned int index)
{
    ValidateConnectionIndex(index);
    return m_Connections[index];
}


//changes has made by removing outhandler n setting tensorinfo here itself
//m 1
void OutputSlot::SetTensorInfo(const TensorInfo& tensorInfo)
{
    m_TensorInfo = tensorInfo;
    m_bTensorInfoSet = true;
}

/// @brief - Gets the matching TensorInfo for the output.
    /// @return - References to the output TensorInfo.
    //m 2
    const TensorInfo& GetTensorInfo() const { return m_TensorInfo; }

  // modified 3
   bool IsTensorInfoSet() const { return m_bTensorInfoSet; }


//modified 4
bool OutputSlot::ValidateTensorShape(const TensorShape& shape) const
{
    BOOST_ASSERT_MSG(IsTensorInfoSet(), "TensorInfo must be set in order to validate the shape.");
    return shape == GetTensorInfo().GetShape();
}

int OutputSlot::Connect(InputSlot& destination)
{
    destination.SetConnection(this);
    m_Connections.push_back(&destination);
    return boost::numeric_cast<int>(m_Connections.size() - 1);
}

void OutputSlot::Disconnect(InputSlot& slot)
{
    slot.SetConnection(nullptr);
    m_Connections.erase(std::remove(m_Connections.begin(), m_Connections.end(), &slot), m_Connections.end());
}

void OutputSlot::DisconnectAll()
{
    while (GetNumConnections() > 0)
    {
        InputSlot& connection = *GetConnection(0);
        Disconnect(connection);
    }
}

void OutputSlot::MoveAllConnections(OutputSlot& destination)
{
    while (GetNumConnections() > 0)
    {
        InputSlot& connection = *GetConnection(0);
        Disconnect(connection);
        destination.Connect(connection);
    }
}

void OutputSlot::ValidateConnectionIndex(unsigned int index) const
{
    if (boost::numeric_cast<std::size_t>(index) >= m_Connections.size())
    {
        throw InvalidArgumentException(
            boost::str(boost::format("GetConnection: Invalid index %1% provided") % index));
    }
}

namespace {
LayerGuid GenerateLayerGuid()
{
    // Note: Not thread safe.
    static LayerGuid newGuid=0;
    return newGuid++;
}
} // namespace

Layer::Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name)
: m_LayerName(name ? name : "")
, m_Type(type)
, m_Guid(GenerateLayerGuid())
{
    m_InputSlots.reserve(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; ++i)
    {
        m_InputSlots.emplace_back(*this, i);
    }

    m_OutputSlots.reserve(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; ++i)
    {
        m_OutputSlots.emplace_back(*this, m_OutputHandlers[i]);
    }
}


DataType Layer::GetDataType() const
{
    if (GetNumInputSlots() > 0) // Ignore the input layer.
    {
        return GetInputSlot(0).GetConnection()->GetTensorInfo().GetDataType();
    }
    return GetOutputSlot(0).GetTensorInfo().GetDataType();
}

void Layer::ResetPriority() const
{
    m_Priority = 0;
    m_Visiting = false;
}




} // namespace armnn
