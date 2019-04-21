
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetwork.hpp>


#include <algorithm>

#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <list>

namespace armnn
{

    class InputSlot final : public IInputSlot
{
public:
    explicit InputSlot(Layer& owner, unsigned int slotIndex)
    : m_OwningLayer(owner)
    , m_Connection(nullptr)
    , m_SlotIndex(slotIndex)
    {}

    ~InputSlot();

    Layer& GetOwningLayer() const { return m_OwningLayer; }
    unsigned int GetSlotIndex() const { return m_SlotIndex; }

    const OutputSlot* GetConnectedOutputSlot() const { return m_Connection; }
    OutputSlot* GetConnectedOutputSlot() { return m_Connection; }

    /// Links the slot to an output slot or breaks an existing link if passing nullptr.
    void SetConnection(OutputSlot* source)
    {
        if (m_Connection != nullptr && source != nullptr)
        {
            throw InvalidArgumentException("Tried to connect an output slot to an input slot, "
                "but the latter already has a connection");
        }
        m_Connection = source;
    }

    // Inserts single-output existing layer at this point in the graph.
    void Insert(Layer& layer);

    // IInputSlot

    const IOutputSlot* GetConnection() const override;
    IOutputSlot* GetConnection() override;

private:
    Layer& m_OwningLayer;
    OutputSlot* m_Connection;
    const unsigned int m_SlotIndex;
};


class OutputSlot final : public IOutputSlot
{
public:
    explicit OutputSlot(Layer& owner)
    : m_OwningLayer(owner)
     ,m_TensorInfo()
    {}

    ~OutputSlot()
    {
        try
        {
            // Coverity fix: DisconnectAll() may throw uncaught exceptions.
            DisconnectAll();
        }
        catch (const std::exception& e)
        {
            // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
            // exception of type std::length_error.
            // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
            std::cerr << "WARNING: An error has occurred when disconnecting all output slots: "
                      << e.what() << std::endl;
        }
    }

    Layer& GetOwningLayer() const { return m_OwningLayer; }



    int Connect(InputSlot& destination);
    void Disconnect(InputSlot& slot);

    const std::vector<InputSlot*>& GetConnections() const { return m_Connections; }

    bool ValidateTensorShape(const TensorShape& shape) const;

    // Disconnect all conections.
    void DisconnectAll();

    /// Moves all connections to another OutputSlot.
    void MoveAllConnections(OutputSlot& destination);

    // IOutputSlot

    unsigned int GetNumConnections() const override { return boost::numeric_cast<unsigned int>(m_Connections.size()); }
    const InputSlot* GetConnection(unsigned int index) const override;
    InputSlot* GetConnection(unsigned int index) override;


  /// @brief - Sets the TensorInfo used by this output handler.
    /// @param tensorInfo - TensorInfo for the output.
    void SetTensorInfo(const TensorInfo& tensorInfo) override;
    const TensorInfo& GetTensorInfo() const override;
    bool IsTensorInfoSet() const override;

    int Connect(IInputSlot& destination) override
    {
        return Connect(*boost::polymorphic_downcast<InputSlot*>(&destination));
    }

    void Disconnect(IInputSlot& slot) override
    {
        return Disconnect(*boost::polymorphic_downcast<InputSlot*>(&slot));
    }

    

private:
    void ValidateConnectionIndex(unsigned int index) const;
    Layer& m_OwningLayer;
    TensorInfo m_TensorInfo;
    bool m_bTensorInfoSet = false;
    std::vector<InputSlot*> m_Connections;
};


// Base layer class

using LayerPriority = unsigned int;

class Layer : public IConnectableLayer
{
public:
    /// @param name - Optional name for the layer (may be nullptr).
    Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name);
    Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, DataLayout layout, const char* name);

    const std::string& GetNameStr() const
    {
        return m_LayerName;
    }

    

    const std::vector<InputSlot>& GetInputSlots() const { return m_InputSlots; }
    const std::vector<OutputSlot>& GetOutputSlots() const { return m_OutputSlots; }

    // Allows non-const access to input slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<InputSlot>::iterator BeginInputSlots() { return m_InputSlots.begin(); }
    std::vector<InputSlot>::iterator EndInputSlots() { return m_InputSlots.end(); }

    // Allows non-const access to output slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<OutputSlot>::iterator BeginOutputSlots() { return m_OutputSlots.begin(); }
    std::vector<OutputSlot>::iterator EndOutputSlots() { return m_OutputSlots.end(); }

    // Checks whether the outputs of this layer don't have any connection.
    bool IsOutputUnconnected()
    {
        unsigned int numConnections = 0;

        for (auto&& output : GetOutputSlots())
        {
            numConnections += output.GetNumConnections();
        }

        return (GetNumOutputSlots() > 0) && (numConnections == 0);
    }


    LayerType GetType() const { return m_Type; }

    DataType GetDataType() const;


    // IConnectableLayer

    const char* GetName() const override { return m_LayerName.c_str(); }

    unsigned int GetNumInputSlots() const override { return static_cast<unsigned int>(m_InputSlots.size()); }
    unsigned int GetNumOutputSlots() const override { return static_cast<unsigned int>(m_OutputSlots.size()); }

    const InputSlot& GetInputSlot(unsigned int index) const override { return m_InputSlots.at(index); }
    InputSlot& GetInputSlot(unsigned int index) override { return  m_InputSlots.at(index); }
    const OutputSlot& GetOutputSlot(unsigned int index = 0) const override { return m_OutputSlots.at(index); }
    OutputSlot& GetOutputSlot(unsigned int index = 0) override { return m_OutputSlots.at(index); }

    void SetGuid(LayerGuid guid) { m_Guid = guid; }
    LayerGuid GetGuid() const final { return m_Guid; }

    void AddRelatedLayerName(const std::string layerName) { m_RelatedLayerNames.emplace_back(layerName); }

    const std::list<std::string>& GetRelatedLayerNames() { return m_RelatedLayerNames; }


private:
    const std::string m_LayerName;

    std::vector<InputSlot> m_InputSlots;
    std::vector<OutputSlot> m_OutputSlots;

    const LayerType m_Type;


    /// Used for sorting.
   
   

    LayerGuid m_Guid;

    //std::list<std::string> m_RelatedLayerNames;
};

// A layer user-provided data can be bound to (e.g. inputs, outputs).
class BindableLayer : public Layer
{
public:
    BindableLayer(unsigned int numInputSlots,
        unsigned int numOutputSlots,
        LayerType type,
        const char* name,
        LayerBindingId id)
    : Layer(numInputSlots, numOutputSlots, type, name)
    , m_Id(id)
    {
    }

    LayerBindingId GetBindingId() const { return m_Id; };

protected:
    ~BindableLayer() = default;

private:
    LayerBindingId m_Id;
};


}