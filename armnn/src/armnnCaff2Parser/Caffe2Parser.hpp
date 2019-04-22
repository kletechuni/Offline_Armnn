#pragma once
#include "armnnCaffe2Parser/ICaffe2Parser.hpp"

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"

#include <memory>
#include <vector>
#include <unordered_map>
#include "caffe2.pb.h"





namespace armnnCaffe2Parser
{

    using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;
    
    class Caffe2ParserBase: public ICaffe2Parser
    {
     public:
        
         armnn::INetworkPtr CreateNetworkFromNetDef(caffe2::NetDef& init,caffe2::NetDef& predict,const std::map<std::string, armnn::TensorShape>& inputShapes,
         const std::vector<std::string>& requestedOutputs);
         
    /// Retrieves binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;
/// Retrieves binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;

        Caffe2ParserBase();
        
    protected:

        void ParseInputLayer();
        void ParseReluLayer(const caffe2::OperatorDef& op);
        void ParseFCLayer(const caffe2::OperatorDef& op);
        void ParseConvLayer(const caffe2::OperatorDef& op);
        void AddConvLayerWithDepthwiseConv(const caffe2::OperatorDef& op,
                                            const armnn::Convolution2dDescriptor convDesc,
                                            unsigned int kernel);
        void AddConvLayerWithSplits(const caffe2::OperatorDef& op,
                                        const armnn::Convolution2dDescriptor convDesc,
                                        unsigned int kernel, unsigned int numGroups);                                   
        void ParseAvePoolingLayer(const caffe2::OperatorDef& op);
        void ParseSoftmaxLayer(const caffe2::OperatorDef& op);
        void ParseSumLayer(const caffe2::OperatorDef& op);
        void ParseLRNLayer(const caffe2::OperatorDef& op);
        void ParseDropoutLayer(const caffe2::OperatorDef& op);
        void ParseMaxPoolingLayer(const caffe2::OperatorDef& op);



    void ResolveInplaceLayers(caffe2::NetDef& predict);
    armnn::TensorInfo  ArgumentToTensorInfo(const caffe2::Argument& arg);
    armnn::IOutputSlot& GetArmnnOutputSlotForCaffe2Output(const std::string& caffe2outputName) const;  
    void SetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName, armnn::IOutputSlot& armnnOutputSlot);
    void TrackBindingPoint(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

    static std::pair<armnn::LayerBindingId, armnn::TensorInfo> GetBindingInfo(
        const std::string& layerName,
        const char* bindingPointDesc,
        const std::unordered_map<std::string, BindingPointInfo>& bindingInfos);

        void TrackInputBinding(armnn::IConnectableLayer* layer,
            armnn::LayerBindingId id,
            const armnn::TensorInfo& tensorInfo);
        void LoadNetDef(caffe2::NetDef& init,caffe2::NetDef& predict);
        armnn::INetworkPtr m_Network;

        std::map<std::string, const caffe2::OperatorDef*> m_Caffe2OperatorsByOutputName;
        using OperationParsingFunction = void(Caffe2ParserBase::*)(const caffe2::OperatorDef& op);
        static const std::map<std::string, OperationParsingFunction> ms_Caffe2OperatorToParsingFunctions;
        std::map<std::string, armnn::TensorShape> m_InputShapes;

        std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

        /// As we add armnn layers we store the armnn IOutputSlot which corresponds to the Caffe2 tops.
        std::unordered_map<std::string, armnn::IOutputSlot*> m_ArmnnOutputSlotForCaffe2Output;

        std::map<std::string, const caffe2::OperatorDef*> blobs;

         /// maps output layer names to their corresponding ids and tensor infos
        std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;

        std::vector<std::string> m_RequestedOutputs;

       void TrackOutputBinding(armnn::IConnectableLayer* layer,
                               armnn::LayerBindingId id,
                                const armnn::TensorInfo& tensorInfo);

/*
        /// Retrieves binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;

    /// Retrieves binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;
    Caffe2ParserBase();*/
    };


    class Caffe2Parser : public Caffe2ParserBase
    {
    public:

        virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,
                const std::map<std::string, armnn::TensorShape>& inputShapes,const std::vector<std::string>& requestedOutputs)override;
    // public:
         Caffe2Parser();
    };
    }

 