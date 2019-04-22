#include "Caffe2Parser.hpp"
//#include "RecordByRecordCaffeParser.hpp"
#include "armnnCaffe2Parser/ICaffe2Parser.hpp"
#include "armnn/Descriptors.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Utils.hpp"
#include "armnn/Exceptions.hpp"

#include "GraphTopologicalSort.hpp"
#include "VerificationHelpers.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
// ProtoBuf
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>

#include <cmath>
#include <sstream>
#include <queue>
#include <fcntl.h>

#include <iostream>



#include "caffe2.pb.h"






namespace armnnCaffe2Parser{

using namespace armnn;
using namespace caffe2;
using namespace std;
using namespace google::protobuf::io;


namespace
{
    const float* GetArrayPtrFromBlob(const caffe2::Argument arg)
    {
        BOOST_ASSERT(arg.name()=="values");
        
        const float* arrayPtr = arg.floats().data();
        return arrayPtr;
    }

    void GetDataFromBlob(const caffe2::Argument arg, std::vector<float>& outData)
    {
        BOOST_ASSERT(arg.name()=="values");

        size_t blobSize = boost::numeric_cast<size_t>(arg.floats_size());
        if (blobSize != outData.size())
        {
            throw ParseException(
                boost::str(
                    boost::format(
                        "Data blob  in layer %2% has an unexpected size. "
                        "Expected %3% elements but got %4% elements. %5%") %
                        arg.name() %
                        outData.size() %
                        blobSize %
                        CHECK_LOCATION().AsString()));
        }

        int outSizeInt = boost::numeric_cast<int>(outData.size());
        for(int i = 0 ; i < outSizeInt; ++i)
        {
            outData[static_cast<size_t>(i)] = arg.floats(i);
        }

    }



}


const std::map<std::string, Caffe2ParserBase::OperationParsingFunction>
    Caffe2ParserBase::ms_Caffe2OperatorToParsingFunctions = {
    { "Relu",           &Caffe2ParserBase::ParseReluLayer },
    { "Conv",           &Caffe2ParserBase::ParseConvLayer},
    { "AveragePool",    &Caffe2ParserBase::ParseAvePoolingLayer },
    { "FC",             &Caffe2ParserBase::ParseFCLayer },
    { "Softmax",        &Caffe2ParserBase::ParseSoftmaxLayer },
    { "Sum",            &Caffe2ParserBase::ParseSumLayer },
    { "LRN",            &Caffe2ParserBase::ParseLRNLayer },
    { "Dropout",        &Caffe2ParserBase::ParseDropoutLayer },
    { "MaxPool",        &Caffe2ParserBase::ParseMaxPoolingLayer},
    };
    
    Caffe2ParserBase::Caffe2ParserBase()
        :m_Network(nullptr,nullptr)
    {

    }

    Caffe2Parser::Caffe2Parser() : Caffe2ParserBase()
    {}

ICaffe2Parser* ICaffe2Parser::Create()
{
    return new Caffe2Parser();
}



BindingPointInfo Caffe2ParserBase::GetNetworkInputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "data", m_NetworkInputsBindingInfo);
}

BindingPointInfo Caffe2ParserBase::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
}



std::pair<armnn::LayerBindingId, armnn::TensorInfo> Caffe2ParserBase::GetBindingInfo(const std::string& layerName,
    const char* bindingPointDesc,
    const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    auto it = nameToBindingInfo.find(layerName);
     if (it == nameToBindingInfo.end())
    {
        throw InvalidArgumentException(
            boost::str(
                boost::format(
                    "Unknown binding %1% for layer '%2%'. %3%") %
                    bindingPointDesc %
                    layerName %
                    CHECK_LOCATION().AsString()));
    }
    return it->second;
}



void Caffe2ParserBase::ResolveInplaceLayers(caffe2::NetDef& predict)
{
    std::map<std::string, std::vector<caffe2::OperatorDef*>> layersbyop;

    for(int layerIdx = 0; layerIdx < predict.op_size(); ++layerIdx)
    {
        //finds the layer with same output
        caffe2::OperatorDef& op = *predict.mutable_op(layerIdx);
        std::string name = op.type();
        for(int i=0 ; i < op.output_size(); ++i)
        {
            layersbyop[op.output(i)].push_back(&op);
        }
    }

    for (auto layersWithSameopIt : layersbyop)
    {
        const std::string& output = layersWithSameopIt.first;
        const std::vector<caffe2::OperatorDef*>& layersWithSameop = layersWithSameopIt.second;

        for (unsigned int layerIdx = 0; layerIdx < layersWithSameop.size() - 1; ++layerIdx)
        {
                caffe2::OperatorDef& op1 = *layersWithSameop[layerIdx];
                caffe2::OperatorDef& op2 = *layersWithSameop[layerIdx + 1];
              
                if (op1.output_size() != 1)
                {
                    throw ParseException(
                        boost::str(
                            boost::format(
                                "Node '%1%' is an in-place layer but doesn't have exactly one "
                                "top. It has %2% instead. %3%") %
                                op1.type() %
                                op1.output_size() %
                                CHECK_LOCATION().AsString()));
                }
            
            stringstream ss;
            ss<<op1.output(0)<<"_inplace"<<layerIdx;
            std::string newOutput = ss.str();
            op1.set_output(0, newOutput);

            if (op2.input_size() != 1 || op2.input(0) != output)
            {
                throw ParseException(
                        boost::str(
                            boost::format(
                                "Node '%1%' is an in-place layer but "
                                "doesn't have exactly one bottom, or it doesn't match its top. "
                                "#bottoms=%2%, first bottom is %3%, top is %4% %5%") %
                                op2.type() %
                                op2.input(0) %
                                output %
                                CHECK_LOCATION().AsString()));
            }
            op2.set_input(0, newOutput);
        }
    }


}


caffe2::Argument TensorDescToArguementShape(const TensorInfo& desc)
{
    caffe2::Argument arg;
    arg.set_name("shape");
    for(unsigned int i=0; i<desc.GetNumDimensions(); ++i)
    {
        arg.add_ints(i);
        arg.set_ints(boost::numeric_cast<int>(i), desc.GetShape()[i]);
    }

    return arg;

}

TensorInfo Caffe2ParserBase::ArgumentToTensorInfo(const caffe2::Argument& arg)
{
    BOOST_ASSERT(arg.name()=="shape");
    std::vector<unsigned int> shape;
    for(int j=0; j<arg.ints_size();++j)
    {
        shape.push_back(static_cast<unsigned int>(arg.ints(j)));
    }
    return TensorInfo(boost::numeric_cast<unsigned int>(shape.size()),shape.data(),DataType::Float32);
}

void Caffe2ParserBase::TrackBindingPoint(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    const std::string layerName = layer->GetName();
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        nameToBindingInfo[layerName] = std::make_pair(id, tensorInfo);
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Id %1% used by more than one %2% layer %3%") %
                    id %
                    bindingPointDesc %
                    CHECK_LOCATION().AsString()));
    }
}




void Caffe2ParserBase::TrackInputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkInputsBindingInfo);
}


void Caffe2ParserBase::SetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName, armnn::IOutputSlot& armnnOutputSlot)
{
    std::cout<<caffe2OutputName<<"  "<<armnnOutputSlot.GetNumConnections()<<std::endl;   
    auto it = m_ArmnnOutputSlotForCaffe2Output.find(caffe2OutputName);
    if (it == m_ArmnnOutputSlotForCaffe2Output.end())
    {
        m_ArmnnOutputSlotForCaffe2Output[caffe2OutputName] = &armnnOutputSlot;
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Attempting to add duplicate entry for Caffe top '%1%' %2%") %
                    caffe2OutputName %
                    CHECK_LOCATION().AsString()));
    }
}


armnn::IOutputSlot& Caffe2ParserBase::GetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName) const
{
   
    auto it = m_ArmnnOutputSlotForCaffe2Output.find(caffe2OutputName);
    if (it != m_ArmnnOutputSlotForCaffe2Output.end())
    {
        return *it->second;
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Could not find armnn output slot for Caffe2 output '%1%' %2%") %
                    caffe2OutputName %
                    CHECK_LOCATION().AsString()));
    }
}

void Caffe2ParserBase::ParseInputLayer()
{
    const armnn::LayerBindingId inputId=boost::numeric_cast<armnn::LayerBindingId>(
        m_NetworkInputsBindingInfo.size());
    armnn::IConnectableLayer* const inputLayer = m_Network->AddInputLayer(inputId,"data");
    armnn::TensorInfo inputTensorInfo;
    auto overrideIt = m_InputShapes.find("data");
    const armnn::TensorShape& overrideShape=overrideIt->second;
    inputTensorInfo.SetShape(overrideShape);
    TrackInputBinding(inputLayer, inputId, inputTensorInfo);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    SetArmnnOutputSlotForCaffe2Output("data", inputLayer->GetOutputSlot(0));
}

 void Caffe2ParserBase::ParseReluLayer(const caffe2::OperatorDef& op)
 {

     BOOST_ASSERT(op.type()=="Relu");
     ActivationDescriptor activationDescriptor;
     const string& name = op.type();
     activationDescriptor.m_Function = ActivationFunction::ReLu;
     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
     IConnectableLayer* const activationLayer = m_Network->AddActivationLayer(activationDescriptor, name.c_str());
     GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(activationLayer->GetInputSlot(0));
      activationLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
      SetArmnnOutputSlotForCaffe2Output(op.output(0), activationLayer->GetOutputSlot(0));
 }


 void Caffe2ParserBase::ParseFCLayer(const caffe2::OperatorDef& op)
 {
     FullyConnectedDescriptor tensorFullyConnectedDescriptor;
     tensorFullyConnectedDescriptor.m_TransposeWeightMatrix=true;

     //the weights name is stored at index 1
     
     auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in FC Layer")%
                    op.input(1).c_str()
                    ));
     }
     const caffe2::OperatorDef& w=*it->second;

     //the biases are stored at the index 2
    auto it1 = blobs.find(op.input(2));
     if(it1 == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in FC Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& b=*it1->second;

     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
    //at the index 1 the data is stored
     //const float* weightDataPtr = GetArrayPtrFromBlob(w.arg(1));
    //at the index 0 the shape info is defined

     vector<float> weightData(boost::numeric_cast<size_t>(w.arg(0).ints(0) *
                                                        w.arg(0).ints(1)));
     GetDataFromBlob(w.arg(1),weightData);
     ConstTensor weights(ArgumentToTensorInfo(w.arg(0)), weightData.data());

     tensorFullyConnectedDescriptor.m_BiasEnabled = true;
    vector<float> biasData(boost::numeric_cast<size_t>(b.arg(0).ints(0)));
     //const float* biasDataPtr = GetArrayPtrFromBlob(b.arg(1));
      GetDataFromBlob(b.arg(1),biasData);
    
     ConstTensor biases(ArgumentToTensorInfo(b.arg(0)), biasData.data());
     armnn::IConnectableLayer* fullyConnectedLayer = m_Network->AddFullyConnectedLayer(tensorFullyConnectedDescriptor, weights, biases,op.type().c_str());
     //the output shape = M x shape of bias
     unsigned int outputsize = b.arg(1).floats_size();
     TensorInfo outputInfo({inputInfo.GetShape()[0],outputsize}, DataType::Float32);
     GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(fullyConnectedLayer->GetInputSlot(0));
     fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
     SetArmnnOutputSlotForCaffe2Output(op.output(0),fullyConnectedLayer->GetOutputSlot(0));


 }



void Caffe2ParserBase::AddConvLayerWithDepthwiseConv(const caffe2::OperatorDef& op,
                                            const armnn::Convolution2dDescriptor convDesc,
                                            unsigned int kernel)
{
   
    BOOST_ASSERT(op.type()=="conv");
    DepthwiseConvolution2dDescriptor desc;
    desc.m_PadLeft      = convDesc.m_PadLeft;
    desc.m_PadRight     = convDesc.m_PadRight;
    desc.m_PadTop       = convDesc.m_PadTop;
    desc.m_PadBottom    = convDesc.m_PadBottom;
    desc.m_StrideX      = convDesc.m_StrideX;
    desc.m_StrideY      = convDesc.m_StrideY;
    desc.m_BiasEnabled  = convDesc.m_BiasEnabled;

    auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& w = *it->second;

     unsigned int numFilters = boost::numeric_cast<unsigned int>(w.arg(0).ints(0));

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();

    caffe2::Argument outputShape;
    outputShape.set_name("shape");
    outputShape.add_ints(0);
    outputShape.set_ints(0, inputInfo.GetShape()[0]);
    outputShape.add_ints(1);
    outputShape.set_ints(1, numFilters);
    outputShape.add_ints(2);
    outputShape.set_ints(
        2, (static_cast<int>(
                static_cast<float>(inputInfo.GetShape()[2] + 2 * desc.m_PadBottom - kernel) /
                static_cast<float>(desc.m_StrideY)) + 1));
    outputShape.add_ints(3);
    outputShape.set_ints(
        3, (static_cast<int>(
                static_cast<float>(inputInfo.GetShape()[3] + 2 * desc.m_PadRight - kernel) /
                static_cast<float>(desc.m_StrideX)) + 1));

     size_t allWeightsSize = boost::numeric_cast<size_t>(w.arg(0).ints(0) * kernel * kernel);
    vector<float> weightData(allWeightsSize);
    GetDataFromBlob(w.arg(1), weightData);
    armnn::IConnectableLayer* returnLayer = nullptr;
    ConstTensor weights(ArgumentToTensorInfo(w.arg(0)),weightData.data());

    if(desc.m_BiasEnabled)
    {

        TensorInfo biasInfo ;
        auto it = blobs.find(op.input(2));
        if(it == blobs.end())
        {
           
            throw ParseException(
                boost::str(
                    boost::format(
                        "Could not find the '%1%' in conv Layer")%
                        op.input(2).c_str()
                        ));
        }
        const caffe2::OperatorDef& b = *it->second;

        vector<float> biasData;
        biasData.resize(boost::numeric_cast<size_t>(outputShape.ints(1)), 1.f);
        GetDataFromBlob(b.arg(1),biasData);
        biasInfo = ArgumentToTensorInfo(b.arg(0));
        ConstTensor biases(biasInfo, biasData.data());

        returnLayer = m_Network->AddDepthwiseConvolution2dLayer(desc, weights, biases, op.type().c_str());

    }

    else
    {
        returnLayer = m_Network->AddDepthwiseConvolution2dLayer(desc, weights, op.type().c_str());
    }


    if (!returnLayer)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to create depthwise convolution layer. "
                    "Layer=%1% #filters=%2% %3%") %
                    op.type() %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }

    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffe2Output(op.input(0));
    inputConnection.Connect(returnLayer->GetInputSlot(0));
    returnLayer->GetOutputSlot(0).SetTensorInfo(ArgumentToTensorInfo(outputShape));
    SetArmnnOutputSlotForCaffe2Output(op.output(0),returnLayer->GetOutputSlot(0));

}




void Caffe2ParserBase::AddConvLayerWithSplits(const caffe2::OperatorDef& op,
                                            const armnn::Convolution2dDescriptor convDesc,
                                            unsigned int kernel, unsigned int numGroups)
{
    caffe2::Argument inputShape = TensorDescToArguementShape(GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo());
    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffe2Output(op.input(0));

    vector<string> convLayerNames(numGroups);
    vector<armnn::IConnectableLayer*> convLayers(numGroups);
    convLayerNames[0] = op.type();

    unsigned int splitterDimSizes[4] = {static_cast<unsigned int>(inputShape.ints(0)),
                                        static_cast<unsigned int>(inputShape.ints(1)),
                                        static_cast<unsigned int>(inputShape.ints(2)),
                                        static_cast<unsigned int>(inputShape.ints(3))};

    splitterDimSizes[1] /= numGroups;
    inputShape.set_ints(1, splitterDimSizes[1]);

    ViewsDescriptor splitterDesc(numGroups);

    for(unsigned int g=0; g<numGroups; ++g)
    {
        stringstream ss;
        ss << op.type() << "_" << g;
        convLayerNames[g] = ss.str();

        splitterDesc.SetViewOriginCoord(g, 1, splitterDimSizes[1] * g);

        for(unsigned int dimIdx=0; dimIdx < 4; dimIdx++)
        {
            splitterDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
    }

    const std::string splitterLayerName = std::string("splitter_")+op.type();
    armnn::IConnectableLayer* splitterLayer = m_Network->AddSplitterLayer(splitterDesc, splitterLayerName.c_str());

    inputConnection.Connect(splitterLayer->GetInputSlot(0));
    for (unsigned int i = 0; i < splitterLayer->GetNumOutputSlots(); i++)
    {
        splitterLayer->GetOutputSlot(i).SetTensorInfo(ArgumentToTensorInfo(inputShape));
    }

    auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(1).c_str()
                    ));
     }
     const caffe2::OperatorDef& w = *it->second;


     auto it1 = blobs.find(op.input(2));
     if(it1 == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& b = *it1->second;

     unsigned int numFilters = w.arg(0).ints(0);

    caffe2::Argument outputShape;
    outputShape.set_name("shape");
    outputShape.add_ints(0);
    outputShape.set_ints(0,inputShape.ints(0));
    outputShape.add_ints(1);
    outputShape.set_ints(1,numFilters/numGroups);
    outputShape.add_ints(2);
    outputShape.set_ints(
        2, (static_cast<int>(static_cast<float>(inputShape.ints(2) +  2 * convDesc.m_PadBottom - kernel)/
                            static_cast<float>(convDesc.m_StrideY))+1));
    outputShape.add_ints(3);
    outputShape.set_ints(
        3, (static_cast<int>(static_cast<float>(inputShape.ints(3) +  2 * convDesc.m_PadRight - kernel)/
                            static_cast<float>(convDesc.m_StrideX))+1));

    // Load the weight data for ALL groups
    vector<float> weightData(boost::numeric_cast<size_t>(numGroups *
                                                         inputShape.ints(1) *  // number of input channels
                                                         outputShape.ints(1) * // number of output channels
                                                         kernel *
                                                         kernel));
    GetDataFromBlob(w.arg(1), weightData);
    const unsigned int weightDimSizes[4] = {
        static_cast<unsigned int>(outputShape.ints(1)),
        static_cast<unsigned int>(inputShape.ints(1)),
        kernel,
        kernel};
    TensorInfo biasInfo;
    vector<float> biasData;

     if (convDesc.m_BiasEnabled)
    {
        biasData.resize(boost::numeric_cast<size_t>(numGroups * outputShape.ints(1)), 1.f);
        GetDataFromBlob(b.arg(1), biasData);

        const unsigned int biasDimSizes[1] = {static_cast<unsigned int>(outputShape.ints(1))};
        biasInfo = TensorInfo(1, biasDimSizes, DataType::Float32);
    }

    const unsigned int numWeightsPerGroup = boost::numeric_cast<unsigned int>(weightData.size()) / numGroups;
    const unsigned int numBiasesPerGroup  = boost::numeric_cast<unsigned int>(biasData.size()) / numGroups;

     for (unsigned int g = 0; g < numGroups; ++g)
    {
        // Sets the slot index, group 0 should be connected to the 0th output of the splitter
        // group 1 should be connected to the 1st output of the splitter.

        // Pulls out the weights for this group from that loaded from the model file earlier.
        ConstTensor weights(TensorInfo(4, weightDimSizes, DataType::Float32),
                            weightData.data() + numWeightsPerGroup * g);

        IConnectableLayer* convLayer = nullptr;
        if (convDesc.m_BiasEnabled)
        {
            // Pulls out the biases for this group from that loaded from the model file earlier.
            ConstTensor biases(biasInfo, biasData.data() + numBiasesPerGroup * g);

            convLayer =
                m_Network->AddConvolution2dLayer(convDesc, weights, biases, convLayerNames[g].c_str());
        }
        else
        {
            convLayer =
                m_Network->AddConvolution2dLayer(convDesc, weights, convLayerNames[g].c_str());
        }
        convLayers[g] = convLayer;

        // If we have more than one group then the input to the nth convolution the splitter layer's nth output,
        // otherwise it's the regular input to this layer.
        armnn::IOutputSlot& splitterInputConnection =
            splitterLayer ? splitterLayer->GetOutputSlot(g) : inputConnection;
        splitterInputConnection.Connect(convLayer->GetInputSlot(0));
        convLayer->GetOutputSlot(0).SetTensorInfo(ArgumentToTensorInfo(outputShape));
    }

     // If the convolution was performed in chunks, add a layer to merge the results

    // The merge input shape matches that of the convolution output
    unsigned int mergeDimSizes[4] = {static_cast<unsigned int>(outputShape.ints(0)),
                                        static_cast<unsigned int>(outputShape.ints(1)),
                                        static_cast<unsigned int>(outputShape.ints(2)),
                                        static_cast<unsigned int>(outputShape.ints(3))};
    // This is used to describe how the input is to be merged
    OriginsDescriptor mergeDesc(numGroups);
    // Now create an input node for each group, using the name from
    // the output of the corresponding convolution
    for (unsigned int g = 0; g < numGroups; ++g)
    {
        mergeDesc.SetViewOriginCoord(g, 1, mergeDimSizes[1] * g);
    }

      // Make sure the output from the merge is the correct size to hold the data for all groups
    mergeDimSizes[1] *= numGroups;
    outputShape.set_ints(1, mergeDimSizes[1]);
    // Finally add the merge layer
    IConnectableLayer* mergerLayer = m_Network->AddMergerLayer(mergeDesc, op.type().c_str());

    if (!mergerLayer)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to create final merger layer for Split+Convolution+Merger. "
                    "Layer=%1% #groups=%2% #filters=%3% %4%") %
                    op.type() %
                    numGroups %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }

    for (unsigned int g = 0; g < numGroups; ++g)
    {
        convLayers[g]->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(g));
    }
     mergerLayer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(4, mergeDimSizes, DataType::Float32));
  //std::cout<<op.output(0)<<" dfjgdfg\n";
     SetArmnnOutputSlotForCaffe2Output(op.output(0), mergerLayer->GetOutputSlot(0));
      
}

 void Caffe2ParserBase::ParseConvLayer(const caffe2::OperatorDef& op)
 {    
     BOOST_ASSERT(op.type()=="Conv");
     //create a map of arg name and arg
     
     std::map<std::string, const caffe2::Argument*> args;
     for(int i=0; i<op.arg_size(); ++i)
     {
         args.insert({op.arg(i).name(),&op.arg(i)});
     }
     auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& w = *it->second;
      //std::cout<<"weights name "<<w.output(0)<<std::endl;  
     unsigned int numFilters = boost::numeric_cast<unsigned int>(w.arg(0).ints(0));

      auto it1 = args.find("group");
     
     unsigned int numGroups = 1;
     if(it1!=args.end())
     {
         const caffe2::Argument& a = *it1->second;
         numGroups = boost::numeric_cast<unsigned int>(a.i());
     }
    // std::cout<<"num groups "<<numGroups<<std::endl; 

     unsigned int kernel = 0;
     auto it2 = args.find("kernel");
     if(it2!=args.end())
     {
         const caffe2::Argument& a = *it2->second;
         kernel = boost::numeric_cast<unsigned int>(a.i());
     }
   // std::cout<<"kernel "<<kernel<<std::endl; 

     unsigned int stride = 1;
     auto it3 = args.find("stride");
     if(it3!=args.end())
     {
         const caffe2::Argument& a = *it3->second;
         stride = boost::numeric_cast<unsigned int>(a.i());
     }

    // std::cout<<"stride "<<stride<<std::endl; 

     unsigned int pad = 0;
     auto it4 = args.find("pad");
     if(it4!=args.end())
     {
         const caffe2::Argument& a = *it4->second;
         pad = boost::numeric_cast<unsigned int>(a.i());
     }

     //std::cout<<"pad "<<pad<<std::endl; 
     Convolution2dDescriptor convolution2dDescriptor;

     convolution2dDescriptor.m_PadLeft = pad;
     convolution2dDescriptor.m_PadRight = pad;
     convolution2dDescriptor.m_PadTop = pad;
     convolution2dDescriptor.m_PadBottom = pad;
     convolution2dDescriptor.m_StrideX = stride;
     convolution2dDescriptor.m_StrideY = stride;
     convolution2dDescriptor.m_BiasEnabled = op.input_size()==3 ? true : false;
    
// std::cout<<"bias_enab "<<convolution2dDescriptor.m_BiasEnabled<<std::endl; 
    if (numGroups > numFilters)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Error parsing Convolution: %1%. "
                    "The 'group'=%2% parameter cannot be larger than the "
                    "number of filters supplied ='%3%'. %4%") %
                    op.name() %
                    numGroups %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }

    
      armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffe2Output(op.input(0));
          // std::cout<<"input "<<op.input(0)<<std::endl; 

    const TensorInfo& inputInfo = inputConnection.GetTensorInfo();
  
     if (inputInfo.GetNumDimensions() != 4)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Convolution input shape is expected to have 4 dimensions. "
                    "%1%'s input has only %2%. %3%") %
                    op.name() %
                    inputInfo.GetNumDimensions() %
                    CHECK_LOCATION().AsString()));
    }


    if (numGroups > 1)
    {
        if (numGroups > inputInfo.GetShape()[1])
        {
            throw ParseException(
                boost::str(
                    boost::format(
                        "Error parsing Convolution: %1%. "
                        "The 'group'=%2% parameter cannot be larger than the "
                        "channel of the input shape=%3% (in NCHW format). %4%") %
                        op.name() %
                        numGroups %
                        inputInfo.GetShape()[1] %
                        CHECK_LOCATION().AsString()));
        }
        else if (numGroups == inputInfo.GetShape()[1])
        {
             
            // we use a depthwise convolution here, because the number of groups equals to the
            // input channels
            AddConvLayerWithDepthwiseConv(op, convolution2dDescriptor, kernel);
            return;
             
        }
        else
        {
            AddConvLayerWithSplits(op,convolution2dDescriptor, kernel, numGroups);
            return;
        }
        

    }
    
   
    caffe2::Argument outputShape;
    outputShape.set_name("shape");
    outputShape.add_ints(0);
    outputShape.set_ints(0,inputInfo.GetShape()[0]);
    outputShape.add_ints(1);
    outputShape.set_ints(1,numFilters);
    outputShape.add_ints(2);
    outputShape.set_ints(
        2, (static_cast<int>(static_cast<float>(inputInfo.GetShape()[2]) +  2 * pad - kernel)/
                            static_cast<float>(stride))+1);
    outputShape.add_ints(3);
    outputShape.set_ints(
        3, (static_cast<int>(static_cast<float>(inputInfo.GetShape()[2]) +  2 * pad - kernel)/
                            static_cast<float>(stride))+1);
    
    vector<float> weightData(boost::numeric_cast<size_t>(w.arg(0).ints(0) *
                                                        w.arg(0).ints(1) *
                                                        w.arg(0).ints(2) *
                                                        w.arg(0).ints(3)));

    // std::cout<<"output_shape "<<outputShape.ints(0)<<" "<<outputShape.ints(1)<<" "<<outputShape.ints(2)<<" "<<outputShape.ints(3)<<" "<<std::endl; 


    GetDataFromBlob(w.arg(1),weightData);
   // std::cout<<"weight_capacity "<<weightData.capacity()<<std::endl; 
    armnn::IConnectableLayer* returnLayer = nullptr;
//// std::cout<<"weight_size "<<weightData.size()<<std::endl; 

    ConstTensor weights(ArgumentToTensorInfo(w.arg(0)),weightData.data());

    if (convolution2dDescriptor.m_BiasEnabled)
    {
         TensorInfo biasInfo ;
        auto it = blobs.find(op.input(2));
        if(it == blobs.end())
        {
           
            throw ParseException(
                boost::str(
                    boost::format(
                        "Could not find the '%1%' in conv Layer")%
                        op.input(2).c_str()
                        ));
        }
        const caffe2::OperatorDef& b = *it->second;

        vector<float> biasData;
        biasData.resize(boost::numeric_cast<size_t>(outputShape.ints(1)), 1.f);
        GetDataFromBlob(b.arg(1),biasData);
       // std::cout<<"biases"<<std::endl;
        // for(int i=0;i<10;i++)
        // {
        //      std::cout<<biasData.at(i)<<std::endl;
        // }
        biasInfo = ArgumentToTensorInfo(b.arg(0));
        ConstTensor biases(biasInfo, biasData.data());

        returnLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor, weights, biases, op.type().c_str());

    }
    else
    {
        returnLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor, weights, op.type().c_str());
    }
   
  
    inputConnection.Connect(returnLayer->GetInputSlot(0));
    returnLayer->GetOutputSlot(0).SetTensorInfo(ArgumentToTensorInfo(outputShape));

    if (!returnLayer)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to create Convolution layer. "
                    "Layer=%1% #groups=%2% #filters=%3% %4%") %
                    op.name() %
                    numGroups %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }
   
    SetArmnnOutputSlotForCaffe2Output(op.output(0), returnLayer->GetOutputSlot(0));
 }





void Caffe2ParserBase::ParseAvePoolingLayer(const caffe2::OperatorDef& op)
{
    BOOST_ASSERT(op.type()=="AveragePool");
    
    const string& name = op.type();
    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();

    std::map<std::string, const caffe2::Argument*> args;

    for (int i=0 ; i<op.arg_size() ;++i)
    {
        args.insert({op.arg(i).name(),&op.arg(i)});
        
    }

    unsigned int kernel_h = 0;
    auto k = args.find("kernel");
    if(k!=args.end())
    {
        const caffe2::Argument& a = *k->second;
        kernel_h = boost::numeric_cast<unsigned int>(a.i());
    }
    unsigned int kernel_w = kernel_h;

    
    unsigned int stride_h = 1;
    auto s = args.find("stride");
    if(s!=args.end())
    {
        const caffe2::Argument& a = *s->second;
        stride_h = boost::numeric_cast<unsigned int>(a.i());
    }
    unsigned int stride_w = stride_h;




    unsigned int pad_l = 0, pad_r = 0, pad_t1 = 0, pad_b = 0;
    auto p = args.find("pad");
    if(p!=args.end())
    {
        const caffe2::Argument& a = *p->second;
        pad_l = boost::numeric_cast<unsigned int>(a.i());
        pad_r = pad_l;
        pad_t1 = pad_r;
        pad_b = pad_t1;
    }
    else
    {
         auto p1 = args.find("pad_l");
         if(p1!=args.end())
        {
            const caffe2::Argument& a = *p1->second;
            pad_l = boost::numeric_cast<unsigned int>(a.i());
        }
        auto p2 = args.find("pad_r");
         if(p2!=args.end())
        {
            const caffe2::Argument& a = *p2->second;
            pad_r = boost::numeric_cast<unsigned int>(a.i());
        }

        auto p3 = args.find("pad_t1");
         if(p3!=args.end())
        {
            const caffe2::Argument& a = *p3->second;
            pad_t1 = boost::numeric_cast<unsigned int>(a.i());
        }

        auto p4 = args.find("pad_b");
         if(p4!=args.end())
        {
            const caffe2::Argument& a = *p4->second;
            pad_b = boost::numeric_cast<unsigned int>(a.i());
        }
    }
    
   



    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType = PoolingAlgorithm::Average;



    pooling2dDescriptor.m_PadLeft     = pad_l;
    pooling2dDescriptor.m_PadRight    = pad_r;
    pooling2dDescriptor.m_PadTop      = pad_t1;
    pooling2dDescriptor.m_PadBottom   = pad_b;
    pooling2dDescriptor.m_StrideX     = stride_w;
    pooling2dDescriptor.m_StrideY     = stride_h;
    pooling2dDescriptor.m_PoolWidth   = kernel_w;
    pooling2dDescriptor.m_PoolHeight  = kernel_h;

    pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Ceiling;
    pooling2dDescriptor.m_PaddingMethod  = PaddingMethod::IgnoreValue;

    
    
    TensorInfo outputInfo(
        { inputInfo.GetShape()[0],
          inputInfo.GetShape()[1],
          static_cast<unsigned int>(
              static_cast<float>(inputInfo.GetShape()[2] + pad_t1 + pad_b - kernel_h) /
              boost::numeric_cast<float>(stride_h)) + 1,
          static_cast<unsigned int>(
              static_cast<float>(inputInfo.GetShape()[3] + pad_l + pad_r - kernel_w) /
              boost::numeric_cast<float>(stride_w)) + 1 },
        DataType::Float32);



    armnn::IConnectableLayer* poolingLayer = m_Network->AddPooling2dLayer(pooling2dDescriptor,
        name.c_str());
    GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(poolingLayer->GetInputSlot(0));
     poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    SetArmnnOutputSlotForCaffe2Output(op.output(0), poolingLayer->GetOutputSlot(0));

 }


void Caffe2ParserBase::ParseMaxPoolingLayer(const caffe2::OperatorDef& op)
{
     BOOST_ASSERT(op.type()=="MaxPool");
    
    const string& name = op.type();
    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();

    std::map<std::string, const caffe2::Argument*> args;

    for (int i=0 ; i<op.arg_size() ;++i)
    {
        args.insert({op.arg(i).name(),&op.arg(i)});
        
    }

    unsigned int kernel_h = 0;
    auto k = args.find("kernel");
    if(k!=args.end())
    {
        const caffe2::Argument& a = *k->second;
        kernel_h = boost::numeric_cast<unsigned int>(a.i());
    }
    unsigned int kernel_w = kernel_h;

    
    unsigned int stride_h = 1;
    auto s = args.find("stride");
    if(s!=args.end())
    {
        const caffe2::Argument& a = *s->second;
        stride_h = boost::numeric_cast<unsigned int>(a.i());
    }
    unsigned int stride_w = stride_h;




    unsigned int pad_l = 0, pad_r = 0, pad_t1 = 0, pad_b = 0;
    auto p = args.find("pad");
    if(p!=args.end())
    {
        const caffe2::Argument& a = *p->second;
        pad_l = boost::numeric_cast<unsigned int>(a.i());
        pad_r = pad_l;
        pad_t1 = pad_r;
        pad_b = pad_t1;
    }
    else
    {
         auto p1 = args.find("pad_l");
         if(p1!=args.end())
        {
            const caffe2::Argument& a = *p1->second;
            pad_l = boost::numeric_cast<unsigned int>(a.i());
        }
        auto p2 = args.find("pad_r");
         if(p2!=args.end())
        {
            const caffe2::Argument& a = *p2->second;
            pad_r = boost::numeric_cast<unsigned int>(a.i());
        }

        auto p3 = args.find("pad_t1");
         if(p3!=args.end())
        {
            const caffe2::Argument& a = *p3->second;
            pad_t1 = boost::numeric_cast<unsigned int>(a.i());
        }

        auto p4 = args.find("pad_b");
         if(p4!=args.end())
        {
            const caffe2::Argument& a = *p4->second;
            pad_b = boost::numeric_cast<unsigned int>(a.i());
        }
    }
    
   



    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType = PoolingAlgorithm::Max;



    pooling2dDescriptor.m_PadLeft     = pad_l;
    pooling2dDescriptor.m_PadRight    = pad_r;
    pooling2dDescriptor.m_PadTop      = pad_t1;
    pooling2dDescriptor.m_PadBottom   = pad_b;
    pooling2dDescriptor.m_StrideX     = stride_w;
    pooling2dDescriptor.m_StrideY     = stride_h;
    pooling2dDescriptor.m_PoolWidth   = kernel_w;
    pooling2dDescriptor.m_PoolHeight  = kernel_h;

    pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Floor;
    pooling2dDescriptor.m_PaddingMethod  = PaddingMethod::IgnoreValue;

    
    
    TensorInfo outputInfo(
        { inputInfo.GetShape()[0],
          inputInfo.GetShape()[1],
          static_cast<unsigned int>(
              static_cast<float>(inputInfo.GetShape()[2] + pad_t1 + pad_b - kernel_h) /
              boost::numeric_cast<float>(stride_h)) + 1,
          static_cast<unsigned int>(
              static_cast<float>(inputInfo.GetShape()[3] + pad_l + pad_r - kernel_w) /
              boost::numeric_cast<float>(stride_w)) + 1 },
        DataType::Float32);

    armnn::IConnectableLayer* poolingLayer = m_Network->AddPooling2dLayer(pooling2dDescriptor,
        name.c_str());
    GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(poolingLayer->GetInputSlot(0));
     poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    SetArmnnOutputSlotForCaffe2Output(op.output(0), poolingLayer->GetOutputSlot(0));
}


void Caffe2ParserBase::ParseSoftmaxLayer(const caffe2::OperatorDef& op)
 {
     armnn::SoftmaxDescriptor softmaxDescriptor;
     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
     
     const string& name = op.type();
     IConnectableLayer* const softmaxLayer = m_Network->AddSoftmaxLayer(softmaxDescriptor, name.c_str());
    GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(softmaxLayer->GetInputSlot(0));

    softmaxLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffe2Output(op.output(0), softmaxLayer->GetOutputSlot(0));
 }



void Caffe2ParserBase::ParseSumLayer(const caffe2::OperatorDef& op)
{
    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
     armnn::IConnectableLayer* newLayer =  m_Network->AddAdditionLayer(op.type().c_str());

    GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(newLayer->GetInputSlot(0));
    GetArmnnOutputSlotForCaffe2Output(op.input(1)).Connect(newLayer->GetInputSlot(1));
    newLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffe2Output(op.output(0), newLayer->GetOutputSlot(0));
}



void Caffe2ParserBase::ParseLRNLayer(const caffe2::OperatorDef& op)
{

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
    NormalizationDescriptor normalizationDescriptor;
    normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    normalizationDescriptor.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
   
    std::map<std::string, const caffe2::Argument*> args;

    for (int i=0 ; i<op.arg_size() ;++i)
    {
        args.insert({op.arg(i).name(),&op.arg(i)});
        
    }

     auto k = args.find("size");
    if(k!=args.end())
    {
        const caffe2::Argument& a = *k->second;
        normalizationDescriptor.m_NormSize = boost::numeric_cast<unsigned int>(a.i());
    }
     else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "size not defined for LRN layer %1% %2%") %
                    op.type() %
                    CHECK_LOCATION().AsString()));
    }


    auto k1 = args.find("alpha");
    if(k1!=args.end())
    {
        const caffe2::Argument& a = *k1->second;
        normalizationDescriptor.m_Alpha = boost::numeric_cast<float>(a.f());
        //in the formula the alpha is divided by the size
        normalizationDescriptor.m_Alpha /= boost::numeric_cast<float>( normalizationDescriptor.m_NormSize);
    }
     else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "alpha not defined for LRN layer %1% %2%") %
                    op.type() %
                    CHECK_LOCATION().AsString()));
    }


     auto k2 = args.find("beta");
    if(k2!=args.end())
    {
        const caffe2::Argument& a = *k2->second;
        normalizationDescriptor.m_Beta = boost::numeric_cast<float>(a.f());
        
    }
     else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "beta not defined for LRN layer %1% %2%") %
                    op.type() %
                    CHECK_LOCATION().AsString()));
    }


      auto k3 = args.find("bias");
    if(k3!=args.end())
    {
        const caffe2::Argument& a = *k3->second;
        normalizationDescriptor.m_K = boost::numeric_cast<float>(a.f());
        
    }
     else
    {
         normalizationDescriptor.m_K = 1;
    }

    IConnectableLayer* const normLayer = m_Network->AddNormalizationLayer(normalizationDescriptor,op.type().c_str());
    GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    SetArmnnOutputSlotForCaffe2Output(op.output(0),normLayer->GetOutputSlot(0));
    
}


void Caffe2ParserBase::ParseDropoutLayer(const caffe2::OperatorDef& op)
{
    if (op.input_size() != 1 || op.output_size() > 2 )
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Dropout layer '%1%' should have exactly 1 bottom and 1 top. "
                    "#bottoms=%2% #tops=%3% %4%") %
                    op.type() %
                    op.input_size() %
                    op.output_size() %
                    CHECK_LOCATION().AsString()));
    
    }

    SetArmnnOutputSlotForCaffe2Output(op.output(0), GetArmnnOutputSlotForCaffe2Output(op.input(0)));
}

void Caffe2ParserBase::TrackOutputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkOutputsBindingInfo);
}


void Caffe2ParserBase::LoadNetDef(caffe2::NetDef& init,caffe2::NetDef& predict)
{

    Caffe2ParserBase::ResolveInplaceLayers(predict);
   
    //Create a lookup of Caff2 layers by output name
    for (int i=0;i<predict.op_size(); ++i)
    {
        const caffe2::OperatorDef& op=predict.op(i);
        
        for(int i=0 ; i<op.output_size();++i)
        {
            m_Caffe2OperatorsByOutputName[op.output(i)]=&op;

        }

    }

    
    std::vector<const caffe2::OperatorDef*> nodes;
    for(int i=0;i<predict.op_size();i++)
    {
        nodes.push_back(&predict.op(i));
    }

    //stores the corresponding name and index of blobs in init_net
    for(int i=0;i<init.op_size();++i)
    {
        blobs.insert({init.op(i).output(0),&init.op(i)});
    }

    this->ParseInputLayer();
  // int k=0;
   for(const caffe2::OperatorDef* current : nodes)
   {
    //    if(k>=2)
    //    {
    //        break;
    //    }
    //    k++;
       auto it = ms_Caffe2OperatorToParsingFunctions.find(current->type());
       if (it == ms_Caffe2OperatorToParsingFunctions.end())
        {
            throw ParseException(
                boost::str(
                    boost::format("Unsupported layer type: '%1%' for layer %2% %3%") %
                    current->type() %
                    current->name() %
                    CHECK_LOCATION().AsString()));
        }
        auto func = it->second;
        (this->*func)(*current);
   }
  
    for (const std::string& requestedOutput : m_RequestedOutputs)
    {
        armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffe2Output(requestedOutput);

        const armnn::LayerBindingId outputId = boost::numeric_cast<armnn::LayerBindingId>(
            m_NetworkOutputsBindingInfo.size());
        armnn::IConnectableLayer* const outputLayer = m_Network->AddOutputLayer(outputId, requestedOutput.c_str());
        outputSlot.Connect(outputLayer->GetInputSlot(0));

        TrackOutputBinding(outputLayer, outputId, outputLayer->GetInputSlot(0).GetConnection()->GetTensorInfo());
    }



}

armnn::INetworkPtr Caffe2Parser::CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,const std::map<std::string, armnn::TensorShape>& inputShapes,
                                                    const std::vector<std::string>& requestedOutputs)
{
    //reading the predict net
    FILE* fd = fopen(predict_net, "rb");
    
    if (fd == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
                    "Failed to open predict_net file at: %1% %2%") %
                    predict_net %
                    CHECK_LOCATION().AsString()));
    }
     
    // Parses the file into a message.
    NetDef predict;

    FileInputStream  inStream(fileno(fd));
    CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success = predict.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to parse predict net protobuf file: %1% %2%") %
                    predict_net %
                    CHECK_LOCATION().AsString()));
    }

    //reading the init net



    FILE* fd1 = fopen(init_net, "rb");

    if (fd1 == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
                    "Failed to open init_net file at: %1% %2%") %
                    init_net %
                    CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    NetDef init;

    FileInputStream  inStream1(fileno(fd1));
    CodedInputStream codedStream1(&inStream1);
    codedStream1.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success1 = init.ParseFromCodedStream(&codedStream1);
    fclose(fd1);

    if (!success1)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to parse init net protobuf file: %1% %2%") %
                    init_net %
                    CHECK_LOCATION().AsString()));
    }


    return CreateNetworkFromNetDef(init,predict,inputShapes,requestedOutputs);
}
armnn::INetworkPtr Caffe2ParserBase::CreateNetworkFromNetDef(caffe2::NetDef& init,caffe2::NetDef& predict,const std::map<std::string, armnn::TensorShape>& inputShapes,
                                                const std::vector<std::string>& requestedOutputs)
{


    m_NetworkInputsBindingInfo.clear();

    m_Network=INetwork::Create();
 
  
    m_InputShapes=inputShapes;

    if (requestedOutputs.size() == 0)
    {
        throw ParseException("requestedOutputs must have at least one entry");
    }
    m_RequestedOutputs = requestedOutputs;

    try
    {
        LoadNetDef(init,predict);

    }catch(const ParseException& e)
    {
        throw e;
    }

    return move(m_Network);

}

}

