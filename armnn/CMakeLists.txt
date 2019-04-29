#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
cmake_minimum_required (VERSION 3.0.2) # 3.0.2 required for return() statement used in AddDllCopyCommands.cmake
project(armnn)
set(additional_cmake_files)
list(APPEND additional_cmake_files
    cmake/Utils.cmake
    cmake/GlobalConfig.cmake
)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()


list(APPEND armnn_sources
    include/armnn/Armnn.hpp
    include/armnn/Descriptors.hpp
    include/armnn/DescriptorsFwd.hpp
    include/armnn/INetwork.hpp
    include/armnn/Tensor.hpp
    include/armnn/TensorFwd.hpp
    include/armnn/Types.hpp
    include/armnn/NetworkFwd.hpp
    src/armnn/layers/LayerWithParameters.hpp
    src/armnn/layers/ActivationLayer.hpp
    src/armnn/layers/ActivationLayer.cpp
    src/armnn/layers/Convolution2dLayer.hpp
    src/armnn/layers/Convolution2dLayer.cpp
    src/armnn/layers/DepthwiseConvolution2dLayer.hpp
    src/armnn/layers/DepthwiseConvolution2dLayer.cpp
    src/armnn/layers/FullyConnectedLayer.hpp
    src/armnn/layers/FullyConnectedLayer.cpp
    src/armnn/layers/InputLayer.hpp
    src/armnn/layers/InputLayer.cpp
    src/armnn/layers/NormalizationLayer.hpp
    src/armnn/layers/NormalizationLayer.cpp
    src/armnn/layers/OutputLayer.hpp
    src/armnn/layers/OutputLayer.cpp
    src/armnn/layers/Pooling2dLayer.hpp
    src/armnn/layers/Pooling2dLayer.cpp
    src/armnn/layers/SoftmaxLayer.hpp
    src/armnn/layers/SoftmaxLayer.cpp
    src/armnn/Layer.cpp
    src/armnn/LayerFwd.hpp
    src/armnn/Layer.hpp
    src/armnn/Network.cpp
    src/armnn/Network.hpp
    src/armnn/LayerFwd.hpp
    src/armnn/LayersFwd.hpp
    src/armnn/InternalTypes.hpp
    src/armnn/InternalTypes.cpp
    )



add_library_ex(armnn SHARED ${armnn_sources})

target_include_directories(armnn PRIVATE src/armnn)


target_link_libraries(armnn ${CMAKE_DL_LIBS})

