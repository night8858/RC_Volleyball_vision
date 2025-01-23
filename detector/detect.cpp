#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"

#include "detect.hpp"
#include "process.h"
#include "common.h"
#include "postprocess.hpp"

using namespace nvinfer1;

detect_nx::detect_nx(void)
{
    
    m_param.batch_size= 1;
    m_param.dst_h = 640;
    m_param.dst_w = 640;
    m_param.src_h = 480;
    m_param.src_w = 640;

    runtime_ = nullptr;
    engine_ = nullptr;
    context_ = nullptr;
    ////////// init cuda ///////////

    // input_src_device = nullptr;
    // input_resize_device = nullptr;
    // input_rgb_device = nullptr;
    // input_norm_device = nullptr;
    // input_device = nullptr;

    int max_img_size = 3000000;
    // CHECK(cudaMallocHost((void **)&input_src_host, 1280 * 720 * 3 * sizeof(unsigned char)));
    // CHECK(cudaMalloc(&input_src_device, 3 * 640 * 480 * sizeof(unsigned char)));
    // CHECK(cudaMalloc(&input_resize_device, 3 * 640 * 640 * sizeof(float)));
    // CHECK(cudaMalloc(&input_rgb_device, 3 * 640 * 640 * sizeof(float)));
    // CHECK(cudaMalloc(&input_norm_device, 3 * 640 * 640 * sizeof(float)));
    //CHECK(cudaMalloc(&input_device, 3 * 640 * 640 * sizeof(float)));

    // output_device = nullptr;
    // output_src_transpose_device = nullptr;
    // output_objects_device = nullptr;
    // output_objects_host = nullptr;
    output_objects_width = 7;
    // output_idx_device = nullptr;
    // output_conf_device = nullptr;

    int output_objects_size = 1 * (1 + 300 * output_objects_width); // 1: count
    // CHECK(cudaMalloc(&output_objects_device, output_objects_size * sizeof(float)));
    // CHECK(cudaMalloc(&output_idx_device, 1 * 300 * sizeof(int)));
    // CHECK(cudaMalloc(&output_conf_device, 1 * 300 * sizeof(float)));
    output_objects_host = new float[output_objects_size];

    /// CHECK(cudaStreamCreate(&stream_));

    ////////////////////////////////
}

detect_nx::~detect_nx()
{
    // // input
    // CHECK(cudaFree(input_src_device));
    // CHECK(cudaFree(input_resize_device));
    // CHECK(cudaFree(input_rgb_device));
    // CHECK(cudaFree(input_norm_device));
    // CHECK(cudaFree(input_device));
    // // output
    // CHECK(cudaFree(output_device));
    // CHECK(cudaFree(output_src_transpose_device));
    // CHECK(cudaFree(output_objects_device));
    // CHECK(cudaFree(output_idx_device));
    // CHECK(cudaFree(output_conf_device));
    // delete[] output_objects_host;
}

void prepare_buffer(std::shared_ptr<nvinfer1::ICudaEngine> engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex("images");
    const int outputIndex = engine->getBindingIndex("output0");
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, 1 * 3 * 640 * 640 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, 6 * 300 * sizeof(float)));

    *output_buffer_host = new float[6 * 300];
}

void detect_nx::RT_engine_init(std::string engine_path)
{

    std::ifstream engine_file(engine_path, std::ios::binary);
    // std::cout << engine_path << std::endl;//打印引擎路径（个人理解---night）
    assert(engine_file.is_open() && "Unable to load engine_ file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data_.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data_.data()), length);


    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime_)
    {
        std::cout << "runtime_ create failed" << std::endl;
        exit(-1);
    }

    auto plan = engine_data_;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_)
    {
        exit(-1);
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        std::cout << "context_ create failed" << std::endl;
        exit(-1);
    }

    m_output_dims = this->context_->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[2];
    std::cout << "total_objects: " << m_total_objects << std::endl;
    // assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    std::cout << m_output_area << std::endl;
    detect_nx();

    // Create GPU buffers on device

    prepare_buffer(engine_ , &device_buffers[0] , &device_buffers[1] , &output_device_host);
    CHECK(cudaStreamCreate(&stream_));


}



void detect_nx::preprocesss(cv::Mat &imgsBatch)
{
    cuda_batch_preprocess(imgsBatch ,device_buffers[0] ,640 ,640 ,stream_ );
    
}

bool detect_nx::infer(void)
{

    bool context = detect_nx::context_->enqueueV2((void **)device_buffers, stream_,nullptr);

    CHECK(cudaStreamSynchronize(stream_));
    return context;
}

void detect_nx::postprocess(void)
{
    // CHECK(cudaMallocHost(&output_objects_host, 1 * (1 + 5 * 8400) * sizeof(float)));
    output_device_host = new float[1800];

    CHECK(cudaMemcpy(output_device_host, device_buffers[1], m_output_area * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < 60; i++)
    {
        //  if (output_device_host[i * 6 + 4] > 0.6)
        //  {
        for (size_t j = 0; j < 6; j++)
        {

            std::cout << output_device_host[i * 6 + j] << "  ";
        }
        std::cout << std::endl;
        //}
    }
}

// 现在要完成的是：
//
// 1 解码函数输出和nms