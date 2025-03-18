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
    

    runtime_ = nullptr;
    engine_ = nullptr;
    context_ = nullptr;
    ////////// init cuda ///////////

    int max_img_size = 3000000;

    output_objects_width = 7;

    int output_objects_size = 1 * (1 + 300 * output_objects_width); // 1: count
    output_objects_host = new float[output_objects_size];

}

detect_nx::~detect_nx(void)
{}


void detect_nx::push_img(cv::Mat& img)
{
    img_mutex_.lock();
    if (input_imgs_.size() == max_size_) input_imgs_.clear();
    input_imgs_.emplace_back(img);
    img_mutex_.unlock();
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


/// @brief 初始化tensorrt引擎，以及内存分配空间的准备
/// @param engine_path 引擎路径
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

    cuda_preprocess_init(640 * 640 * 3);
}


/// @brief 输入图像预处理
/// @param imgsBatch 输入图像（单张）
void detect_nx::preprocess(cv::Mat &imgsBatch)
{
    cuda_batch_preprocess(imgsBatch ,device_buffers[0] ,640 ,640 ,stream_ );
}

bool detect_nx::infer(void)
{

    bool context = detect_nx::context_->enqueueV2((void **)device_buffers, stream_,nullptr);

    CHECK(cudaStreamSynchronize(stream_));
    return context;
}


void detect_nx::postprocess(cv::Mat &imgsBatch)
{

    output_device_host = new float[1800];

    CHECK(cudaMemcpy(output_device_host, device_buffers[1], m_output_area * sizeof(float), cudaMemcpyDeviceToHost));
    
    decode(det , &output_device_host[0] ,0.7 ,1);
    //topk( res, output_device_host, 0, 0.8 ,100);
    volley = get_ball(input_img_ , det);
    //draw_bbox(imgsBatch, res);


}

void detect_nx::show_result(cv::Mat &show_img)
{
    draw_bbox_single(show_img, det);
    

    std::cout << "X : " << volley.center_x 
    << " Y : " << volley.center_y << std::endl;

    std::cout << "deepth : " << volley.deepth << std::endl;
    
    cv::imshow("1" ,show_img);
    cv::waitKey(1);
}