#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"

#include "common_struct.hpp"

class detect_nx
{
public:
    detect_nx();
    ~detect_nx();

public:
    void RT_engine_init(std::string engine_path);
    // void batch_copy(cv::Mat &imgsBatch);
    void preprocesss(cv::Mat &imgsBatch);
    bool infer(void);
    void postprocess(cv::Mat &imgsBatch);

private:
    cv::Mat m_img_src;
    InitParameter m_param;
    AffineMat m_dst2src;
    
    std::vector<Detection> res;

    nvinfer1::Dims m_output_dims;
    int m_output_area;
    int m_total_objects;

    std::vector<unsigned char> engine_data_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;

    int inputIndex;
    int outputIndex;

    // input
    float *input_host;

    float* device_buffers[2];
    // output
    float *output_device_host;

    float *output_device;
    float *output_src_transpose_device;
    float *output_objects_device;
    float *output_objects_host;
    int output_objects_width;
    int *output_idx_device;
    float *output_conf_device;
};
