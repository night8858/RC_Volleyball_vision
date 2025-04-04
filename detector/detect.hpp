#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <mutex>
#include <vector>

#include "common_struct.hpp"

class detect_nx
{
public:
    detect_nx();
    ~detect_nx();

    const int max_size_ = 10;

    std::vector<cv::Mat> input_imgs_;
    std::vector<cv::Mat> input_cam1_imgs_;
    std::vector<cv::Mat> input_cam2_imgs_;

    
    std::mutex img_mutex_;
    std::mutex img_cam1_mutex_;
    std::mutex img_cam2_mutex_;

    int hik_img_flag; // hik相机图像标志位
    int usb_img_flag; // usb相机图像标志位

    cv::Mat input_img_;
    cv::Mat input_cam1_img_;
    cv::Mat input_cam2_img_;

    cv::Mat show_img_;
    cv::Mat show_cam1_img_;
    cv::Mat show_cam2_img_;

    volleyball volley;      // 存储排球数据
    volleyball volley_cam1; // 存储排球数据
    volleyball volley_cam2; // 存储排球数据

    void push_img(cv::Mat &img ,  int cam_id);

public:
    void RT_engine_init(std::string engine_path);
    void preprocess(void);
    bool infer(void);
    void postprocess(void);
    void show_result(cv::Mat &show_img , Detection &det);

private:

    int flag = 2;    //用来判断单路推理双路推理

    cv::Mat m_img_src;
    AffineMat m_dst2src;

    Detection det;
    Detection det1;
    Detection det2;

    std::vector<Detection> res;
    std::vector<Detection> res1;
    std::vector<Detection> res2;

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

    float *device_buffers[2];          //存储输入输出的数据
    // output
    float *output_device_host;

    uint8_t *img_buffer_host_1;
    uint8_t *img_buffer_device_1;

    uint8_t *img_buffer_host_2;
    uint8_t *img_buffer_device_2;


    float *output_device;
    float *output_objects_device;
    float *output_objects_host;

};
